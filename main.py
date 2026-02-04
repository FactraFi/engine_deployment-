from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import os
import sys

# Add current directory to path so we can import the engine
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Standard import now that the file is renamed to unified_scoring_engine.py
# In this source dir it is 'unified_scoring_engine Final.py' but in Git-Push it is 'unified_scoring_engine.py'
# Logic to handle both for local testing vs deployment structure
try:
    from unified_scoring_engine import UnifiedEngine
except ImportError:
    # Use importlib for local testing with space in filename
    import importlib.util
    file_path = "unified_scoring_engine Final.py"
    module_name = "unified_scoring_engine"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    UnifiedEngine = module.UnifiedEngine

engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global engine
    engine = UnifiedEngine()
    yield
    # Shutdown
    pass

# Fix 1: Single app creation with title
app = FastAPI(title="FactraFi Scoring Service", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0.1"}

from pydantic import BaseModel
from typing import List, Optional

class CalculateRequest(BaseModel):
    tickers: Optional[List[str]] = None

@app.post("/calculate-scores")
async def calculate_scores(request: CalculateRequest = None):
    """Calculate V-Scores and RETURN them (Edge Function handles DB write)"""
    try:
        if not engine:
            raise HTTPException(status_code=500, detail="Engine not initialized")
        
        # Calculate scores (run all universes)
        # Note: We currently ignore request.tickers and run the designated universes
        results = await engine.run()
        
        # Filter if tickers provided (optional optimization)
        if request and request.tickers:
            results = [r for r in results if r['Ticker'] in request.tickers]
        
        # Return results directly
        return {
            "success": True, 
            "scores": results,
            "regime": "Multi-Strategy" # Simplified as engine handles multiple regimes
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
