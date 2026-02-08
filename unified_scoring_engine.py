"""
FactraFi Unified Scoring Engine (V2)
------------------------------------
A self-contained, audit-ready script for the FactraFi Scoring Engine.
Implements:
1. Data Ingestion (Alpha Vantage) with Interactive API Key Input
2. V2 Pillar Calculations (Fundamentals, Momentum, Risk, Sentiment)
3. Regime Detection (XGBoost + SHAP)
4. Portfolio Optimization (CVXPY)
5. Cross-Sectional Z-Scoring & Sector Neutralization
6. Backtesting & Reporting

Dependencies:
- pandas, numpy, xgboost, shap, cvxpy, httpx, requests
- pip install pandas numpy xgboost shap cvxpy httpx requests

Usage:
python unified_scoring_engine.py
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

# Third-party imports
try:
    import pandas as pd
    import numpy as np
    import httpx
    import xgboost as xgb
    import shap
    import cvxpy as cp
except ImportError as e:
    print(f"CRITICAL ERROR: Missing dependency: {e}")
    print("Please install required packages: pip install pandas numpy xgboost shap cvxpy httpx requests")
    sys.exit(1)

# Set Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FactraFiEngine")

# --- configuration & Constants ---
DATA_DIR = Path("data_unified")
PRICES_DIR = DATA_DIR / "prices"
FINANCIALS_DIR = DATA_DIR / "financials"
RESULTS_DIR = DATA_DIR / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PRICES_DIR.mkdir(parents=True, exist_ok=True)
FINANCIALS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Multi-Universe Configuration (Bucketed Scoring)
UNIVERSES = {
    "Growth_Tech":     ["NVDA", "AMD", "MSFT", "PLTR", "QCOM", "AVGO", "CRM", "NOW", "ADBE"],
    "Value_Tech":      ["IBM", "CSCO", "INTC", "TXN", "ORCL", "SNPS", "LRCX"],
    "Defensive":       ["PG", "KO", "JNJ", "PFE", "WM", "VZ", "MRK", "UNH"],
    "Cyclical":        ["CAT", "DE", "F", "GM", "XOM", "CVX", "COP", "SLB"],
    "Financials":      ["JPM", "BAC", "V", "MA", "GS", "MS", "BLK", "AXP"],
    "Consumer_Growth": ["AMZN", "TSLA", "NFLX", "SBUX", "NKE", "LULU", "CMG"]
}

# --- 1. Pillar Calculator Class (V2 Logic) ---
# --- 1. Pillar Calculator Class (V2 Logic) ---
class PillarCalculator:
    """Calculates individual pillar scores (0-100) using V2 definitions."""
    
    SECTOR_MAP = {
        "TECHNOLOGY": "Technology", "FINANCIAL": "Financials", "HEALTHCARE": "Healthcare",
        "CONSUMER_DISCRETIONARY": "Consumer Discretionary", "CONSUMER_STAPLES": "Consumer Staples",
        "ENERGY": "Energy", "UTILITIES": "Utilities", "INDUSTRIALS": "Industrials",
        "MATERIALS": "Materials", "REAL_ESTATE": "Real Estate", "COMMUNICATION_SERVICES": "Communication Services"
    }
    
    def __init__(self, api_key: str):
        self.api_key = api_key

    def calculate_historical_pillars(self, price_history: pd.DataFrame, financials: Dict, date_str: str, style: str = "General") -> Optional[Dict]:
        """Calculate Point-in-Time scores with Style-Specific Logic."""
        if date_str not in price_history.index:
            try:
                dt = pd.to_datetime(date_str)
                avail = pd.to_datetime(price_history.index)
                idx = avail.searchsorted(dt)
                if idx > 0: date_str = str(price_history.index[idx-1])
                else: return None
            except: return None
            
        history = price_history.loc[:date_str]
        if len(history) < 50: return None
        
        # --- 1. Momentum (Style: Growth_Tech uses Volume) ---
        momentum = self._compute_momentum(history, style)
        
        # --- 2. Risk (Style: Cyclical uses Beta Proxy) ---
        risk = self._compute_risk(history, style)
        
        # --- 3. Fundamentals (Style: Financials=P/B, Consumer=RevGrowth) ---
        fund_data = self._get_pit_fundamentals(financials, date_str)
        curr_price = history['adjusted_close'].iloc[-1]
        fundamentals = self._compute_fundamentals(fund_data, curr_price, style)
        
        # --- 4. Sentiment (Style: Defensive checks Payout Ratio) ---
        sentiment = self._compute_sentiment(fund_data, style) 
        
        return {
            "fundamentals": fundamentals if not np.isnan(fundamentals) else 50,
            "momentum": momentum if not np.isnan(momentum) else 50,
            "risk": risk if not np.isnan(risk) else 50,
            "sentiment": sentiment if not np.isnan(sentiment) else 50
        }

    def _compute_momentum(self, history, style):
        # Base: RSI + MACD
        delta = history['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        curr_rsi = rsi.iloc[-1]
        
        ema_12 = history['close'].ewm(span=12).mean()
        ema_26 = history['close'].ewm(span=26).mean()
        macd = (ema_12 - ema_26).iloc[-1]
        
        rsi_s = curr_rsi if not np.isnan(curr_rsi) else 50.0 
        macd_s = max(0, min(100, 50 + (macd * 10)))
        
        base_score = (rsi_s * 0.5) + (macd_s * 0.5)
        
        # Tech Fix: Volume Trend
        if style == "Growth_Tech":
            vol_sma10 = history['volume'].rolling(10).mean().iloc[-1]
            vol_sma50 = history['volume'].rolling(50).mean().iloc[-1]
            vol_score = 60 if vol_sma10 > vol_sma50 else 40
            return (base_score * 0.7) + (vol_score * 0.3)
            
        return base_score

    def _compute_risk(self, history, style):
        ret = history['close'].pct_change()
        vol = ret.std() * np.sqrt(252)
        
        # Cyclical Fix: Less penalizing on Volatility, favor relative strength?
        # User asked for Beta. Approx Beta: Volatility relative to market (using 0.15 const as SPY vol)
        if style == "Cyclical":
            # If Beta is high (>1.5), penalize strongly. If Beta ~1, neutral.
            # Approx Beta = StockVol / 0.15
            beta_proxy = vol / 0.15
            if beta_proxy > 2.0: return 20  # High Risk
            elif beta_proxy < 0.8: return 40 # Dead money?
            else: return 80 # Moving with market
            
        risk_score = max(0, 100 - (vol * 200)) # Default
        return risk_score

    def _compute_fundamentals(self, data, price, style):
        scores = []
        shares = self._safe_float(data.get('commonStockSharesOutstanding'))
        total_eq = self._safe_float(data.get('totalShareholderEquity'))
        
        # --- Financials Fix: Price to Book ---
        if style == "Financials":
            if price and shares and total_eq and total_eq > 0:
                bps = total_eq / shares
                pb = price / bps
                # P/B < 1 is great (100), > 3 is expensive (0)
                scores.append(100 if pb < 1 else max(0, 100 - ((pb - 1) * 33)))
            return np.mean(scores) if scores else 50.0

        # --- Standard EV/EBITDA ---
        debt = self._safe_float(data.get('shortLongTermDebtTotal')) or self._safe_float(data.get('totalLiabilities'))
        cash = self._safe_float(data.get('cashAndCashEquivalentsAtCarryingValue')) or self._safe_float(data.get('cashAndShortTermInvestments'))
        ebitda = self._safe_float(data.get('ebitda')) or self._safe_float(data.get('netIncome')) # Fallback for banks if missed
        
        if price and shares and debt is not None and cash is not None and ebitda and ebitda != 0:
            ev = (price * shares) + debt - cash
            ratio = ev / ebitda
            if ratio <= 10: s = 100
            elif ratio >= 25: s = 0
            else: s = 100 - ((ratio - 10)/15 * 100)
            scores.append(s)
            
        # --- D/E ---
        liab = self._safe_float(data.get('totalLiabilities'))
        if liab is not None and total_eq and total_eq != 0:
            ratio = liab / total_eq
            s = max(0, 100 - (ratio * 33))
            scores.append(s)

        # --- Consumer Fix: Revenue Growth ---
        if style == "Consumer_Growth":
            rev_now = self._safe_float(data.get('totalRevenue'))
            # Need previous year/qtr. Simplified: Use 'QuarterlyRevenueGrowthYOY' if avail in overview fallback
            # Or assume current data dict has 'totalRevenue' of latest Q. 
            # Impl limitation: We need historical growth. Data dict is a single PIT snapshot.
            # We will use ROE as proxy for Growth Quality if growth history missing, 
            # Or rely on Overview's Growth metrics if we trust they aren't too forward looking
            pass # Keep standard + ROE later?

        return np.mean(scores) if scores else 50.0

    def _compute_sentiment(self, data, style):
        # Base: Div Yield
        dy = self._safe_float(data.get('DividendYield'))
        base_s = min(100, 50 + (dy * 500)) if dy else 50.0
        
        # Defensive Fix: Payout Ratio
        if style == "Defensive":
            payout = self._safe_float(data.get('PayoutRatio')) # From Overview
            if payout:
                if payout > 1.0: return 20 # Distressed / Cut risk
                if payout > 0.8: return 40 # Warning
                # Else yield is safe
            
        return base_s

    def _get_pit_fundamentals(self, financials: Dict, date_str: str) -> Dict:
        """Get latest report before date with +2 day lag."""
        target = pd.to_datetime(date_str)
        # Better finding logic
        inc = self._find_report(financials.get('income_statement', []), target)
        bal = self._find_report(financials.get('balance_sheet', []), target)
        ovr = financials.get('overview', {})
        return {**ovr, **inc, **bal}

    def _find_report(self, reports, target_date):
        for r in reports:
            rd = pd.to_datetime(r['fiscalDateEnding']) + pd.Timedelta(days=2)
            if rd <= target_date: return r
        return {}

    def _safe_float(self, v):
        try: return float(v)
        except: return None

# --- 2. Model Trainer (XGBoost) ---
class RegimeDetector:
    """Trains XGBoost to determine market regime (Feature Importance)."""
    def train_and_analyze(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        logger.info(f"Training XGBoost on {len(X)} historical samples...")
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, n_jobs=-1
        )
        model.fit(X, y)
        
        # SHAP Analysis
        explainer = shap.TreeExplainer(model)
        sample = X.iloc[:1000] if len(X) > 1000 else X
        shap_values = explainer.shap_values(sample)
        
        final_signals = {}
        feature_names = X.columns.tolist()
        
        for i, feature in enumerate(feature_names):
            vals = sample.iloc[:, i].values
            shaps = shap_values[:, i]
            
            # Signed Importance
            corr = np.corrcoef(vals, shaps)[0, 1] if np.std(vals) > 0 else 0
            importance = np.mean(np.abs(shaps))
            
            if np.isnan(corr): corr = 0
            
            # V2 Logic: Toxic factor check
            if corr < -0.05:
                final_signals[feature] = 0.0 # Toxic
            else:
                final_signals[feature] = float(importance)
        
        total = sum(final_signals.values())
        return {k: v/total for k, v in final_signals.items()} if total > 0 else {k: 0.25 for k in feature_names}

# --- 3. Optimizer (CVXPY) ---
class PortfolioOptimizer:
    """Optimizes weights using QP."""
    GUARDRAILS = {
        "fundamentals": (0.1, 0.4), "momentum": (0.1, 0.5),
        "risk": (0.1, 0.5), "sentiment": (0.05, 0.25)
    }
    
    def optimize(self, signals: Dict[str, float]) -> Dict[str, float]:
        if "cvxpy" not in sys.modules:
            logger.warning("CVXPY missing. Using Defaults.")
            return signals # Fallback
            
        w = cp.Variable(4)
        targets = np.array([signals.get(k, 0.25) for k in ["fundamentals", "momentum", "risk", "sentiment"]])
        targets = targets / targets.sum()
        
        obj = cp.Minimize(cp.sum_squares(w - targets))
        constr = [
            cp.sum(w) == 1, w >= 0,
            w[0] >= self.GUARDRAILS["fundamentals"][0], w[0] <= self.GUARDRAILS["fundamentals"][1],
            w[1] >= self.GUARDRAILS["momentum"][0], w[1] <= self.GUARDRAILS["momentum"][1],
            w[2] >= self.GUARDRAILS["risk"][0], w[2] <= self.GUARDRAILS["risk"][1],
            w[3] >= self.GUARDRAILS["sentiment"][0], w[3] <= self.GUARDRAILS["sentiment"][1]
        ]
        
        prob = cp.Problem(obj, constr)
        prob.solve()
        
        res = w.value
        res = res / res.sum()
        return {
            "fundamentals": round(res[0], 3), "momentum": round(res[1], 3),
            "risk": round(res[2], 3), "sentiment": round(res[3], 3)
        }

# --- 4. Main Engine ---
class UnifiedEngine:
    def __init__(self):
        self.api_key = self._get_api_key()
        self.pillar_calc = PillarCalculator(self.api_key)
        self.detector = RegimeDetector()
        self.optimizer = PortfolioOptimizer()
        
    def _get_api_key(self):
        return os.getenv("ALPHA_VANTAGE_API_KEY")

    async def fetch_data(self, tickers):
        if not self.api_key:
             raise ValueError("ALPHA_VANTAGE_API_KEY environment variable required")
             
        logger.info(f"Fetching data for {len(tickers)} unique tickers...")
        client = httpx.AsyncClient()
        for t in tickers:
            p_file = PRICES_DIR / f"{t}.parquet"
            f_file = FINANCIALS_DIR / f"{t}.json"
            
            if not p_file.exists():
                logger.info(f"Downloading Price: {t}")
                url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={t}&apikey={self.api_key}&outputsize=full&datatype=csv"
                try:
                    r = await client.get(url, timeout=10.0)
                    from io import StringIO
                    df = pd.read_csv(StringIO(r.text))
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                        df.sort_index(inplace=True)
                        df.to_parquet(p_file)
                    else:
                        logger.error(f"Invalid price data for {t}: {r.text[:50]}")
                except Exception as e:
                    logger.error(f"Failed to fetch price {t}: {e}")
            
            if not f_file.exists():
                logger.info(f"Downloading Financials: {t}")
                try:
                    inc = (await client.get(f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={t}&apikey={self.api_key}", timeout=10.0)).json()
                    bal = (await client.get(f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={t}&apikey={self.api_key}", timeout=10.0)).json()
                    ovr = (await client.get(f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={t}&apikey={self.api_key}", timeout=10.0)).json()
                    
                    data = {
                        "income_statement": inc.get("quarterlyReports", []),
                        "balance_sheet": bal.get("quarterlyReports", []),
                        "overview": ovr
                    }
                    with open(f_file, "w") as f:
                        json.dump(data, f)
                except Exception as e:
                    logger.error(f"Failed to fetch financials {t}: {e}")
            
            await asyncio.sleep(0.5) # Rate limit
        await client.aclose()

    async def store_to_supabase(self, results: List[Dict]):
        """Store calculated scores to Supabase stock_scores table"""
        from supabase import create_client
        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        
        if not url or not key:
            logger.warning("Supabase credentials missing. Skipping DB upload.")
            return

        supabase = create_client(url, key)
        
        for row in results:
            # Parse weights/scores for storage
            try:
                # Weights are stored as string in result, parsed here if needed, 
                # but map requests raw pillar scores (0-100) to columns: fundamentals, momentum, risk, sentiment
                # The dictionary returned by engine has 'Total Score', 'Style', 'Rank IC'
                # But it doesn't pass the raw pillar scores in the final dict?
                # Wait, the `master_results` append currently only has: Ticker, Style, Total Score, Rank IC, Weights.
                # I need to modify the run loop to include pillar scores in the output dict.
                pass 
            except:
                pass

            data = {
                "ticker": row["Ticker"],
                "name": row["Ticker"], # Overview fetch expensive here, using ticker as fallback
                "sector": row.get("Style", "Unknown"), # Using Style as Sector proxy or passed sector?
                "v_score": int(float(row["Total Score"])),
                "fundamentals": int(float(row.get("fundamentals", 50))),
                "momentum": int(float(row.get("momentum", 50))),
                "risk": int(float(row.get("risk", 50))),
                "sentiment": int(float(row.get("sentiment", 50))),
                "why_explanation": f"Style: {row['Style']}. Rank IC: {row['Rank IC']}",
                "updated_at": datetime.now().isoformat()
            }
            try:
                supabase.table("stock_scores").upsert(data).execute()
            except Exception as e:
                logger.error(f"Failed to upsert {row['Ticker']}: {e}")

    def _generate_explanation(self, ticker: str, style: str, pillars: Dict, weights: Dict, total_score: float, rank_ic: float) -> str:
        """Generate natural language explanation for V-Score"""
        # Find strongest and weakest pillars
        pillar_scores = {
            "fundamentals": pillars["fundamentals"],
            "momentum": pillars["momentum"],
            "risk": pillars["risk"],
            "sentiment": pillars["sentiment"]
        }
        sorted_pillars = sorted(pillar_scores.items(), key=lambda x: x[1], reverse=True)
        strongest = sorted_pillars[0]
        weakest = sorted_pillars[-1]

        # Overall conviction
        if total_score >= 80:
            overall = "Strong conviction"
        elif total_score >= 60:
            overall = "Moderate conviction"
        elif total_score >= 40:
            overall = "Neutral outlook"
        else:
            overall = "Cautious stance"

        # Strength descriptions
        strength_map = {
            "fundamentals": "solid fundamentals",
            "momentum": "strong price momentum", 
            "risk": "favorable risk profile",
            "sentiment": "positive market sentiment"
        }
        weakness_map = {
            "fundamentals": "weaker fundamentals",
            "momentum": "momentum concerns",
            "risk": "elevated risk levels", 
            "sentiment": "cautious sentiment"
        }

        strength_desc = strength_map.get(strongest[0], strongest[0])
        weakness_desc = weakness_map.get(weakest[0], weakest[0])

        # Most influential pillar
        contributions = {k: weights.get(k, 0.25) * v for k, v in pillar_scores.items()}
        most_influential = max(contributions, key=contributions.get)
        weight_pct = int(weights.get(most_influential, 0.25) * 100)

        return f"{overall}. {style} stock showing {strength_desc} but {weakness_desc}. {most_influential.capitalize()} driving {weight_pct}% of score."

    async def run(self):
        # 1. Gather all unique tickers
        all_tickers = list(set([t for tickers in UNIVERSES.values() for t in tickers]))
        
        # 2. Fetch all data
        await self.fetch_data(all_tickers)
        
        master_results = []
        
        print("\n" + "="*80)
        print("STARTING MULTI-UNIVERSE (STYLE BUCKETED) SCORING")
        print("="*80)
        
        # 3. Iterate through Buckets
        for style, tickers in UNIVERSES.items():
            print(f"\n>>> PROCESSING STYLE BUCKET: {style} ({len(tickers)} Stocks)")
            
            # --- Build History (Subset) ---
            history_rows = []
            for t in tickers:
                try:
                    price = pd.read_parquet(PRICES_DIR / f"{t}.parquet")
                    with open(FINANCIALS_DIR / f"{t}.json") as f: fin = json.load(f)
                    
                    dates = pd.date_range(end=price.index.max(), periods=52, freq='W-FRI')
                    for d in dates:
                        d_str = d.strftime("%Y-%m-%d")
                        scores = self.pillar_calc.calculate_historical_pillars(price, fin, d_str, style=style)
                        if not scores: continue
                        
                        try:
                            idx = price.index.searchsorted(d)
                            curr = price.iloc[idx]['adjusted_close']
                            fut = price.iloc[idx+5]['adjusted_close']
                            ret = (fut - curr)/curr
                        except: continue
                        
                        history_rows.append({
                            "ticker": t, "date": d, "sector": fin.get('overview', {}).get('Sector', 'Unknown'),
                            **scores, "fwd_ret": ret
                        })
                except Exception as e:
                    continue
            
            df = pd.DataFrame(history_rows)
            if df.empty:
                logger.warning(f"No history for {style}. Skipping.")
                continue
            
            # --- Z-Score & Neutralization (Style Relative) ---
            cols = ['fundamentals', 'momentum', 'risk', 'sentiment']
            # Global Z within this bucket = Style Z
            df[cols] = df.groupby('date')[cols].transform(lambda x: (x - x.mean()) / x.std() if x.std()!=0 else 0)
            df[cols] = df[cols].fillna(0)
            
            # Optimization
            signals = self.detector.train_and_analyze(df[cols], df['fwd_ret'])
            weights = self.optimizer.optimize(signals)
            print(f"   Detected Weights: {weights}")
            
            # Validation Metric for this bucket
            df['score'] = (df['fundamentals']*weights['fundamentals'] + df['momentum']*weights['momentum'] + 
                           df['risk']*weights['risk'] + df['sentiment']*weights['sentiment'])
            ic = df[['score', 'fwd_ret']].corr(method='spearman').iloc[0,1]
            print(f"   Bucket Rank IC: {ic:.4f}")
            
            # Scoring Current
            latest_date = df['date'].max()
            latest_df = df[df['date'] == latest_date].copy()
            
            for idx, row in latest_df.iterrows():
                z = (row['fundamentals'] * weights['fundamentals'] +
                     row['momentum'] * weights['momentum'] +
                     row['risk'] * weights['risk'] +
                     row['sentiment'] * weights['sentiment'])
                score = (1 / (1 + np.exp(-z))) * 100
                
                def z_to_100(v):
                    return max(0, min(100, 50 + (v * 20)))

                pillars_100 = {
                    "fundamentals": z_to_100(row['fundamentals']),
                    "momentum": z_to_100(row['momentum']),
                    "risk": z_to_100(row['risk']),
                    "sentiment": z_to_100(row['sentiment'])
                }

                explanation = self._generate_explanation(
                    ticker=row['ticker'],
                    style=style,
                    pillars=pillars_100,
                    weights=weights,
                    total_score=round(score, 1),
                    rank_ic=round(ic, 4)
                )

                master_results.append({
                    "Ticker": row['ticker'],
                    "Style": style,
                    "Total Score": round(score, 1),
                    "Rank IC": round(ic, 4),
                    "Weights": str(weights),
                    "fundamentals": pillars_100["fundamentals"],
                    "momentum": pillars_100["momentum"],
                    "risk": pillars_100["risk"],
                    "sentiment": pillars_100["sentiment"],
                    "Explanation": explanation
                })
        
        # 4. Final Aggregation
        final_df = pd.DataFrame(master_results).sort_values("Total Score", ascending=False)
        
        print("\n" + "="*80)
        print("FINAL MASTER LEADERBOARD (Aggregated from 4 Models)")
        print("="*80)
        print(final_df[["Ticker", "Style", "Total Score", "Rank IC", "Weights", "Explanation"]].to_string(index=False))
        
        return final_df.to_dict(orient='records')

if __name__ == "__main__":
    engine = UnifiedEngine()
    engine.run()
