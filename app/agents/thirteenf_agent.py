import os
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# We use the openai package which supports OpenAI, Groq, and other OpenAI-compatible APIs.
# The user can swap out the base_url or just use OPENAI_API_KEY/GROQ_API_KEY.
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# We default to Groq for speed if the key is present, otherwise fallback to OpenAI
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if GROQ_API_KEY:
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )
    # Using Groq's GPT-OSS 120B reasoning model as it excels at step-by-step logic and macro deduction.
    DEFAULT_MODEL = "openai/gpt-oss-120b" 
else:
    client = OpenAI(api_key=OPENAI_API_KEY)
    DEFAULT_MODEL = "gpt-4o-mini"

# --- Structured Output Models ---

class FundPostureReport(BaseModel):
    risk_appetite: str = Field(description="Aggressive, Neutral, or Defensive based on beta and sector rotations.")
    sector_rotation_logic: str = Field(description="A 2-3 sentence explanation of the macro logic behind their sector shifts.")
    top_thesis: str = Field(description="A 3-sentence rationale explaining the thesis for their largest new or increased position, citing recent news, macro data, and valuation.")

class DivergenceBrief(BaseModel):
    overall_sentiment: str = Field(description="The aggregated risk-on or risk-off posture across all tracked funds.")
    consensus_buys: str = Field(description="A 2-sentence summary of what the smartest funds universally agree is underpriced.")
    battlegrounds: str = Field(description="A 3-sentence summary highlighting tickers or sectors where funds are in heavy disagreement (e.g., Value buying while Quants sell).")

# --- Agent Skills ---

def generate_fund_posture_report(filer_name: str, sector_weights: Dict[str, float], portfolio_beta: float, top_holdings: List[Dict], recent_news: str) -> Optional[Dict[str, Any]]:
    """
    Skill A (Fund Profiler): Generates a synthesis for a single fund.
    """
    if not GROQ_API_KEY and not OPENAI_API_KEY:
        print("[Agent] Skipping LLM generation. No GROQ_API_KEY or OPENAI_API_KEY found.")
        return None
        
    # DeepSeek R1 performs best when ALL instructions are in the user prompt.
    user_prompt = f"""
    You are an elite quantitative analyst profiling a hedge fund's latest 13F filing.
    You output JSON conforming to the requested schema. Be extremely direct, insightful,
    and focus on the underlying macro thesis that drove these trades.

    Analyze the recent quarter 13F filing for the following hedge fund:
    Fund: {filer_name}
    
    Portfolio Risk Profile:
    - Weighted Beta: {portfolio_beta}
    - Sector Weights: {json.dumps(sector_weights, indent=2)}
    
    Top Holdings & Moves:
    {json.dumps(top_holdings[:5], indent=2)}
    
    Recent News Context on their top Buys:
    {recent_news}
    
    Generate the Fund Posture Report.
    """
    
    try:
        kwargs = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": user_prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.6,
        }
        
        # If using DeepSeek R1 on Groq, we pass reasoning_format via extra_body
        if "deepseek" in DEFAULT_MODEL or "qwen" in DEFAULT_MODEL or "gpt-oss" in DEFAULT_MODEL:
            kwargs["extra_body"] = {"reasoning_format": "parsed"}
            
        completion = client.chat.completions.create(**kwargs)
        # Parse the output
        result_text = completion.choices[0].message.content
        return json.loads(result_text)
    except Exception as e:
        print(f"[Agent] Failed to generate Fund Posture Report: {e}")
        return None

def generate_divergence_brief(aggregated_shifts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Skill B (Consensus Engine): Generates a cross-fund summary of disagreements and overlaps.
    """
    if not GROQ_API_KEY and not OPENAI_API_KEY:
        print("[Agent] Skipping LLM generation. No GROQ_API_KEY or OPENAI_API_KEY found.")
        return None
        
    # DeepSeek R1 performs best when ALL instructions are in the user prompt.
    user_prompt = f"""
    You are the Chief Investment Officer (CIO) AI for a multi-manager hedge fund.
    You analyze the aggregated 13F flow data of the top 30 funds world-wide.
    Identify the consensus, but more importantly, identify the battlegrounds where funds disagree.
    Output JSON.

    Aggregated 13F Data:
    {json.dumps(aggregated_shifts, indent=2)}
    
    Generate the Divergence Brief.
    """
    
    try:
        kwargs = {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": user_prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.6,
        }
        
        # If using DeepSeek R1 on Groq, we pass reasoning_format via extra_body
        if "deepseek" in DEFAULT_MODEL or "qwen" in DEFAULT_MODEL or "gpt-oss" in DEFAULT_MODEL:
            kwargs["extra_body"] = {"reasoning_format": "parsed"}
            
        completion = client.chat.completions.create(**kwargs)
        result_text = completion.choices[0].message.content
        return json.loads(result_text)
    except Exception as e:
        print(f"[Agent] Failed to generate Divergence Brief: {e}")
        return None

# --- Orchestrators ---

def analyze_fund_posture(cik: str, year: int, quarter: int) -> Optional[Dict[str, Any]]:
    """
    Orchestrates Skill A: 
    1. Fetches holdings
    2. Calculates shifts
    3. Fetches news for top 3 buys
    4. Calls LLM
    """

    from thirteenf_service import get_13f_holdings_by_quarter, calculate_portfolio_shifts
    from openbb_adapter import fetch_news_openbb

    data = get_13f_holdings_by_quarter(cik, year, quarter)
    if not data:
        print(f"[Agent] No data found for CIK {cik} in {year} Q{quarter}")
        return None
        
    holdings = data.get("holdings", [])
    shifts = calculate_portfolio_shifts(holdings)
    
    top_holdings = shifts.get("holdings_enriched", [])[:5]
    
    # Grab news for top 3
    recent_news = ""
    for h in top_holdings[:3]:
        ticker = h.get("ticker", "")
        if ticker and ticker != "UNKNOWN":
            news_items = fetch_news_openbb(ticker, limit=3)
            recent_news += f"\\n--- {ticker} News ---\\n"
            for n in news_items:
                recent_news += f"- {n.get('headline')} ({n.get('datetime')})\\n"
                
    # Define JSON schema explicitly in prompt so standard models format it right
    schema_prompt = (
        "\\nReturn a strict JSON object with these keys:\\n"
        "- risk_appetite: string\\n"
        "- sector_rotation_logic: string\\n"
        "- top_thesis: string"
    )
                
    return generate_fund_posture_report(
        filer_name=data.get("filer_name", "Unknown Fund"),
        sector_weights=shifts.get("sector_weights", {}),
        portfolio_beta=shifts.get("portfolio_beta", 1.0),
        top_holdings=top_holdings,
        recent_news=recent_news + schema_prompt
    )

def analyze_smart_money_consensus(ciks: List[str], year: int, quarter: int) -> Optional[Dict[str, Any]]:
    """
    Orchestrates Skill B:
    1. Grabs shifts for all funds
    2. Calls LLM
    """

    from thirteenf_service import get_13f_holdings_by_quarter, calculate_portfolio_shifts
    
    aggregated_shifts = {}
    for cik in ciks:
        data = get_13f_holdings_by_quarter(cik, year, quarter)
        if data:
            shifts = calculate_portfolio_shifts(data.get("holdings", []))
            # Just grab top 3 sectors to save token space
            top_sectors = dict(list(shifts.get("sector_weights", {}).items())[:3])
            aggregated_shifts[data.get("filer_name", cik)] = {
                "beta": shifts.get("portfolio_beta"),
                "top_sectors": top_sectors,
                "top_holdings": [h.get("ticker") for h in shifts.get("holdings_enriched", [])[:5] if h.get("ticker") != "UNKNOWN"]
            }
            
    schema_prompt = (
        "\\nReturn a strict JSON object with these keys ONLY:\\n"
        "- overall_sentiment: string\\n"
        "- consensus_buys: string\\n"
        "- battlegrounds: string"
    )
    
    payload = {
        "data": aggregated_shifts,
        "instructions": schema_prompt
    }
            
    return generate_divergence_brief(payload)
