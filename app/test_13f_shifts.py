import sys
from pathlib import Path

from thirteenf_service import get_13f_holdings_by_quarter, calculate_portfolio_shifts
from thirteenf_config import THIRTEENF_INSTITUTIONS

def test_portfolio_shifts():
    # Let's test Berkshire Hathaway (Buffett) - CIK 0001067983
    # Use 2024 Q4 (latest available usually)
    cik = "0001067983"
    print(f"Fetching 13F Holdings for CIK: {cik} in 2024 Q4...")
    
    data = get_13f_holdings_by_quarter(cik, 2024, 4)
    if not data:
        print("Data not found for 2024 Q4. Trying 2024 Q3...")
        data = get_13f_holdings_by_quarter(cik, 2024, 3)
        
    if not data:
        print("Could not find recent filings for testing.")
        return

    holdings = data.get("holdings", [])
    
    # Slice to top 15 for speed during testing, as yfinance takes time
    top_holdings = sorted(holdings, key=lambda x: x.get("value_thousands", 0), reverse=True)[:15]
    
    print(f"\nAnalyzing Top 15 Holdings for {data.get('filer_name')}...")
    shifts = calculate_portfolio_shifts(top_holdings)
    
    print("\n--- SECTOR WEIGHTS ---")
    for sector, weight in shifts.get("sector_weights", {}).items():
        print(f"{sector}: {weight}%")
        
    print(f"\n--- PORTFOLIO BETA ---")
    print(f"Weighted Beta: {shifts.get('portfolio_beta')}")

if __name__ == "__main__":
    test_portfolio_shifts()
