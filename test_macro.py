import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'app'))
from app.macro_data import fetch_macro_indicator
metrics = ['gdp', 'cpi', 'unemployment', 'treasury_10y', 'm2', 'pmi', 'retail_sales', 'consumer_sentiment']
for m in metrics:
    try:
        df = fetch_macro_indicator(m)
        print(f"--- {m} ---")
        if df is not None and not df.empty:
            print(df.tail(3))
        else:
            print("No data")
    except Exception as e:
        print(f"Error {m}: {e}")
