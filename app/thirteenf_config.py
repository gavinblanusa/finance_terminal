"""
13F Institutional Holdings configuration.

Edit THIRTEENF_INSTITUTIONS here; the app loads this at runtime.
- THIRTEENF_INSTITUTIONS: list of {name, cik} for institutions we track (13F-HR filings).
  CIK is stored as 10-digit zero-padded string for SEC API calls.
"""

# Institutions to track for 13F holdings. CIK zero-padded to 10 digits.
THIRTEENF_INSTITUTIONS = [
    {"name": "Berkshire Hathaway Inc", "cik": "0001067983"},
    {"name": "Duquesne Family Office LLC", "cik": "0001536411"},
    {"name": "Soros Fund Management LLC", "cik": "0001029160"},
    {"name": "Pershing Square Capital Management, L.P.", "cik": "0001336528"},
    {"name": "Bridgewater Associates, LP", "cik": "0001350694"},
    {"name": "Scion Asset Management, LLC", "cik": "0001649339"},
    {"name": "Appaloosa LP", "cik": "0001656456"},
    {"name": "Situational Awareness LP", "cik": "0002045724"},
    {"name": "Atreides Management, LP", "cik": "0001777813"},
    {"name": "SCGE MANAGEMENT, L.P.", "cik": "0001537530"},
]
