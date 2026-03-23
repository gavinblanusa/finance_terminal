"""
13F Institutional Holdings configuration.

Edit THIRTEENF_INSTITUTIONS here; the app loads this at runtime.
- THIRTEENF_INSTITUTIONS: list of {name, cik} for institutions we track (13F-HR filings).
  CIK is stored as 10-digit zero-padded string for SEC API calls.
"""

# Institutions to track for 13F holdings. CIK zero-padded to 10 digits.
# Organized by strategy to help the agent detect specific fund-type flows.
THIRTEENF_INSTITUTIONS = [
    # --- Macro / Multi-Strategy / Legends ---
    {"name": "Berkshire Hathaway Inc (Buffett)", "cik": "0001067983"},
    {"name": "Duquesne Family Office LLC (Druckenmiller)", "cik": "0001536411"},
    {"name": "Soros Fund Management LLC", "cik": "0001029160"},
    {"name": "Appaloosa LP (Tepper)", "cik": "0001656456"},
    {"name": "Bridgewater Associates, LP (Dalio)", "cik": "0001350694"},
    {"name": "Point72 Asset Management, L.P. (Cohen)", "cik": "0001603466"},
    {"name": "Millennium Management Llc (England)", "cik": "0001258170"},
    {"name": "Citadel Advisors Llc (Griffin)", "cik": "0001423053"},

    # --- Activist / Value / Contrarian ---
    {"name": "Pershing Square Capital Management, L.P. (Ackman)", "cik": "0001336528"},
    {"name": "Scion Asset Management, LLC (Burry)", "cik": "0001649339"},
    {"name": "Elliott Investment Management L.P. (Singer)", "cik": "0001095079"},
    {"name": "Third Point LLC (Loeb)", "cik": "0001040273"},
    {"name": "Baupost Group Llc/ma (Klarman)", "cik": "0001061768"},
    {"name": "Greenlight Capital Inc (Einhorn)", "cik": "0001079114"},
    {"name": "Icahn Carl C", "cik": "0000921669"},
    {"name": "Starboard Value LP (Smith)", "cik": "0001532585"},

    # --- Growth / Tech / Tiger Cubs ---
    {"name": "Tiger Global Management Llc (Coleman)", "cik": "0001167483"},
    {"name": "Coatue Management Llc (Laffont)", "cik": "0001121150"},
    {"name": "Viking Global Investors Lp (Halvorsen)", "cik": "0001103804"},
    {"name": "Lone Pine Capital Llc (Mandel)", "cik": "0001048883"},
    {"name": "Altimeter Capital Management, LP (Gerstner)", "cik": "0001569695"},
    {"name": "Situational Awareness LP", "cik": "0002045724"},
    {"name": "Atreides Management, LP", "cik": "0001777813"},
    {"name": "SCGE MANAGEMENT, L.P.", "cik": "0001537530"},

    # --- Quant / Algo ---
    {"name": "Renaissance Technologies Llc (Simons)", "cik": "0001037389"},
    {"name": "Two Sigma Investments, Lp (Siegel/Overdeck)", "cik": "0001179392"},
    {"name": "AQR Capital Management, LLC (Asness)", "cik": "0001167557"},
    {"name": "Jane Street Group, LLC", "cik": "0001597503"},
]
