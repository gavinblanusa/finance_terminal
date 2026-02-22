"""
Partnerships (8-K Watch) configuration.

Edit WATCH_TICKERS and COUNTERPARTY_INTEREST_NAMES here; the app loads this at runtime.
- WATCH_TICKERS: public company tickers we poll SEC submissions for (8-K filings).
- COUNTERPARTY_INTEREST_NAMES: private/other names we highlight when they appear as counterparty.
"""

# Public companies we watch for 8-K filings (tickers only; de-duplicated and normalized).
WATCH_TICKERS = [
    "NVDA",   # NVIDIA
    "TSLA",   # Tesla
    "APP",    # AppLovin
    "PLTR",   # Palantir
    "AVGO",   # Broadcom
    "ORCL",   # Oracle
    "VST",    # Vistra
    "MSFT",   # Microsoft
    "AMZN",   # Amazon
    "META",   # Meta Platforms
    "TSM",    # Taiwan Semiconductor
    "LLY",    # Eli Lilly
    "COIN",   # Coinbase
    "NVO",    # Novo Nordisk
    "UBER",   # Uber
    "NFLX",   # Netflix
    "ANET",   # Arista Networks
    "ARM",    # Arm Holdings
    "CEG",    # Constellation Energy
    "GOOG",   # Google (Alphabet)
    "ASML",   # ASML
    "MU",     # Micron
    "IBM",    # IBM (normalized from IMB)
    "INTC",   # Intel
    "MELI",   # MercadoLibre
    "HOOD",   # Robinhood
    "SNOW",   # Snowflake
    "RKLB",   # Rocket Lab
    "ASTS",   # AST SpaceMobile
    "SYM",    # Symbiotic
    "LITE",   # Lumentum Holdings
    "NBIS",   # Nebius
    "IREN",   # Iren
    "CRCL",   # Circle
    "AVAV",   # AeroVironment
    "OKLO",   # Oklo
    "HIMS",   # Hims and Hers
    "USAR",   # USA Rare Earth
    "FLY",    # Firefly Aerospace
    "VOYG",   # Voyager Technologies
    "TE",     # T1 Energy
    "FIGR",   # Figure
]

# Private/interest counterparty names we highlight when they appear in 8-K Item 1.01.
COUNTERPARTY_INTEREST_NAMES = [
    "SpaceX",
    "OpenAI",
    "Stripe",
    "Anduril Industries",
    "xAI",
    "Databricks",
    "Anthropic",
    "ByteDance",
    "Perplexity",
    "Cerebras Systems",
    "Groq",
    "Shield AI",
    "Revolut",
    "Canva",
    "Discord",
    "Epic Games",
    "Lambda Labs",
    "Hugging Face",
    "Midjourney",
    "Ramp",
    "Neuralink",
    "Polymarket",
    "Kalshi",
    "Replit",
    "Cursor",
]
