"""
Scraper contract used by the inference API.

Implement `scrape_market_data` so it returns:
{
  "ohlcv": [
    {"date": "...", "bank": "NABIL", "open": ..., "high": ..., "low": ..., "close": ..., "volume": ..., "amount": ...},
    ...
  ],
  "nepse": [
    {"date": "...", "nepse_close": ...},
    ...
  ],
  "fundamentals": {
    "NABIL": {"car": 12.4, "npl": 3.85}
  },
  "policy_rate": 4.5
}
"""


def scrape_market_data(symbol: str, timeframe: str = "1d", lookback_days: int = 320):
    raise NotImplementedError(
        "Implement scrape_market_data in src/scripts/nepse_scraper.py and return "
        "ohlcv/nepse/fundamentals/policy_rate as documented."
    )
