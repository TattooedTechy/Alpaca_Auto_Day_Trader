# symbol_loader.py
import requests
import pandas as pd
from yahoo_fin import stock_info as si

def get_nasdaq100_tickers():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(requests.get(url).text)
    for table in tables:
        if "Ticker" in table.columns:
            return [t.strip().replace(".", "-") for t in table["Ticker"]]
    raise ValueError("Could not find the Nasdaq-100 table on Wikipedia")

def build_symbols(raw_symbols: list[str]) -> list[str]:
    symbols: list[str] = []
    for s in raw_symbols:
        key = s.strip().upper()
        if key == "RUN":
            continue
        try:
            if key == "DOW":
                symbols.extend(si.tickers_dow())
            elif key == "SP500":
                symbols.extend(si.tickers_sp500())
            elif key == "NASDAQ100":
                symbols.extend(get_nasdaq100_tickers())
            elif key == "NASDAQ":
                symbols.extend(si.tickers_nasdaq())
            else:
                symbols.append(key)
        except Exception as e:
            print(f"⚠️ Failed to fetch {key}: {e}")
            symbols.append(key)

    # dedupe & only alphanumeric
    return list(dict.fromkeys(sym for sym in symbols if sym.isalnum()))
