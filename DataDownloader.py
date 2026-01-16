'''
currently used trading pairs:

AAVEUSDC
BCHUSDC
BNBUSDC
BTCUSDC
ETHUSDC
SOLUSDC
ZENUSDC

author: T. Kowalski, with help of ChatGPT & Microsoft Copilot - thanks to them code looks much nicer.
'''


import os
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta, timezone
import time

# -----------------------------
# Config
# -----------------------------
SYMBOLS = [
    "BTCUSDC", "ETHUSDC", "BNBUSDC", "BCHUSDC",
    "ZECUSDC", "SOLUSDC", "ZENUSDC", "AAVEUSDC"
]

INTERVALS = {
    "5m":  Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "30m": Client.KLINE_INTERVAL_30MINUTE,
    "1h":  Client.KLINE_INTERVAL_1HOUR,
    "4h":  Client.KLINE_INTERVAL_4HOUR,
}

DAYS = 60
EXCHANGE = "binance"

API_KEY = ""      # optional
API_SECRET = ""

# -----------------------------
# Time range (UTC)
# -----------------------------
end_date = datetime.now(timezone.utc)
start_date = end_date - timedelta(days=DAYS)

# -----------------------------
# Binance client
# -----------------------------
client = Client(API_KEY, API_SECRET)

# -----------------------------
# Save path
# -----------------------------
base_dir = "all_data"
save_path = os.path.join(base_dir, EXCHANGE)
os.makedirs(save_path, exist_ok=True)

# -----------------------------
# Functions
# -----------------------------
def interval_to_timedelta(interval):
    mapping = {
        Client.KLINE_INTERVAL_5MINUTE: timedelta(minutes=5),
        Client.KLINE_INTERVAL_15MINUTE: timedelta(minutes=15),
        Client.KLINE_INTERVAL_30MINUTE: timedelta(minutes=30),
        Client.KLINE_INTERVAL_1HOUR: timedelta(hours=1),
        Client.KLINE_INTERVAL_4HOUR: timedelta(hours=4),
    }
    return mapping[interval]

def download_klines(symbol, interval, start_time, end_time):
    """Download all klines in the range by batching 1000 per request."""
    klines = []
    temp_start = start_time

    while temp_start < end_time:
        try:
            batch = client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=int(temp_start.timestamp() * 1000),
                endTime=int(end_time.timestamp() * 1000),
                limit=1000
            )
        except Exception as e:
            print(f"Error fetching {symbol} {interval}: {e}")
            time.sleep(1)  # backoff
            continue

        if not batch:
            break

        klines.extend(batch)

        # Move to next candle (avoid overlap)
        last_open_time = batch[-1][0]  # in ms
        temp_start = datetime.fromtimestamp(last_open_time / 1000, tz=timezone.utc) + interval_to_timedelta(interval)

        # Stop if API returns fewer than 1000 candles (end of data)
        if len(batch) < 1000:
            break

    return klines

def klines_to_df(klines, interval_name):
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["interval"] = interval_name
    df = df[["timestamp", "interval", "open", "high", "low", "close", "volume"]]
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

# -----------------------------
# Main
# -----------------------------
def main():
    for symbol in SYMBOLS:
        print(f"\nDownloading {symbol}...")
        all_dfs = []

        for interval_name, interval_code in INTERVALS.items():
            print(f"  → {interval_name}")
            klines = download_klines(symbol, interval_code, start_date, end_date)

            if not klines:
                print(f"    No data")
                continue

            df = klines_to_df(klines, interval_name)
            print(f"    Downloaded {len(df)} rows")
            all_dfs.append(df)

        if not all_dfs:
            print(f"No data for {symbol}")
            continue

        # Combine all intervals, no strict intersection yet
        final_df = pd.concat(all_dfs).drop_duplicates(subset=["timestamp", "interval"])
        final_df = final_df.sort_values(["interval", "timestamp"]).reset_index(drop=True)

        filename = f"{symbol.lower()}_multi_interval_{start_date.date()}_{end_date.date()}.txt"
        filepath = os.path.join(save_path, filename)
        final_df.to_csv(filepath, sep="\t", index=False, header=False)
        print(f"Saved {symbol} → {filepath} ({len(final_df)} rows)")

if __name__ == "__main__":
    main()
