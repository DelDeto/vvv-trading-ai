import requests
import pandas as pd

BASE_URL = "https://fapi.binance.com"
SYMBOL = "VVVUSDT"


def get_klines(interval="4h", limit=200):

    url = f"{BASE_URL}/fapi/v1/klines"

    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "limit": limit
    }

    print(f"Requesting Binance: {SYMBOL} {interval}")

    response = requests.get(
        url,
        params=params,
        timeout=15
    )

    print("Binance HTTP status:", response.status_code)

    if response.status_code != 200:
        print("Binance response:")
        print(response.text)

        raise Exception(
            f"Binance request failed with HTTP "
            f"{response.status_code}"
        )

    data = response.json()

    print(
        f"Successfully received "
        f"{len(data)} candles from Binance."
    )

    df = pd.DataFrame(
        data,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore"
        ]
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:
        df[column] = pd.to_numeric(df[column])

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True
    )

    df.set_index(
        "open_time",
        inplace=True
    )

    return df
