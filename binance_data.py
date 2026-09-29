import requests
import pandas as pd


BASE_URL = "https://fapi.binance.com"
SYMBOL = "VVVUSDT"


def get_klines(interval="4h", limit=200):
    """
    Lấy dữ liệu OHLCV của VVVUSDT Perpetual
    từ Binance USD-M Futures.
    """

    url = f"{BASE_URL}/fapi/v1/klines"

    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "limit": limit
    }

    print(
        f"Requesting Binance data: "
        f"{SYMBOL} | {interval} | {limit} candles"
    )

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    print(
        "Binance HTTP status:",
        response.status_code
    )

    # Không dùng raise_for_status ở đây
    # để nếu Binance trả lỗi 451, 403... ta thấy nội dung lỗi.
    if response.status_code != 200:

        print("Binance response:")
        print(response.text)

        raise RuntimeError(
            f"Binance request failed "
            f"with HTTP {response.status_code}"
        )

    data = response.json()

    print(
        f"Successfully received "
        f"{len(data)} candles."
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

    # Chuyển giá/volume từ text thành số
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Chuyển timestamp Binance thành UTC datetime
    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
        utc=True
    )

    # Dùng thời gian mở nến làm index
    df.set_index(
        "open_time",
        inplace=True
    )

    return df
