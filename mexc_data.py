import time
import requests
import pandas as pd


BASE_URL = "https://api.mexc.com"
SYMBOL = "VVV_USDT"

INTERVAL_MAP = {
    "15m": "Min15",
    "1h": "Min60",
    "4h": "Hour4",
}

INTERVAL_SECONDS = {
    "15m": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
}


def get_klines(interval="4h", limit=200):
    """
    Lấy dữ liệu OHLCV của VVV_USDT Perpetual
    từ MEXC Futures public API.
    """

    if interval not in INTERVAL_MAP:
        raise ValueError(
            f"Unsupported interval: {interval}. "
            f"Use one of: {list(INTERVAL_MAP)}"
        )

    if limit <= 0:
        raise ValueError("limit must be > 0")

    mexc_interval = INTERVAL_MAP[interval]

    # MEXC Futures K-line dùng start/end theo Unix timestamp (giây).
    end = int(time.time())
    start = end - (INTERVAL_SECONDS[interval] * (limit + 2))

    url = f"{BASE_URL}/api/v1/contract/kline/{SYMBOL}"

    params = {
        "interval": mexc_interval,
        "start": start,
        "end": end,
    }

    print(
        f"Requesting MEXC Futures data: "
        f"{SYMBOL} | {interval} ({mexc_interval}) | ~{limit} candles"
    )

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    print("MEXC HTTP status:", response.status_code)

    if response.status_code != 200:
        print("MEXC response:")
        print(response.text)
        raise RuntimeError(
            f"MEXC request failed with HTTP {response.status_code}"
        )

    payload = response.json()

    if not payload.get("success") or payload.get("code") != 0:
        print("MEXC response:")
        print(payload)
        raise RuntimeError(
            f"MEXC API error: code={payload.get('code')}"
        )

    data = payload.get("data") or {}

    required = ["time", "open", "high", "low", "close", "vol"]
    missing = [key for key in required if key not in data]

    if missing:
        raise RuntimeError(
            f"MEXC response missing fields: {missing}"
        )

    lengths = [len(data[key]) for key in required]
    row_count = min(lengths) if lengths else 0

    if row_count == 0:
        raise RuntimeError("MEXC returned no candle data")

    df = pd.DataFrame({
        "open_time": data["time"][:row_count],
        "open": data["open"][:row_count],
        "high": data["high"][:row_count],
        "low": data["low"][:row_count],
        "close": data["close"][:row_count],
        "volume": data["vol"][:row_count],
    })

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="s",
        utc=True
    )

    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)

    # Giữ đúng số nến gần nhất mà caller yêu cầu.
    df = df.tail(limit)

    if df.empty:
        raise RuntimeError("No valid MEXC candles after parsing")

    print(
        f"Successfully received {len(df)} "
        f"{interval} candles from MEXC."
    )

    return df



def get_closed_klines(
    interval="4h",
    limit=420,
    min_required=20,
):
    """
    Return only fully closed MEXC candles.

    The exchange can include the currently-forming candle in the
    kline response. Signal generation must never use that candle,
    because its OHLC values can still change and would create
    repainting BOS/CHoCH/FVG/zone/setup signals.
    """

    if interval not in INTERVAL_SECONDS:
        raise ValueError(
            f"Unsupported interval: {interval}"
        )

    raw = get_klines(
        interval,
        limit + 4,
    )

    now = pd.Timestamp.now(
        tz="UTC"
    )

    candle_delta = pd.to_timedelta(
        INTERVAL_SECONDS[interval],
        unit="s",
    )

    close_times = (
        raw.index + candle_delta
    )

    closed = raw.loc[
        close_times <= now
    ].tail(limit)

    if closed.empty:
        raise RuntimeError(
            (
                "No fully closed MEXC candles "
                f"available for {interval}."
            )
        )

    if len(closed) < min_required:
        raise RuntimeError(
            (
                f"Only {len(closed)} closed {interval} candles "
                f"were returned; at least {min_required} are "
                "required for the configured analysis horizon."
            )
        )

    dropped = len(raw) - len(closed)

    print(
        (
            f"Closed-candle filter: {interval} | "
            f"{len(closed)} closed | "
            f"{dropped} forming/extra excluded"
        )
    )

    return closed


def get_contract_snapshot():
    """
    Public MEXC Futures ticker snapshot for VVV_USDT.

    Returns real-time-ish contract fields used by the chart header
    and derivatives panel: last price, 24h high/low/change,
    open interest proxy (holdVol), and funding rate.
    """

    url = (
        f"{BASE_URL}/api/v1/contract/ticker"
    )

    response = requests.get(
        url,
        params={
            "symbol": SYMBOL,
        },
        timeout=20,
    )

    print(
        "MEXC ticker HTTP status:",
        response.status_code,
    )

    if response.status_code != 200:
        print(
            "MEXC ticker response:"
        )
        print(response.text)

        raise RuntimeError(
            (
                "MEXC ticker request "
                f"failed with HTTP "
                f"{response.status_code}"
            )
        )

    payload = response.json()

    if (
        not payload.get("success")
        or payload.get("code") != 0
    ):
        print(
            "MEXC ticker response:"
        )
        print(payload)

        raise RuntimeError(
            (
                "MEXC ticker API error: "
                f"code={payload.get('code')}"
            )
        )

    data = payload.get("data")

    if isinstance(data, list):
        matches = [
            item
            for item in data
            if item.get("symbol")
            == SYMBOL
        ]

        data = (
            matches[0]
            if matches
            else None
        )

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            (
                "MEXC ticker returned "
                "unexpected data."
            )
        )

    def num(key):
        value = data.get(key)

        if value is None:
            return None

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return None

    snapshot = {
        "symbol": (
            data.get("symbol")
            or SYMBOL
        ),
        "last_price": num(
            "lastPrice"
        ),
        "high_24h": num(
            "high24Price"
        ),
        "low_24h": num(
            "lower24Price"
        ),
        "change_rate_24h": num(
            "riseFallRate"
        ),
        "change_value_24h": num(
            "riseFallValue"
        ),
        "hold_vol": num(
            "holdVol"
        ),
        "funding_rate": num(
            "fundingRate"
        ),
        "index_price": num(
            "indexPrice"
        ),
        "fair_price": num(
            "fairPrice"
        ),
        "timestamp": (
            data.get("timestamp")
        ),
    }

    print(
        (
            "MEXC ticker snapshot: "
            f"last={snapshot['last_price']} | "
            f"holdVol={snapshot['hold_vol']} | "
            f"funding={snapshot['funding_rate']}"
        )
    )

    return snapshot
