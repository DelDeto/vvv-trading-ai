import numpy as np
import pandas as pd


def _atr(df, period=14):
    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period, min_periods=1).mean()


def _swing_masks(df, window=2):
    high_mask = pd.Series(True, index=df.index)
    low_mask = pd.Series(True, index=df.index)

    for offset in range(1, window + 1):
        high_mask &= df["high"] > df["high"].shift(offset)
        high_mask &= df["high"] > df["high"].shift(-offset)

        low_mask &= df["low"] < df["low"].shift(offset)
        low_mask &= df["low"] < df["low"].shift(-offset)

    return high_mask.fillna(False), low_mask.fillna(False)


def _infer_trend(swing_highs, swing_lows):
    """
    HH + HL -> bullish
    LH + LL -> bearish
    otherwise -> mixed
    """

    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        prev_high = swing_highs[-2][1]
        last_high = swing_highs[-1][1]

        prev_low = swing_lows[-2][1]
        last_low = swing_lows[-1][1]

        if last_high > prev_high and last_low > prev_low:
            return "bullish"

        if last_high < prev_high and last_low < prev_low:
            return "bearish"

        return "mixed"

    return "neutral"


def _structure_events(df, high_mask, low_mask, window=2):
    """
    Rule-based BOS / CHoCH:
    - Close breaks a confirmed swing level.
    - Break in the active direction = BOS.
    - Break opposite active direction = CHoCH.
    """

    events = []

    last_high = None
    last_low = None

    broken_highs = set()
    broken_lows = set()

    bias = "neutral"

    for i in range(len(df)):
        # A centered swing needs 'window' future candles before it is confirmed.
        confirmed_index = i - window

        if confirmed_index >= 0:
            if bool(high_mask.iloc[confirmed_index]):
                last_high = (
                    confirmed_index,
                    float(df["high"].iloc[confirmed_index]),
                )

            if bool(low_mask.iloc[confirmed_index]):
                last_low = (
                    confirmed_index,
                    float(df["low"].iloc[confirmed_index]),
                )

        if i == 0:
            continue

        close_now = float(df["close"].iloc[i])
        close_prev = float(df["close"].iloc[i - 1])

        if last_high is not None:
            swing_index, level = last_high

            if (
                swing_index not in broken_highs
                and i > swing_index
                and close_now > level
                and close_prev <= level
            ):
                kind = (
                    "BOS"
                    if bias in ("neutral", "bullish")
                    else "CHoCH"
                )

                events.append(
                    {
                        "index": i,
                        "time": df.index[i].isoformat(),
                        "direction": "bullish",
                        "kind": kind,
                        "level": level,
                        "close": close_now,
                    }
                )

                broken_highs.add(swing_index)
                bias = "bullish"

        if last_low is not None:
            swing_index, level = last_low

            if (
                swing_index not in broken_lows
                and i > swing_index
                and close_now < level
                and close_prev >= level
            ):
                kind = (
                    "BOS"
                    if bias in ("neutral", "bearish")
                    else "CHoCH"
                )

                events.append(
                    {
                        "index": i,
                        "time": df.index[i].isoformat(),
                        "direction": "bearish",
                        "kind": kind,
                        "level": level,
                        "close": close_now,
                    }
                )

                broken_lows.add(swing_index)
                bias = "bearish"

    return events


def _find_liquidity(current_price, swing_highs, swing_lows):
    """
    BSL = nearest confirmed swing high above current price.
    SSL = nearest confirmed swing low below current price.
    """

    overhead = [
        point
        for point in swing_highs
        if point[1] > current_price
    ]

    below = [
        point
        for point in swing_lows
        if point[1] < current_price
    ]

    if overhead:
        bsl = min(overhead, key=lambda item: item[1])
    elif swing_highs:
        bsl = swing_highs[-1]
    else:
        bsl = None

    if below:
        ssl = max(below, key=lambda item: item[1])
    elif swing_lows:
        ssl = swing_lows[-1]
    else:
        ssl = None

    def pack(point):
        if point is None:
            return None

        index, price, timestamp = point

        return {
            "index": index,
            "price": price,
            "time": timestamp,
        }

    return pack(bsl), pack(ssl)


def _find_fvgs(df):
    """
    Three-candle Fair Value Gap heuristic:
    bullish FVG: low[i] > high[i-2]
    bearish FVG: high[i] < low[i-2]
    """

    fvgs = []

    for i in range(2, len(df)):
        high_two_back = float(df["high"].iloc[i - 2])
        low_two_back = float(df["low"].iloc[i - 2])

        high_now = float(df["high"].iloc[i])
        low_now = float(df["low"].iloc[i])

        # Bullish FVG
        if low_now > high_two_back:
            lower = high_two_back
            upper = low_now

            future = df.iloc[i + 1 :]

            filled = (
                not future.empty
                and float(future["low"].min()) <= lower
            )

            mitigated = (
                not future.empty
                and float(future["low"].min()) < upper
            )

            fvgs.append(
                {
                    "type": "bullish",
                    "index": i,
                    "time": df.index[i].isoformat(),
                    "lower": lower,
                    "upper": upper,
                    "filled": bool(filled),
                    "mitigated": bool(mitigated),
                }
            )

        # Bearish FVG
        if high_now < low_two_back:
            lower = high_now
            upper = low_two_back

            future = df.iloc[i + 1 :]

            filled = (
                not future.empty
                and float(future["high"].max()) >= upper
            )

            mitigated = (
                not future.empty
                and float(future["high"].max()) > lower
            )

            fvgs.append(
                {
                    "type": "bearish",
                    "index": i,
                    "time": df.index[i].isoformat(),
                    "lower": lower,
                    "upper": upper,
                    "filled": bool(filled),
                    "mitigated": bool(mitigated),
                }
            )

    return fvgs


def _find_zones(df, atr):
    """
    Supply/Demand heuristic:
    last base candle before an ATR-sized displacement candle.
    """

    supply = []
    demand = []

    body = (df["close"] - df["open"]).abs()

    candle_range = (
        df["high"] - df["low"]
    ).replace(0, np.nan)

    close_location = (
        (df["close"] - df["low"])
        / candle_range
    )

    for i in range(1, len(df)):
        atr_value = float(atr.iloc[i])

        if pd.isna(atr_value) or atr_value <= 0:
            continue

        bullish_displacement = (
            df["close"].iloc[i] > df["open"].iloc[i]
            and float(body.iloc[i]) >= 1.15 * atr_value
            and float(close_location.iloc[i]) >= 0.70
        )

        bearish_displacement = (
            df["close"].iloc[i] < df["open"].iloc[i]
            and float(body.iloc[i]) >= 1.15 * atr_value
            and float(close_location.iloc[i]) <= 0.30
        )

        base_index = i - 1

        base_open = float(df["open"].iloc[base_index])
        base_close = float(df["close"].iloc[base_index])
        base_low = float(df["low"].iloc[base_index])
        base_high = float(df["high"].iloc[base_index])

        if bullish_displacement:
            lower = base_low
            upper = max(base_open, base_close)

            later = df.iloc[i + 1 :]

            invalidated = (
                not later.empty
                and bool((later["close"] < lower).any())
            )

            demand.append(
                {
                    "index": base_index,
                    "time": df.index[base_index].isoformat(),
                    "lower": lower,
                    "upper": upper,
                    "invalidated": invalidated,
                    "strength": float(
                        body.iloc[i] / atr_value
                    ),
                }
            )

        if bearish_displacement:
            lower = min(base_open, base_close)
            upper = base_high

            later = df.iloc[i + 1 :]

            invalidated = (
                not later.empty
                and bool((later["close"] > upper).any())
            )

            supply.append(
                {
                    "index": base_index,
                    "time": df.index[base_index].isoformat(),
                    "lower": lower,
                    "upper": upper,
                    "invalidated": invalidated,
                    "strength": float(
                        body.iloc[i] / atr_value
                    ),
                }
            )

    return supply, demand


def _nearest_zone(zones, current_price, side):
    active = [
        zone
        for zone in zones
        if not zone.get("invalidated", False)
    ]

    if not active:
        return None

    if side == "supply":
        above = [
            zone
            for zone in active
            if zone["upper"] >= current_price
        ]

        if above:
            return min(
                above,
                key=lambda zone: max(
                    zone["lower"],
                    current_price,
                ),
            )

    if side == "demand":
        below = [
            zone
            for zone in active
            if zone["lower"] <= current_price
        ]

        if below:
            return max(
                below,
                key=lambda zone: min(
                    zone["upper"],
                    current_price,
                ),
            )

    return active[-1]


def analyze_smc(df, swing_window=2):
    """
    Return conservative rule-based SMC structure for one timeframe.
    """

    if df is None or len(df) < 20:
        raise ValueError(
            "Need at least 20 candles for SMC analysis"
        )

    work = df[
        ["open", "high", "low", "close", "volume"]
    ].copy()

    work = work.dropna().copy()

    atr = _atr(work)

    high_mask, low_mask = _swing_masks(
        work,
        window=swing_window,
    )

    swing_highs = [
        (
            i,
            float(work["high"].iloc[i]),
            work.index[i].isoformat(),
        )
        for i in range(len(work))
        if bool(high_mask.iloc[i])
    ]

    swing_lows = [
        (
            i,
            float(work["low"].iloc[i]),
            work.index[i].isoformat(),
        )
        for i in range(len(work))
        if bool(low_mask.iloc[i])
    ]

    trend = _infer_trend(
        swing_highs,
        swing_lows,
    )

    events = _structure_events(
        work,
        high_mask,
        low_mask,
        window=swing_window,
    )

    current_price = float(
        work["close"].iloc[-1]
    )

    bsl, ssl = _find_liquidity(
        current_price,
        swing_highs,
        swing_lows,
    )

    fvgs = _find_fvgs(work)

    active_fvgs = [
        fvg
        for fvg in fvgs
        if not fvg["filled"]
    ]

    supply, demand = _find_zones(
        work,
        atr,
    )

    nearest_supply = _nearest_zone(
        supply,
        current_price,
        "supply",
    )

    nearest_demand = _nearest_zone(
        demand,
        current_price,
        "demand",
    )

    last_event = (
        events[-1]
        if events
        else None
    )

    return {
        "current_price": current_price,
        "atr": float(atr.iloc[-1]),
        "trend": trend,
        "bsl": bsl,
        "ssl": ssl,
        "swing_highs": [
            {
                "index": index,
                "price": price,
                "time": timestamp,
            }
            for index, price, timestamp
            in swing_highs[-8:]
        ],
        "swing_lows": [
            {
                "index": index,
                "price": price,
                "time": timestamp,
            }
            for index, price, timestamp
            in swing_lows[-8:]
        ],
        "events": events[-10:],
        "last_event": last_event,
        "active_fvgs": active_fvgs[-6:],
        "nearest_supply": nearest_supply,
        "nearest_demand": nearest_demand,
        "supply_zones": [
            zone
            for zone in supply
            if not zone["invalidated"]
        ][-4:],
        "demand_zones": [
            zone
            for zone in demand
            if not zone["invalidated"]
        ][-4:],
    }


def derive_overall_status(
    analysis_4h,
    analysis_1h,
    analysis_15m,
):
    """
    Conservative multi-timeframe status.
    """

    trend_4h = analysis_4h["trend"]
    trend_1h = analysis_1h["trend"]

    event_15m = analysis_15m.get(
        "last_event"
    )

    if event_15m:
        direction = event_15m["direction"]
        kind = event_15m["kind"]

        if direction == "bullish":
            if (
                trend_4h == "bullish"
                and trend_1h == "bullish"
                and kind == "BOS"
            ):
                return "CONFIRMED LONG"

            return "DEVELOPING LONG"

        if direction == "bearish":
            if (
                trend_4h == "bearish"
                and trend_1h == "bearish"
                and kind == "BOS"
            ):
                return "CONFIRMED SHORT"

            return "DEVELOPING SHORT"

    return "WAIT"
