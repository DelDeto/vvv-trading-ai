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

    return tr.rolling(
        period,
        min_periods=1,
    ).mean()


def _swing_masks(
    df,
    window=2,
):
    high_mask = pd.Series(
        True,
        index=df.index,
    )

    low_mask = pd.Series(
        True,
        index=df.index,
    )

    for offset in range(
        1,
        window + 1,
    ):
        high_mask &= (
            df["high"]
            > df["high"].shift(offset)
        )

        high_mask &= (
            df["high"]
            > df["high"].shift(-offset)
        )

        low_mask &= (
            df["low"]
            < df["low"].shift(offset)
        )

        low_mask &= (
            df["low"]
            < df["low"].shift(-offset)
        )

    return (
        high_mask.fillna(False),
        low_mask.fillna(False),
    )


def _infer_trend(
    swing_highs,
    swing_lows,
):
    """
    HH + HL -> bullish
    LH + LL -> bearish
    otherwise -> mixed
    """

    if (
        len(swing_highs) >= 2
        and len(swing_lows) >= 2
    ):
        prev_high = swing_highs[-2][1]
        last_high = swing_highs[-1][1]

        prev_low = swing_lows[-2][1]
        last_low = swing_lows[-1][1]

        if (
            last_high > prev_high
            and last_low > prev_low
        ):
            return "bullish"

        if (
            last_high < prev_high
            and last_low < prev_low
        ):
            return "bearish"

        return "mixed"

    return "neutral"


def _structure_events(
    df,
    high_mask,
    low_mask,
    window=2,
):
    """
    Rule-based BOS / CHoCH:
    - Close breaks a confirmed swing level.
    - Break in active direction = BOS.
    - Break opposite active direction = CHoCH.
    """

    events = []

    last_high = None
    last_low = None

    broken_highs = set()
    broken_lows = set()

    bias = "neutral"

    for i in range(len(df)):
        confirmed_index = (
            i - window
        )

        if confirmed_index >= 0:
            if bool(
                high_mask.iloc[
                    confirmed_index
                ]
            ):
                last_high = (
                    confirmed_index,
                    float(
                        df["high"].iloc[
                            confirmed_index
                        ]
                    ),
                )

            if bool(
                low_mask.iloc[
                    confirmed_index
                ]
            ):
                last_low = (
                    confirmed_index,
                    float(
                        df["low"].iloc[
                            confirmed_index
                        ]
                    ),
                )

        if i == 0:
            continue

        close_now = float(
            df["close"].iloc[i]
        )

        close_prev = float(
            df["close"].iloc[i - 1]
        )

        if last_high is not None:
            swing_index, level = (
                last_high
            )

            if (
                swing_index
                not in broken_highs
                and i > swing_index
                and close_now > level
                and close_prev <= level
            ):
                kind = (
                    "BOS"
                    if bias
                    in (
                        "neutral",
                        "bullish",
                    )
                    else "CHoCH"
                )

                events.append(
                    {
                        "index": i,
                        "time": (
                            df.index[
                                i
                            ].isoformat()
                        ),
                        "direction": (
                            "bullish"
                        ),
                        "kind": kind,
                        "level": level,
                        "close": close_now,
                    }
                )

                broken_highs.add(
                    swing_index
                )

                bias = "bullish"

        if last_low is not None:
            swing_index, level = (
                last_low
            )

            if (
                swing_index
                not in broken_lows
                and i > swing_index
                and close_now < level
                and close_prev >= level
            ):
                kind = (
                    "BOS"
                    if bias
                    in (
                        "neutral",
                        "bearish",
                    )
                    else "CHoCH"
                )

                events.append(
                    {
                        "index": i,
                        "time": (
                            df.index[
                                i
                            ].isoformat()
                        ),
                        "direction": (
                            "bearish"
                        ),
                        "kind": kind,
                        "level": level,
                        "close": close_now,
                    }
                )

                broken_lows.add(
                    swing_index
                )

                bias = "bearish"

    return events


def _untouched_swings(
    df,
    swings,
    side,
):
    """
    Keep only liquidity pools that have not already been traded through
    after the swing was formed.
    """

    valid = []

    for point in swings:
        index, price, timestamp = (
            point
        )

        future = df.iloc[
            index + 1 :
        ]

        if future.empty:
            untouched = True

        elif side == "high":
            untouched = (
                float(
                    future[
                        "high"
                    ].max()
                )
                < price
            )

        else:
            untouched = (
                float(
                    future[
                        "low"
                    ].min()
                )
                > price
            )

        if untouched:
            valid.append(
                point
            )

    return valid


def _find_liquidity(
    df,
    current_price,
    swing_highs,
    swing_lows,
):
    """
    BSL / SSL are selected only from untouched confirmed swings.
    This prevents stale historical highs/lows that were already swept
    from being displayed as active liquidity.
    """

    active_highs = (
        _untouched_swings(
            df,
            swing_highs,
            "high",
        )
    )

    active_lows = (
        _untouched_swings(
            df,
            swing_lows,
            "low",
        )
    )

    overhead = [
        point
        for point in active_highs
        if point[1] > current_price
    ]

    below = [
        point
        for point in active_lows
        if point[1] < current_price
    ]

    bsl = (
        min(
            overhead,
            key=lambda item: item[1],
        )
        if overhead
        else None
    )

    ssl = (
        max(
            below,
            key=lambda item: item[1],
        )
        if below
        else None
    )

    def pack(point):
        if point is None:
            return None

        index, price, timestamp = (
            point
        )

        return {
            "index": index,
            "price": price,
            "time": timestamp,
        }

    return (
        pack(bsl),
        pack(ssl),
    )


def _find_fvgs(df):
    """
    Three-candle Fair Value Gap:
    bullish FVG: low[i] > high[i-2]
    bearish FVG: high[i] < low[i-2]
    """

    fvgs = []

    for i in range(
        2,
        len(df),
    ):
        high_two_back = float(
            df["high"].iloc[i - 2]
        )

        low_two_back = float(
            df["low"].iloc[i - 2]
        )

        high_now = float(
            df["high"].iloc[i]
        )

        low_now = float(
            df["low"].iloc[i]
        )

        if (
            low_now
            > high_two_back
        ):
            lower = high_two_back
            upper = low_now

            future = df.iloc[
                i + 1 :
            ]

            filled = (
                not future.empty
                and float(
                    future[
                        "low"
                    ].min()
                )
                <= lower
            )

            mitigated = (
                not future.empty
                and float(
                    future[
                        "low"
                    ].min()
                )
                < upper
            )

            fvgs.append(
                {
                    "type": (
                        "bullish"
                    ),
                    "index": i,
                    "time": (
                        df.index[
                            i
                        ].isoformat()
                    ),
                    "lower": lower,
                    "upper": upper,
                    "filled": bool(
                        filled
                    ),
                    "mitigated": bool(
                        mitigated
                    ),
                }
            )

        if (
            high_now
            < low_two_back
        ):
            lower = high_now
            upper = low_two_back

            future = df.iloc[
                i + 1 :
            ]

            filled = (
                not future.empty
                and float(
                    future[
                        "high"
                    ].max()
                )
                >= upper
            )

            mitigated = (
                not future.empty
                and float(
                    future[
                        "high"
                    ].max()
                )
                > lower
            )

            fvgs.append(
                {
                    "type": (
                        "bearish"
                    ),
                    "index": i,
                    "time": (
                        df.index[
                            i
                        ].isoformat()
                    ),
                    "lower": lower,
                    "upper": upper,
                    "filled": bool(
                        filled
                    ),
                    "mitigated": bool(
                        mitigated
                    ),
                }
            )

    return fvgs


def _find_displacements(
    df,
    atr,
):
    """
    Strong directional candle heuristic:
    body >= 1.15 ATR,
    body >= 65% of range,
    close near the directional extreme.
    """

    events = []

    body = (
        df["close"]
        - df["open"]
    ).abs()

    candle_range = (
        df["high"]
        - df["low"]
    ).replace(
        0,
        np.nan,
    )

    close_location = (
        (
            df["close"]
            - df["low"]
        )
        / candle_range
    )

    body_ratio = (
        body
        / candle_range
    )

    for i in range(
        1,
        len(df),
    ):
        atr_value = float(
            atr.iloc[i]
        )

        if (
            pd.isna(atr_value)
            or atr_value <= 0
        ):
            continue

        bullish = (
            df["close"].iloc[i]
            > df["open"].iloc[i]
            and float(
                body.iloc[i]
            )
            >= 1.15 * atr_value
            and float(
                body_ratio.iloc[i]
            )
            >= 0.65
            and float(
                close_location.iloc[i]
            )
            >= 0.75
        )

        bearish = (
            df["close"].iloc[i]
            < df["open"].iloc[i]
            and float(
                body.iloc[i]
            )
            >= 1.15 * atr_value
            and float(
                body_ratio.iloc[i]
            )
            >= 0.65
            and float(
                close_location.iloc[i]
            )
            <= 0.25
        )

        if bullish or bearish:
            events.append(
                {
                    "index": i,
                    "time": (
                        df.index[
                            i
                        ].isoformat()
                    ),
                    "direction": (
                        "bullish"
                        if bullish
                        else "bearish"
                    ),
                    "strength": float(
                        body.iloc[i]
                        / atr_value
                    ),
                    "open": float(
                        df["open"].iloc[i]
                    ),
                    "close": float(
                        df["close"].iloc[i]
                    ),
                    "high": float(
                        df["high"].iloc[i]
                    ),
                    "low": float(
                        df["low"].iloc[i]
                    ),
                }
            )

    return events


def _find_sweeps(
    df,
    high_mask,
    low_mask,
    window=2,
):
    """
    Liquidity Sweep heuristic:
    - BSL sweep: wick above confirmed swing high, close back below.
    - SSL sweep: wick below confirmed swing low, close back above.
    """

    sweeps = []

    last_high = None
    last_low = None

    swept_highs = set()
    swept_lows = set()

    for i in range(len(df)):
        confirmed_index = (
            i - window
        )

        if confirmed_index >= 0:
            if bool(
                high_mask.iloc[
                    confirmed_index
                ]
            ):
                last_high = (
                    confirmed_index,
                    float(
                        df["high"].iloc[
                            confirmed_index
                        ]
                    ),
                )

            if bool(
                low_mask.iloc[
                    confirmed_index
                ]
            ):
                last_low = (
                    confirmed_index,
                    float(
                        df["low"].iloc[
                            confirmed_index
                        ]
                    ),
                )

        high_now = float(
            df["high"].iloc[i]
        )

        low_now = float(
            df["low"].iloc[i]
        )

        close_now = float(
            df["close"].iloc[i]
        )

        if last_high:
            swing_index, level = (
                last_high
            )

            if (
                swing_index
                not in swept_highs
                and i > swing_index
                and high_now > level
                and close_now < level
            ):
                sweeps.append(
                    {
                        "index": i,
                        "time": (
                            df.index[
                                i
                            ].isoformat()
                        ),
                        "type": (
                            "BSL"
                        ),
                        "direction": (
                            "bearish"
                        ),
                        "level": level,
                        "extreme": high_now,
                    }
                )

                swept_highs.add(
                    swing_index
                )

        if last_low:
            swing_index, level = (
                last_low
            )

            if (
                swing_index
                not in swept_lows
                and i > swing_index
                and low_now < level
                and close_now > level
            ):
                sweeps.append(
                    {
                        "index": i,
                        "time": (
                            df.index[
                                i
                            ].isoformat()
                        ),
                        "type": (
                            "SSL"
                        ),
                        "direction": (
                            "bullish"
                        ),
                        "level": level,
                        "extreme": low_now,
                    }
                )

                swept_lows.add(
                    swing_index
                )

    return sweeps


def _find_retests(
    df,
    structure_events,
    atr,
    max_bars=8,
):
    """
    Retest of a broken structure level after BOS / CHoCH.
    """

    retests = []

    for event in structure_events:
        event_index = event[
            "index"
        ]

        level = float(
            event["level"]
        )

        end = min(
            len(df),
            event_index
            + max_bars
            + 1,
        )

        for i in range(
            event_index + 1,
            end,
        ):
            atr_value = float(
                atr.iloc[i]
            )

            tolerance = max(
                0.001,
                atr_value * 0.12,
            )

            low_now = float(
                df["low"].iloc[i]
            )

            high_now = float(
                df["high"].iloc[i]
            )

            close_now = float(
                df["close"].iloc[i]
            )

            if (
                event["direction"]
                == "bullish"
            ):
                touched = (
                    low_now
                    <= level + tolerance
                )

                held = (
                    close_now
                    >= level
                )

            else:
                touched = (
                    high_now
                    >= level - tolerance
                )

                held = (
                    close_now
                    <= level
                )

            if touched and held:
                retests.append(
                    {
                        "index": i,
                        "time": (
                            df.index[
                                i
                            ].isoformat()
                        ),
                        "direction": (
                            event[
                                "direction"
                            ]
                        ),
                        "structure_kind": (
                            event["kind"]
                        ),
                        "level": level,
                        "source_index": (
                            event_index
                        ),
                    }
                )
                break

    return retests


def _find_zones(
    df,
    atr,
):
    """
    Supply/Demand:
    base candle immediately before
    an ATR-sized displacement candle.
    """

    supply = []
    demand = []

    displacements = (
        _find_displacements(
            df,
            atr,
        )
    )

    by_index = {
        item["index"]: item
        for item
        in displacements
    }

    for i in range(
        1,
        len(df),
    ):
        displacement = (
            by_index.get(i)
        )

        if not displacement:
            continue

        base_index = i - 1

        base_open = float(
            df["open"].iloc[
                base_index
            ]
        )

        base_close = float(
            df["close"].iloc[
                base_index
            ]
        )

        base_low = float(
            df["low"].iloc[
                base_index
            ]
        )

        base_high = float(
            df["high"].iloc[
                base_index
            ]
        )

        if (
            displacement[
                "direction"
            ]
            == "bullish"
        ):
            lower = base_low

            upper = max(
                base_open,
                base_close,
            )

            later = df.iloc[
                i + 1 :
            ]

            invalidated = (
                not later.empty
                and bool(
                    (
                        later[
                            "close"
                        ]
                        < lower
                    ).any()
                )
            )

            demand.append(
                {
                    "index": (
                        base_index
                    ),
                    "time": (
                        df.index[
                            base_index
                        ].isoformat()
                    ),
                    "lower": lower,
                    "upper": upper,
                    "invalidated": (
                        invalidated
                    ),
                    "strength": (
                        displacement[
                            "strength"
                        ]
                    ),
                }
            )

        else:
            lower = min(
                base_open,
                base_close,
            )

            upper = base_high

            later = df.iloc[
                i + 1 :
            ]

            invalidated = (
                not later.empty
                and bool(
                    (
                        later[
                            "close"
                        ]
                        > upper
                    ).any()
                )
            )

            supply.append(
                {
                    "index": (
                        base_index
                    ),
                    "time": (
                        df.index[
                            base_index
                        ].isoformat()
                    ),
                    "lower": lower,
                    "upper": upper,
                    "invalidated": (
                        invalidated
                    ),
                    "strength": (
                        displacement[
                            "strength"
                        ]
                    ),
                }
            )

    return (
        supply,
        demand,
        displacements,
    )


def _nearest_zone(
    zones,
    current_price,
    side,
):
    active = [
        zone
        for zone in zones
        if not zone.get(
            "invalidated",
            False,
        )
    ]

    if not active:
        return None

    if side == "supply":
        above = [
            zone
            for zone in active
            if zone["upper"]
            >= current_price
        ]

        if above:
            return min(
                above,
                key=lambda zone: (
                    abs(
                        zone["lower"]
                        - current_price
                    )
                    if zone[
                        "lower"
                    ]
                    >= current_price
                    else 0
                ),
            )

    if side == "demand":
        below = [
            zone
            for zone in active
            if zone["lower"]
            <= current_price
        ]

        if below:
            return min(
                below,
                key=lambda zone: (
                    abs(
                        current_price
                        - zone["upper"]
                    )
                    if zone[
                        "upper"
                    ]
                    <= current_price
                    else 0
                ),
            )

    return active[-1]


def _latest_recent(
    events,
    current_index,
    bars,
    direction=None,
):
    candidates = [
        item
        for item in events
        if (
            current_index
            - item["index"]
            <= bars
        )
        and (
            direction is None
            or item.get(
                "direction"
            )
            == direction
        )
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: item[
            "index"
        ],
    )


def _setup_signal(
    current_index,
    sweeps,
    displacements,
    structure_events,
    retests,
    direction,
):
    sweep = _latest_recent(
        sweeps,
        current_index,
        20,
        direction,
    )

    displacement = (
        _latest_recent(
            displacements,
            current_index,
            12,
            direction,
        )
    )

    structure = _latest_recent(
        structure_events,
        current_index,
        12,
        direction,
    )

    retest = _latest_recent(
        retests,
        current_index,
        8,
        direction,
    )

    score = sum(
        item is not None
        for item in [
            sweep,
            displacement,
            structure,
            retest,
        ]
    )

    ordered = False

    if (
        displacement
        and structure
        and retest
    ):
        ordered = (
            displacement["index"]
            <= structure["index"]
            < retest["index"]
        )

    sweep_before_structure = (
        sweep is not None
        and structure is not None
        and sweep["index"]
        <= structure["index"]
    )

    confirmed = (
        ordered
        and (
            sweep_before_structure
            or (
                structure.get(
                    "kind"
                )
                == "BOS"
            )
        )
    )

    latest_index = max(
        [
            item["index"]
            for item in [
                sweep,
                displacement,
                structure,
                retest,
            ]
            if item is not None
        ],
        default=-1,
    )

    return {
        "direction": direction,
        "score": score,
        "confirmed": confirmed,
        "latest_index": (
            latest_index
        ),
        "sweep": sweep,
        "displacement": (
            displacement
        ),
        "structure": structure,
        "retest": retest,
    }


def analyze_smc(
    df,
    swing_window=2,
):
    """
    Conservative, rule-based SMC analysis.
    """

    if (
        df is None
        or len(df) < 20
    ):
        raise ValueError(
            (
                "Need at least 20 "
                "candles for SMC "
                "analysis"
            )
        )

    work = df[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].copy()

    work = (
        work.dropna().copy()
    )

    atr = _atr(work)

    (
        high_mask,
        low_mask,
    ) = _swing_masks(
        work,
        window=swing_window,
    )

    swing_highs = [
        (
            i,
            float(
                work[
                    "high"
                ].iloc[i]
            ),
            work.index[
                i
            ].isoformat(),
        )
        for i in range(len(work))
        if bool(
            high_mask.iloc[i]
        )
    ]

    swing_lows = [
        (
            i,
            float(
                work[
                    "low"
                ].iloc[i]
            ),
            work.index[
                i
            ].isoformat(),
        )
        for i in range(len(work))
        if bool(
            low_mask.iloc[i]
        )
    ]

    trend = _infer_trend(
        swing_highs,
        swing_lows,
    )

    structure_events = (
        _structure_events(
            work,
            high_mask,
            low_mask,
            window=swing_window,
        )
    )

    sweeps = _find_sweeps(
        work,
        high_mask,
        low_mask,
        window=swing_window,
    )

    supply, demand, displacements = (
        _find_zones(
            work,
            atr,
        )
    )

    retests = _find_retests(
        work,
        structure_events,
        atr,
    )

    current_price = float(
        work["close"].iloc[-1]
    )

    bsl, ssl = _find_liquidity(
        work,
        current_price,
        swing_highs,
        swing_lows,
    )

    fvgs = _find_fvgs(
        work
    )

    active_fvgs = [
        fvg
        for fvg in fvgs
        if not fvg["filled"]
    ]

    nearest_supply = (
        _nearest_zone(
            supply,
            current_price,
            "supply",
        )
    )

    nearest_demand = (
        _nearest_zone(
            demand,
            current_price,
            "demand",
        )
    )

    last_event = (
        structure_events[-1]
        if structure_events
        else None
    )

    current_index = (
        len(work) - 1
    )

    bullish_setup = (
        _setup_signal(
            current_index,
            sweeps,
            displacements,
            structure_events,
            retests,
            "bullish",
        )
    )

    bearish_setup = (
        _setup_signal(
            current_index,
            sweeps,
            displacements,
            structure_events,
            retests,
            "bearish",
        )
    )

    preferred_setup = max(
        [
            bullish_setup,
            bearish_setup,
        ],
        key=lambda item: (
            item["score"],
            item["latest_index"],
        ),
    )

    return {
        "current_price": (
            current_price
        ),
        "atr": float(
            atr.iloc[-1]
        ),
        "trend": trend,
        "bsl": bsl,
        "ssl": ssl,
        "swing_highs": [
            {
                "index": index,
                "price": price,
                "time": timestamp,
            }
            for (
                index,
                price,
                timestamp,
            )
            in swing_highs[-8:]
        ],
        "swing_lows": [
            {
                "index": index,
                "price": price,
                "time": timestamp,
            }
            for (
                index,
                price,
                timestamp,
            )
            in swing_lows[-8:]
        ],
        "events": (
            structure_events[
                -10:
            ]
        ),
        "last_event": (
            last_event
        ),
        "sweeps": (
            sweeps[-8:]
        ),
        "last_sweep": (
            sweeps[-1]
            if sweeps
            else None
        ),
        "displacements": (
            displacements[-8:]
        ),
        "last_displacement": (
            displacements[-1]
            if displacements
            else None
        ),
        "retests": (
            retests[-8:]
        ),
        "last_retest": (
            retests[-1]
            if retests
            else None
        ),
        "active_fvgs": (
            active_fvgs[-6:]
        ),
        "nearest_supply": (
            nearest_supply
        ),
        "nearest_demand": (
            nearest_demand
        ),
        "supply_zones": [
            zone
            for zone in supply
            if not zone[
                "invalidated"
            ]
        ][-4:],
        "demand_zones": [
            zone
            for zone in demand
            if not zone[
                "invalidated"
            ]
        ][-4:],
        "setup": (
            preferred_setup
        ),
        "bullish_setup": (
            bullish_setup
        ),
        "bearish_setup": (
            bearish_setup
        ),
    }


def derive_overall_status(
    analysis_4h,
    analysis_1h,
    analysis_15m,
):
    """
    Multi-timeframe status:
    confirmation requires a 15m sequence,
    while higher timeframes act as a filter.
    """

    trend_4h = analysis_4h[
        "trend"
    ]

    trend_1h = analysis_1h[
        "trend"
    ]

    setup_15m = (
        analysis_15m.get(
            "setup",
            {},
        )
    )

    direction = (
        setup_15m.get(
            "direction"
        )
    )

    score = setup_15m.get(
        "score",
        0,
    )

    confirmed = (
        setup_15m.get(
            "confirmed",
            False,
        )
    )

    if (
        direction == "bullish"
    ):
        hard_opposition = (
            trend_4h == "bearish"
            and trend_1h == "bearish"
        )

        if (
            confirmed
            and not hard_opposition
        ):
            return (
                "CONFIRMED LONG"
            )

        if score >= 2:
            return (
                "DEVELOPING LONG"
            )

    if (
        direction == "bearish"
    ):
        hard_opposition = (
            trend_4h == "bullish"
            and trend_1h == "bullish"
        )

        if (
            confirmed
            and not hard_opposition
        ):
            return (
                "CONFIRMED SHORT"
            )

        if score >= 2:
            return (
                "DEVELOPING SHORT"
            )

    return "WAIT"
