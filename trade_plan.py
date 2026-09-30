def _price(point):
    if not point:
        return None
    return point.get("price")


def _zone_distance(zone, current_price):
    if not zone:
        return None

    lower = float(zone["lower"])
    upper = float(zone["upper"])

    if lower <= current_price <= upper:
        return 0.0

    if current_price < lower:
        return lower - current_price

    return current_price - upper


def _nearby_fvg(analysis, direction, current_price, atr):
    wanted = (
        "bullish"
        if direction == "long"
        else "bearish"
    )

    candidates = []

    for fvg in analysis.get(
        "active_fvgs",
        [],
    ):
        if fvg.get("type") != wanted:
            continue

        lower = float(
            fvg["lower"]
        )

        upper = float(
            fvg["upper"]
        )

        if (
            current_price
            < lower
        ):
            distance = (
                lower
                - current_price
            )

        elif (
            current_price
            > upper
        ):
            distance = (
                current_price
                - upper
            )

        else:
            distance = 0.0

        if distance <= 1.5 * atr:
            candidates.append(
                (
                    distance,
                    lower,
                    upper,
                )
            )

    if not candidates:
        return None

    _, lower, upper = min(
        candidates,
        key=lambda item: item[0],
    )

    return {
        "lower": lower,
        "upper": upper,
        "source": (
            f"{wanted} FVG"
        ),
    }


def _candidate_entry_zones(
    analysis_15m,
    direction,
):
    current_price = float(
        analysis_15m[
            "current_price"
        ]
    )

    atr = max(
        float(
            analysis_15m.get(
                "atr",
                0,
            )
        ),
        0.001,
    )

    setup = analysis_15m.get(
        "setup",
        {},
    )

    candidates = []

    retest = setup.get(
        "retest"
    )

    structure = setup.get(
        "structure"
    )

    if retest:
        center = float(
            retest["level"]
        )

        half = max(
            atr * 0.10,
            center * 0.0005,
        )

        candidates.append(
            {
                "lower": (
                    center - half
                ),
                "upper": (
                    center + half
                ),
                "source": "retest",
                "priority": 0,
            }
        )

    zone = (
        analysis_15m.get(
            "nearest_demand"
        )
        if direction == "long"
        else analysis_15m.get(
            "nearest_supply"
        )
    )

    if zone:
        distance = _zone_distance(
            zone,
            current_price,
        )

        if (
            distance is not None
            and distance <= 1.5 * atr
        ):
            candidates.append(
                {
                    "lower": float(
                        zone["lower"]
                    ),
                    "upper": float(
                        zone["upper"]
                    ),
                    "source": (
                        "demand"
                        if direction
                        == "long"
                        else "supply"
                    ),
                    "priority": 1,
                }
            )

    fvg = _nearby_fvg(
        analysis_15m,
        direction,
        current_price,
        atr,
    )

    if fvg:
        fvg["priority"] = 2
        candidates.append(
            fvg
        )

    if structure:
        center = float(
            structure["level"]
        )

        half = max(
            atr * 0.08,
            center * 0.0004,
        )

        candidates.append(
            {
                "lower": (
                    center - half
                ),
                "upper": (
                    center + half
                ),
                "source": (
                    f"{structure['kind']} "
                    "reclaim"
                ),
                "priority": 3,
            }
        )

    if not candidates:
        half = atr * 0.12

        candidates.append(
            {
                "lower": (
                    current_price - half
                ),
                "upper": (
                    current_price + half
                ),
                "source": (
                    "ATR pullback"
                ),
                "priority": 4,
            }
        )

    def score(candidate):
        distance = _zone_distance(
            candidate,
            current_price,
        )

        # Prefer a retracement zone on the correct side of price.
        if direction == "long":
            wrong_side = (
                candidate["lower"]
                > current_price
            )
        else:
            wrong_side = (
                candidate["upper"]
                < current_price
            )

        return (
            1 if wrong_side else 0,
            candidate["priority"],
            distance,
        )

    best = min(
        candidates,
        key=score,
    )

    best = dict(best)

    best.pop(
        "priority",
        None,
    )

    return best


def _build_stop(
    analysis_15m,
    entry_zone,
    direction,
):
    atr = max(
        float(
            analysis_15m.get(
                "atr",
                0,
            )
        ),
        0.001,
    )

    setup = analysis_15m.get(
        "setup",
        {},
    )

    sweep = setup.get(
        "sweep"
    )

    if direction == "long":
        references = [
            float(
                entry_zone["lower"]
            )
        ]

        ssl = _price(
            analysis_15m.get(
                "ssl"
            )
        )

        if ssl is not None:
            references.append(
                float(ssl)
            )

        demand = (
            analysis_15m.get(
                "nearest_demand"
            )
        )

        if demand:
            references.append(
                float(
                    demand["lower"]
                )
            )

        if (
            sweep
            and sweep.get(
                "direction"
            )
            == "bullish"
        ):
            references.append(
                float(
                    sweep.get(
                        "extreme",
                        sweep["level"],
                    )
                )
            )

        invalidation = min(
            references
        )

        return (
            invalidation
            - 0.10 * atr
        )

    references = [
        float(
            entry_zone["upper"]
        )
    ]

    bsl = _price(
        analysis_15m.get(
            "bsl"
        )
    )

    if bsl is not None:
        references.append(
            float(bsl)
        )

    supply = (
        analysis_15m.get(
            "nearest_supply"
        )
    )

    if supply:
        references.append(
            float(
                supply["upper"]
            )
        )

    if (
        sweep
        and sweep.get(
            "direction"
        )
        == "bearish"
    ):
        references.append(
            float(
                sweep.get(
                    "extreme",
                    sweep["level"],
                )
            )
        )

    invalidation = max(
        references
    )

    return (
        invalidation
        + 0.10 * atr
    )


def _liquidity_targets(
    analysis_4h,
    analysis_1h,
    analysis_15m,
    direction,
    entry_mid,
    risk,
):
    candidates = []

    if direction == "long":
        raw = [
            (
                _price(
                    analysis_15m.get(
                        "bsl"
                    )
                ),
                "15M BSL",
            ),
            (
                _price(
                    analysis_1h.get(
                        "bsl"
                    )
                ),
                "1H BSL",
            ),
            (
                _price(
                    analysis_4h.get(
                        "bsl"
                    )
                ),
                "4H BSL",
            ),
        ]

        for value, source in raw:
            if (
                value is not None
                and float(value)
                > entry_mid
            ):
                candidates.append(
                    (
                        float(value),
                        source,
                    )
                )

        candidates.sort(
            key=lambda item: (
                item[0]
            )
        )

    else:
        raw = [
            (
                _price(
                    analysis_15m.get(
                        "ssl"
                    )
                ),
                "15M SSL",
            ),
            (
                _price(
                    analysis_1h.get(
                        "ssl"
                    )
                ),
                "1H SSL",
            ),
            (
                _price(
                    analysis_4h.get(
                        "ssl"
                    )
                ),
                "4H SSL",
            ),
        ]

        for value, source in raw:
            if (
                value is not None
                and float(value)
                < entry_mid
            ):
                candidates.append(
                    (
                        float(value),
                        source,
                    )
                )

        candidates.sort(
            key=lambda item: (
                -item[0]
            )
        )

    deduped = []

    for value, source in candidates:
        if not any(
            abs(
                value
                - item[0]
            )
            < 1e-8
            for item
            in deduped
        ):
            deduped.append(
                (
                    value,
                    source,
                )
            )

    rr_multiple = 1

    while len(deduped) < 3:
        if direction == "long":
            value = (
                entry_mid
                + risk
                * rr_multiple
            )
        else:
            value = (
                entry_mid
                - risk
                * rr_multiple
            )

        if not any(
            abs(
                value
                - item[0]
            )
            < 1e-8
            for item
            in deduped
        ):
            deduped.append(
                (
                    value,
                    f"{rr_multiple}R",
                )
            )

        rr_multiple += 1

    return deduped[:3]


def build_trade_plan(
    analysis_4h,
    analysis_1h,
    analysis_15m,
    status,
):
    """
    Build a rule-based execution map from the active 15M SMC setup.

    This is not an order instruction. It is a deterministic map used
    for charting, monitoring, and comparing setup development over time.
    """

    setup = analysis_15m.get(
        "setup",
        {},
    )

    score = int(
        setup.get(
            "score",
            0,
        )
        or 0
    )

    if (
        status == "WAIT"
        or score < 2
    ):
        return {
            "active": False,
            "status": status,
            "reason": (
                "No active 15M setup "
                "with at least 2/4 signals."
            ),
        }

    direction = (
        "long"
        if "LONG" in status
        else "short"
    )

    entry_zone = (
        _candidate_entry_zones(
            analysis_15m,
            direction,
        )
    )

    entry_mid = (
        float(
            entry_zone["lower"]
        )
        + float(
            entry_zone["upper"]
        )
    ) / 2

    stop_loss = _build_stop(
        analysis_15m,
        entry_zone,
        direction,
    )

    if direction == "long":
        risk = (
            entry_mid
            - stop_loss
        )
    else:
        risk = (
            stop_loss
            - entry_mid
        )

    atr = max(
        float(
            analysis_15m.get(
                "atr",
                0,
            )
        ),
        0.001,
    )

    # Never publish a zero/negative-risk plan.
    if risk <= 0:
        risk = atr * 0.5

        stop_loss = (
            entry_mid - risk
            if direction == "long"
            else entry_mid + risk
        )

    target_pairs = (
        _liquidity_targets(
            analysis_4h,
            analysis_1h,
            analysis_15m,
            direction,
            entry_mid,
            risk,
        )
    )

    targets = []

    for index, (
        value,
        source,
    ) in enumerate(
        target_pairs,
        start=1,
    ):
        reward = (
            value - entry_mid
            if direction == "long"
            else entry_mid - value
        )

        rr = (
            reward / risk
            if risk > 0
            else None
        )

        targets.append(
            {
                "name": (
                    f"TP{index}"
                ),
                "price": value,
                "source": source,
                "rr": rr,
            }
        )

    setup_state = (
        "confirmed"
        if setup.get(
            "confirmed",
            False,
        )
        else "developing"
    )

    trigger = (
        "Wait for price to trade into the entry zone "
        "and hold/reject in the setup direction."
    )

    return {
        "active": True,
        "direction": direction,
        "setup_state": (
            setup_state
        ),
        "setup_score": score,
        "entry_zone": {
            "lower": float(
                entry_zone["lower"]
            ),
            "upper": float(
                entry_zone["upper"]
            ),
            "source": (
                entry_zone["source"]
            ),
        },
        "entry_mid": entry_mid,
        "stop_loss": (
            stop_loss
        ),
        "risk": risk,
        "targets": targets,
        "trigger": trigger,
    }
