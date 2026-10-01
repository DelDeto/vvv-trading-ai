import json
import os
from datetime import datetime, timezone

from mexc_data import (
    get_closed_klines,
    get_contract_snapshot,
)
from chart import create_chart
from trade_plan import build_trade_plan
from signal_journal import (
    update_signal_journal,
)
from smc_analysis import (
    analyze_smc,
    derive_overall_status,
    derive_mtf_alignment,
)


STATE_PATH = "state.json"
HISTORY_LIMIT = 420


def _ohlc_rows(
    df,
    limit=160,
):
    rows = []

    for timestamp, row in (
        df.tail(limit)
        .iterrows()
    ):
        rows.append(
            {
                "time": (
                    timestamp.isoformat()
                ),
                "open": float(
                    row["open"]
                ),
                "high": float(
                    row["high"]
                ),
                "low": float(
                    row["low"]
                ),
                "close": float(
                    row["close"]
                ),
                "volume": float(
                    row["volume"]
                ),
            }
        )

    return rows


def _fmt(value):
    if value is None:
        return "-"

    return f"{value:.3f}"


def _zone_text(zone):
    if not zone:
        return "-"

    quality = zone.get(
        "quality"
    )

    grade = zone.get(
        "grade"
    )

    pattern = zone.get(
        "pattern"
    )

    mitigations = zone.get(
        "mitigations"
    )

    suffix = ""

    if quality:
        suffix += (
            f" [{quality}"
        )

        if grade:
            suffix += (
                f", {grade}"
            )

        if pattern:
            suffix += (
                f", {pattern}"
            )

        if mitigations is not None:
            suffix += (
                f", M{mitigations}"
            )

        suffix += "]"

    return (
        f"{_fmt(zone['lower'])}"
        f"-{_fmt(zone['upper'])}"
        f"{suffix}"
    )


def _fvg_text(analysis):
    active = analysis.get(
        "active_fvgs",
        [],
    )[-2:]

    if not active:
        return "-"

    parts = []

    for fvg in active:
        parts.append(
            (
                f"{fvg['type'].upper()} "
                f"{_fmt(fvg['lower'])}-"
                f"{_fmt(fvg['upper'])}"
            )
        )

    return " | ".join(parts)


def _event_text(analysis):
    event = analysis.get(
        "last_event"
    )

    if not event:
        return "None"

    return (
        f"{event['kind']} "
        f"{event['direction'].upper()} "
        f"@ {_fmt(event['level'])}"
    )


def _sweep_text(analysis):
    event = (
        analysis.get(
            "setup",
            {},
        ).get(
            "sweep"
        )
    )

    if not event:
        return "None"

    return (
        f"{event['type']} SWEEP "
        f"-> {event['direction'].upper()} "
        f"@ {_fmt(event['level'])}"
    )


def _displacement_text(analysis):
    event = (
        analysis.get(
            "setup",
            {},
        ).get(
            "displacement"
        )
    )

    if not event:
        return "None"

    return (
        f"{event['direction'].upper()} "
        f"x{event['strength']:.2f} ATR"
    )


def _retest_text(analysis):
    event = (
        analysis.get(
            "setup",
            {},
        ).get(
            "retest"
        )
    )

    if not event:
        return "None"

    return (
        f"{event['structure_kind']} "
        f"{event['direction'].upper()} "
        f"@ {_fmt(event['level'])}"
    )


def _setup_text(analysis):
    setup = analysis.get(
        "setup",
        {},
    )

    if not setup:
        return "None"

    direction = setup.get(
        "direction",
        "-",
    ).upper()

    score = setup.get(
        "score",
        0,
    )

    if setup.get(
        "confirmed",
        False,
    ):
        state = "CONFIRMED"

    elif score >= 2:
        state = "DEVELOPING"

    elif score == 1:
        state = "WATCH"

    else:
        state = "WAIT"

    return (
        f"{state} {direction} "
        f"({score}/4 signals)"
    )


def _reference_zone_text(
    analysis,
    side,
):
    key = (
        "reference_demand_zones"
        if side == "demand"
        else "reference_supply_zones"
    )

    zones = analysis.get(
        key,
        [],
    )

    if not zones:
        return "-"

    zone = zones[0]

    pattern = zone.get(
        "pattern",
        "?",
    )

    mitigations = zone.get(
        "mitigations",
        0,
    )

    return (
        f"{_fmt(zone['lower'])}-"
        f"{_fmt(zone['upper'])} "
        f"[HISTORICAL, {pattern}, M{mitigations}]"
    )


def _rejected_zone_text(
    analysis,
    side,
):
    key = (
        "rejected_demand_zones"
        if side == "demand"
        else "rejected_supply_zones"
    )

    zones = analysis.get(
        key,
        [],
    )

    if not zones:
        return "-"

    zone = zones[0]

    reasons = ",".join(
        zone.get(
            "rejection_reasons",
            [],
        )
    )

    pattern = zone.get(
        "pattern",
        "?",
    )

    return (
        f"{_fmt(zone['lower'])}-"
        f"{_fmt(zone['upper'])} "
        f"[{pattern}; {reasons}]"
    )


def _print_summary(
    timeframe,
    analysis,
):
    bsl = analysis.get("bsl")
    ssl = analysis.get("ssl")

    print()
    print(
        f"[{timeframe}] "
        f"Trend: "
        f"{analysis['trend'].upper()}"
    )

    print(
        f"[{timeframe}] "
        f"Regime: "
        f"{analysis.get('regime', '-').upper()}"
    )

    print(
        f"[{timeframe}] "
        f"Current: "
        f"{_fmt(analysis['current_price'])}"
    )

    print(
        f"[{timeframe}] "
        f"BSL: "
        f"{_fmt(bsl['price']) if bsl else '-'}"
    )

    print(
        f"[{timeframe}] "
        f"SSL: "
        f"{_fmt(ssl['price']) if ssl else '-'}"
    )

    print(
        f"[{timeframe}] "
        f"Demand: "
        f"{_zone_text(analysis.get('nearest_demand'))}"
    )

    print(
        f"[{timeframe}] "
        f"Supply: "
        f"{_zone_text(analysis.get('nearest_supply'))}"
    )

    print(
        f"[{timeframe}] "
        f"Reference Demand: "
        f"{_reference_zone_text(analysis, 'demand')}"
    )

    print(
        f"[{timeframe}] "
        f"Reference Supply: "
        f"{_reference_zone_text(analysis, 'supply')}"
    )

    print(
        f"[{timeframe}] "
        f"Rejected Demand: "
        f"{_rejected_zone_text(analysis, 'demand')}"
    )

    print(
        f"[{timeframe}] "
        f"Rejected Supply: "
        f"{_rejected_zone_text(analysis, 'supply')}"
    )

    print(
        f"[{timeframe}] "
        f"FVG: "
        f"{_fvg_text(analysis)}"
    )

    print(
        f"[{timeframe}] "
        f"Last structure: "
        f"{_event_text(analysis)}"
    )

    print(
        f"[{timeframe}] "
        f"Liquidity sweep: "
        f"{_sweep_text(analysis)}"
    )

    print(
        f"[{timeframe}] "
        f"Displacement: "
        f"{_displacement_text(analysis)}"
    )

    print(
        f"[{timeframe}] "
        f"Retest: "
        f"{_retest_text(analysis)}"
    )

    print(
        f"[{timeframe}] "
        f"Setup sequence: "
        f"{_setup_text(analysis)}"
    )


def _load_state():
    if not os.path.exists(
        STATE_PATH
    ):
        return None

    try:
        with open(
            STATE_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except Exception as exc:
        print(
            "State load warning:",
            exc,
        )
        return None


def _compact_event(analysis):
    event = analysis.get(
        "last_event"
    )

    if not event:
        return None

    return {
        "time": event.get("time"),
        "direction": event.get(
            "direction"
        ),
        "kind": event.get("kind"),
        "level": event.get("level"),
    }


def _compact_zone(zone):
    if not zone:
        return None

    return {
        "lower": zone.get("lower"),
        "upper": zone.get("upper"),
        "quality": zone.get(
            "quality"
        ),
        "grade": zone.get(
            "grade"
        ),
        "mitigations": zone.get(
            "mitigations"
        ),
    }


def _compact_liquidity(point):
    if not point:
        return None

    return {
        "price": point.get("price"),
        "time": point.get("time"),
    }


def _make_state(report):
    state = {
        "generated_at_utc": (
            report[
                "generated_at_utc"
            ]
        ),
        "current_price": (
            report[
                "current_price"
            ]
        ),
        "status": (
            report[
                "status"
            ]
        ),
        "mtf_alignment": (
            report.get(
                "mtf_alignment"
            )
        ),
        "trade_plan": (
            report.get(
                "trade_plan"
            )
        ),
        "market_snapshot": (
            report.get(
                "market_snapshot"
            )
        ),
        "timeframes": {},
    }

    for timeframe, analysis in (
        report["timeframes"].items()
    ):
        state["timeframes"][
            timeframe
        ] = {
            "trend": (
                analysis.get("trend")
            ),
            "regime": (
                analysis.get(
                    "regime"
                )
            ),
            "bsl": _compact_liquidity(
                analysis.get("bsl")
            ),
            "ssl": _compact_liquidity(
                analysis.get("ssl")
            ),
            "last_event": (
                _compact_event(
                    analysis
                )
            ),
            "last_sweep": (
                analysis.get(
                    "setup",
                    {},
                ).get(
                    "sweep"
                )
            ),
            "last_displacement": (
                analysis.get(
                    "setup",
                    {},
                ).get(
                    "displacement"
                )
            ),
            "last_retest": (
                analysis.get(
                    "setup",
                    {},
                ).get(
                    "retest"
                )
            ),
            "setup": {
                "direction": (
                    analysis.get(
                        "setup",
                        {},
                    ).get(
                        "direction"
                    )
                ),
                "score": (
                    analysis.get(
                        "setup",
                        {},
                    ).get(
                        "score"
                    )
                ),
                "confirmed": (
                    analysis.get(
                        "setup",
                        {},
                    ).get(
                        "confirmed"
                    )
                ),
            },
            "nearest_supply": (
                _compact_zone(
                    analysis.get(
                        "nearest_supply"
                    )
                )
            ),
            "nearest_demand": (
                _compact_zone(
                    analysis.get(
                        "nearest_demand"
                    )
                )
            ),
        }

    return state


def _event_signature(event):
    if not event:
        return None

    return (
        event.get("time"),
        event.get("kind"),
        event.get("direction"),
        round(
            float(
                event.get(
                    "level",
                    0,
                )
            ),
            6,
        ),
    )


def _trade_plan_signature(plan):
    if not plan:
        return None

    if not plan.get(
        "active",
        False,
    ):
        return (
            False,
            plan.get("status"),
        )

    entry = plan.get(
        "entry_zone",
        {},
    )

    targets = plan.get(
        "targets",
        [],
    )

    return (
        True,
        plan.get("direction"),
        plan.get("setup_state"),
        plan.get(
            "execution_ready"
        ),
        tuple(
            plan.get(
                "blockers",
                [],
            )
        ),
        round(
            float(
                entry.get(
                    "lower",
                    0,
                )
            ),
            3,
        ),
        round(
            float(
                entry.get(
                    "upper",
                    0,
                )
            ),
            3,
        ),
        round(
            float(
                plan.get(
                    "stop_loss",
                    0,
                )
            ),
            3,
        ),
        tuple(
            round(
                float(
                    item.get(
                        "price",
                        0,
                    )
                ),
                3,
            )
            for item
            in targets
        ),
    )


def _compare_state(
    previous,
    current,
):
    changes = []

    if not previous:
        return (
            changes,
            True,
            None,
        )

    previous_price = previous.get(
        "current_price"
    )

    current_price = current.get(
        "current_price"
    )

    price_change_pct = None

    if (
        previous_price
        not in (None, 0)
    ):
        price_change_pct = (
            (
                current_price
                - previous_price
            )
            / previous_price
            * 100
        )

        if abs(
            price_change_pct
        ) >= 1.0:
            changes.append(
                (
                    "Price moved "
                    f"{price_change_pct:+.2f}% "
                    "since prior run"
                )
            )

    if (
        previous.get("status")
        != current.get("status")
    ):
        changes.append(
            (
                "Status changed: "
                f"{previous.get('status')} "
                "-> "
                f"{current.get('status')}"
            )
        )

    if (
        (
            previous.get(
                "mtf_alignment",
                {},
            )
            or {}
        ).get(
            "label"
        )
        !=
        (
            current.get(
                "mtf_alignment",
                {},
            )
            or {}
        ).get(
            "label"
        )
    ):
        changes.append(
            (
                "MTF alignment changed: "
                f"{(previous.get('mtf_alignment') or {}).get('label')} "
                "-> "
                f"{(current.get('mtf_alignment') or {}).get('label')}"
            )
        )

    if (
        _trade_plan_signature(
            previous.get(
                "trade_plan"
            )
        )
        !=
        _trade_plan_signature(
            current.get(
                "trade_plan"
            )
        )
    ):
        changes.append(
            "Trade plan changed"
        )

    for timeframe in [
        "4H",
        "1H",
        "15M",
    ]:
        previous_tf = (
            previous
            .get(
                "timeframes",
                {},
            )
            .get(
                timeframe,
                {},
            )
        )

        current_tf = (
            current
            .get(
                "timeframes",
                {},
            )
            .get(
                timeframe,
                {},
            )
        )

        if (
            previous_tf.get("trend")
            != current_tf.get(
                "trend"
            )
        ):
            changes.append(
                (
                    f"{timeframe} trend: "
                    f"{previous_tf.get('trend')} "
                    "-> "
                    f"{current_tf.get('trend')}"
                )
            )

        if (
            previous_tf.get(
                "regime"
            )
            != current_tf.get(
                "regime"
            )
        ):
            changes.append(
                (
                    f"{timeframe} regime: "
                    f"{previous_tf.get('regime')} "
                    "-> "
                    f"{current_tf.get('regime')}"
                )
            )

        if (
            _event_signature(
                previous_tf.get(
                    "last_event"
                )
            )
            !=
            _event_signature(
                current_tf.get(
                    "last_event"
                )
            )
        ):
            changes.append(
                (
                    f"{timeframe} structure "
                    "event changed"
                )
            )

        for field, label in [
            ("last_sweep", "liquidity sweep"),
            ("last_displacement", "displacement"),
            ("last_retest", "retest"),
        ]:
            previous_signal = (
                previous_tf.get(field)
                or {}
            )

            current_signal = (
                current_tf.get(field)
                or {}
            )

            previous_signature = (
                previous_signal.get("time"),
                previous_signal.get("direction"),
                previous_signal.get("level"),
                previous_signal.get("type"),
                previous_signal.get("structure_kind"),
            )

            current_signature = (
                current_signal.get("time"),
                current_signal.get("direction"),
                current_signal.get("level"),
                current_signal.get("type"),
                current_signal.get("structure_kind"),
            )

            if (
                previous_signature
                != current_signature
            ):
                changes.append(
                    f"{timeframe} {label} changed"
                )

        if (
            previous_tf.get("setup")
            != current_tf.get("setup")
        ):
            changes.append(
                f"{timeframe} setup sequence changed"
            )

    material_change = bool(
        changes
    )

    return (
        changes,
        material_change,
        price_change_pct,
    )


def _invalidation_and_targets(
    status,
    analyses,
):
    a15 = analyses["15M"]
    a1h = analyses["1H"]

    bsl_15 = (
        a15.get("bsl") or {}
    ).get("price")

    ssl_15 = (
        a15.get("ssl") or {}
    ).get("price")

    bsl_1h = (
        a1h.get("bsl") or {}
    ).get("price")

    ssl_1h = (
        a1h.get("ssl") or {}
    ).get("price")

    supply_15 = (
        a15.get(
            "nearest_supply"
        )
        or {}
    )

    demand_15 = (
        a15.get(
            "nearest_demand"
        )
        or {}
    )

    if "LONG" in status:
        invalidation = (
            demand_15.get("lower")
            or ssl_15
        )

        targets = [
            value
            for value in [
                bsl_15,
                bsl_1h,
            ]
            if value is not None
        ]

    elif "SHORT" in status:
        invalidation = (
            supply_15.get("upper")
            or bsl_15
        )

        targets = [
            value
            for value in [
                ssl_15,
                ssl_1h,
            ]
            if value is not None
        ]

    else:
        invalidation = None

        targets = [
            value
            for value in [
                ssl_15,
                bsl_15,
            ]
            if value is not None
        ]

    # Preserve order, remove duplicates.
    deduped = []

    for value in targets:
        if value not in deduped:
            deduped.append(value)

    return (
        invalidation,
        deduped,
    )


def _market_snapshot_lines(
    snapshot,
):
    if not snapshot:
        return []

    lines = []

    change_rate = snapshot.get(
        "change_rate_24h"
    )

    hold_vol = snapshot.get(
        "hold_vol"
    )

    hold_change = snapshot.get(
        "hold_vol_change_pct"
    )

    funding = snapshot.get(
        "funding_rate"
    )

    if change_rate is not None:
        lines.append(
            (
                "24h change: "
                f"{change_rate * 100:+.2f}%"
            )
        )

    if hold_vol is not None:
        oi_text = (
            f"{hold_vol:,.0f}"
        )

        if hold_change is not None:
            oi_text += (
                f" ({hold_change:+.2f}%)"
            )

        lines.append(
            "MEXC holdVol (OI proxy): "
            + oi_text
        )

    if funding is not None:
        lines.append(
            (
                "Funding: "
                f"{funding * 100:+.4f}%"
            )
        )

    return lines


def _build_hourly_update(
    report,
    previous_state,
    changes,
    material_change,
    price_change_pct,
):
    analyses = report[
        "timeframes"
    ]

    status = report["status"]

    invalidation, targets = (
        _invalidation_and_targets(
            status,
            analyses,
        )
    )

    lines = []

    lines.append(
        "VVV_USDT MEXC HOURLY SMC UPDATE"
    )

    lines.append(
        "=" * 42
    )

    lines.append(
        (
            "Generated UTC: "
            f"{report['generated_at_utc']}"
        )
    )

    lines.append(
        (
            "Current price: "
            f"{_fmt(report['current_price'])}"
        )
    )

    if (
        previous_state
        and price_change_pct
        is not None
    ):
        lines.append(
            (
                "Change vs prior run: "
                f"{price_change_pct:+.2f}%"
            )
        )

    else:
        lines.append(
            (
                "Change vs prior run: "
                "N/A"
            )
        )

    lines.append(
        (
            "Status: "
            f"{status}"
        )
    )

    alignment = (
        report.get(
            "mtf_alignment",
            {}
        )
        or {}
    )

    lines.append(
        (
            "MTF Alignment: "
            f"{alignment.get('label', 'NEUTRAL')}"
        )
    )

    snapshot = report.get(
        "market_snapshot"
    )

    lines.extend(
        _market_snapshot_lines(
            snapshot
        )
    )

    participation = (
        (snapshot or {}).get(
            "participation_context",
            {},
        )
    )

    if participation:
        lines.append(
            (
                "Participation: "
                f"{participation.get('regime', 'NEUTRAL_MIXED')}"
            )
        )

    evaluation = (
        report.get(
            "evaluation_summary",
            {},
        )
        or {}
    )

    if evaluation.get(
        "total_signals",
        0,
    ):
        lines.append(
            (
                "Forward eval: "
                f"{evaluation.get('total_signals', 0)} signals | "
                f"{evaluation.get('resolved', 0)} resolved | "
                f"{evaluation.get('calibration_status', 'COLLECTING_DATA')}"
            )
        )

    trade_plan = report.get(
        "trade_plan",
        {},
    )

    if trade_plan.get(
        "active",
        False,
    ):
        entry = trade_plan[
            "entry_zone"
        ]

        lines.append("")
        lines.append(
            (
                "TRADE MAP: "
                f"{trade_plan['direction'].upper()} "
                f"({trade_plan['setup_state'].upper()})"
            )
        )

        execution_ready = (
            trade_plan.get(
                "execution_ready",
                False,
            )
        )

        lines.append(
            (
                "Execution: "
                + (
                    "READY"
                    if execution_ready
                    else "WAIT"
                )
            )
        )

        blockers = trade_plan.get(
            "blockers",
            [],
        )

        if blockers:
            lines.append(
                "Blockers:"
            )

            for blocker in blockers:
                lines.append(
                    f"- {blocker}"
                )

        lines.append(
            (
                "Entry zone: "
                f"{_fmt(entry['lower'])}-"
                f"{_fmt(entry['upper'])} "
                f"[{entry['source']}"
                + (
                    f", grade {entry.get('zone_grade')}"
                    if entry.get(
                        "zone_grade"
                    )
                    else ""
                )
                + "]"
            )
        )

        lines.append(
            (
                "SL / invalidation: "
                f"{_fmt(trade_plan['stop_loss'])}"
            )
        )

        for target in (
            trade_plan.get(
                "targets",
                []
            )
        ):
            rr = target.get(
                "rr"
            )

            rr_text = (
                f"{rr:.2f}R"
                if rr is not None
                else "-"
            )

            lines.append(
                (
                    f"{target['name']}: "
                    f"{_fmt(target['price'])} "
                    f"[{target['source']}, "
                    f"{rr_text}]"
                )
            )

        lines.append(
            (
                "Trigger: "
                f"{trade_plan['trigger']}"
            )
        )

    lines.append("")

    for timeframe in [
        "4H",
        "1H",
        "15M",
    ]:
        analysis = analyses[
            timeframe
        ]

        bsl = analysis.get("bsl")
        ssl = analysis.get("ssl")

        lines.append(
            (
                f"[{timeframe}] "
                f"Trend: "
                f"{analysis['trend'].upper()}"
            )
        )

        lines.append(
            (
                f"[{timeframe}] "
                "Regime: "
                f"{analysis.get('regime', '-').upper()}"
            )
        )

        lines.append(
            (
                f"[{timeframe}] "
                "Momentum/structure: "
                f"{_event_text(analysis)}"
            )
        )

        lines.append(
            (
                f"[{timeframe}] "
                "BSL: "
                f"{_fmt(bsl['price']) if bsl else '-'}"
                " | SSL: "
                f"{_fmt(ssl['price']) if ssl else '-'}"
            )
        )

        lines.append(
            (
                f"[{timeframe}] "
                "Demand: "
                f"{_zone_text(analysis.get('nearest_demand'))}"
                " | Supply: "
                f"{_zone_text(analysis.get('nearest_supply'))}"
            )
        )

        lines.append(
            (
                f"[{timeframe}] "
                "Active FVG: "
                f"{_fvg_text(analysis)}"
            )
        )

        lines.append(
            (
                f"[{timeframe}] "
                "Liquidity Sweep: "
                f"{_sweep_text(analysis)}"
            )
        )

        lines.append(
            (
                f"[{timeframe}] "
                "Displacement: "
                f"{_displacement_text(analysis)}"
            )
        )

        lines.append(
            (
                f"[{timeframe}] "
                "Retest: "
                f"{_retest_text(analysis)}"
            )
        )

        lines.append(
            (
                f"[{timeframe}] "
                "Setup: "
                f"{_setup_text(analysis)}"
            )
        )

        lines.append("")

    if invalidation is not None:
        lines.append(
            (
                "Working invalidation: "
                f"{_fmt(invalidation)}"
            )
        )

    if targets:
        lines.append(
            (
                "Nearby liquidity targets: "
                + " -> ".join(
                    _fmt(value)
                    for value
                    in targets
                )
            )
        )

    lines.append("")

    if material_change:
        lines.append(
            "Material changes:"
        )

        if changes:
            for change in changes:
                lines.append(
                    f"- {change}"
                )

        else:
            lines.append(
                "- First baseline run"
            )

    else:
        lines.append(
            (
                "Nothing materially changed "
                "since the previous run."
            )
        )

    return "\n".join(lines)


def _participation_context(
    previous_state,
    market_snapshot,
    current_price,
):
    """
    Price + MEXC holdVol context. This is descriptive only and never
    acts as a standalone LONG/SHORT signal.
    """

    previous_snapshot = (
        previous_state.get(
            "market_snapshot",
            {},
        )
        if previous_state
        else {}
    )

    previous_hold = (
        previous_snapshot.get(
            "hold_vol"
        )
        if previous_snapshot
        else None
    )

    previous_price = (
        previous_state.get(
            "current_price"
        )
        if previous_state
        else None
    )

    current_hold = (
        market_snapshot.get(
            "hold_vol"
        )
    )

    hold_change_pct = None
    price_change_pct = None

    if (
        previous_hold
        not in (None, 0)
        and current_hold
        is not None
    ):
        hold_change_pct = (
            (
                current_hold
                - previous_hold
            )
            / previous_hold
            * 100
        )

    if (
        previous_price
        not in (None, 0)
        and current_price
        is not None
    ):
        price_change_pct = (
            (
                current_price
                - previous_price
            )
            / previous_price
            * 100
        )

    threshold = 0.10

    price_up = (
        price_change_pct
        is not None
        and price_change_pct
        >= threshold
    )

    price_down = (
        price_change_pct
        is not None
        and price_change_pct
        <= -threshold
    )

    hold_up = (
        hold_change_pct
        is not None
        and hold_change_pct
        >= threshold
    )

    hold_down = (
        hold_change_pct
        is not None
        and hold_change_pct
        <= -threshold
    )

    if price_up and hold_up:
        regime = "PRICE_UP_HOLD_UP"
        note = (
            "Giá tăng cùng holdVol tăng; "
            "mức tham gia vị thế đang mở rộng."
        )

    elif price_down and hold_up:
        regime = "PRICE_DOWN_HOLD_UP"
        note = (
            "Giá giảm cùng holdVol tăng; "
            "mức tham gia vị thế đang mở rộng theo nhịp giảm."
        )

    elif price_up and hold_down:
        regime = "PRICE_UP_HOLD_DOWN"
        note = (
            "Giá tăng nhưng holdVol giảm; "
            "có thể phản ánh đóng vị thế/short covering."
        )

    elif price_down and hold_down:
        regime = "PRICE_DOWN_HOLD_DOWN"
        note = (
            "Giá giảm nhưng holdVol giảm; "
            "có thể phản ánh đóng vị thế/long liquidation."
        )

    else:
        regime = "NEUTRAL_MIXED"
        note = (
            "Biến động giá/holdVol chưa đủ rõ "
            "để tạo bối cảnh participation."
        )

    return {
        "regime": regime,
        "price_change_pct": (
            price_change_pct
        ),
        "hold_vol_change_pct": (
            hold_change_pct
        ),
        "note": note,
        "signal_weight": "context_only",
    }


def main():

    print("=" * 50)
    print(
        "VVV_USDT MEXC FUTURES "
        "+ SMC ANALYSIS"
    )
    print("=" * 50)

    previous_state = (
        _load_state()
    )

    # Real MEXC Futures OHLC.
    # Signals are calculated ONLY from fully closed candles.
    # 420 bars covers the longest current zone-policy horizon
    # (max_age 360) with enough warm-up history.
    df_4h = get_closed_klines(
        "4h",
        HISTORY_LIMIT,
        min_required=360,
    )

    df_1h = get_closed_klines(
        "1h",
        HISTORY_LIMIT,
        min_required=360,
    )

    df_15m = get_closed_klines(
        "15m",
        HISTORY_LIMIT,
        min_required=360,
    )

    market_snapshot = (
        get_contract_snapshot()
    )

    closed_analysis_price = float(
        df_15m.iloc[-1][
            "close"
        ]
    )

    live_price = (
        market_snapshot.get(
            "last_price"
        )
    )

    if live_price is None:
        live_price = (
            closed_analysis_price
        )

    participation = (
        _participation_context(
            previous_state,
            market_snapshot,
            live_price,
        )
    )

    market_snapshot[
        "hold_vol_change_pct"
    ] = participation.get(
        "hold_vol_change_pct"
    )

    market_snapshot[
        "participation_context"
    ] = participation

    # Rule-based SMC analysis
    smc_4h = analyze_smc(
        df_4h,
        timeframe="4H",
    )

    smc_1h = analyze_smc(
        df_1h,
        timeframe="1H",
    )

    smc_15m = analyze_smc(
        df_15m,
        timeframe="15M",
    )

    analyses = {
        "4H": smc_4h,
        "1H": smc_1h,
        "15M": smc_15m,
    }

    status = derive_overall_status(
        smc_4h,
        smc_1h,
        smc_15m,
    )

    mtf_alignment = (
        derive_mtf_alignment(
            smc_4h,
            smc_1h,
            smc_15m,
        )
    )

    analysis_price = float(
        df_15m.iloc[-1]["close"]
    )

    current_price = live_price

    trade_plan = build_trade_plan(
        smc_4h,
        smc_1h,
        smc_15m,
        status,
        mtf_alignment=(
            mtf_alignment
        ),
    )

    generated_at = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    print()
    print(
        f"Live VVV_USDT price: "
        f"{current_price}"
    )

    print(
        (
            "Last closed 15M: "
            f"{analysis_price} @ "
            f"{df_15m.index[-1].isoformat()}"
        )
    )

    _print_summary(
        "4H",
        smc_4h,
    )

    _print_summary(
        "1H",
        smc_1h,
    )

    _print_summary(
        "15M",
        smc_15m,
    )

    print()
    print(
        "OVERALL STATUS:",
        status,
    )

    print(
        "MTF ALIGNMENT:",
        mtf_alignment.get(
            "label"
        ),
    )

    os.makedirs(
        "output",
        exist_ok=True,
    )

    report = {
        "generated_at_utc": (
            generated_at
        ),
        "exchange": "MEXC",
        "symbol": "VVV_USDT",
        "current_price": (
            current_price
        ),
        "analysis_price": (
            analysis_price
        ),
        "last_closed_candle": {
            "4H": (
                df_4h.index[-1]
                .isoformat()
            ),
            "1H": (
                df_1h.index[-1]
                .isoformat()
            ),
            "15M": (
                df_15m.index[-1]
                .isoformat()
            ),
        },
        "data_integrity": {
            "closed_candles_only": True,
            "history_bars_requested": (
                HISTORY_LIMIT
            ),
            "actual_history_bars": {
                "4H": len(df_4h),
                "1H": len(df_1h),
                "15M": len(df_15m),
            },
            "chart_bars": 160,
            "ai_raw_context_bars": 64,
        },
        "status": status,
        "mtf_alignment": (
            mtf_alignment
        ),
        "trade_plan": (
            trade_plan
        ),
        "market_snapshot": (
            market_snapshot
        ),
        "timeframes": analyses,
    }

    evaluation_summary = (
        update_signal_journal(
            report,
            df_15m,
        )
    )

    report[
        "evaluation_summary"
    ] = evaluation_summary

    current_state = (
        _make_state(
            report
        )
    )

    (
        changes,
        material_change,
        price_change_pct,
    ) = _compare_state(
        previous_state,
        current_state,
    )

    update_text = (
        _build_hourly_update(
            report,
            previous_state,
            changes,
            material_change,
            price_change_pct,
        )
    )

    print()
    print(update_text)

    with open(
        "output/VVVUSDT_SMC_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    ai_input = {
        "report": report,
        "ohlc": {
            "4H": _ohlc_rows(
                df_4h,
                160,
            ),
            "1H": _ohlc_rows(
                df_1h,
                160,
            ),
            "15M": _ohlc_rows(
                df_15m,
                160,
            ),
        },
    }

    with open(
        "output/VVVUSDT_ai_input.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            ai_input,
            file,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        "output/VVVUSDT_hourly_update.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            update_text
        )

    # Persist compact state for next workflow run.
    with open(
        STATE_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            current_state,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # Real candlestick charts + SMC overlay
    create_chart(
        df_4h,
        "4H",
        "output/VVVUSDT_4H.png",
        candles=160,
        exchange="MEXC",
        analysis=smc_4h,
        market_snapshot=(
            market_snapshot
        ),
    )

    create_chart(
        df_1h,
        "1H",
        "output/VVVUSDT_1H.png",
        candles=160,
        exchange="MEXC",
        analysis=smc_1h,
        market_snapshot=(
            market_snapshot
        ),
    )

    create_chart(
        df_15m,
        "15M",
        "output/VVVUSDT_15M.png",
        candles=160,
        exchange="MEXC",
        analysis=smc_15m,
        trade_plan=trade_plan,
        market_snapshot=(
            market_snapshot
        ),
    )

    print()
    print("=" * 50)
    print(
        "ALL MEXC SMC CHARTS CREATED"
    )
    print("=" * 50)


if __name__ == "__main__":
    main()
