import json
import math
import os
import re
from pathlib import Path

from openai import OpenAI
from signal_journal import (
    attach_ai_review,
)


OUTPUT_DIR = Path("output")
INPUT_PATH = OUTPUT_DIR / "VVVUSDT_ai_input.json"
AI_JSON_PATH = OUTPUT_DIR / "VVVUSDT_ai_analysis.json"
AI_TEXT_PATH = OUTPUT_DIR / "VVVUSDT_ai_analysis.txt"

MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "qwen/qwen3.8-27b:free",
)

FALLBACK_MODEL = os.getenv(
    "OPENROUTER_FALLBACK_MODEL",
    "openrouter/free",
)

OPENROUTER_BASE_URL = (
    "https://openrouter.ai/api/v1"
)

ANALYSIS_METHOD = "PA-MTF Hybrid V1"


SYSTEM_PROMPT = """
You are the qualitative analysis layer of a hybrid crypto market-analysis system.

SOURCE-OF-TRUTH RULES
- Every candle and every numerical market fact in the input came from deterministic Python using real MEXC OHLC/market data.
- Never invent, estimate, adjust, replace, or reinterpret a numerical price level.
- Do NOT create new support/resistance, entry, stop, target, Supply/Demand, FVG, BSL/SSL, BOS/CHoCH, OI, funding, VWAP, EMA, RSI, ATR, or volume values.
- Do not quote numerical price levels in your prose. Exact levels will be appended separately by Python.
- A Historical/Reference zone is context only, never an active execution zone.
- You do not draw or modify charts.
- Python execution_ready is the hard execution gate. If it is false, you must not recommend executing a trade now.
- AI bias is an interpretation, not a probability or certainty.

ANALYSIS FRAMEWORK: PA-MTF HYBRID V1
Evidence priority, from strongest to weaker:
1. Multi-timeframe Price Action structure: HH/HL vs LH/LL, BOS, CHoCH.
2. Location: active Supply/Demand V3 zones, DBR/RBR/RBD/DBD pattern, freshness and mitigation.
3. Liquidity: BSL/SSL location and liquidity sweeps.
4. Impulse/imbalance: displacement, FVG and retest.
5. Technical confirmation: EMA20/EMA50, anchored VWAP, ATR regime, RSI14 and relative volume.
6. Derivatives context: MEXC holdVol/open-position proxy and funding. Use only as context, never as a standalone signal.

TOP-DOWN PROCESS
- 4H defines macro context and major location.
- 1H defines intermediate structure and whether price is at a meaningful decision zone.
- 15M defines execution structure and timing.
- Explicitly identify timeframe conflicts. A 15M signal against 1H/4H context is lower quality unless the higher timeframe is at a reversal location and lower timeframe structure confirms.
- Prefer WAIT when structure is conflicted, displacement/retest is missing, or Python execution gate is WAIT.
- Do not force a directional view.

OUTPUT STYLE
- ALL explanatory string values MUST be written in Vietnamese. English is allowed only for standard technical acronyms/tokens such as BOS, CHoCH, FVG, EMA, VWAP, RSI, ATR, BSL, SSL, DBR, RBR, RBD, DBD, LONG, SHORT, WAIT.
- Compact, professional, trading-desk style.
- Explain why, not just the label.
- Never use certainty language such as "chắc chắn", "sẽ tăng", or "sẽ giảm".
- Never use unsupported probability language such as "xác suất cao", "high probability", or percentage-like confidence.
- Do not invent or nominate AI targets. Entry, SL and TP belong to the deterministic Python trade plan only.
- In bullish_scenario and bearish_scenario, describe confirmation/invalidation CONDITIONS only; do not propose new price targets.
- When describing whether price is inside/near/above/below a zone, use the supplied price_vs_active_demand / price_vs_active_supply fields instead of inferring location from raw numbers.
"""


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "ai_bias": {
            "type": "string",
            "enum": [
                "BULLISH",
                "BEARISH",
                "MIXED",
                "WAIT",
            ],
        },
        "market_context": {
            "type": "string",
        },
        "view_4h": {
            "type": "string",
        },
        "view_1h": {
            "type": "string",
        },
        "view_15m": {
            "type": "string",
        },
        "confluence": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "conflicts": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "preferred_scenario": {
            "type": "string",
        },
        "bullish_scenario": {
            "type": "string",
        },
        "bearish_scenario": {
            "type": "string",
        },
        "execution_comment": {
            "type": "string",
        },
        "risk_note": {
            "type": "string",
        },
    },
    "required": [
        "ai_bias",
        "market_context",
        "view_4h",
        "view_1h",
        "view_15m",
        "confluence",
        "conflicts",
        "preferred_scenario",
        "bullish_scenario",
        "bearish_scenario",
        "execution_comment",
        "risk_note",
    ],
    "additionalProperties": False,
}


def _safe_float(value):
    try:
        value = float(value)

        if math.isfinite(value):
            return value

    except (
        TypeError,
        ValueError,
    ):
        pass

    return None


def _ema(values, span):
    if not values:
        return None

    alpha = 2.0 / (
        span + 1.0
    )

    ema = float(values[0])

    for value in values[1:]:
        ema = (
            alpha
            * float(value)
            + (
                1.0 - alpha
            )
            * ema
        )

    return ema


def _rsi(values, period=14):
    if len(values) <= period:
        return None

    deltas = [
        float(values[i])
        - float(values[i - 1])
        for i in range(
            1,
            len(values),
        )
    ]

    gains = [
        max(delta, 0.0)
        for delta in deltas
    ]

    losses = [
        max(-delta, 0.0)
        for delta in deltas
    ]

    avg_gain = sum(
        gains[:period]
    ) / period

    avg_loss = sum(
        losses[:period]
    ) / period

    for i in range(
        period,
        len(deltas),
    ):
        avg_gain = (
            (
                avg_gain
                * (
                    period - 1
                )
            )
            + gains[i]
        ) / period

        avg_loss = (
            (
                avg_loss
                * (
                    period - 1
                )
            )
            + losses[i]
        ) / period

    if avg_loss == 0:
        return 100.0

    rs = (
        avg_gain
        / avg_loss
    )

    return (
        100.0
        - (
            100.0
            / (
                1.0 + rs
            )
        )
    )


def _anchored_vwap(
    candles,
    analysis,
):
    if not candles:
        return None

    anchor_times = []

    setup = analysis.get(
        "setup",
        {},
    )

    for field in [
        "sweep",
        "structure",
        "displacement",
    ]:
        event = setup.get(field)

        if (
            event
            and event.get("time")
        ):
            anchor_times.append(
                event["time"]
            )

    time_to_index = {
        candle["time"]: i
        for i, candle
        in enumerate(candles)
    }

    anchor_indices = [
        time_to_index[time]
        for time in anchor_times
        if time in time_to_index
    ]

    if anchor_indices:
        anchor = min(
            anchor_indices
        )
    else:
        anchor = max(
            0,
            len(candles) - 96,
        )

    pv = 0.0
    volume_sum = 0.0

    for candle in candles[
        anchor:
    ]:
        volume = (
            _safe_float(
                candle.get(
                    "volume"
                )
            )
            or 0.0
        )

        high = _safe_float(
            candle.get("high")
        )

        low = _safe_float(
            candle.get("low")
        )

        close = _safe_float(
            candle.get("close")
        )

        if (
            high is None
            or low is None
            or close is None
        ):
            continue

        typical = (
            high
            + low
            + close
        ) / 3.0

        pv += (
            typical
            * volume
        )

        volume_sum += volume

    if volume_sum <= 0:
        return None

    return (
        pv
        / volume_sum
    )


def _technical_context(
    candles,
    analysis,
):
    closes = [
        _safe_float(
            candle.get("close")
        )
        for candle in candles
    ]

    closes = [
        value
        for value in closes
        if value is not None
    ]

    volumes = [
        _safe_float(
            candle.get("volume")
        )
        or 0.0
        for candle in candles
    ]

    if not closes:
        return {}

    close = closes[-1]

    ema20 = _ema(
        closes[-80:],
        20,
    )

    ema50 = _ema(
        closes[-120:],
        50,
    )

    rsi14 = _rsi(
        closes[-80:],
        14,
    )

    vwap = (
        _anchored_vwap(
            candles,
            analysis,
        )
    )

    recent_volumes = (
        volumes[-20:]
    )

    average_volume = (
        sum(recent_volumes)
        / len(recent_volumes)
        if recent_volumes
        else None
    )

    relative_volume = None

    if (
        average_volume
        not in (None, 0)
        and volumes
    ):
        relative_volume = (
            volumes[-1]
            / average_volume
        )

    lookback = closes[-20:]

    range_high = max(
        lookback
    )

    range_low = min(
        lookback
    )

    range_position = None

    if range_high > range_low:
        range_position = (
            (
                close
                - range_low
            )
            / (
                range_high
                - range_low
            )
        )

    return {
        "close_vs_ema20": (
            "above"
            if (
                ema20 is not None
                and close >= ema20
            )
            else "below"
            if ema20 is not None
            else "n/a"
        ),
        "ema20_vs_ema50": (
            "above"
            if (
                ema20 is not None
                and ema50 is not None
                and ema20 >= ema50
            )
            else "below"
            if (
                ema20 is not None
                and ema50 is not None
            )
            else "n/a"
        ),
        "close_vs_anchored_vwap": (
            "above"
            if (
                vwap is not None
                and close >= vwap
            )
            else "below"
            if vwap is not None
            else "n/a"
        ),
        "rsi14_regime": (
            "strong"
            if (
                rsi14 is not None
                and rsi14 >= 60
            )
            else "weak"
            if (
                rsi14 is not None
                and rsi14 <= 40
            )
            else "neutral"
            if rsi14 is not None
            else "n/a"
        ),
        "relative_volume_regime": (
            "expanding"
            if (
                relative_volume
                is not None
                and relative_volume
                >= 1.25
            )
            else "contracting"
            if (
                relative_volume
                is not None
                and relative_volume
                <= 0.75
            )
            else "normal"
            if relative_volume
            is not None
            else "n/a"
        ),
        "range20_position": (
            "upper"
            if (
                range_position
                is not None
                and range_position
                >= 0.67
            )
            else "lower"
            if (
                range_position
                is not None
                and range_position
                <= 0.33
            )
            else "middle"
            if range_position
            is not None
            else "n/a"
        ),
    }


def _compact_zone(zone):
    if not zone:
        return None

    return {
        "lower": zone.get(
            "lower"
        ),
        "upper": zone.get(
            "upper"
        ),
        "quality": zone.get(
            "quality"
        ),
        "grade": zone.get(
            "grade"
        ),
        "quality_score": zone.get(
            "quality_score"
        ),
        "pattern": zone.get(
            "pattern"
        ),
        "mitigations": zone.get(
            "mitigations"
        ),
        "qualified": zone.get(
            "qualified"
        ),
    }


def _compact_event(event):
    if not event:
        return None

    keep = [
        "time",
        "kind",
        "type",
        "direction",
        "level",
        "extreme",
        "strength",
        "structure_kind",
    ]

    return {
        key: event.get(key)
        for key in keep
        if event.get(key)
        is not None
    }


def _zone_relation(
    current_price,
    zone,
    atr,
    side,
):
    if not zone:
        return "none"

    try:
        price = float(
            current_price
        )

        lower = float(
            zone["lower"]
        )

        upper = float(
            zone["upper"]
        )

        atr_value = max(
            float(atr or 0.0),
            1e-9,
        )

    except (
        TypeError,
        ValueError,
        KeyError,
    ):
        return "unknown"

    width = max(
        upper - lower,
        1e-9,
    )

    if (
        lower
        <= price
        <= upper
    ):
        position = (
            price - lower
        ) / width

        if side == "demand":
            if position <= 0.33:
                return (
                    "inside_near_distal"
                )

            if position >= 0.67:
                return (
                    "inside_near_proximal"
                )

        else:
            if position <= 0.33:
                return (
                    "inside_near_proximal"
                )

            if position >= 0.67:
                return (
                    "inside_near_distal"
                )

        return "inside_middle"

    near_threshold = (
        0.25 * atr_value
    )

    if price > upper:
        distance = (
            price - upper
        )

        if side == "demand":
            return (
                "above_near_proximal"
                if distance
                <= near_threshold
                else "above_far"
            )

        return (
            "above_beyond_distal"
            if distance
            <= near_threshold
            else "above_far"
        )

    distance = (
        lower - price
    )

    if side == "supply":
        return (
            "below_near_proximal"
            if distance
            <= near_threshold
            else "below_far"
        )

    return (
        "below_beyond_distal"
        if distance
        <= near_threshold
        else "below_far"
    )


def _timeframe_packet(
    timeframe,
    analysis,
    candles,
):
    return {
        "timeframe": timeframe,
        "trend": analysis.get(
            "trend"
        ),
        "regime": analysis.get(
            "regime"
        ),
        "atr": analysis.get(
            "atr"
        ),
        "current_price": (
            analysis.get(
                "current_price"
            )
        ),
        "bsl": analysis.get(
            "bsl"
        ),
        "ssl": analysis.get(
            "ssl"
        ),
        "active_demand": (
            _compact_zone(
                analysis.get(
                    "nearest_demand"
                )
            )
        ),
        "active_supply": (
            _compact_zone(
                analysis.get(
                    "nearest_supply"
                )
            )
        ),
        "price_vs_active_demand": (
            _zone_relation(
                analysis.get(
                    "current_price"
                ),
                analysis.get(
                    "nearest_demand"
                ),
                analysis.get(
                    "atr"
                ),
                "demand",
            )
        ),
        "price_vs_active_supply": (
            _zone_relation(
                analysis.get(
                    "current_price"
                ),
                analysis.get(
                    "nearest_supply"
                ),
                analysis.get(
                    "atr"
                ),
                "supply",
            )
        ),
        "reference_demand": [
            _compact_zone(zone)
            for zone in analysis.get(
                "reference_demand_zones",
                [],
            )[:2]
        ],
        "reference_supply": [
            _compact_zone(zone)
            for zone in analysis.get(
                "reference_supply_zones",
                [],
            )[:2]
        ],
        "last_structure": (
            _compact_event(
                analysis.get(
                    "last_event"
                )
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
            "sweep": (
                _compact_event(
                    analysis.get(
                        "setup",
                        {},
                    ).get(
                        "sweep"
                    )
                )
            ),
            "displacement": (
                _compact_event(
                    analysis.get(
                        "setup",
                        {},
                    ).get(
                        "displacement"
                    )
                )
            ),
            "structure": (
                _compact_event(
                    analysis.get(
                        "setup",
                        {},
                    ).get(
                        "structure"
                    )
                )
            ),
            "retest": (
                _compact_event(
                    analysis.get(
                        "setup",
                        {},
                    ).get(
                        "retest"
                    )
                )
            ),
        },
        "active_fvgs": [
            {
                "type": fvg.get(
                    "type"
                ),
                "lower": fvg.get(
                    "lower"
                ),
                "upper": fvg.get(
                    "upper"
                ),
            }
            for fvg in analysis.get(
                "active_fvgs",
                [],
            )[-3:]
        ],
        "technical_confirmation": (
            _technical_context(
                candles,
                analysis,
            )
        ),
        "ohlc_recent": candles[
            -64:
        ],
    }


def _build_model_input(
    payload,
):
    report = payload[
        "report"
    ]

    candles = payload[
        "ohlc"
    ]

    timeframes = (
        report[
            "timeframes"
        ]
    )

    packet = {
        "method": (
            ANALYSIS_METHOD
        ),
        "symbol": (
            report.get(
                "symbol"
            )
        ),
        "exchange": (
            report.get(
                "exchange"
            )
        ),
        "generated_at_utc": (
            report.get(
                "generated_at_utc"
            )
        ),
        "python_status": (
            report.get(
                "status"
            )
        ),
        "mtf_alignment": (
            report.get(
                "mtf_alignment"
            )
        ),
        "data_integrity": (
            report.get(
                "data_integrity"
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
        "timeframes": {
            timeframe: (
                _timeframe_packet(
                    timeframe,
                    timeframes[
                        timeframe
                    ],
                    candles[
                        timeframe
                    ],
                )
            )
            for timeframe in [
                "4H",
                "1H",
                "15M",
            ]
        },
    }

    return packet


def _fmt(value):
    if value is None:
        return "-"

    try:
        return (
            f"{float(value):.3f}"
        )

    except (
        TypeError,
        ValueError,
    ):
        return str(value)


def _zone_line(
    label,
    zone,
):
    if not zone:
        return (
            f"{label}: -"
        )

    bits = []

    if zone.get(
        "quality"
    ):
        bits.append(
            zone["quality"]
        )

    if zone.get(
        "grade"
    ):
        bits.append(
            zone["grade"]
        )

    if zone.get(
        "pattern"
    ):
        bits.append(
            zone["pattern"]
        )

    if zone.get(
        "mitigations"
    ) is not None:
        bits.append(
            "M"
            + str(
                zone[
                    "mitigations"
                ]
            )
        )

    suffix = (
        " ["
        + ", ".join(bits)
        + "]"
        if bits
        else ""
    )

    return (
        f"{label}: "
        f"{_fmt(zone.get('lower'))}-"
        f"{_fmt(zone.get('upper'))}"
        f"{suffix}"
    )


def _python_facts_text(
    report,
):
    lines = [
        "VVV_USDT HYBRID PRICE ACTION",
        "=" * 38,
        (
            "Nguồn giá/chart: "
            "MEXC OHLC -> Python"
        ),
        (
            "AI model: "
            f"{MODEL}"
        ),
        (
            "Method: "
            f"{ANALYSIS_METHOD}"
        ),
        (
            "Python status: "
            f"{report.get('status', 'WAIT')}"
        ),
        (
            "MTF Alignment: "
            f"{(report.get('mtf_alignment') or {}).get('label', 'NEUTRAL')}"
        ),
        (
            "Data mode: CLOSED CANDLES ONLY"
        ),
    ]

    trade_plan = (
        report.get(
            "trade_plan",
            {},
        )
    )

    lines.append(
        (
            "Execution gate: "
            + (
                "READY"
                if trade_plan.get(
                    "execution_ready",
                    False,
                )
                else "WAIT"
            )
        )
    )

    snapshot = (
        report.get(
            "market_snapshot",
            {}
        )
    )

    lines.append(
        (
            "Price: "
            f"{_fmt(report.get('current_price'))}"
        )
    )

    funding = snapshot.get(
        "funding_rate"
    )

    if funding is not None:
        lines.append(
            (
                "Funding: "
                f"{float(funding) * 100:+.4f}%"
            )
        )

    hold_vol = snapshot.get(
        "hold_vol"
    )

    hold_change = snapshot.get(
        "hold_vol_change_pct"
    )

    participation = (
        snapshot.get(
            "participation_context",
            {},
        )
        or {}
    )

    if participation:
        lines.append(
            (
                "Participation: "
                f"{participation.get('regime', 'NEUTRAL_MIXED')}"
            )
        )

    if hold_vol is not None:
        text = (
            "MEXC holdVol: "
            f"{float(hold_vol):,.0f}"
        )

        if hold_change is not None:
            text += (
                f" ({float(hold_change):+.2f}%)"
            )

        lines.append(text)

    for timeframe in [
        "4H",
        "1H",
        "15M",
    ]:
        analysis = (
            report[
                "timeframes"
            ][timeframe]
        )

        lines.append("")

        lines.append(
            (
                f"[{timeframe}] "
                f"Trend {str(analysis.get('trend', '-')).upper()} "
                f"| Regime {str(analysis.get('regime', '-')).upper()}"
            )
        )

        lines.append(
            _zone_line(
                "Demand",
                analysis.get(
                    "nearest_demand"
                ),
            )
        )

        lines.append(
            _zone_line(
                "Supply",
                analysis.get(
                    "nearest_supply"
                ),
            )
        )

        bsl = (
            analysis.get(
                "bsl"
            )
            or {}
        ).get(
            "price"
        )

        ssl = (
            analysis.get(
                "ssl"
            )
            or {}
        ).get(
            "price"
        )

        lines.append(
            (
                "Liquidity: "
                f"BSL {_fmt(bsl)}"
                " | "
                f"SSL {_fmt(ssl)}"
            )
        )

        event = (
            analysis.get(
                "last_event"
            )
        )

        if event:
            lines.append(
                (
                    "Structure: "
                    f"{event.get('kind')} "
                    f"{str(event.get('direction', '')).upper()} "
                    f"@ {_fmt(event.get('level'))}"
                )
            )

        setup = (
            analysis.get(
                "setup",
                {}
            )
        )

        lines.append(
            (
                "Setup: "
                f"{str(setup.get('direction', '-')).upper()} "
                f"{setup.get('score', 0)}/4"
            )
        )

    return "\n".join(lines)


def _ai_text(
    result,
):
    lines = [
        "",
        "AI NHẬN ĐỊNH",
        "=" * 38,
        (
            "AI bias: "
            f"{result['ai_bias']}"
        ),
        (
            "Tổng quan: "
            f"{result['market_context']}"
        ),
        "",
        (
            "4H: "
            f"{result['view_4h']}"
        ),
        (
            "1H: "
            f"{result['view_1h']}"
        ),
        (
            "15M: "
            f"{result['view_15m']}"
        ),
    ]

    if result[
        "confluence"
    ]:
        lines.append("")
        lines.append(
            "Confluence:"
        )

        lines.extend(
            "- " + item
            for item in result[
                "confluence"
            ]
        )

    if result[
        "conflicts"
    ]:
        lines.append("")
        lines.append(
            "Xung đột:"
        )

        lines.extend(
            "- " + item
            for item in result[
                "conflicts"
            ]
        )

    lines.extend(
        [
            "",
            (
                "Kịch bản ưu tiên: "
                f"{result['preferred_scenario']}"
            ),
            (
                "Bullish scenario: "
                f"{result['bullish_scenario']}"
            ),
            (
                "Bearish scenario: "
                f"{result['bearish_scenario']}"
            ),
            (
                "Execution: "
                f"{result['execution_comment']}"
            ),
            (
                "Risk: "
                f"{result['risk_note']}"
            ),
        ]
    )

    return "\n".join(lines)


def _parse_json_content(
    content,
):
    if not content:
        raise ValueError(
            "OpenRouter returned empty AI content."
        )

    if isinstance(
        content,
        list,
    ):
        parts = []

        for item in content:
            if isinstance(
                item,
                dict,
            ):
                text_value = (
                    item.get("text")
                    or item.get(
                        "content"
                    )
                )

                if text_value:
                    parts.append(
                        str(text_value)
                    )
            else:
                parts.append(
                    str(item)
                )

        content = "\n".join(
            parts
        )

    content = str(
        content
    ).strip()

    if content.startswith(
        "```"
    ):
        lines = (
            content.splitlines()
        )

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1]
            .strip()
            .startswith(
                "```"
            )
        ):
            lines = lines[:-1]

        content = "\n".join(
            lines
        ).strip()

    try:
        return json.loads(
            content
        )

    except json.JSONDecodeError:
        start = content.find(
            "{"
        )

        end = content.rfind(
            "}"
        )

        if (
            start >= 0
            and end > start
        ):
            return json.loads(
                content[
                    start:
                    end + 1
                ]
            )

        raise


def _result_strings(
    result,
):
    values = []

    def walk(
        value,
    ):
        if isinstance(
            value,
            str,
        ):
            values.append(
                value
            )

        elif isinstance(
            value,
            list,
        ):
            for item in value:
                walk(item)

        elif isinstance(
            value,
            dict,
        ):
            for item in (
                value.values()
            ):
                walk(item)

    walk(result)

    return values


def _validate_result(
    result,
    model_input,
):
    """
    Deterministic guardrail after the LLM and before Telegram.

    Hard failures cause the workflow to try the fallback model.
    The Python execution gate always wins over AI wording.
    """

    if not isinstance(
        result,
        dict,
    ):
        raise ValueError(
            "AI result is not a JSON object."
        )

    required = set(
        OUTPUT_SCHEMA[
            "required"
        ]
    )

    missing = (
        required
        - set(
            result.keys()
        )
    )

    if missing:
        raise ValueError(
            (
                "AI result missing fields: "
                + ", ".join(
                    sorted(missing)
                )
            )
        )

    if result.get(
        "ai_bias"
    ) not in {
        "BULLISH",
        "BEARISH",
        "MIXED",
        "WAIT",
    }:
        raise ValueError(
            "AI result has invalid ai_bias."
        )

    for key in [
        "confluence",
        "conflicts",
    ]:
        if not isinstance(
            result.get(key),
            list,
        ):
            raise ValueError(
                (
                    f"AI result field "
                    f"{key} must be a list."
                )
            )

    strings = (
        _result_strings(
            result
        )
    )

    combined = " ".join(
        strings
    )

    lower = combined.lower()

    # AI is not allowed to publish a new numerical trading level.
    # Technical timeframe tokens such as 4H/1H/15M are unaffected.
    decimal_level = re.search(
        r"(?<![A-Za-z])\d+\.\d{2,}",
        combined,
    )

    if decimal_level:
        raise ValueError(
            (
                "AI output contains a numerical "
                "price-like level."
            )
        )

    if "%" in combined:
        raise ValueError(
            (
                "AI output contains unsupported "
                "percentage/confidence wording."
            )
        )

    probability_phrases = [
        "xác suất cao",
        "xác suất thấp",
        "high probability",
        "low probability",
        "confidence score",
    ]

    if any(
        phrase in lower
        for phrase
        in probability_phrases
    ):
        raise ValueError(
            (
                "AI output contains unsupported "
                "probability language."
            )
        )

    vietnamese_markers = [
        "giá",
        "xu hướng",
        "cấu trúc",
        "thanh khoản",
        "không",
        "đang",
        "cần",
        "vùng",
        "tăng",
        "giảm",
        "khung",
        "chờ",
        "xung đột",
        "ưu tiên",
    ]

    marker_hits = sum(
        marker in lower
        for marker
        in vietnamese_markers
    )

    if marker_hits < 2:
        raise ValueError(
            (
                "AI output does not appear to be "
                "Vietnamese as required."
            )
        )

    trade_plan = (
        model_input.get(
            "trade_plan",
            {},
        )
        or {}
    )

    execution_ready = bool(
        trade_plan.get(
            "execution_ready",
            False,
        )
    )

    if not execution_ready:
        # Do not merely trust the model wording: replace the execution
        # line with deterministic Python-gated language.
        result[
            "execution_comment"
        ] = (
            "Python execution gate = WAIT. "
            "Chưa thực thi; chỉ theo dõi "
            "điều kiện xác nhận."
        )

    alignment = (
        model_input.get(
            "mtf_alignment",
            {},
        )
        or {}
    )

    context_conflict = (
        alignment.get(
            "label"
        )
        == "CONFLICT"
        and result.get(
            "ai_bias"
        )
        in (
            "BULLISH",
            "BEARISH",
        )
    )

    return {
        "passed": True,
        "execution_gate_enforced": (
            not execution_ready
        ),
        "context_conflict_flag": (
            context_conflict
        ),
        "numeric_level_check": "passed",
        "language_check": "passed",
        "probability_check": "passed",
    }


def _request_analysis(
    client,
    model,
    model_input,
):
    response = (
        client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        SYSTEM_PROMPT
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Analyze this exact "
                        "machine-generated market packet. "
                        "Do not add numerical levels. "
                        "IMPORTANT: every explanatory JSON string MUST be Vietnamese; "
                        "standard technical acronyms may remain English. "
                        "Return a concise Vietnamese trading-desk review.\n\n"
                        "MARKET PACKET:\n"
                        + json.dumps(
                            model_input,
                            ensure_ascii=False,
                            separators=(
                                ",",
                                ":",
                            ),
                        )
                    ),
                },
            ],
            response_format={
                "type": (
                    "json_schema"
                ),
                "json_schema": {
                    "name": (
                        "vvv_pa_analysis"
                    ),
                    "strict": True,
                    "schema": (
                        OUTPUT_SCHEMA
                    ),
                },
            },
            max_tokens=1800,
            temperature=0.1,
            extra_body={
                "reasoning": {
                    "enabled": False,
                },
                "provider": {
                    "require_parameters": True,
                },
            },
        )
    )

    if not response.choices:
        raise ValueError(
            (
                "OpenRouter returned "
                "no choices."
            )
        )

    choice = response.choices[0]

    finish_reason = getattr(
        choice,
        "finish_reason",
        None,
    )

    message = choice.message

    content = getattr(
        message,
        "content",
        None,
    )

    routed_model = (
        getattr(
            response,
            "model",
            None,
        )
        or model
    )

    print(
        (
            "OpenRouter response: "
            f"requested={model} "
            f"routed={routed_model} "
            f"finish={finish_reason}"
        )
    )

    if (
        finish_reason
        == "length"
    ):
        raise ValueError(
            "OpenRouter response hit output length limit."
        )

    result = (
        _parse_json_content(
            content
        )
    )

    validation = (
        _validate_result(
            result,
            model_input,
        )
    )

    return (
        result,
        routed_model,
        validation,
    )


def main():
    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if not api_key:
        print(
            "AI analysis skipped: "
            "OPENROUTER_API_KEY is not configured."
        )

        return 0

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing AI input: {INPUT_PATH}"
        )

    payload = json.loads(
        INPUT_PATH.read_text(
            encoding="utf-8"
        )
    )

    model_input = (
        _build_model_input(
            payload
        )
    )

    client = OpenAI(
        api_key=api_key,
        base_url=(
            OPENROUTER_BASE_URL
        ),
        default_headers={
            "X-Title": (
                "VVV Trading AI"
            ),
        },
    )

    candidate_models = [
        MODEL,
    ]

    if (
        FALLBACK_MODEL
        and FALLBACK_MODEL
        != MODEL
    ):
        candidate_models.append(
            FALLBACK_MODEL
        )

    result = None
    routed_model = None
    validation = None
    errors = []

    for candidate_model in (
        candidate_models
    ):
        try:
            (
                result,
                routed_model,
                validation,
            ) = _request_analysis(
                client,
                candidate_model,
                model_input,
            )

            break

        except Exception as exc:
            errors.append(
                (
                    candidate_model,
                    type(exc).__name__,
                    str(exc),
                )
            )

            print(
                (
                    "OpenRouter AI attempt failed: "
                    f"{candidate_model} | "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
            )

    if result is None:
        raise RuntimeError(
            (
                "All OpenRouter AI attempts failed: "
                + " | ".join(
                    (
                        f"{model}: "
                        f"{error_type}"
                    )
                    for (
                        model,
                        error_type,
                        _,
                    )
                    in errors
                )
            )
        )

    report = payload[
        "report"
    ]

    artifact = {
        "model": routed_model,
        "requested_model": MODEL,
        "fallback_model": (
            FALLBACK_MODEL
        ),
        "method": (
            ANALYSIS_METHOD
        ),
        "python_status": (
            report.get(
                "status"
            )
        ),
        "analysis": result,
        "validation": (
            validation
        ),
        "technical_context": {
            timeframe: (
                model_input[
                    "timeframes"
                ][timeframe][
                    "technical_confirmation"
                ]
            )
            for timeframe in [
                "4H",
                "1H",
                "15M",
            ]
        },
    }

    try:
        attach_ai_review(
            report,
            result,
        )

    except Exception as exc:
        print(
            (
                "Signal journal AI attach "
                f"warning: {exc}"
            )
        )

    AI_JSON_PATH.write_text(
        json.dumps(
            artifact,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    telegram_text = (
        _python_facts_text(
            report
        )
        + "\n"
        + _ai_text(
            result
        )
    )

    AI_TEXT_PATH.write_text(
        telegram_text,
        encoding="utf-8",
    )

    print(
        (
            "AI Price Action analysis "
            "completed via OpenRouter "
            f"with {routed_model}."
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
