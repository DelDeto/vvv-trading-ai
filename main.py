import json
import os
from datetime import datetime, timezone

from mexc_data import get_klines
from chart import create_chart
from smc_analysis import (
    analyze_smc,
    derive_overall_status,
)


def _fmt(value):
    if value is None:
        return "-"

    return f"{value:.3f}"


def _zone_text(zone):
    if not zone:
        return "-"

    return (
        f"{_fmt(zone['lower'])}"
        f"-{_fmt(zone['upper'])}"
    )


def _print_summary(
    timeframe,
    analysis,
):
    event = analysis.get(
        "last_event"
    )

    event_text = "None"

    if event:
        event_text = (
            f"{event['kind']} "
            f"{event['direction'].upper()} "
            f"@ {_fmt(event['level'])}"
        )

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
        f"Last structure: "
        f"{event_text}"
    )

    print(
        f"[{timeframe}] "
        f"Active FVG count: "
        f"{len(analysis.get('active_fvgs', []))}"
    )


def main():

    print("=" * 50)
    print(
        "VVV_USDT MEXC FUTURES "
        "+ SMC ANALYSIS"
    )
    print("=" * 50)

    # Real MEXC Futures OHLC
    df_4h = get_klines(
        "4h",
        200,
    )

    df_1h = get_klines(
        "1h",
        200,
    )

    df_15m = get_klines(
        "15m",
        200,
    )

    # Rule-based SMC analysis
    smc_4h = analyze_smc(
        df_4h
    )

    smc_1h = analyze_smc(
        df_1h
    )

    smc_15m = analyze_smc(
        df_15m
    )

    status = derive_overall_status(
        smc_4h,
        smc_1h,
        smc_15m,
    )

    current_price = float(
        df_15m.iloc[-1]["close"]
    )

    print()
    print(
        f"Current VVV_USDT close: "
        f"{current_price}"
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

    os.makedirs(
        "output",
        exist_ok=True,
    )

    # Save machine-readable SMC report.
    report = {
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "exchange": "MEXC",
        "symbol": "VVV_USDT",
        "current_price": current_price,
        "status": status,
        "timeframes": {
            "4H": smc_4h,
            "1H": smc_1h,
            "15M": smc_15m,
        },
    }

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

    # Real candlestick charts + SMC overlay
    create_chart(
        df_4h,
        "4H",
        "output/VVVUSDT_4H.png",
        candles=100,
        exchange="MEXC",
        analysis=smc_4h,
    )

    create_chart(
        df_1h,
        "1H",
        "output/VVVUSDT_1H.png",
        candles=120,
        exchange="MEXC",
        analysis=smc_1h,
    )

    create_chart(
        df_15m,
        "15M",
        "output/VVVUSDT_15M.png",
        candles=150,
        exchange="MEXC",
        analysis=smc_15m,
    )

    print()
    print("=" * 50)
    print(
        "ALL MEXC SMC CHARTS CREATED"
    )
    print("=" * 50)


if __name__ == "__main__":
    main()
