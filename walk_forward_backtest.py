import json
import os

import pandas as pd

from mexc_data import (
    get_closed_klines,
)
from smc_analysis import (
    analyze_smc,
    derive_overall_status,
    derive_mtf_alignment,
)
from trade_plan import (
    build_trade_plan,
)
from signal_journal import (
    _new_signal,
    _update_signal,
    _stats,
)


OUTPUT_PATH = (
    "output/VVVUSDT_walk_forward_backtest.json"
)


def _closed_asof(
    df,
    interval,
    as_of,
):
    delta = {
        "15M": pd.Timedelta(
            minutes=15
        ),
        "1H": pd.Timedelta(
            hours=1
        ),
        "4H": pd.Timedelta(
            hours=4
        ),
    }[
        interval
    ]

    mask = (
        df.index
        + delta
        <= as_of
    )

    return (
        df.loc[
            mask
        ]
        .tail(420)
        .copy()
    )


def main():
    print(
        "Loading closed MEXC history "
        "for walk-forward evaluation..."
    )

    df_15m = get_closed_klines(
        "15m",
        1500,
    )

    df_1h = get_closed_klines(
        "1h",
        500,
    )

    df_4h = get_closed_klines(
        "4h",
        420,
    )

    signals = []
    last_signature = None
    evaluated_points = 0

    # Evaluate once per hour, matching the production workflow cadence.
    for position in range(
        120,
        len(df_15m) - 96,
        4,
    ):
        candle_time = (
            df_15m.index[
                position
            ]
        )

        as_of = (
            candle_time
            + pd.Timedelta(
                minutes=15
            )
        )

        h15 = (
            df_15m.iloc[
                : position + 1
            ]
            .tail(420)
            .copy()
        )

        h1 = _closed_asof(
            df_1h,
            "1H",
            as_of,
        )

        h4 = _closed_asof(
            df_4h,
            "4H",
            as_of,
        )

        if (
            len(h15) < 100
            or len(h1) < 60
            or len(h4) < 30
        ):
            continue

        analysis_4h = (
            analyze_smc(
                h4,
                timeframe="4H",
            )
        )

        analysis_1h = (
            analyze_smc(
                h1,
                timeframe="1H",
            )
        )

        analysis_15m = (
            analyze_smc(
                h15,
                timeframe="15M",
            )
        )

        status = (
            derive_overall_status(
                analysis_4h,
                analysis_1h,
                analysis_15m,
            )
        )

        alignment = (
            derive_mtf_alignment(
                analysis_4h,
                analysis_1h,
                analysis_15m,
            )
        )

        plan = build_trade_plan(
            analysis_4h,
            analysis_1h,
            analysis_15m,
            status,
            mtf_alignment=(
                alignment
            ),
        )

        evaluated_points += 1

        report = {
            "generated_at_utc": (
                as_of.isoformat()
            ),
            "status": status,
            "mtf_alignment": (
                alignment
            ),
            "trade_plan": plan,
            "market_snapshot": {},
            "last_closed_candle": {
                "15M": (
                    candle_time
                    .isoformat()
                )
            },
            "timeframes": {
                "4H": analysis_4h,
                "1H": analysis_1h,
                "15M": analysis_15m,
            },
        }

        if not plan.get(
            "active",
            False,
        ):
            last_signature = None
            continue

        signal = (
            _new_signal(
                report
            )
        )

        signature = signal.get(
            "signature"
        )

        if (
            signature
            == last_signature
        ):
            continue

        last_signature = (
            signature
        )

        _update_signal(
            signal,
            df_15m,
        )

        signals.append(
            signal
        )

    stats = _stats(
        signals
    )

    result = {
        "method": (
            "WALK_FORWARD_NO_LOOKAHEAD"
        ),
        "cadence": "1H",
        "source": (
            "MEXC closed candles"
        ),
        "evaluated_points": (
            evaluated_points
        ),
        "signals": signals,
        "stats": stats,
        "notes": [
            (
                "Each decision point uses only candles "
                "fully closed by that historical timestamp."
            ),
            (
                "Future 15M candles are used only after "
                "a signal is frozen, to score its outcome."
            ),
            (
                "Same-bar entry plus stop/target is marked "
                "AMBIGUOUS_SAME_BAR rather than guessing order."
            ),
            (
                "Backtest statistics are calibration evidence, "
                "not a guarantee of future performance."
            ),
        ],
    }

    os.makedirs(
        "output",
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        (
            "Walk-forward evaluation complete: "
            f"{len(signals)} signals | "
            f"{stats.get('resolved', 0)} resolved"
        )
    )


if __name__ == "__main__":
    main()
