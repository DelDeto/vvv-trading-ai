import json
import os
from datetime import datetime, timezone

import pandas as pd


JOURNAL_PATH = "signal_history.json"
STATS_PATH = "output/VVVUSDT_signal_stats.json"
MAX_TRACK_BARS_15M = 96


def _load_journal():
    if not os.path.exists(
        JOURNAL_PATH
    ):
        return {
            "version": 1,
            "signals": [],
        }

    try:
        with open(
            JOURNAL_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                "Journal root is not an object"
            )

        data.setdefault(
            "version",
            1,
        )

        data.setdefault(
            "signals",
            [],
        )

        return data

    except Exception as exc:
        print(
            "Signal journal load warning:",
            exc,
        )

        return {
            "version": 1,
            "signals": [],
        }


def _save_journal(
    journal,
):
    with open(
        JOURNAL_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            journal,
            file,
            ensure_ascii=False,
            indent=2,
        )


def _round_key(
    value,
):
    if value is None:
        return None

    return round(
        float(value),
        6,
    )


def _plan_signature(
    report,
):
    plan = report.get(
        "trade_plan",
        {},
    )

    if not plan.get(
        "active",
        False,
    ):
        return None

    entry = plan.get(
        "entry_zone",
        {},
    )

    return "|".join(
        str(item)
        for item in [
            plan.get(
                "direction"
            ),
            _round_key(
                entry.get(
                    "lower"
                )
            ),
            _round_key(
                entry.get(
                    "upper"
                )
            ),
            _round_key(
                plan.get(
                    "stop_loss"
                )
            ),
            tuple(
                _round_key(
                    target.get(
                        "price"
                    )
                )
                for target
                in plan.get(
                    "targets",
                    [],
                )
            ),
        ]
    )


def _new_signal(
    report,
):
    plan = report[
        "trade_plan"
    ]

    entry = plan[
        "entry_zone"
    ]

    targets = [
        {
            "name": item.get(
                "name"
            ),
            "price": float(
                item[
                    "price"
                ]
            ),
            "source": item.get(
                "source"
            ),
            "rr": item.get(
                "rr"
            ),
        }
        for item
        in plan.get(
            "targets",
            [],
        )
    ]

    entry_mid = (
        float(
            entry["lower"]
        )
        + float(
            entry["upper"]
        )
    ) / 2.0

    stop = float(
        plan[
            "stop_loss"
        ]
    )

    risk = abs(
        entry_mid - stop
    )

    timeframes = report.get(
        "timeframes",
        {},
    )

    snapshot = report.get(
        "market_snapshot",
        {},
    )

    participation = snapshot.get(
        "participation_context",
        {},
    )

    created_at = (
        report.get(
            "last_closed_candle",
            {},
        ).get(
            "15M"
        )
        or report.get(
            "generated_at_utc"
        )
    )

    return {
        "id": (
            report.get(
                "generated_at_utc"
            )
        ),
        "signature": (
            _plan_signature(
                report
            )
        ),
        "created_at": (
            created_at
        ),
        "generated_at_utc": (
            report.get(
                "generated_at_utc"
            )
        ),
        "status": (
            report.get(
                "status"
            )
        ),
        "direction": (
            plan.get(
                "direction"
            )
        ),
        "execution_ready": (
            plan.get(
                "execution_ready",
                False,
            )
        ),
        "mtf_alignment": (
            report.get(
                "mtf_alignment",
                {},
            ).get(
                "label"
            )
        ),
        "regimes": {
            timeframe: (
                timeframes.get(
                    timeframe,
                    {},
                ).get(
                    "regime"
                )
            )
            for timeframe
            in [
                "4H",
                "1H",
                "15M",
            ]
        },
        "setup_score": (
            plan.get(
                "setup_score"
            )
        ),
        "entry_zone": {
            "lower": float(
                entry[
                    "lower"
                ]
            ),
            "upper": float(
                entry[
                    "upper"
                ]
            ),
            "source": (
                entry.get(
                    "source"
                )
            ),
            "grade": (
                entry.get(
                    "zone_grade"
                )
                or plan.get(
                    "entry_zone_grade"
                )
            ),
        },
        "entry_mid": (
            entry_mid
        ),
        "stop_loss": stop,
        "risk": risk,
        "targets": targets,
        "participation_regime": (
            participation.get(
                "regime"
            )
        ),
        "funding_rate": (
            snapshot.get(
                "funding_rate"
            )
        ),
        "hold_vol": (
            snapshot.get(
                "hold_vol"
            )
        ),
        "ai_bias": None,
        "entered_at": None,
        "last_evaluated_at": None,
        "bars_observed": 0,
        "mfe_r": 0.0,
        "mae_r": 0.0,
        "targets_hit": [],
        "outcome": "OPEN",
        "closed_at": None,
    }


def _touches_zone(
    row,
    lower,
    upper,
):
    return (
        float(
            row["high"]
        )
        >= lower
        and float(
            row["low"]
        )
        <= upper
    )


def _update_signal(
    signal,
    df_15m,
):
    if signal.get(
        "outcome"
    ) not in (
        "OPEN",
        "ENTERED",
    ):
        return

    created_at = pd.Timestamp(
        signal[
            "created_at"
        ]
    )

    if created_at.tzinfo is None:
        created_at = (
            created_at.tz_localize(
                "UTC"
            )
        )

    candles = df_15m.loc[
        df_15m.index
        > created_at
    ]

    last_evaluated = (
        signal.get(
            "last_evaluated_at"
        )
    )

    if last_evaluated:
        last_timestamp = (
            pd.Timestamp(
                last_evaluated
            )
        )

        if (
            last_timestamp.tzinfo
            is None
        ):
            last_timestamp = (
                last_timestamp
                .tz_localize(
                    "UTC"
                )
            )

        candles = candles.loc[
            candles.index
            > last_timestamp
        ]

    if candles.empty:
        return

    direction = signal[
        "direction"
    ]

    entry = float(
        signal[
            "entry_mid"
        ]
    )

    lower = float(
        signal[
            "entry_zone"
        ][
            "lower"
        ]
    )

    upper = float(
        signal[
            "entry_zone"
        ][
            "upper"
        ]
    )

    stop = float(
        signal[
            "stop_loss"
        ]
    )

    risk = max(
        float(
            signal.get(
                "risk",
                0.0,
            )
        ),
        1e-9,
    )

    target_prices = [
        float(
            item["price"]
        )
        for item
        in signal.get(
            "targets",
            [],
        )
    ]

    targets_hit = set(
        signal.get(
            "targets_hit",
            [],
        )
    )

    for timestamp, row in (
        candles.iterrows()
    ):
        signal[
            "bars_observed"
        ] = int(
            signal.get(
                "bars_observed",
                0,
            )
        ) + 1

        newly_entered = False

        if not signal.get(
            "entered_at"
        ):
            if _touches_zone(
                row,
                lower,
                upper,
            ):
                signal[
                    "entered_at"
                ] = (
                    timestamp
                    .isoformat()
                )

                signal[
                    "outcome"
                ] = "ENTERED"

                newly_entered = True

            else:
                signal[
                    "last_evaluated_at"
                ] = (
                    timestamp
                    .isoformat()
                )

                if (
                    signal[
                        "bars_observed"
                    ]
                    >= MAX_TRACK_BARS_15M
                ):
                    signal[
                        "outcome"
                    ] = (
                        "EXPIRED_NO_ENTRY"
                    )

                    signal[
                        "closed_at"
                    ] = (
                        timestamp
                        .isoformat()
                    )

                    break

                continue

        high = float(
            row["high"]
        )

        low = float(
            row["low"]
        )

        if direction == "long":
            favorable_r = (
                high - entry
            ) / risk

            adverse_r = (
                entry - low
            ) / risk

            stop_hit = (
                low <= stop
            )

            hit_indexes = [
                index
                for index, price
                in enumerate(
                    target_prices,
                    start=1,
                )
                if high >= price
            ]

        else:
            favorable_r = (
                entry - low
            ) / risk

            adverse_r = (
                high - entry
            ) / risk

            stop_hit = (
                high >= stop
            )

            hit_indexes = [
                index
                for index, price
                in enumerate(
                    target_prices,
                    start=1,
                )
                if low <= price
            ]

        signal[
            "mfe_r"
        ] = max(
            float(
                signal.get(
                    "mfe_r",
                    0.0,
                )
            ),
            favorable_r,
        )

        signal[
            "mae_r"
        ] = max(
            float(
                signal.get(
                    "mae_r",
                    0.0,
                )
            ),
            adverse_r,
        )

        newly_hit = [
            index
            for index
            in hit_indexes
            if index
            not in targets_hit
        ]

        # OHLC has no intrabar event order. If a candle can hit both
        # stop and a new target after entry, label it ambiguous rather
        # than fabricating which happened first.
        if (
            stop_hit
            and newly_hit
        ):
            signal[
                "outcome"
            ] = (
                "AMBIGUOUS_SAME_BAR"
            )

            signal[
                "closed_at"
            ] = (
                timestamp
                .isoformat()
            )

            break

        if stop_hit:
            signal[
                "outcome"
            ] = "STOP"

            signal[
                "closed_at"
            ] = (
                timestamp
                .isoformat()
            )

            break

        for index in newly_hit:
            targets_hit.add(
                index
            )

        signal[
            "targets_hit"
        ] = sorted(
            targets_hit
        )

        if (
            target_prices
            and len(
                targets_hit
            )
            >= len(
                target_prices
            )
        ):
            signal[
                "outcome"
            ] = (
                f"TP{len(target_prices)}"
            )

            signal[
                "closed_at"
            ] = (
                timestamp
                .isoformat()
            )

            break

        signal[
            "last_evaluated_at"
        ] = (
            timestamp.isoformat()
        )

        if (
            signal[
                "bars_observed"
            ]
            >= MAX_TRACK_BARS_15M
        ):
            signal[
                "outcome"
            ] = (
                "TIMEOUT_ENTERED"
            )

            signal[
                "closed_at"
            ] = (
                timestamp
                .isoformat()
            )

            break

        if newly_entered:
            # Continue evaluating later candles. The entry candle itself
            # is included for MFE/MAE but same-bar stop/target ambiguity
            # is handled above.
            pass

    if not signal.get(
        "last_evaluated_at"
    ):
        signal[
            "last_evaluated_at"
        ] = (
            candles.index[-1]
            .isoformat()
        )


def _stats(
    signals,
):
    entered = [
        item
        for item in signals
        if item.get(
            "entered_at"
        )
    ]

    resolved = [
        item
        for item in entered
        if item.get(
            "outcome"
        )
        not in (
            "OPEN",
            "ENTERED",
        )
    ]

    def count_outcome(
        name,
    ):
        return sum(
            item.get(
                "outcome"
            )
            == name
            for item
            in signals
        )

    def target_count(
        minimum,
    ):
        return sum(
            max(
                item.get(
                    "targets_hit",
                    [],
                ),
                default=0,
            )
            >= minimum
            for item
            in signals
        )

    mfe_values = [
        float(
            item.get(
                "mfe_r",
                0.0,
            )
        )
        for item
        in entered
    ]

    mae_values = [
        float(
            item.get(
                "mae_r",
                0.0,
            )
        )
        for item
        in entered
    ]

    by_grade = {}

    for item in signals:
        grade = (
            item.get(
                "entry_zone",
                {},
            ).get(
                "grade"
            )
            or "NA"
        )

        bucket = (
            by_grade.setdefault(
                grade,
                {
                    "total": 0,
                    "entered": 0,
                    "stop": 0,
                    "tp1_plus": 0,
                },
            )
        )

        bucket[
            "total"
        ] += 1

        if item.get(
            "entered_at"
        ):
            bucket[
                "entered"
            ] += 1

        if item.get(
            "outcome"
        ) == "STOP":
            bucket[
                "stop"
            ] += 1

        if max(
            item.get(
                "targets_hit",
                [],
            ),
            default=0,
        ) >= 1:
            bucket[
                "tp1_plus"
            ] += 1

    return {
        "total_signals": len(
            signals
        ),
        "entered": len(
            entered
        ),
        "resolved": len(
            resolved
        ),
        "stops": count_outcome(
            "STOP"
        ),
        "tp1_plus": target_count(
            1
        ),
        "tp2_plus": target_count(
            2
        ),
        "tp3_plus": target_count(
            3
        ),
        "ambiguous": count_outcome(
            "AMBIGUOUS_SAME_BAR"
        ),
        "expired_no_entry": (
            count_outcome(
                "EXPIRED_NO_ENTRY"
            )
        ),
        "timeout_entered": (
            count_outcome(
                "TIMEOUT_ENTERED"
            )
        ),
        "avg_mfe_r": (
            sum(
                mfe_values
            )
            / len(
                mfe_values
            )
            if mfe_values
            else None
        ),
        "avg_mae_r": (
            sum(
                mae_values
            )
            / len(
                mae_values
            )
            if mae_values
            else None
        ),
        "by_zone_grade": (
            by_grade
        ),
        "calibration_note": (
            "Outcome data is observational forward-test data. "
            "Do not auto-tune thresholds until sample size is sufficient."
        ),
    }


def update_signal_journal(
    report,
    df_15m,
):
    journal = (
        _load_journal()
    )

    signals = journal[
        "signals"
    ]

    for signal in signals:
        _update_signal(
            signal,
            df_15m,
        )

    signature = (
        _plan_signature(
            report
        )
    )

    if signature:
        matching = None

        for signal in reversed(
            signals
        ):
            if (
                signal.get(
                    "signature"
                )
                == signature
                and signal.get(
                    "outcome"
                )
                in (
                    "OPEN",
                    "ENTERED",
                )
            ):
                matching = signal
                break

        if matching:
            matching[
                "status"
            ] = report.get(
                "status"
            )

            matching[
                "execution_ready"
            ] = (
                report.get(
                    "trade_plan",
                    {},
                ).get(
                    "execution_ready",
                    False,
                )
            )

            matching[
                "mtf_alignment"
            ] = (
                report.get(
                    "mtf_alignment",
                    {},
                ).get(
                    "label"
                )
            )

        else:
            signals.append(
                _new_signal(
                    report
                )
            )

    journal[
        "updated_at_utc"
    ] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    stats = _stats(
        signals
    )

    journal[
        "stats"
    ] = stats

    _save_journal(
        journal
    )

    os.makedirs(
        "output",
        exist_ok=True,
    )

    with open(
        STATS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            stats,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return stats


def attach_ai_review(
    report,
    ai_result,
):
    signature = (
        _plan_signature(
            report
        )
    )

    if not signature:
        return

    journal = (
        _load_journal()
    )

    for signal in reversed(
        journal.get(
            "signals",
            [],
        )
    ):
        if signal.get(
            "signature"
        ) == signature:
            signal[
                "ai_bias"
            ] = ai_result.get(
                "ai_bias"
            )

            signal[
                "ai_reviewed_at_utc"
            ] = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

            break

    journal[
        "stats"
    ] = _stats(
        journal.get(
            "signals",
            [],
        )
    )

    _save_journal(
        journal
    )
