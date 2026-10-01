import json
from datetime import datetime, timezone
from pathlib import Path


REPORT_PATH = Path(
    "output/VVVUSDT_SMC_report.json"
)
AI_INPUT_PATH = Path(
    "output/VVVUSDT_ai_input.json"
)

INTERVAL_SECONDS = {
    "4H": 4 * 60 * 60,
    "1H": 60 * 60,
    "15M": 15 * 60,
}

VALID_STATUS = {
    "WAIT",
    "DEVELOPING LONG",
    "DEVELOPING SHORT",
    "CONFIRMED LONG",
    "CONFIRMED SHORT",
}


def _fail(
    message,
):
    raise RuntimeError(
        "SELF_CHECK_FAILED: "
        + message
    )


def main():
    if not REPORT_PATH.exists():
        _fail(
            "Missing SMC report"
        )

    if not AI_INPUT_PATH.exists():
        _fail(
            "Missing AI input packet"
        )

    report = json.loads(
        REPORT_PATH.read_text(
            encoding="utf-8"
        )
    )

    packet = json.loads(
        AI_INPUT_PATH.read_text(
            encoding="utf-8"
        )
    )

    if report.get(
        "status"
    ) not in VALID_STATUS:
        _fail(
            (
                "Invalid status vocabulary: "
                f"{report.get('status')}"
            )
        )

    integrity = report.get(
        "data_integrity",
        {},
    )

    if not integrity.get(
        "closed_candles_only",
        False,
    ):
        _fail(
            "closed_candles_only is not enabled"
        )

    if int(
        integrity.get(
            "history_bars",
            0,
        )
        or 0
    ) < 360:
        _fail(
            "History is shorter than zone max_age"
        )

    now = datetime.now(
        timezone.utc
    ).timestamp()

    ohlc = packet.get(
        "ohlc",
        {},
    )

    for timeframe in [
        "4H",
        "1H",
        "15M",
    ]:
        rows = ohlc.get(
            timeframe,
            [],
        )

        if not rows:
            _fail(
                f"No OHLC rows for {timeframe}"
            )

        timestamps = [
            datetime.fromisoformat(
                row["time"]
            ).timestamp()
            for row in rows
        ]

        if timestamps != sorted(
            timestamps
        ):
            _fail(
                (
                    f"OHLC order is not sorted "
                    f"for {timeframe}"
                )
            )

        last_open = (
            timestamps[-1]
        )

        close_time = (
            last_open
            + INTERVAL_SECONDS[
                timeframe
            ]
        )

        if close_time > (
            now + 5
        ):
            _fail(
                (
                    f"Forming candle leaked into "
                    f"{timeframe} AI/chart packet"
                )
            )

    plan = report.get(
        "trade_plan",
        {},
    )

    alignment = report.get(
        "mtf_alignment",
        {},
    )

    if (
        plan.get(
            "execution_ready",
            False,
        )
        and alignment.get(
            "label"
        )
        == "CONFLICT"
    ):
        _fail(
            (
                "Execution READY while "
                "MTF alignment is CONFLICT"
            )
        )

    if (
        plan.get(
            "execution_ready",
            False,
        )
        and plan.get(
            "blockers"
        )
    ):
        _fail(
            "Execution READY with blockers"
        )

    if plan.get(
        "active",
        False,
    ):
        direction = plan.get(
            "direction"
        )

        entry_mid = float(
            plan.get(
                "entry_mid"
            )
        )

        targets = [
            float(
                item["price"]
            )
            for item in plan.get(
                "targets",
                [],
            )
        ]

        if direction == "long":
            if any(
                target <= entry_mid
                for target
                in targets
            ):
                _fail(
                    "LONG target is not above entry"
                )

            if targets != sorted(
                targets
            ):
                _fail(
                    "LONG targets are not monotonic"
                )

        elif direction == "short":
            if any(
                target >= entry_mid
                for target
                in targets
            ):
                _fail(
                    "SHORT target is not below entry"
                )

            if targets != sorted(
                targets,
                reverse=True,
            ):
                _fail(
                    "SHORT targets are not monotonic"
                )

        else:
            _fail(
                "Active trade plan has invalid direction"
            )

        source = (
            plan.get(
                "entry_zone",
                {},
            ).get(
                "source",
                "",
            )
        )

        if "historical" in str(
            source
        ).lower():
            _fail(
                "Historical/reference zone used as entry"
            )

    print(
        "SELF_CHECK_OK: "
        "closed candles, status, MTF gate, "
        "and trade-map invariants passed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
