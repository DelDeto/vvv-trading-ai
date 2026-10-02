import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


STATE_PATH = Path("telegram_state.json")
LOCAL_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def _current_auto_slot():
    return datetime.now(
        LOCAL_TZ
    ).strftime("%Y-%m-%dT%H")


def main():
    event = os.getenv(
        "VVV_RUN_EVENT",
        "",
    )
    source = os.getenv(
        "VVV_RUN_SOURCE",
        "",
    )

    # Explicit user-triggered runs always execute, but they do not
    # consume the automatic hourly delivery slot.
    manual = (
        event == "push"
        or (
            event == "workflow_dispatch"
            and source != "watchdog"
        )
    )

    if manual:
        print("run=true")
        print("reason=Manual run")
        return 0

    current_slot = (
        _current_auto_slot()
    )

    if not STATE_PATH.exists():
        print("run=true")
        print(
            "reason=No delivery state"
        )
        return 0

    try:
        data = json.loads(
            STATE_PATH.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        print("run=true")
        print(
            "reason=Unreadable delivery state"
        )
        return 0

    last_auto_slot = data.get(
        "last_auto_slot"
    )

    # Migration support for state files written before last_auto_slot
    # existed: infer the slot only if the previous delivery was
    # automatic.
    if not last_auto_slot:
        last_event = data.get(
            "last_event"
        )
        last_sent = data.get(
            "last_sent_utc"
        )

        if (
            last_event
            in ("schedule", "watchdog")
            and last_sent
        ):
            try:
                previous = (
                    datetime
                    .fromisoformat(
                        last_sent.replace(
                            "Z",
                            "+00:00",
                        )
                    )
                    .astimezone(
                        LOCAL_TZ
                    )
                )
                last_auto_slot = (
                    previous.strftime(
                        "%Y-%m-%dT%H"
                    )
                )
            except Exception:
                last_auto_slot = None

    if (
        last_auto_slot
        == current_slot
    ):
        print("run=false")
        print(
            "reason=Automatic delivery "
            f"already completed for {current_slot}"
        )
        return 0

    print("run=true")
    print(
        "reason=Automatic delivery "
        f"missing for {current_slot}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
