import json
import os
from datetime import datetime, timezone
from pathlib import Path


STATE_PATH = Path("telegram_state.json")
MIN_INTERVAL_MINUTES = 50


def main():
    event = os.getenv("VVV_RUN_EVENT", "")

    # Manual runs should always execute.
    if event != "schedule":
        print("run=true")
        return 0

    if not STATE_PATH.exists():
        print("run=true")
        return 0

    try:
        data = json.loads(
            STATE_PATH.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        print("run=true")
        return 0

    last_sent = data.get(
        "last_sent_utc"
    )

    if not last_sent:
        print("run=true")
        return 0

    try:
        previous = datetime.fromisoformat(
            last_sent
        )

        if previous.tzinfo is None:
            previous = previous.replace(
                tzinfo=timezone.utc
            )

        minutes = (
            datetime.now(
                timezone.utc
            )
            - previous
        ).total_seconds() / 60.0

    except Exception:
        print("run=true")
        return 0

    if minutes < MIN_INTERVAL_MINUTES:
        print(
            (
                "run=false\n"
                f"reason=Last Telegram delivery was "
                f"{minutes:.1f} minutes ago"
            )
        )
        return 0

    print(
        (
            "run=true\n"
            f"reason=Last Telegram delivery was "
            f"{minutes:.1f} minutes ago"
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
