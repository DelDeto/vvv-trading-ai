import json
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import requests


BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

OUTPUT_DIR = Path("output")
REPORT_PATH = OUTPUT_DIR / "VVVUSDT_SMC_report.json"
UPDATE_PATH = OUTPUT_DIR / "VVVUSDT_hourly_update.txt"
AI_UPDATE_PATH = OUTPUT_DIR / "VVVUSDT_ai_analysis.txt"
AI_JSON_PATH = OUTPUT_DIR / "VVVUSDT_ai_analysis.json"
TELEGRAM_STATE_PATH = Path("telegram_state.json")
RUN_EVENT = os.getenv("VVV_RUN_EVENT", "")
RUN_SOURCE = os.getenv("VVV_RUN_SOURCE", "")
LOCAL_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

CHARTS = [
    ("4H", OUTPUT_DIR / "VVVUSDT_4H.png"),
    ("1H", OUTPUT_DIR / "VVVUSDT_1H.png"),
    ("15M", OUTPUT_DIR / "VVVUSDT_15M.png"),
]


def _load_delivery_state():
    if not TELEGRAM_STATE_PATH.exists():
        return {
            "last_sent_utc": None,
            "last_event": None,
        }

    try:
        return json.loads(
            TELEGRAM_STATE_PATH.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return {
            "last_sent_utc": None,
            "last_event": None,
        }


def _save_delivery_state(receipt, slot):
    previous = _load_delivery_state()

    event_label = (
        RUN_SOURCE
        or RUN_EVENT
        or "unknown"
    )

    last_auto_slot = (
        previous.get(
            "last_auto_slot"
        )
    )

    is_auto_delivery = (
        RUN_EVENT == "schedule"
        or RUN_SOURCE == "watchdog"
    )

    if is_auto_delivery:
        last_auto_slot = slot

    TELEGRAM_STATE_PATH.write_text(
        json.dumps(
            {
                "last_sent_utc": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
                "last_event": event_label,
                "message_id": receipt["result"]["message_id"],
                "last_auto_slot": (
                    last_auto_slot
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _api(method):
    return f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"


def _check_config():
    missing = []

    if not BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")

    if not CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")

    if missing:
        print(
            "Telegram notification skipped. "
            "Missing secrets: "
            + ", ".join(missing)
        )
        return False

    return True


def _post(method, **kwargs):
    response = requests.post(
        _api(method),
        timeout=30,
        **kwargs,
    )

    if response.status_code != 200:
        print(
            f"Telegram {method} failed "
            f"with HTTP {response.status_code}"
        )
        print(response.text)
        raise RuntimeError(
            f"Telegram {method} request failed"
        )

    payload = response.json()

    if not payload.get("ok"):
        print(payload)
        raise RuntimeError(
            f"Telegram {method} returned ok=false"
        )

    return payload


def _split_message(text, limit=3900):
    text = text.strip()

    if not text:
        return []

    chunks = []

    while len(text) > limit:
        cut = text.rfind(
            "\n",
            0,
            limit,
        )

        if cut <= 0:
            cut = limit

        chunks.append(
            text[:cut].strip()
        )

        text = text[cut:].strip()

    if text:
        chunks.append(text)

    return chunks


def _short_caption(report, timeframe):
    current_price = report.get(
        "current_price"
    )

    status = report.get(
        "status",
        "WAIT",
    )

    trade_plan = report.get(
        "trade_plan",
        {},
    )

    execution = (
        "READY"
        if trade_plan.get(
            "execution_ready",
            False,
        )
        else "WAIT"
    )

    price_text = (
        f"{current_price:.3f}"
        if isinstance(
            current_price,
            (int, float),
        )
        else "-"
    )

    ai_bias = None

    if AI_JSON_PATH.exists():
        try:
            ai_payload = json.loads(
                AI_JSON_PATH.read_text(
                    encoding="utf-8"
                )
            )

            ai_bias = (
                ai_payload
                .get(
                    "analysis",
                    {},
                )
                .get(
                    "ai_bias"
                )
            )

        except Exception:
            ai_bias = None

    lines = [
        (
            f"VVV_USDT {timeframe} | "
            "MEXC"
        ),
        f"Price: {price_text}",
        f"Python: {status}",
        f"Execution: {execution}",
    ]

    if ai_bias:
        lines.append(
            f"AI bias: {ai_bias}"
        )

    return "\n".join(lines)


def _send_text_update():
    source = (
        AI_UPDATE_PATH
        if AI_UPDATE_PATH.exists()
        else UPDATE_PATH
    )

    if not source.exists():
        raise FileNotFoundError("Missing VVV text report")
    text = source.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError("Empty VVV text report")
    print("Telegram text mode:", "hybrid AI" if source == AI_UPDATE_PATH else "Python fallback")
    # One acknowledged core report per slot. Long reports use a document,
    # avoiding partial multi-message delivery and duplicate text on retry.
    if len(text.encode("utf-16-le")) // 2 <= 3900:
        return _post("sendMessage", data={"chat_id": CHAT_ID, "text": text,
                     "disable_web_page_preview": "true"})
    import io
    with io.BytesIO(text.encode("utf-8")) as stream:
        return _post("sendDocument", data={"chat_id": CHAT_ID,
                     "caption": "VVV_USDT | MEXC hourly report"},
                     files={"document": ("VVV_hourly_report.txt", stream, "text/plain")})



def _send_chart(report, timeframe, path):
    if not path.exists():
        print(f"::warning::Chart missing: {path}")
        return
    with path.open("rb") as image_file:
        _post("sendPhoto", data={"chat_id": CHAT_ID,
              "caption": _short_caption(report, timeframe)}, files={"photo": image_file})
    print(f"Telegram sent: {path}")



def main():
    if not _check_config():
        raise RuntimeError("VVV Telegram secrets missing")

    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Missing report: {REPORT_PATH}"
        )

    report = json.loads(
        REPORT_PATH.read_text(
            encoding="utf-8"
        )
    )

    slot = datetime.now(LOCAL_TZ).strftime("%Y-%m-%dT%H")
    automatic = RUN_EVENT == "schedule" or RUN_SOURCE == "watchdog"
    if automatic and _load_delivery_state().get("last_auto_slot") == slot:
        print("VVV automatic delivery already completed for", slot)
        return 0
    receipt = _send_text_update()
    _save_delivery_state(receipt, slot)
    print("VVV Telegram acknowledged message_id=", receipt["result"]["message_id"])
    # Charts are supplemental. Never resend a successful core report because
    # an optional photo failed.
    for item in CHARTS:
        try:
            _send_chart(report, *item)
        except Exception as exc:
            print(f"::warning::VVV photo unavailable {item[0]}: {type(exc).__name__}")

    print(
        "Telegram notification completed."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
