import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch, Mock

import mexc_data as m
import schedule_gate as g
import telegram_notify as t


class ReliabilityTests(unittest.TestCase):
    def test_retry_transient_mexc(self):
        with patch.object(m.requests, "get", side_effect=[
            Mock(status_code=503), Mock(status_code=200)
        ]) as get, patch.object(m.time, "sleep"):
            self.assertEqual(m._get_with_retry("https://example.invalid").status_code, 200)
            self.assertEqual(get.call_count, 2)

    def test_gate(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "state.json"
            path.write_text(json.dumps({"last_auto_slot": "2026-11-26T09"}))
            with patch.object(g, "STATE_PATH", path), patch.object(g, "_current_auto_slot", return_value="2026-11-26T09"), patch.dict(g.os.environ, {"VVV_RUN_EVENT":"schedule","VVV_RUN_SOURCE":""}), redirect_stdout(io.StringIO()) as out:
                g.main()
                self.assertIn("run=false", out.getvalue())

    def test_delivery_missing_secret(self):
        with patch.object(t, "BOT_TOKEN", None), patch.object(t, "CHAT_ID", None):
            with self.assertRaises(RuntimeError):
                t.main()

    def test_delivery_core_then_photo_failure_and_dedupe(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            report = root / "report.json"
            report.write_text("{}")
            update = root / "update.txt"
            update.write_text("VVV test")
            state = root / "state.json"
            with patch.multiple(t, BOT_TOKEN="mock", CHAT_ID="mock", REPORT_PATH=report,
                    UPDATE_PATH=update, AI_UPDATE_PATH=root/"missing.txt", TELEGRAM_STATE_PATH=state,
                    RUN_EVENT="schedule", RUN_SOURCE="", CHARTS=[("4H",root/"chart.png")]), \
                    patch.object(t, "_post", return_value={"ok":True,"result":{"message_id":123}}) as post, \
                    patch.object(t, "_send_chart", side_effect=RuntimeError("photo error")):
                self.assertEqual(t.main(), 0)
                self.assertEqual(json.loads(state.read_text())["message_id"], 123)
                self.assertEqual(t.main(), 0)
                post.assert_called_once()

    def test_no_ack_no_state(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            report = root / "report.json"
            report.write_text("{}")
            state = root / "state.json"
            with patch.multiple(t, BOT_TOKEN="mock", CHAT_ID="mock", REPORT_PATH=report, TELEGRAM_STATE_PATH=state), patch.object(t, "_send_text_update", side_effect=RuntimeError("network")):
                with self.assertRaises(RuntimeError):
                    t.main()
                self.assertFalse(state.exists())

    def test_long_report_single_document(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "text"
            p.write_text("x"*5000)
            with patch.multiple(t, UPDATE_PATH=p, AI_UPDATE_PATH=p.with_suffix(".missing")), patch.object(t, "_post", return_value={"ok":True}) as post:
                t._send_text_update()
                post.assert_called_once()
                self.assertEqual(post.call_args.args[0], "sendDocument")


if __name__ == "__main__":
    unittest.main()
