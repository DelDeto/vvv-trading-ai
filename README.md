# vvv-trading-ai

Automated VVV_USDT Futures Price Action / SMC monitoring on MEXC.

Production flow:

```text
MEXC closed OHLC
-> deterministic Python PA/SMC
-> Market Regime
-> Multi-Timeframe Alignment
-> rule-based Trade Plan
-> production self-check
-> OpenRouter qualitative review
-> deterministic AI validator
-> Telegram
```

Key principles:
- No forming candle is used for signals.
- Python owns all numerical levels and chart rendering.
- AI never invents candles, Supply/Demand, Entry, SL or TP.
- Signal status and execution readiness are separate.
- MTF conflict blocks execution.
- Forward outcomes are persisted in `signal_history.json`.
- Weekly no-lookahead walk-forward evaluation is available via GitHub Actions.

See `ANALYSIS_METHOD.md` for the full PA-MTF Hybrid V2 specification.
