# PA-MTF Hybrid V1

This repository uses a hybrid architecture for VVV_USDT market analysis.

## 1. Deterministic Python layer — source of truth

The Python layer is responsible for every numerical market fact and every chart.

Data source:
- MEXC Futures OHLC and contract snapshot.
- The same exact OHLC packet used to build the PNG charts is persisted for the AI layer.
- AI never creates, edits, interpolates, or replaces candles.

Core Price Action / SMC calculations:
- HH / HL / LH / LL trend context.
- BOS and CHoCH from confirmed swing breaks.
- Supply / Demand V3 using DBR, RBR, RBD, DBD price-action origins.
- Zone freshness and mitigation count.
- BSL / SSL liquidity pools.
- Liquidity sweep.
- Displacement.
- Fair Value Gap (FVG).
- Retest.
- ATR-based volatility context.
- Rule-based setup and execution gate.

Supplementary technical confirmation:
- EMA20 / EMA50 relationship.
- Anchored VWAP relationship.
- RSI14 regime.
- Relative-volume regime.
- 20-bar range location.
- MEXC holdVol/open-position proxy.
- Funding rate.

Python owns:
- Candles.
- Chart rendering.
- Supply/Demand boundaries.
- BSL/SSL.
- FVG boundaries.
- BOS/CHoCH.
- Entry / SL / TP from the rule-based Trade Map.
- WAIT / execution_ready gate.

## 2. AI interpretation layer

Default AI route:
- Primary: Qwen3.8 27B (free), `qwen/qwen3.8-27b:free`.
- Fallback: OpenRouter Free Models Router, `openrouter/free`.
- OpenRouter OpenAI-compatible Chat Completions API (`/api/v1/chat/completions`).
- Structured output is enforced with JSON Schema and provider routing requires support for the requested parameters.
- Model reasoning is disabled for this task to keep the hourly response short and deterministic; the model still receives Python-derived PA/SMC features plus the most recent 64 real candles per timeframe.

The model receives structured JSON, not an invented chart.

Top-down reasoning order:
1. 4H — macro structure and major location.
2. 1H — intermediate structure and decision area.
3. 15M — execution structure and timing.

Evidence hierarchy:
1. Market structure and BOS/CHoCH.
2. Active Supply/Demand location and zone freshness.
3. BSL/SSL and liquidity sweep.
4. Displacement, FVG and retest.
5. EMA, anchored VWAP, ATR, RSI and relative volume.
6. holdVol and funding as secondary context only.

Rules:
- AI cannot invent numerical levels.
- AI cannot promote Historical/Reference zones to active zones.
- AI cannot override Python execution_ready.
- When timeframes conflict, the report must explicitly mention the conflict.
- If the evidence is incomplete, WAIT is preferred over forcing a direction.
- AI bias is qualitative and is not a probability.

## 3. Telegram delivery

Hourly workflow:
MEXC -> Python analysis -> PNG charts -> exact AI input packet -> AI qualitative review -> Telegram.

If OPENROUTER_API_KEY is absent or the AI step fails:
- The deterministic Python pipeline still completes.
- Telegram automatically falls back to the Python hourly update.
- Charts remain unaffected.

Required GitHub Actions secret:
- OPENROUTER_API_KEY

Optional model overrides:
- OPENROUTER_MODEL
- Default: qwen/qwen3.8-27b:free
- OPENROUTER_FALLBACK_MODEL
- Default: openrouter/free

Telegram secrets remain:
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID
