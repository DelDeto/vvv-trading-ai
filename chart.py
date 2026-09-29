import os
import mplfinance as mpf


def create_chart(df, interval, output_path, candles=100):
    """
    Vẽ candlestick chart từ OHLC Binance thật.
    """

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plot_df = df[
        ["open", "high", "low", "close", "volume"]
    ].copy()

    plot_df.columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    plot_df = plot_df.tail(candles)

    current_price = plot_df.iloc[-1]["Close"]

    print(
        f"Creating {interval} chart | "
        f"Current close: {current_price}"
    )

    mpf.plot(
        plot_df,
        type="candle",
        volume=True,
        style="nightclouds",
        title=f"VVVUSDT PERPETUAL - BINANCE - {interval}",
        ylabel="Price (USDT)",
        ylabel_lower="Volume",
        figsize=(16, 9),
        tight_layout=True,
        savefig=dict(
            fname=output_path,
            dpi=150,
            bbox_inches="tight"
        )
    )

    print(f"Saved chart: {output_path}")
