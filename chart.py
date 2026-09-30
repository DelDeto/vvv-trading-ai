import os
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd


def _fmt(value):
    if value is None:
        return "-"

    return f"{value:.3f}"


def create_chart(
    df,
    interval,
    output_path,
    candles=100,
    exchange="MEXC",
    analysis=None,
):
    """
    Vẽ candlestick chart từ OHLC Futures thật
    và overlay SMC rule-based.
    """

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    plot_df = df[
        ["open", "high", "low", "close", "volume"]
    ].copy()

    plot_df.columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    plot_df = plot_df.tail(candles)

    current_price = float(
        plot_df.iloc[-1]["Close"]
    )

    print(
        f"Creating {interval} chart | "
        f"{exchange} | "
        f"Current close: {current_price}"
    )

    fig, axes = mpf.plot(
        plot_df,
        type="candle",
        volume=True,
        style="nightclouds",
        title=(
            f"VVV_USDT PERPETUAL - "
            f"{exchange} - {interval}"
        ),
        ylabel="Price (USDT)",
        ylabel_lower="Volume",
        figsize=(16, 9),
        tight_layout=True,
        returnfig=True,
    )

    ax = axes[0]

    x_right = len(plot_df) - 1

    # Current price
    ax.axhline(
        current_price,
        linewidth=1.0,
        linestyle=":",
        alpha=0.75,
    )

    ax.text(
        x_right,
        current_price,
        f" PRICE {_fmt(current_price)}",
        ha="right",
        va="bottom",
        fontsize=8,
    )

    if analysis:
        bsl = analysis.get("bsl")
        ssl = analysis.get("ssl")

        # Buy-side liquidity
        if bsl:
            ax.axhline(
                bsl["price"],
                linewidth=1.1,
                linestyle="--",
                alpha=0.85,
            )

            ax.text(
                x_right,
                bsl["price"],
                f" BSL {_fmt(bsl['price'])}",
                ha="right",
                va="bottom",
                fontsize=8,
            )

        # Sell-side liquidity
        if ssl:
            ax.axhline(
                ssl["price"],
                linewidth=1.1,
                linestyle="--",
                alpha=0.85,
            )

            ax.text(
                x_right,
                ssl["price"],
                f" SSL {_fmt(ssl['price'])}",
                ha="right",
                va="top",
                fontsize=8,
            )

        demand = analysis.get(
            "nearest_demand"
        )

        supply = analysis.get(
            "nearest_supply"
        )

        # Demand zone
        if demand:
            ax.axhspan(
                demand["lower"],
                demand["upper"],
                alpha=0.10,
            )

            ax.text(
                0,
                demand["upper"],
                (
                    f"Demand "
                    f"{_fmt(demand['lower'])}-"
                    f"{_fmt(demand['upper'])}"
                ),
                ha="left",
                va="bottom",
                fontsize=8,
            )

        # Supply zone
        if supply:
            ax.axhspan(
                supply["lower"],
                supply["upper"],
                alpha=0.10,
            )

            ax.text(
                0,
                supply["lower"],
                (
                    f"Supply "
                    f"{_fmt(supply['lower'])}-"
                    f"{_fmt(supply['upper'])}"
                ),
                ha="left",
                va="top",
                fontsize=8,
            )

        # Last two active FVGs
        active_fvgs = analysis.get(
            "active_fvgs",
            [],
        )[-2:]

        for fvg in active_fvgs:
            ax.axhspan(
                fvg["lower"],
                fvg["upper"],
                alpha=0.06,
            )

            ax.text(
                max(
                    0,
                    x_right - 18,
                ),
                (
                    fvg["lower"]
                    + fvg["upper"]
                )
                / 2,
                (
                    f"{fvg['type'].upper()} FVG "
                    f"{_fmt(fvg['lower'])}-"
                    f"{_fmt(fvg['upper'])}"
                ),
                ha="left",
                va="center",
                fontsize=7,
            )

        # Latest BOS / CHoCH
        last_event = analysis.get(
            "last_event"
        )

        if last_event:
            event_time = last_event.get(
                "time"
            )

            try:
                timestamp = pd.Timestamp(
                    event_time
                )

                if timestamp in plot_df.index:
                    event_x = plot_df.index.get_loc(
                        timestamp
                    )

                    event_y = float(
                        plot_df.loc[
                            timestamp,
                            "Close",
                        ]
                    )

                    marker = (
                        "^"
                        if last_event[
                            "direction"
                        ]
                        == "bullish"
                        else "v"
                    )

                    ax.scatter(
                        [event_x],
                        [event_y],
                        marker=marker,
                        s=70,
                        zorder=5,
                    )

                    ax.text(
                        event_x,
                        event_y,
                        (
                            f" "
                            f"{last_event['kind']} "
                            f"{last_event['direction'].upper()}"
                        ),
                        fontsize=8,
                        va=(
                            "bottom"
                            if marker == "^"
                            else "top"
                        ),
                    )

            except Exception as exc:
                print(
                    "SMC event overlay warning:",
                    exc,
                )

        event_text = "None"

        if last_event:
            event_text = (
                f"{last_event['kind']} "
                f"{last_event['direction'].upper()}"
            )

        summary = (
            f"Trend: "
            f"{analysis.get('trend', 'neutral').upper()}\n"
            f"Last structure: {event_text}\n"
            f"ATR: {_fmt(analysis.get('atr'))}"
        )

        ax.text(
            0.012,
            0.98,
            summary,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.4",
                alpha=0.35,
            ),
        )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved chart: {output_path}"
    )
