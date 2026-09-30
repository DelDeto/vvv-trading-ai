import os
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd


# Binance dark-theme palette
BG = "#0B0E11"
PANEL = "#0B0E11"
GRID = "#1E2329"
TEXT = "#EAECEF"
MUTED = "#848E9C"

GREEN = "#0ECB81"
RED = "#F6465D"
YELLOW = "#F0B90B"
BLUE = "#2B7FFF"

SUPPLY = "#F6465D"
DEMAND = "#0ECB81"
BULL_FVG = "#0ECB81"
BEAR_FVG = "#F6465D"


binance_mc = mpf.make_marketcolors(
    up=GREEN,
    down=RED,
    edge={"up": GREEN, "down": RED},
    wick={"up": GREEN, "down": RED},
    volume={"up": GREEN, "down": RED},
    ohlc={"up": GREEN, "down": RED},
)

binance_style = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    marketcolors=binance_mc,
    facecolor=PANEL,
    figcolor=BG,
    gridcolor=GRID,
    gridstyle="-",
    y_on_right=True,
    rc={
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": TEXT,
        "axes.grid": True,
        "axes.grid.axis": "both",
        "axes.grid.which": "major",
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": TEXT,
        "font.size": 9,
    },
)


def _fmt(value):
    if value is None:
        return "-"

    return f"{value:.3f}"


def _style_axes(axes):
    """
    Remove heavy borders and keep the chart close to Binance dark mode.
    """

    for axis in axes:
        axis.set_facecolor(PANEL)

        for spine in axis.spines.values():
            spine.set_visible(False)

        axis.tick_params(
            colors=MUTED,
            labelsize=8,
            length=0,
        )

        axis.grid(
            True,
            color=GRID,
            linewidth=0.6,
            alpha=0.55,
        )


def _draw_price_tag(
    ax,
    current_price,
    last_open,
):
    """
    Binance-like current-price dotted line + price tag on right axis.
    """

    price_color = (
        GREEN
        if current_price >= last_open
        else RED
    )

    ax.axhline(
        current_price,
        color=price_color,
        linewidth=0.9,
        linestyle=(0, (2, 2)),
        alpha=0.95,
        zorder=2,
    )

    ax.text(
        1.002,
        current_price,
        f" {_fmt(current_price)} ",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="center",
        fontsize=8,
        color=BG,
        clip_on=False,
        bbox=dict(
            boxstyle="square,pad=0.22",
            facecolor=price_color,
            edgecolor=price_color,
            linewidth=0,
        ),
        zorder=7,
    )


def _draw_liquidity_line(
    ax,
    price,
    label,
    color,
    va,
):
    ax.axhline(
        price,
        color=color,
        linewidth=0.9,
        linestyle=(0, (5, 4)),
        alpha=0.85,
        zorder=2,
    )

    ax.text(
        0.995,
        price,
        f" {label} {_fmt(price)} ",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va=va,
        fontsize=7.5,
        color=color,
        bbox=dict(
            boxstyle="round,pad=0.20",
            facecolor=BG,
            edgecolor=color,
            linewidth=0.7,
            alpha=0.92,
        ),
        zorder=6,
    )


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
    theo phong cách Binance và overlay SMC rule-based.
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

    last_open = float(
        plot_df.iloc[-1]["Open"]
    )

    last_high = float(
        plot_df.iloc[-1]["High"]
    )

    last_low = float(
        plot_df.iloc[-1]["Low"]
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
        style=binance_style,
        ylabel="",
        ylabel_lower="",
        datetime_format="%m-%d %H:%M",
        xrotation=0,
        figsize=(16, 9),
        tight_layout=True,
        returnfig=True,
        update_width_config={
            "candle_width": 0.68,
            "candle_linewidth": 0.75,
            "volume_width": 0.68,
        },
    )

    fig.patch.set_facecolor(BG)

    _style_axes(axes)

    ax = axes[0]
    x_right = len(plot_df) - 1

    # Binance-like title/header
    fig.text(
        0.065,
        0.972,
        (
            f"VVV_USDT Perpetual  ·  "
            f"{interval}  ·  {exchange}"
        ),
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=TEXT,
    )

    candle_color = (
        GREEN
        if current_price >= last_open
        else RED
    )

    fig.text(
        0.065,
        0.947,
        (
            f"O {_fmt(last_open)}    "
            f"H {_fmt(last_high)}    "
            f"L {_fmt(last_low)}    "
            f"C {_fmt(current_price)}"
        ),
        ha="left",
        va="top",
        fontsize=8.5,
        color=candle_color,
    )

    _draw_price_tag(
        ax,
        current_price,
        last_open,
    )

    if analysis:
        bsl = analysis.get("bsl")
        ssl = analysis.get("ssl")

        if bsl:
            _draw_liquidity_line(
                ax,
                bsl["price"],
                "BSL",
                YELLOW,
                "bottom",
            )

        if ssl:
            _draw_liquidity_line(
                ax,
                ssl["price"],
                "SSL",
                BLUE,
                "top",
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
                color=DEMAND,
                alpha=0.075,
                zorder=0,
            )

            ax.axhline(
                demand["upper"],
                color=DEMAND,
                linewidth=0.7,
                alpha=0.65,
            )

            ax.text(
                0.008,
                demand["upper"],
                (
                    f" DEMAND "
                    f"{_fmt(demand['lower'])}-"
                    f"{_fmt(demand['upper'])} "
                ),
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="bottom",
                fontsize=7.2,
                color=DEMAND,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor=BG,
                    edgecolor=DEMAND,
                    linewidth=0.6,
                    alpha=0.88,
                ),
            )

        # Supply zone
        if supply:
            ax.axhspan(
                supply["lower"],
                supply["upper"],
                color=SUPPLY,
                alpha=0.075,
                zorder=0,
            )

            ax.axhline(
                supply["lower"],
                color=SUPPLY,
                linewidth=0.7,
                alpha=0.65,
            )

            ax.text(
                0.008,
                supply["lower"],
                (
                    f" SUPPLY "
                    f"{_fmt(supply['lower'])}-"
                    f"{_fmt(supply['upper'])} "
                ),
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="top",
                fontsize=7.2,
                color=SUPPLY,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor=BG,
                    edgecolor=SUPPLY,
                    linewidth=0.6,
                    alpha=0.88,
                ),
            )

        # Last two active FVGs
        active_fvgs = analysis.get(
            "active_fvgs",
            [],
        )[-2:]

        for fvg in active_fvgs:
            fvg_color = (
                BULL_FVG
                if fvg["type"] == "bullish"
                else BEAR_FVG
            )

            ax.axhspan(
                fvg["lower"],
                fvg["upper"],
                color=fvg_color,
                alpha=0.045,
                zorder=0,
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
                fontsize=6.8,
                color=fvg_color,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor=BG,
                    edgecolor=fvg_color,
                    linewidth=0.5,
                    alpha=0.82,
                ),
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
                    event_x = (
                        plot_df.index.get_loc(
                            timestamp
                        )
                    )

                    event_y = float(
                        plot_df.loc[
                            timestamp,
                            "Close",
                        ]
                    )

                    bullish = (
                        last_event["direction"]
                        == "bullish"
                    )

                    marker = (
                        "^"
                        if bullish
                        else "v"
                    )

                    event_color = (
                        GREEN
                        if bullish
                        else RED
                    )

                    ax.scatter(
                        [event_x],
                        [event_y],
                        marker=marker,
                        s=58,
                        color=event_color,
                        edgecolors=BG,
                        linewidths=0.6,
                        zorder=6,
                    )

                    ax.text(
                        event_x,
                        event_y,
                        (
                            f" {last_event['kind']} "
                            f"{last_event['direction'].upper()}"
                        ),
                        fontsize=7.2,
                        color=event_color,
                        va=(
                            "bottom"
                            if bullish
                            else "top"
                        ),
                        bbox=dict(
                            boxstyle="round,pad=0.15",
                            facecolor=BG,
                            edgecolor=event_color,
                            linewidth=0.45,
                            alpha=0.85,
                        ),
                    )

            except Exception as exc:
                print(
                    "SMC event overlay warning:",
                    exc,
                )


        # Latest Liquidity Sweep
        last_sweep = analysis.get(
            "last_sweep"
        )

        if last_sweep:
            try:
                timestamp = pd.Timestamp(
                    last_sweep["time"]
                )

                if timestamp in plot_df.index:
                    sweep_x = (
                        plot_df.index.get_loc(
                            timestamp
                        )
                    )

                    sweep_y = float(
                        last_sweep.get(
                            "extreme",
                            last_sweep[
                                "level"
                            ],
                        )
                    )

                    ax.scatter(
                        [sweep_x],
                        [sweep_y],
                        marker="o",
                        s=42,
                        facecolors="none",
                        edgecolors=YELLOW,
                        linewidths=1.2,
                        zorder=7,
                    )

                    ax.text(
                        sweep_x,
                        sweep_y,
                        (
                            f" {last_sweep['type']} "
                            "SWEEP"
                        ),
                        fontsize=6.8,
                        color=YELLOW,
                        va=(
                            "bottom"
                            if last_sweep[
                                "direction"
                            ]
                            == "bearish"
                            else "top"
                        ),
                        bbox=dict(
                            boxstyle="round,pad=0.13",
                            facecolor=BG,
                            edgecolor=YELLOW,
                            linewidth=0.45,
                            alpha=0.85,
                        ),
                        zorder=8,
                    )

            except Exception as exc:
                print(
                    "Sweep overlay warning:",
                    exc,
                )

        # Latest displacement
        last_displacement = analysis.get(
            "last_displacement"
        )

        if last_displacement:
            try:
                timestamp = pd.Timestamp(
                    last_displacement[
                        "time"
                    ]
                )

                if timestamp in plot_df.index:
                    disp_x = (
                        plot_df.index.get_loc(
                            timestamp
                        )
                    )

                    disp_y = float(
                        last_displacement[
                            "close"
                        ]
                    )

                    bullish_disp = (
                        last_displacement[
                            "direction"
                        ]
                        == "bullish"
                    )

                    disp_color = (
                        GREEN
                        if bullish_disp
                        else RED
                    )

                    ax.scatter(
                        [disp_x],
                        [disp_y],
                        marker="*",
                        s=72,
                        color=disp_color,
                        edgecolors=BG,
                        linewidths=0.5,
                        zorder=7,
                    )

                    ax.text(
                        disp_x,
                        disp_y,
                        (
                            " DISP "
                            f"{last_displacement['strength']:.1f}x"
                        ),
                        fontsize=6.8,
                        color=disp_color,
                        va=(
                            "bottom"
                            if bullish_disp
                            else "top"
                        ),
                        bbox=dict(
                            boxstyle="round,pad=0.13",
                            facecolor=BG,
                            edgecolor=disp_color,
                            linewidth=0.45,
                            alpha=0.85,
                        ),
                        zorder=8,
                    )

            except Exception as exc:
                print(
                    "Displacement overlay warning:",
                    exc,
                )

        # Latest retest
        last_retest = analysis.get(
            "last_retest"
        )

        if last_retest:
            try:
                timestamp = pd.Timestamp(
                    last_retest["time"]
                )

                if timestamp in plot_df.index:
                    retest_x = (
                        plot_df.index.get_loc(
                            timestamp
                        )
                    )

                    retest_y = float(
                        last_retest["level"]
                    )

                    ax.scatter(
                        [retest_x],
                        [retest_y],
                        marker="D",
                        s=34,
                        color=BLUE,
                        edgecolors=BG,
                        linewidths=0.5,
                        zorder=7,
                    )

                    ax.text(
                        retest_x,
                        retest_y,
                        " RETEST",
                        fontsize=6.8,
                        color=BLUE,
                        va="bottom",
                        bbox=dict(
                            boxstyle="round,pad=0.13",
                            facecolor=BG,
                            edgecolor=BLUE,
                            linewidth=0.45,
                            alpha=0.85,
                        ),
                        zorder=8,
                    )

            except Exception as exc:
                print(
                    "Retest overlay warning:",
                    exc,
                )

        event_text = "None"

        if last_event:
            event_text = (
                f"{last_event['kind']} "
                f"{last_event['direction'].upper()}"
            )

        trend_text = analysis.get(
            "trend",
            "neutral",
        ).upper()

        trend_color = MUTED

        if trend_text == "BULLISH":
            trend_color = GREEN

        elif trend_text == "BEARISH":
            trend_color = RED

        setup = analysis.get(
            "setup",
            {},
        )

        setup_state = (
            "CONFIRMED"
            if setup.get(
                "confirmed",
                False,
            )
            else "DEVELOPING"
        )

        setup_direction = (
            setup.get(
                "direction",
                "-",
            ).upper()
        )

        setup_score = setup.get(
            "score",
            0,
        )

        summary = (
            f"TREND  {trend_text}\n"
            f"STRUCTURE  {event_text}\n"
            f"SETUP  {setup_state} "
            f"{setup_direction} {setup_score}/4\n"
            f"ATR  {_fmt(analysis.get('atr'))}"
        )

        ax.text(
            0.012,
            0.900,
            summary,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.6,
            color=trend_color,
            bbox=dict(
                boxstyle="round,pad=0.42",
                facecolor="#181A20",
                edgecolor=GRID,
                linewidth=0.8,
                alpha=0.92,
            ),
            zorder=8,
        )

    fig.savefig(
        output_path,
        dpi=160,
        facecolor=BG,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved chart: {output_path}"
    )
