from mexc_data import get_klines
from chart import create_chart


def main():

    print("=" * 40)
    print("VVV_USDT MEXC FUTURES DATA")
    print("=" * 40)

    # Lấy OHLC thật từ MEXC Futures
    df_4h = get_klines("4h", 200)
    df_1h = get_klines("1h", 200)
    df_15m = get_klines("15m", 200)

    current_price = df_15m.iloc[-1]["close"]

    print()
    print(f"Current VVV_USDT close: {current_price}")
    print()

    print("Latest 5 x 4H candles:")
    print(
        df_4h[
            ["open", "high", "low", "close", "volume"]
        ].tail(5)
    )

    print()

    create_chart(
        df_4h,
        "4H",
        "output/VVVUSDT_4H.png",
        candles=100,
        exchange="MEXC"
    )

    create_chart(
        df_1h,
        "1H",
        "output/VVVUSDT_1H.png",
        candles=120,
        exchange="MEXC"
    )

    create_chart(
        df_15m,
        "15M",
        "output/VVVUSDT_15M.png",
        candles=150,
        exchange="MEXC"
    )

    print()
    print("=" * 40)
    print("ALL MEXC CHARTS CREATED")
    print("=" * 40)


if __name__ == "__main__":
    main()
