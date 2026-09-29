from binance_data import get_klines
from chart import create_chart


def main():

    print("================================")
    print("VVVUSDT BINANCE DATA")
    print("================================")

    # Lấy dữ liệu OHLC thật từ Binance Futures
    df_4h = get_klines("4h", 200)
    df_1h = get_klines("1h", 200)
    df_15m = get_klines("15m", 200)

    # Giá Close của candle 15m đang chạy
    current_price = df_15m.iloc[-1]["close"]

    print(f"\nCurrent VVVUSDT: {current_price} USDT")

    # Hiển thị 5 candle 4H cuối để kiểm tra
    print("\nLatest 5 x 4H candles:")

    print(
        df_4h[
            ["open", "high", "low", "close", "volume"]
        ].tail(5)
    )

    # Tạo chart
    create_chart(
        df_4h,
        "4H",
        "output/VVVUSDT_4H.png"
    )

    create_chart(
        df_1h,
        "1H",
        "output/VVVUSDT_1H.png"
    )

    create_chart(
        df_15m,
        "15M",
        "output/VVVUSDT_15M.png"
    )

    print("\n================================")
    print("ALL CHARTS CREATED SUCCESSFULLY")
    print("================================")


if __name__ == "__main__":
    main()
