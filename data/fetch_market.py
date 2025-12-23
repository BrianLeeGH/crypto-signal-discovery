"""
市场数据采集模块
使用 ccxt 从 Binance 获取K线数据
"""

import os
import sys
import time

import ccxt
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    DEFAULT_DAYS,
    DEFAULT_EXCHANGE,
    DEFAULT_TIMEFRAME,
    RAW_DATA_DIR,
    SYMBOLS,
)


def get_exchange(exchange_id="binance"):
    """初始化交易所连接"""
    exchange_class = getattr(ccxt, exchange_id)
    options = {"enableRateLimit": True}
    if exchange_id == "binance":
        options["options"] = {"defaultType": "future"}
    exchange = exchange_class(options)
    return exchange


def fetch_ohlcv(exchange, symbol, timeframe="1h", days=180):
    """
    获取K线数据

    Parameters:
    -----------
    exchange : ccxt.Exchange
    symbol : str, 如 'BTC/USDT'
    timeframe : str, 如 '1h', '4h', '1d'
    days : int, 获取多少天的数据

    Returns:
    --------
    pd.DataFrame
    """
    print(f"Fetching {symbol} {timeframe} data for {days} days...")

    # 计算需要获取的K线数量
    timeframe_hours = {
        "1m": 1 / 60,
        "5m": 5 / 60,
        "15m": 15 / 60,
        "30m": 30 / 60,
        "1h": 1,
        "4h": 4,
        "1d": 24,
    }
    hours_per_candle = timeframe_hours.get(timeframe, 1)
    total_candles = int(days * 24 / hours_per_candle)

    # Binance 每次最多返回 1000 条
    limit = 1000
    all_data = []

    # 从现在往回取
    since = None

    for _ in range(0, total_candles, limit):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break

            all_data = ohlcv + all_data  # 新数据在前

            # 更新 since 为最早数据的前一个时间点
            since = ohlcv[0][0] - 1

            print(f"  Fetched {len(all_data)} candles...")
            time.sleep(0.1)  # 避免触发限流

            if len(ohlcv) < limit:
                break

        except Exception as e:
            print(f"  Error: {e}")
            break

    # 转换为 DataFrame
    df = pd.DataFrame(
        all_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    df = df.sort_index()

    # 去重
    df = df[~df.index.duplicated(keep="last")]

    print(f"  Done. Total {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    return df


def save_data(df, filename):
    """保存数据到CSV"""
    filepath = os.path.join(RAW_DATA_DIR, filename)
    df.to_csv(filepath)
    print(f"Saved to {filepath}")


def main():
    """主函数：采集所有配置的交易对数据"""
    exchange = get_exchange(DEFAULT_EXCHANGE)

    for symbol in SYMBOLS:
        safe_symbol = symbol.replace("/", "_")

        # K线数据
        ohlcv = fetch_ohlcv(exchange, symbol, DEFAULT_TIMEFRAME, DEFAULT_DAYS)
        if not ohlcv.empty:
            save_data(ohlcv, f"{safe_symbol}_ohlcv_{DEFAULT_TIMEFRAME}.csv")

        print(f"\n{'='*50}\n")

    print("Market data collection complete!")


if __name__ == "__main__":
    main()
