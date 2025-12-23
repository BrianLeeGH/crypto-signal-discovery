"""
合约数据采集模块
从交易所获取资金费率、持仓量、多空比等数据
支持 Binance 和 OKX
"""

import os
import sys
import time
from datetime import datetime

import ccxt
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DEFAULT_DAYS, DEFAULT_EXCHANGE, RAW_DATA_DIR, SYMBOLS


def get_exchange(exchange_id="binance"):
    """初始化交易所连接"""
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class(
        {"enableRateLimit": True, "options": {"defaultType": "future"}}
    )
    return exchange


def fetch_funding_rate_history(exchange, symbol, days=180):
    """获取历史资金费率"""
    print(f"Fetching historical funding rate for {symbol}...")
    try:
        # 尝试使用 ccxt 统一接口
        since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
        try:
            funding_history = exchange.fetch_funding_rate_history(
                symbol, since=since, limit=100
            )
            if funding_history:
                df = pd.DataFrame(funding_history)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df["fundingRate"] = df["fundingRate"].astype(float)
                df = df.set_index("timestamp")
                df = df.sort_index()
                return df[["fundingRate"]]
        except:
            pass

        # 交易所特定接口
        if exchange.id == "binance":
            market_symbol = symbol.replace("/", "")
            res = exchange.fapiPublicGetFundingRate(
                {"symbol": market_symbol, "limit": 1000}
            )
            df = pd.DataFrame(res)
            df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
            df["fundingRate"] = df["fundingRate"].astype(float)
            df = df.set_index("timestamp")
            return df[["fundingRate"]]

        elif exchange.id == "okx":
            instId = symbol.replace("/", "-").upper()
            if "USDT" in instId and "SWAP" not in instId:
                instId += "-SWAP"
            res = exchange.publicGetPublicFundingRateHistory({"instId": instId})
            df = pd.DataFrame(res["data"])
            df["timestamp"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms")
            df["fundingRate"] = df["fundingRate"].astype(float)
            df = df.set_index("timestamp")
            return df[["fundingRate"]]

    except Exception as e:
        print(f"  Error fetching funding rate: {e}")
    return pd.DataFrame()


def fetch_open_interest_history(exchange, symbol, timeframe="1h", days=30):
    """获取历史持仓量"""
    print(f"Fetching historical open interest for {symbol}...")
    try:
        if exchange.id == "binance":
            market_symbol = symbol.replace("/", "")
            res = exchange.fapiDataGetOpenInterestHist(
                {"symbol": market_symbol, "period": timeframe, "limit": 500}
            )
            df = pd.DataFrame(res)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
            df = df.set_index("timestamp")
            return df[["sumOpenInterest"]]

        elif exchange.id == "okx":
            instId = symbol.replace("/", "-").upper()
            if "USDT" in instId and "SWAP" not in instId:
                instId += "-SWAP"
            res = exchange.publicGetPublicOpenInterest({"instId": instId})
            df = pd.DataFrame(res["data"])
            df["timestamp"] = pd.to_datetime(df["ts"].astype(int), unit="ms")
            df["sumOpenInterest"] = df["oi"].astype(float)
            df = df.set_index("timestamp")
            return df[["sumOpenInterest"]]

    except Exception as e:
        print(f"  Error fetching OI: {e}")
    return pd.DataFrame()


def fetch_long_short_ratio(exchange, symbol, timeframe="1h"):
    """获取多空比 (主要支持 Binance)"""
    if exchange.id != "binance":
        print(
            f"  Long/Short ratio history is primarily supported for Binance in this script."
        )
        return pd.DataFrame()

    print(f"Fetching long/short ratio for {symbol}...")
    try:
        market_symbol = symbol.replace("/", "")
        res = exchange.fapiDataGetTopLongShortAccountRatio(
            {"symbol": market_symbol, "period": timeframe, "limit": 500}
        )
        df = pd.DataFrame(res)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["longShortRatio"] = df["longShortRatio"].astype(float)
        df = df.set_index("timestamp")
        return df[["longShortRatio"]]
    except Exception as e:
        print(f"  Error fetching L/S ratio: {e}")
    return pd.DataFrame()


def save_data(df, filename):
    """保存数据到CSV"""
    if df.empty:
        return
    filepath = os.path.join(RAW_DATA_DIR, filename)
    df.to_csv(filepath)
    print(f"Saved to {filepath}")


def main():
    exchange = get_exchange(DEFAULT_EXCHANGE)
    print(f"Using exchange: {exchange.id}")

    for symbol in SYMBOLS:
        safe_symbol = symbol.replace("/", "_")

        # 1. 资金费率
        funding = fetch_funding_rate_history(exchange, symbol, DEFAULT_DAYS)
        save_data(funding, f"{safe_symbol}_funding.csv")

        # 2. 历史持仓量
        oi = fetch_open_interest_history(exchange, symbol)
        save_data(oi, f"{safe_symbol}_oi.csv")

        # 3. 多空比
        ls = fetch_long_short_ratio(exchange, symbol)
        save_data(ls, f"{safe_symbol}_long_short.csv")

        print(f"\n{'='*50}\n")

    print("Contract data collection complete!")


if __name__ == "__main__":
    main()
