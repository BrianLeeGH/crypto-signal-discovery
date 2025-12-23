"""
链上数据和情绪数据采集模块
"""

import requests
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import RAW_DATA_DIR, FEAR_GREED_API, DEFAULT_DAYS


def fetch_fear_greed_index(days=180):
    """
    获取 Fear & Greed Index
    来源: Alternative.me (免费)
    
    数值含义:
    0-25: Extreme Fear
    25-50: Fear  
    50-75: Greed
    75-100: Extreme Greed
    """
    print("Fetching Fear & Greed Index...")
    
    try:
        url = f"{FEAR_GREED_API}?limit={days}&format=json"
        response = requests.get(url, timeout=30)
        data = response.json()
        
        if 'data' not in data:
            print(f"  Error: Unexpected response format")
            return pd.DataFrame()
        
        records = []
        for item in data['data']:
            records.append({
                'timestamp': datetime.fromtimestamp(int(item['timestamp'])),
                'fear_greed_value': int(item['value']),
                'fear_greed_class': item['value_classification']
            })
        
        df = pd.DataFrame(records)
        df = df.set_index('timestamp')
        df = df.sort_index()
        
        print(f"  Done. {len(df)} days of Fear & Greed data")
        return df
        
    except Exception as e:
        print(f"  Error: {e}")
        return pd.DataFrame()


def fetch_coinglass_funding(symbol='BTC'):
    """
    从 Coinglass 获取综合资金费率
    注意：免费版有请求限制
    """
    print(f"Fetching Coinglass funding rate for {symbol}...")
    
    try:
        url = f"https://open-api.coinglass.com/public/v2/funding"
        params = {'symbol': symbol}
        
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if data.get('code') != '0':
            print(f"  Error: {data.get('msg', 'Unknown error')}")
            return pd.DataFrame()
        
        # 这个接口返回的是各交易所当前费率，不是历史
        print(f"  Current funding rates across exchanges:")
        for item in data.get('data', []):
            print(f"    {item.get('exchangeName')}: {item.get('rate')}")
        
        return data.get('data', [])
        
    except Exception as e:
        print(f"  Error: {e}")
        return []


def fetch_coinglass_long_short_ratio(symbol='BTC'):
    """
    从 Coinglass 获取多空比
    """
    print(f"Fetching long/short ratio for {symbol}...")
    
    try:
        url = f"https://open-api.coinglass.com/public/v2/long_short"
        params = {'symbol': symbol, 'interval': '1h'}
        
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if data.get('code') != '0':
            print(f"  Error: {data.get('msg', 'Unknown error')}")
            return pd.DataFrame()
        
        return data.get('data', [])
        
    except Exception as e:
        print(f"  Error: {e}")
        return []


def fetch_exchange_netflow_mock():
    """
    交易所净流入流出数据
    
    注意：真实的链上数据需要:
    - Glassnode (付费)
    - CryptoQuant (部分免费)
    - Nansen (付费)
    
    这里提供一个数据结构示例，实际使用需要接入真实API
    """
    print("Note: Exchange netflow requires premium API (Glassnode/CryptoQuant)")
    print("This is a placeholder showing expected data structure")
    
    # 预期的数据结构
    example_structure = {
        'timestamp': 'datetime',
        'exchange_inflow_btc': 'float - BTC流入交易所数量',
        'exchange_outflow_btc': 'float - BTC流出交易所数量', 
        'exchange_netflow_btc': 'float - 净流入(正=卖压)',
        'exchange_reserve_btc': 'float - 交易所BTC储备'
    }
    
    print(f"Expected columns: {list(example_structure.keys())}")
    return pd.DataFrame()


def fetch_whale_alerts():
    """
    大额转账监控
    
    可用来源:
    - Whale Alert API (需注册)
    - 直接监控链上数据
    """
    print("Note: Whale alerts require Whale Alert API key")
    print("Register at: https://whale-alert.io/")
    return pd.DataFrame()


def save_data(df, filename):
    """保存数据到CSV"""
    if df.empty:
        print(f"Skipping {filename} - no data")
        return
        
    filepath = os.path.join(RAW_DATA_DIR, filename)
    df.to_csv(filepath)
    print(f"Saved to {filepath}")


def main():
    """主函数"""
    
    # Fear & Greed Index - 免费可用
    fng = fetch_fear_greed_index(DEFAULT_DAYS)
    save_data(fng, 'fear_greed_index.csv')
    
    print("\n" + "="*50 + "\n")
    
    # Coinglass 数据 - 可能有限制
    fetch_coinglass_funding('BTC')
    fetch_coinglass_long_short_ratio('BTC')
    
    print("\n" + "="*50 + "\n")
    
    # 提示需要付费API的数据
    fetch_exchange_netflow_mock()
    fetch_whale_alerts()
    
    print("\n" + "="*50)
    print("On-chain data collection complete!")
    print("\nTo get more comprehensive on-chain data, consider:")
    print("  - Glassnode: https://glassnode.com/")
    print("  - CryptoQuant: https://cryptoquant.com/")
    print("  - Nansen: https://nansen.ai/")


if __name__ == '__main__':
    main()
