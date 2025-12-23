"""
特征工程模块
将原始数据转换为可用于分析的特征
"""

import pandas as pd
import numpy as np
import os
import sys
from glob import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    ROLLING_WINDOWS, TARGET_HORIZONS, SYMBOLS
)


def load_ohlcv(symbol):
    """加载K线数据"""
    safe_symbol = symbol.replace('/', '_')
    filepath = os.path.join(RAW_DATA_DIR, f'{safe_symbol}_ohlcv_1h.csv')
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
    return df


def load_funding(symbol):
    """加载资金费率数据"""
    safe_symbol = symbol.replace('/', '_')
    filepath = os.path.join(RAW_DATA_DIR, f'{safe_symbol}_funding.csv')
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
    return df


def load_fear_greed():
    """加载恐惧贪婪指数"""
    filepath = os.path.join(RAW_DATA_DIR, 'fear_greed_index.csv')
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
    return df


def calculate_price_features(df):
    """
    计算价格相关特征
    """
    features = pd.DataFrame(index=df.index)
    
    # 收益率
    features['returns_1h'] = df['close'].pct_change()
    features['returns_4h'] = df['close'].pct_change(4)
    features['returns_24h'] = df['close'].pct_change(24)
    
    # 对数收益率（更适合统计分析）
    features['log_returns_1h'] = np.log(df['close'] / df['close'].shift(1))
    
    # 波动率（滚动标准差）
    for window in ROLLING_WINDOWS:
        features[f'volatility_{window}h'] = features['log_returns_1h'].rolling(window).std()
    
    # 价格位置（相对于近期高低点）
    for window in ROLLING_WINDOWS:
        rolling_high = df['high'].rolling(window).max()
        rolling_low = df['low'].rolling(window).min()
        features[f'price_position_{window}h'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)
    
    # 动量
    for window in ROLLING_WINDOWS:
        features[f'momentum_{window}h'] = df['close'] / df['close'].shift(window) - 1
    
    return features


def calculate_volume_features(df):
    """
    计算成交量相关特征
    """
    features = pd.DataFrame(index=df.index)
    
    # 成交量变化率
    features['volume_change'] = df['volume'].pct_change()
    
    # 成交量相对均值
    for window in ROLLING_WINDOWS:
        vol_ma = df['volume'].rolling(window).mean()
        features[f'volume_ratio_{window}h'] = df['volume'] / (vol_ma + 1e-10)
    
    # 量价关系
    features['volume_price_corr_24h'] = (
        df['close'].pct_change()
        .rolling(24)
        .corr(df['volume'].pct_change())
    )
    
    # OBV (On-Balance Volume) 简化版
    price_direction = np.sign(df['close'].diff())
    features['obv_slope_24h'] = (price_direction * df['volume']).rolling(24).sum()
    
    return features


def calculate_technical_indicators(df):
    """
    计算常用技术指标
    """
    features = pd.DataFrame(index=df.index)
    
    # RSI
    for window in [6, 14, 24]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-10)
        features[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    features['macd'] = ema12 - ema26
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    features['macd_hist'] = features['macd'] - features['macd_signal']
    
    # Bollinger Bands 位置
    for window in [20, 48]:
        ma = df['close'].rolling(window).mean()
        std = df['close'].rolling(window).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        features[f'bb_position_{window}'] = (df['close'] - lower) / (upper - lower + 1e-10)
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features['atr_14'] = true_range.rolling(14).mean()
    features['atr_ratio'] = true_range / (features['atr_14'] + 1e-10)
    
    return features


def calculate_funding_features(funding_df, target_index):
    """
    计算资金费率相关特征
    """
    if funding_df.empty:
        return pd.DataFrame(index=target_index)
    
    # 重采样到小时级别（资金费率通常8小时一次）
    funding_hourly = funding_df['fundingRate'].resample('1H').last().ffill()
    funding_hourly = funding_hourly.reindex(target_index, method='ffill')
    
    features = pd.DataFrame(index=target_index)
    features['funding_rate'] = funding_hourly
    
    # 累计资金费率
    for window in [24, 72]:
        features[f'funding_cumsum_{window}h'] = funding_hourly.rolling(window).sum()
    
    # 资金费率变化
    features['funding_change'] = funding_hourly.diff()
    
    # 资金费率极端值标记
    features['funding_extreme'] = (abs(funding_hourly) > 0.001).astype(int)
    
    return features


def calculate_fear_greed_features(fng_df, target_index):
    """
    计算恐惧贪婪指数特征
    """
    if fng_df.empty:
        return pd.DataFrame(index=target_index)
    
    # 恐惧贪婪是日度数据，需要填充到小时
    fng_hourly = fng_df['fear_greed_value'].resample('1H').last().ffill()
    fng_hourly = fng_hourly.reindex(target_index, method='ffill')
    
    features = pd.DataFrame(index=target_index)
    features['fear_greed'] = fng_hourly
    
    # 变化率
    features['fear_greed_change_1d'] = fng_hourly.diff(24)
    features['fear_greed_change_7d'] = fng_hourly.diff(168)
    
    # 极端值标记
    features['extreme_fear'] = (fng_hourly < 25).astype(int)
    features['extreme_greed'] = (fng_hourly > 75).astype(int)
    
    return features


def calculate_targets(df):
    """
    计算预测目标（未来收益率）
    """
    targets = pd.DataFrame(index=df.index)
    
    for horizon in TARGET_HORIZONS:
        # 未来收益率
        targets[f'target_return_{horizon}h'] = df['close'].shift(-horizon) / df['close'] - 1
        
        # 二分类：涨跌
        targets[f'target_direction_{horizon}h'] = (targets[f'target_return_{horizon}h'] > 0).astype(int)
    
    return targets


def build_feature_matrix(symbol):
    """
    构建完整的特征矩阵
    """
    print(f"\nBuilding features for {symbol}...")
    
    # 加载数据
    ohlcv = load_ohlcv(symbol)
    if ohlcv.empty:
        print(f"  No OHLCV data for {symbol}")
        return pd.DataFrame()
    
    funding = load_funding(symbol)
    fng = load_fear_greed()
    
    # 计算各类特征
    print("  Calculating price features...")
    price_features = calculate_price_features(ohlcv)
    
    print("  Calculating volume features...")
    volume_features = calculate_volume_features(ohlcv)
    
    print("  Calculating technical indicators...")
    tech_features = calculate_technical_indicators(ohlcv)
    
    print("  Calculating funding features...")
    funding_features = calculate_funding_features(funding, ohlcv.index)
    
    print("  Calculating fear/greed features...")
    fng_features = calculate_fear_greed_features(fng, ohlcv.index)
    
    print("  Calculating targets...")
    targets = calculate_targets(ohlcv)
    
    # 合并所有特征
    feature_matrix = pd.concat([
        ohlcv[['open', 'high', 'low', 'close', 'volume']],
        price_features,
        volume_features,
        tech_features,
        funding_features,
        fng_features,
        targets
    ], axis=1)
    
    # 删除前面的NaN（由于rolling计算）
    max_window = max(ROLLING_WINDOWS)
    feature_matrix = feature_matrix.iloc[max_window:]
    
    print(f"  Feature matrix shape: {feature_matrix.shape}")
    print(f"  Date range: {feature_matrix.index[0]} to {feature_matrix.index[-1]}")
    
    return feature_matrix


def main():
    """主函数"""
    for symbol in SYMBOLS:
        feature_matrix = build_feature_matrix(symbol)
        
        if not feature_matrix.empty:
            safe_symbol = symbol.replace('/', '_')
            filepath = os.path.join(PROCESSED_DATA_DIR, f'{safe_symbol}_features.csv')
            feature_matrix.to_csv(filepath)
            print(f"  Saved to {filepath}")
            
            # 打印特征列表
            print(f"\n  Features ({len(feature_matrix.columns)} total):")
            for col in feature_matrix.columns:
                print(f"    - {col}")
    
    print("\n" + "="*50)
    print("Feature engineering complete!")


if __name__ == '__main__':
    main()
