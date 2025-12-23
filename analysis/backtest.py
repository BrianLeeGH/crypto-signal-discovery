"""
信号回测验证模块
使用 Walk-Forward 方法验证信号的样本外表现
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    PROCESSED_DATA_DIR, SYMBOLS,
    TRAIN_WINDOW, VALID_WINDOW, ROLL_STEP
)


def load_features(symbol):
    """加载特征矩阵"""
    safe_symbol = symbol.replace('/', '_')
    filepath = os.path.join(PROCESSED_DATA_DIR, f'{safe_symbol}_features.csv')
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
    return df


def load_correlation_results(symbol, target):
    """加载相关性分析结果"""
    safe_symbol = symbol.replace('/', '_')
    filepath = os.path.join(PROCESSED_DATA_DIR, f'{safe_symbol}_correlation_{target}.csv')
    
    if not os.path.exists(filepath):
        return pd.DataFrame()
    
    return pd.read_csv(filepath)


def select_top_features(corr_results, n=10, min_mi=0.01):
    """
    选择top特征用于建模
    
    条件：
    1. 互信息 > min_mi
    2. 统计显著
    """
    filtered = corr_results[
        (corr_results['mutual_info'] > min_mi) &
        (corr_results['significant'] == True)
    ]
    
    if len(filtered) < n:
        # 如果显著特征不够，放宽条件
        filtered = corr_results[corr_results['mutual_info'] > min_mi]
    
    return filtered.head(n)['feature'].tolist()


def walk_forward_validation(df, feature_cols, target_col, 
                            train_days=60, valid_days=14, step_days=7,
                            model_type='classification'):
    """
    Walk-Forward 验证
    
    Parameters:
    -----------
    df : pd.DataFrame
    feature_cols : list, 特征列名
    target_col : str, 目标列名
    train_days : int, 训练窗口（天）
    valid_days : int, 验证窗口（天）
    step_days : int, 滚动步长（天）
    model_type : str, 'classification' 或 'regression'
    """
    # 转换为小时（假设1小时K线）
    train_size = train_days * 24
    valid_size = valid_days * 24
    step_size = step_days * 24
    
    # 准备数据
    X = df[feature_cols].copy()
    
    if model_type == 'classification':
        y = (df[target_col] > 0).astype(int)
    else:
        y = df[target_col]
    
    # 去除NaN
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    results = []
    
    # 滚动窗口
    start = 0
    while start + train_size + valid_size <= len(X):
        train_end = start + train_size
        valid_end = train_end + valid_size
        
        X_train = X.iloc[start:train_end]
        y_train = y.iloc[start:train_end]
        X_valid = X.iloc[train_end:valid_end]
        y_valid = y.iloc[train_end:valid_end]
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        
        # 训练模型
        if model_type == 'classification':
            model = LogisticRegression(C=0.1, max_iter=1000)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_valid_scaled)
            y_prob = model.predict_proba(X_valid_scaled)[:, 1]
            
            accuracy = accuracy_score(y_valid, y_pred)
            try:
                auc = roc_auc_score(y_valid, y_prob)
            except:
                auc = 0.5
            
            # 多头信号胜率（预测涨且实际涨）
            long_signals = y_prob > 0.5
            if long_signals.sum() > 0:
                long_accuracy = (y_valid[long_signals] == 1).mean()
            else:
                long_accuracy = 0.5
            
            results.append({
                'period_start': X_valid.index[0],
                'period_end': X_valid.index[-1],
                'accuracy': accuracy,
                'auc': auc,
                'long_accuracy': long_accuracy,
                'n_samples': len(y_valid)
            })
            
        else:
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_valid_scaled)
            
            mse = mean_squared_error(y_valid, y_pred)
            
            # 方向准确率
            direction_acc = ((y_pred > 0) == (y_valid > 0)).mean()
            
            # IC (Information Coefficient)
            ic = np.corrcoef(y_pred, y_valid)[0, 1] if len(y_valid) > 2 else 0
            
            results.append({
                'period_start': X_valid.index[0],
                'period_end': X_valid.index[-1],
                'mse': mse,
                'direction_accuracy': direction_acc,
                'ic': ic,
                'n_samples': len(y_valid)
            })
        
        start += step_size
    
    return pd.DataFrame(results)


def calculate_simple_strategy_returns(df, feature_col, target_return_col, 
                                       threshold_percentile=70):
    """
    基于单一信号的简单策略回测
    
    策略逻辑：
    - 当信号值高于阈值时做多
    - 当信号值低于阈值时空仓
    """
    feature = df[feature_col]
    returns = df[target_return_col]
    
    # 使用滚动分位数作为阈值
    rolling_threshold = feature.rolling(168).quantile(threshold_percentile / 100)
    
    # 生成信号（1=多头，0=空仓）
    signal = (feature > rolling_threshold).astype(int)
    
    # 信号需要滞后一期（避免前视偏差）
    signal = signal.shift(1)
    
    # 策略收益
    strategy_returns = signal * returns
    
    # 计算指标
    valid_mask = ~(strategy_returns.isna() | returns.isna())
    strategy_returns = strategy_returns[valid_mask]
    benchmark_returns = returns[valid_mask]
    
    if len(strategy_returns) < 100:
        return None
    
    # 累计收益
    strategy_cumret = (1 + strategy_returns).cumprod()
    benchmark_cumret = (1 + benchmark_returns).cumprod()
    
    # 年化收益（假设1小时K线，一年约8760小时）
    n_hours = len(strategy_returns)
    strategy_annual = strategy_cumret.iloc[-1] ** (8760 / n_hours) - 1
    benchmark_annual = benchmark_cumret.iloc[-1] ** (8760 / n_hours) - 1
    
    # 夏普比率（简化版，假设无风险利率为0）
    strategy_sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-10) * np.sqrt(8760)
    
    # 最大回撤
    peak = strategy_cumret.cummax()
    drawdown = (strategy_cumret - peak) / peak
    max_drawdown = drawdown.min()
    
    # 胜率
    win_rate = (strategy_returns > 0).sum() / (strategy_returns != 0).sum()
    
    # 交易次数
    signal_changes = signal.diff().abs().sum() / 2
    
    return {
        'feature': feature_col,
        'total_return': strategy_cumret.iloc[-1] - 1,
        'annual_return': strategy_annual,
        'benchmark_annual': benchmark_annual,
        'sharpe': strategy_sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'n_trades': signal_changes,
        'n_hours': n_hours
    }


def backtest_top_features(df, corr_results, target_return_col, n_features=10):
    """
    回测top特征的策略表现
    """
    results = []
    
    top_features = select_top_features(corr_results, n=n_features)
    
    for feat in top_features:
        if feat not in df.columns:
            continue
        
        perf = calculate_simple_strategy_returns(df, feat, target_return_col)
        if perf:
            results.append(perf)
    
    return pd.DataFrame(results)


def print_backtest_results(results_df):
    """打印回测结果"""
    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 70)
    
    if 'accuracy' in results_df.columns:
        # Classification results
        print(f"\nMean Accuracy: {results_df['accuracy'].mean():.2%} ± {results_df['accuracy'].std():.2%}")
        print(f"Mean AUC: {results_df['auc'].mean():.3f} ± {results_df['auc'].std():.3f}")
        print(f"Mean Long Accuracy: {results_df['long_accuracy'].mean():.2%}")
        print(f"\nPeriod-by-period results:")
        for _, row in results_df.iterrows():
            print(f"  {row['period_start'].strftime('%Y-%m-%d')} to {row['period_end'].strftime('%Y-%m-%d')}: "
                  f"Acc={row['accuracy']:.2%}, AUC={row['auc']:.3f}, Long={row['long_accuracy']:.2%}")
    else:
        # Regression results
        print(f"\nMean Direction Accuracy: {results_df['direction_accuracy'].mean():.2%}")
        print(f"Mean IC: {results_df['ic'].mean():.4f}")


def print_strategy_results(strategy_df):
    """打印策略回测结果"""
    print("\n" + "=" * 70)
    print("SINGLE-FEATURE STRATEGY BACKTEST")
    print("=" * 70)
    
    strategy_df = strategy_df.sort_values('sharpe', ascending=False)
    
    for _, row in strategy_df.iterrows():
        print(f"\n{row['feature']}:")
        print(f"    Annual Return: {row['annual_return']:.1%} (Benchmark: {row['benchmark_annual']:.1%})")
        print(f"    Sharpe Ratio: {row['sharpe']:.2f}")
        print(f"    Max Drawdown: {row['max_drawdown']:.1%}")
        print(f"    Win Rate: {row['win_rate']:.1%}")
        print(f"    Trades: {row['n_trades']:.0f}")


def main():
    """主函数"""
    for symbol in SYMBOLS:
        print(f"\n{'#'*70}")
        print(f"# BACKTESTING {symbol}")
        print("#" * 70)
        
        df = load_features(symbol)
        if df.empty:
            continue
        
        target = 'target_return_24h'
        target_direction = 'target_direction_24h'
        
        # 加载相关性分析结果
        corr_results = load_correlation_results(symbol, target)
        if corr_results.empty:
            print(f"No correlation results found. Run correlation analysis first.")
            continue
        
        # 选择top特征
        top_features = select_top_features(corr_results, n=10)
        print(f"\nSelected features: {top_features}")
        
        if not top_features:
            print("No significant features found.")
            continue
        
        # Walk-Forward 验证（分类）
        print("\n>>> Classification Model (Direction Prediction)")
        wf_results = walk_forward_validation(
            df, top_features, target_direction,
            train_days=TRAIN_WINDOW, valid_days=VALID_WINDOW, step_days=ROLL_STEP,
            model_type='classification'
        )
        
        if not wf_results.empty:
            print_backtest_results(wf_results)
            
            # 保存结果
            safe_symbol = symbol.replace('/', '_')
            wf_results.to_csv(
                os.path.join(PROCESSED_DATA_DIR, f'{safe_symbol}_walkforward_results.csv'),
                index=False
            )
        
        # 单特征策略回测
        print("\n>>> Single-Feature Strategy Backtest")
        strategy_results = backtest_top_features(df, corr_results, target, n_features=10)
        
        if not strategy_results.empty:
            print_strategy_results(strategy_results)
            strategy_results.to_csv(
                os.path.join(PROCESSED_DATA_DIR, f'{safe_symbol}_strategy_results.csv'),
                index=False
            )
    
    print("\n" + "=" * 70)
    print("Backtesting complete!")


if __name__ == '__main__':
    main()
