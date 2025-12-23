"""
相关性分析模块
发现特征与目标变量之间的统计关系
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PROCESSED_DATA_DIR, MAX_LAG, SIGNIFICANCE_LEVEL, SYMBOLS


def load_features(symbol):
    """加载特征矩阵"""
    safe_symbol = symbol.replace('/', '_')
    filepath = os.path.join(PROCESSED_DATA_DIR, f'{safe_symbol}_features.csv')
    
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
    return df


def get_feature_columns(df):
    """获取特征列（排除目标列和原始OHLCV）"""
    exclude_prefixes = ['target_', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = []
    
    for col in df.columns:
        if not any(col.startswith(prefix) for prefix in exclude_prefixes):
            feature_cols.append(col)
    
    # 也包含 volume 的衍生特征
    for col in df.columns:
        if col.startswith('volume_') and col != 'volume':
            if col not in feature_cols:
                feature_cols.append(col)
    
    return feature_cols


def get_target_columns(df):
    """获取目标列"""
    return [col for col in df.columns if col.startswith('target_return_')]


def calculate_lagged_correlation(feature_series, target_series, max_lag=48):
    """
    计算滞后相关系数
    
    Parameters:
    -----------
    feature_series : pd.Series, 特征
    target_series : pd.Series, 目标
    max_lag : int, 最大滞后期数
    
    Returns:
    --------
    dict: lag -> (correlation, p_value)
    """
    results = {}
    
    for lag in range(0, max_lag + 1):
        if lag == 0:
            x = feature_series
            y = target_series
        else:
            # 特征滞后于目标（即特征在前，看能否预测未来目标）
            x = feature_series.shift(lag)
            y = target_series
        
        # 对齐并去除NaN
        valid = pd.concat([x, y], axis=1).dropna()
        if len(valid) < 30:
            continue
        
        corr, pval = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
        results[lag] = (corr, pval)
    
    return results


def find_best_lag(lagged_corr_results):
    """找到相关性最强的滞后期"""
    if not lagged_corr_results:
        return None, 0, 1
    
    best_lag = max(lagged_corr_results.keys(), 
                   key=lambda k: abs(lagged_corr_results[k][0]))
    best_corr, best_pval = lagged_corr_results[best_lag]
    
    return best_lag, best_corr, best_pval


def granger_causality_test(feature_series, target_series, max_lag=12):
    """
    Granger因果检验
    检验特征是否"Granger因果"于目标
    
    注意：这不是真正的因果关系，只是预测能力的统计检验
    """
    try:
        data = pd.concat([target_series, feature_series], axis=1).dropna()
        if len(data) < max_lag * 3:
            return None
        
        # grangercausalitytests 要求 [y, x] 顺序
        # 检验 x 是否 Granger-cause y
        result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        
        # 提取每个滞后期的p值（使用F检验）
        p_values = {lag: result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)}
        
        # 找到最显著的滞后期
        best_lag = min(p_values.keys(), key=lambda k: p_values[k])
        
        return {
            'best_lag': best_lag,
            'p_value': p_values[best_lag],
            'all_p_values': p_values
        }
        
    except Exception as e:
        return None


def calculate_mutual_information(feature_series, target_series, n_bins=20):
    """
    计算互信息（捕捉非线性关系）
    """
    from sklearn.metrics import mutual_info_score
    
    # 离散化
    valid = pd.concat([feature_series, target_series], axis=1).dropna()
    if len(valid) < 100:
        return 0
    
    x = pd.cut(valid.iloc[:, 0], bins=n_bins, labels=False)
    y = pd.cut(valid.iloc[:, 1], bins=n_bins, labels=False)
    
    # 去除离散化后的NaN
    mask = ~(x.isna() | y.isna())
    x, y = x[mask], y[mask]
    
    if len(x) < 50:
        return 0
    
    mi = mutual_info_score(x, y)
    return mi


def analyze_all_features(df, target_col='target_return_24h'):
    """
    分析所有特征与目标的关系
    """
    feature_cols = get_feature_columns(df)
    target = df[target_col].dropna()
    
    results = []
    
    print(f"\nAnalyzing {len(feature_cols)} features against {target_col}...")
    print("-" * 70)
    
    for i, feat_col in enumerate(feature_cols):
        feature = df[feat_col]
        
        # 1. 滞后相关性
        lagged_corr = calculate_lagged_correlation(feature, target, MAX_LAG)
        best_lag, best_corr, best_pval = find_best_lag(lagged_corr)
        
        # 2. Granger因果
        granger = granger_causality_test(feature, target, max_lag=12)
        granger_pval = granger['p_value'] if granger else 1.0
        granger_lag = granger['best_lag'] if granger else 0
        
        # 3. 互信息
        mi = calculate_mutual_information(feature, target)
        
        results.append({
            'feature': feat_col,
            'best_lag': best_lag,
            'correlation': best_corr,
            'corr_pvalue': best_pval,
            'granger_lag': granger_lag,
            'granger_pvalue': granger_pval,
            'mutual_info': mi,
            'significant': best_pval < SIGNIFICANCE_LEVEL and granger_pval < SIGNIFICANCE_LEVEL
        })
        
        # 进度显示
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(feature_cols)} features...")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('mutual_info', ascending=False)
    
    return results_df


def print_top_features(results_df, n=20):
    """打印最相关的特征"""
    print("\n" + "=" * 70)
    print("TOP PREDICTIVE FEATURES (sorted by mutual information)")
    print("=" * 70)
    
    top = results_df.head(n)
    
    for _, row in top.iterrows():
        sig_marker = "✓" if row['significant'] else " "
        print(f"\n{sig_marker} {row['feature']}")
        print(f"    Correlation: {row['correlation']:.4f} (lag={row['best_lag']}h, p={row['corr_pvalue']:.4f})")
        print(f"    Granger: p={row['granger_pvalue']:.4f} (lag={row['granger_lag']}h)")
        print(f"    Mutual Info: {row['mutual_info']:.4f}")


def analyze_feature_stability(df, feature_col, target_col, window_size=720):
    """
    分析特征预测能力的稳定性
    将数据分成多个窗口，检查相关性是否稳定
    """
    correlations = []
    
    for start in range(0, len(df) - window_size, window_size // 2):
        end = start + window_size
        window_df = df.iloc[start:end]
        
        feature = window_df[feature_col]
        target = window_df[target_col]
        
        valid = pd.concat([feature, target], axis=1).dropna()
        if len(valid) < 100:
            continue
        
        corr, _ = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
        correlations.append({
            'start': df.index[start],
            'end': df.index[min(end, len(df) - 1)],
            'correlation': corr
        })
    
    if not correlations:
        return None
    
    corr_df = pd.DataFrame(correlations)
    
    return {
        'mean_corr': corr_df['correlation'].mean(),
        'std_corr': corr_df['correlation'].std(),
        'min_corr': corr_df['correlation'].min(),
        'max_corr': corr_df['correlation'].max(),
        'sign_consistency': (np.sign(corr_df['correlation']) == np.sign(corr_df['correlation'].iloc[0])).mean(),
        'details': corr_df
    }


def main():
    """主函数"""
    for symbol in SYMBOLS:
        print(f"\n{'='*70}")
        print(f"ANALYZING {symbol}")
        print("=" * 70)
        
        df = load_features(symbol)
        if df.empty:
            continue
        
        # 分析不同预测周期
        for target in ['target_return_4h', 'target_return_12h', 'target_return_24h']:
            if target not in df.columns:
                continue
            
            print(f"\n>>> Target: {target}")
            results = analyze_all_features(df, target)
            
            # 保存结果
            safe_symbol = symbol.replace('/', '_')
            results_path = os.path.join(PROCESSED_DATA_DIR, f'{safe_symbol}_correlation_{target}.csv')
            results.to_csv(results_path, index=False)
            print(f"\nResults saved to {results_path}")
            
            # 打印top特征
            print_top_features(results, n=15)
            
            # 分析top特征的稳定性
            print("\n" + "-" * 70)
            print("STABILITY ANALYSIS (top 5 features)")
            print("-" * 70)
            
            for feat in results.head(5)['feature']:
                stability = analyze_feature_stability(df, feat, target)
                if stability:
                    print(f"\n{feat}:")
                    print(f"    Mean corr: {stability['mean_corr']:.4f} ± {stability['std_corr']:.4f}")
                    print(f"    Range: [{stability['min_corr']:.4f}, {stability['max_corr']:.4f}]")
                    print(f"    Sign consistency: {stability['sign_consistency']:.1%}")


if __name__ == '__main__':
    main()
