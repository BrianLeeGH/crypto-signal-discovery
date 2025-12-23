# 项目配置

import os
from datetime import datetime, timedelta

# ============ 基础配置 ============

# 数据存储路径
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# 确保目录存在
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# ============ 交易对配置 ============

# 主要关注的交易对
SYMBOLS = [
    'BTC/USDT',
    'ETH/USDT',
]

# 默认交易所
DEFAULT_EXCHANGE = 'binance'

# ============ 时间配置 ============

# 数据时间范围（默认最近180天）
DEFAULT_DAYS = 180
DEFAULT_START_DATE = datetime.utcnow() - timedelta(days=DEFAULT_DAYS)
DEFAULT_END_DATE = datetime.utcnow()

# K线周期
DEFAULT_TIMEFRAME = '1h'  # 1小时K线

# ============ API 配置 ============

# Binance（公开数据不需要API Key）
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_SECRET = os.getenv('BINANCE_SECRET', '')

# Alternative.me Fear & Greed Index（免费，无需Key）
FEAR_GREED_API = 'https://api.alternative.me/fng/'

# Coinglass（部分免费）
COINGLASS_API = 'https://open-api.coinglass.com/public/v2/'

# CryptoQuant（需要注册免费账号）
CRYPTOQUANT_API_KEY = os.getenv('CRYPTOQUANT_API_KEY', '')

# ============ 特征配置 ============

# 滚动窗口大小
ROLLING_WINDOWS = [6, 12, 24, 48, 72]  # 小时

# 预测目标：未来N期收益
TARGET_HORIZONS = [1, 4, 12, 24]  # 小时

# ============ 分析配置 ============

# 滞后相关性检验的最大滞后期数
MAX_LAG = 48  # 小时

# 显著性水平
SIGNIFICANCE_LEVEL = 0.05

# ============ 回测配置 ============

# 训练集大小（天）
TRAIN_WINDOW = 60

# 验证集大小（天）
VALID_WINDOW = 14

# 滚动步长（天）
ROLL_STEP = 7
