# 币圈信号发现项目

## 目标
通过数据驱动的方式，发现与加密货币价格变动相关性高的信号，用于辅助交易决策。

## 项目阶段

### Phase 1: 数据采集 ✅ 已完成
- [x] 市场数据（价格、成交量、K线）
- [x] 合约数据（资金费率、多空比、持仓量）
- [x] 链上数据（活跃地址、哈希率）
- [x] 情绪数据（恐惧贪婪指数）

### Phase 2: 特征工程
- [ ] 原始特征清洗和对齐
- [ ] 衍生特征计算（变化率、滚动统计量）
- [ ] 标签构建（未来N期收益率）

### Phase 3: 相关性分析
- [ ] 滞后相关系数矩阵
- [ ] Granger因果检验
- [ ] 互信息分析（非线性关系）

### Phase 4: 信号验证
- [ ] Walk-forward回测框架
- [ ] 样本外表现评估
- [ ] 信号衰减分析

### Phase 5: 策略构建（可选）
- [ ] 多信号融合
- [ ] 仓位管理
- [ ] 风险控制

## 目录结构

```
crypto-signal-discovery/
├── README.md           # 本文件
├── config/             # 配置文件
│   └── settings.py     # API密钥、参数配置
├── data/               # 数据存储
│   ├── raw/            # 原始数据
│   └── processed/      # 处理后数据
├── features/           # 特征工程模块
├── analysis/           # 分析模块
├── models/             # 模型模块
└── notebooks/          # 探索性分析
```

## 使用方法

每个阶段都有独立的运行脚本，可以单独执行：

```bash
# Phase 1: 采集数据
python -m data.fetch_market      # 市场数据
python -m data.fetch_funding     # 合约数据
python -m data.fetch_onchain     # 链上数据

# Phase 2: 特征工程
python -m features.build_features

# Phase 3: 相关性分析
python -m analysis.correlation

# Phase 4: 信号验证
python -m analysis.backtest
```

## 依赖安装

```bash
pip install ccxt pandas numpy scipy statsmodels requests python-dotenv
```

## 注意事项

1. **过拟合风险**：币圈数据量少，务必严格做样本外验证
2. **数据质量**：不同来源数据可能有出入，需要清洗对齐
3. **API限制**：注意各平台的请求频率限制
4. **时区问题**：统一使用UTC时间
