# Dev Container 配置说明

## 快速开始

1. 确保已安装 Docker Desktop
2. 在 VS Code 中安装 "Dev Containers" 扩展
3. 按 `F1` 或 `Ctrl+Shift+P`，选择 "Dev Containers: Reopen in Container"
4. 等待容器构建完成

## 已包含的工具

### Python 环境
- Python 3.11
- pip 包管理器

### VS Code 扩展
- Python (官方)
- Pylance (类型检查和智能提示)
- Jupyter (笔记本支持)
- Black Formatter (代码格式化)
- Ruff (快速 linter)

### Python 包
所有依赖已在 `requirements.txt` 中定义，容器创建时自动安装：
- ccxt: 加密货币交易所 API
- pandas: 数据处理
- numpy: 数值计算
- scipy: 科学计算
- statsmodels: 统计分析
- scikit-learn: 机器学习
- matplotlib/seaborn: 数据可视化
- jupyter: 交互式笔记本

## 端口转发

- **8888**: Jupyter Notebook 服务器

启动 Jupyter：
```bash
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## 数据持久化

`data/` 目录已配置为绑定挂载，数据会保存在宿主机上，容器重建后不会丢失。

## 环境变量

如需配置 API 密钥，在项目根目录创建 `.env` 文件：
```
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
```

## 常用命令

在容器内运行：
```bash
# 安装额外依赖
pip install package_name

# 运行数据采集
python -m data.fetch_market

# 启动 Jupyter
jupyter notebook --ip=0.0.0.0 --no-browser

# 运行完整流程
python run.py all
```
