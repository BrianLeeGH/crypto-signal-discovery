#!/usr/bin/env python3
"""
快速启动脚本
一键运行完整流程或单独运行某个阶段
"""

import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_command(cmd, description):
    """运行命令并显示状态"""
    print(f"\n{'='*60}")
    print(f">>> {description}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed!")
        return False
    return True


def install_dependencies():
    """安装依赖"""
    print("Installing dependencies...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "ccxt",
            "pandas",
            "numpy",
            "scipy",
            "statsmodels",
            "requests",
            "scikit-learn",
            "python-dotenv",
        ]
    )
    print("Dependencies installed.")


def phase1_data_collection():
    """Phase 1: 数据采集"""
    run_command(
        f"{sys.executable} -m data.fetch_market",
        "Phase 1.1: Fetching market data (OHLCV)",
    )
    run_command(
        f"{sys.executable} -m data.fetch_funding",
        "Phase 1.2: Fetching contract data (Funding, OI, L/S Ratio)",
    )
    run_command(
        f"{sys.executable} -m data.fetch_onchain",
        "Phase 1.3: Fetching on-chain and sentiment data",
    )


def phase2_feature_engineering():
    """Phase 2: 特征工程"""
    run_command(
        f"{sys.executable} -m features.build_features",
        "Phase 2: Building feature matrix",
    )


def phase3_correlation_analysis():
    """Phase 3: 相关性分析"""
    run_command(
        f"{sys.executable} -m analysis.correlation", "Phase 3: Correlation analysis"
    )


def phase4_backtesting():
    """Phase 4: 回测验证"""
    run_command(
        f"{sys.executable} -m analysis.backtest", "Phase 4: Walk-forward backtesting"
    )


def run_all():
    """运行完整流程"""
    install_dependencies()
    phase1_data_collection()
    phase2_feature_engineering()
    phase3_correlation_analysis()
    phase4_backtesting()

    print("\n" + "=" * 60)
    print("ALL PHASES COMPLETE!")
    print("=" * 60)
    print("\nCheck the data/processed/ directory for results:")
    print("  - *_features.csv: Feature matrix")
    print("  - *_correlation_*.csv: Correlation analysis")
    print("  - *_walkforward_results.csv: Model validation")
    print("  - *_strategy_results.csv: Strategy backtest")


def print_help():
    """打印帮助信息"""
    help_text = """
Crypto Signal Discovery - Quick Start Script

Usage:
    python run.py [command]

Commands:
    all         Run complete pipeline (all phases)
    install     Install Python dependencies
    phase1      Data collection only
    phase2      Feature engineering only
    phase3      Correlation analysis only
    phase4      Backtesting only
    help        Show this help message

Examples:
    python run.py all       # Run everything
    python run.py phase1    # Just collect data
    python run.py phase3    # Re-run analysis with existing data

Notes:
    - Phase 1 requires internet connection
    - Phase 2-4 require Phase 1 data to exist
    - Results are saved to data/processed/
    """
    print(help_text)


def main():
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1].lower()

    commands = {
        "all": run_all,
        "install": install_dependencies,
        "phase1": phase1_data_collection,
        "phase2": phase2_feature_engineering,
        "phase3": phase3_correlation_analysis,
        "phase4": phase4_backtesting,
        "help": print_help,
    }

    if command in commands:
        commands[command]()
    else:
        print(f"Unknown command: {command}")
        print_help()


if __name__ == "__main__":
    main()
