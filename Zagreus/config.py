# Kephale/config.py
import os

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 子路径
ENVS_DIR = os.path.join(ROOT_DIR, 'envs')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')

# 示例：日志和权重保存路径
LOG_DIR = os.path.join(ROOT_DIR, 'runs')
WEIGHTS_DIR = os.path.join(ROOT_DIR, 'weights')

print(f"ROOT_DIR: {ROOT_DIR}")