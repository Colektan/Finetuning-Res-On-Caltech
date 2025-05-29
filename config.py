from glob import glob
import os
import json

# Config
EXPERI_NAME = "From-Scratch"

BASE_LR = 1e-5  # 基础层学习率
FC_LR = 1e-4     # 输出层学习率
WEIGHT_DECAY = 0.01
OPTIMIZER = "AdamW" # ["SGD", "AdamW"]
MOMENTUM = 0.9 # 动量法
GAMMA_EXP_SCHEDULER = 0.99
BATCH_SIZE = 32
FROM_SCRATCH = False
EPOCHES = 400
BALANCED_SAMPLER = True

# 搜索logs文件夹，是否已存在同名文件夹
if os.path.exists(os.path.join("logs", EXPERI_NAME)):
    num = len(glob(os.path.join("logs", EXPERI_NAME + "*"))) + 1
    EXPERI_NAME = EXPERI_NAME + "_" + f"{num}"

SAVE_DIR = os.path.join("logs", EXPERI_NAME)
os.makedirs(SAVE_DIR)

config = {
    'BASE_LR': BASE_LR,
    'FC_LR': FC_LR,
    'WEIGHT_DECAY': WEIGHT_DECAY,
    'OPTIMIZER': OPTIMIZER,
    'MOMENTUM': MOMENTUM,
    'GAMMA_EXP_SCHEDULER': GAMMA_EXP_SCHEDULER,
    'BATCH_SIZE': BATCH_SIZE,
    'FROM_SCRATCH': FROM_SCRATCH,
    'EXPERI_NAME': EXPERI_NAME,
    'EPOCHES': EPOCHES,
    'BALANCED_SAMPLER': BALANCED_SAMPLER
}

with open(os.path.join(SAVE_DIR, "configs.json"), "w") as f:
    json.dump(config, f, indent=4)
