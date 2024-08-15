import os
import logging
from sympy import pi


max_workers = 4

os.environ['ENVIRONMENT'] = 'production'
# 创建日志记录器
logger = logging.getLogger('my_logger')

# 创建控制台处理器
console_handler = logging.StreamHandler()

# 创建格式化器并将其添加到处理器
formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(formatter)

# 将处理器添加到日志记录器
logger.addHandler(console_handler)

# 根据环境设置日志级别
environment = os.getenv('ENVIRONMENT', 'development')

if environment == 'production':
    logger.setLevel(logging.CRITICAL)
else:
    logger.setLevel(logging.DEBUG)

# 创建 eval_logger 用于存储实验数据
eval_logger = logging.getLogger('eval_logger')

# 创建文件处理器用于将日志记录到文件
file_handler = logging.FileHandler('experiment.log')

# 创建格式化器并将其添加到文件处理器
file_formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(file_formatter)

# 将文件处理器添加到 eval_logger
eval_logger.addHandler(file_handler)

# 设置 eval_logger 日志级别为 DEBUG
eval_logger.setLevel(logging.DEBUG)

# 创建 train_logger 用于存储训练信息
train_logger = logging.getLogger('train_logger')

# 创建文件处理器用于将日志记录到文件
file_handler = logging.FileHandler('training.log')

# 创建格式化器并将其添加到文件处理器
file_formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(file_formatter)

# 将文件处理器添加到 train_logger
train_logger.addHandler(file_handler)

# 设置 train_logger 日志级别为 DEBUG
train_logger.setLevel(logging.DEBUG)

# 节点类型到整数的映射
NODE_TYPE_TO_INT = {
    "Point": 1,
    "Line": 2,
    "Angle": 3,
    "Triangle": 4,
    "Circle": 5,
    "Arc": 6,
    "Polygon": 7
}

# 边类型到整数的映射
EDGE_TYPE_TO_INT = {
    "Connected": 1,
    "Equal": 2,
    "Perpendicular": 3,
    "Parallel": 4,
    "Congruent": 5,
    "Similar": 6,
    "LiesOnLine": 7,
    "LiesOnCircle": 8,
    "Center": 9,
    "Interior": 10,
    "Endpoint": 11,
    "Sidepoint": 12,
    "Side": 13,
    "Vertex": 14,
    "AngleSide": 15
}

UPPER_BOUND = 10
TOLERANCE = {
    "line": 3,
    "angle": pi/45,
    "arc": pi/45,
    "circle": 5,
    "triangle": 5,
    "polygon": 5
}

# 定义基本文件名
diagram_logic_forms_json_file = "diagram_logic_forms_annot.json"
text_logic_forms_json_file = "text_logic_forms_annot_dissolved.json"
pred_diagram_logic_forms_json_path = "PGDP_logic_forms_pred.json"
pred_text_logic_forms_json_file = "text_logic_forms_pred.json"

# 构建完整路径
db_dir = os.path.join("db", "Geometry3K_logic_forms")
db_dir_single = os.path.join("db", "Geometry3K")

# 构建完整路径
diagram_logic_forms_json_path = os.path.join(db_dir, diagram_logic_forms_json_file)
text_logic_forms_json_path = os.path.join(db_dir, text_logic_forms_json_file)
pred_diagram_logic_forms_json_path = os.path.join(db_dir, pred_diagram_logic_forms_json_path)
pred_text_logic_forms_json_path = os.path.join(db_dir, pred_text_logic_forms_json_file)

# 构建 model_pool 的路径
model_pool_path = os.path.join("reasoner", "graph_models", "graph_models.json")
model_pool_test_path = os.path.join("reasoner", "graph_models", "graph_models_test.json")

error_ids_path = os.path.join("db", "error_ids.txt")