import os
import logging
from sympy import pi


# os.environ['ENVIRONMENT'] = 'production'
# 创建一个日志记录器
logger = logging.getLogger('my_logger')

# 创建一个控制台处理器
console_handler = logging.StreamHandler()

# 创建一个只包含消息内容的格式化器并将其添加到处理器
formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(formatter)

# 将处理器添加到日志记录器
logger.addHandler(console_handler)

# 根据环境设置日志级别
environment = os.getenv('ENVIRONMENT', 'development')

if environment == 'production':
    logger.setLevel(logging.ERROR)
else:
    logger.setLevel(logging.DEBUG)

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
    "angle": pi/60,
    "arc": pi/60,
    "circle": 5,
    "triangle": 5,
    "polygon": 5
}

# 定义基本文件名
diagram_logic_forms_json_file = "diagram_logic_forms_annot.json"
text_logic_forms_json_file = "text_logic_forms_annot_dissolved.json"

# 构建完整路径
db_dir = os.path.join("db", "Geometry3K_logic_forms")
db_dir_single = os.path.join("db", "Geometry3K")

# 构建完整路径
diagram_logic_forms_json_path = os.path.join(db_dir, diagram_logic_forms_json_file)
text_logic_forms_json_path = os.path.join(db_dir, text_logic_forms_json_file)

# 构建 model_pool 的路径
model_pool_path = os.path.join("reasoner", "graph_models", "graph_models.json")
model_pool_test_path = os.path.join("reasoner", "graph_models", "graph_models_test.json")

error_ids_path = os.path.join("db", "error_ids.txt")