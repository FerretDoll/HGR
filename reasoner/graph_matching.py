import ctypes
import re
import sys

import sympy
from sympy import sympify, simplify, And, Or, Eq, Symbol, Le, Ge, Lt, Gt

from reasoner.logic_graph import ModelGraph
from reasoner.utils import filter_duplicates, group_by_id_sets
from reasoner.config import TOLERANCE
from utils.common_utils import calc_cross_angle

# 定义回调函数类型
CALLBACK_FUNC_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_char_p)


def load_models_from_json(json_data):
    model_pool = []  # 用于存储所有实例化的模型图

    # 遍历JSON数据中的每个模型键和对应的值
    for model_id, model_content in json_data.items():
        # 使用ModelGraph的from_json类方法实例化每个模型图
        model_graph = ModelGraph.from_json(model_content, model_id)
        # 将实例化的模型图添加到模型池列表中
        model_pool.append(model_graph)

    return model_pool


def get_candidate_models_from_pool(model_pool, global_graph):
    """
    从模型池中获取候选模型，确保模型的节点和边的类型数量不超过全局图中的对应数量。
    """
    candidate_models = []

    # 遍历模型池中的每个模型图
    for model in model_pool:
        # 检查模型的每种节点类型的数量是否不超过全局图中的相应数量
        node_type_check = all(
            model.node_types.get(nt, 0) <= global_graph.node_types.get(nt, 0) for nt in model.node_types)

        # 检查模型的每种边类型的数量是否不超过全局图中的相应数量
        edge_type_check = all(
            model.edge_types.get(et, 0) <= global_graph.edge_types.get(et, 0) for et in model.edge_types)

        # 如果节点和边的类型数量都不超过全局图，添加到候选模型中
        if node_type_check and edge_type_check:
            candidate_models.append(model)

    return candidate_models


def run_vf3(pattern_data, target_data, options=b'-f vfe -u -s'):
    solutions = []

    # 定义回调函数
    @CALLBACK_FUNC_TYPE
    def result_callback(all_solutions_c_str):
        nonlocal solutions
        all_solutions_str = all_solutions_c_str.decode('utf-8')
        if all_solutions_str:
            solutions = all_solutions_str.split('|')  # 根据分隔符'|'分割字符串

    # 系统判断，以决定加载的共享库是.so还是.dll
    if sys.platform.startswith('win'):
        lib_path = 'vf3/bin/vf3.dll'
        kernel32 = ctypes.WinDLL('kernel32.dll')
        handle = kernel32.LoadLibraryW(lib_path)
        libvf3 = ctypes.CDLL(lib_path)
    else:
        libvf3 = ctypes.CDLL('vf3/bin/vf3.so')

    # 设置run_vf3函数的参数类型
    libvf3.run_vf3.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, CALLBACK_FUNC_TYPE]

    # 调用函数
    libvf3.run_vf3(ctypes.c_char_p(pattern_data.encode('utf-8')), ctypes.c_char_p(target_data.encode('utf-8')), options,
                   result_callback)

    return solutions


def parse_mapping(model_graph, global_graph, mapping_str):
    """
    解析映射字符串，返回一个字典，其中键是子图中的节点索引，值是全局图中对应的节点索引。
    """
    mapping_dict = {}
    mappings = mapping_str.split(':')
    for pair in mappings:
        if pair:
            global_idx, sub_idx = pair.split(',')
            mapping_dict[model_graph.grf_to_id[int(sub_idx)]] = global_graph.grf_to_id[int(global_idx)]
    return mapping_dict


def approximately_equal(value1, value2, tolerance):
    """ 比较两个数值是否在指定的误差范围内相等。 """
    if tolerance:
        return abs(value1 - value2) <= tolerance
    else:
        return False


def eval_with_none_check(expression, substitutions):
    """对表达式进行求值，先检查是否有None值。"""
    if any(value == 'None' or value is None for value in substitutions.values()):
        return None  # 如果任何一个替换值是None，则返回None
    return sympify(expression).evalf(subs=substitutions)


def parse_expression(expr_str, _placeholders, key_value_pair=None, is_visual=False, point_positions=None):
    while '(' in expr_str:
        open_index = expr_str.rfind('(')
        close_index = expr_str.find(')', open_index)
        inner_expr = expr_str[open_index + 1:close_index]
        parsed_inner_expr = parse_expression(inner_expr, _placeholders, key_value_pair, is_visual, point_positions)
        _placeholder = f"__placeholder_{len(_placeholders)}__"
        _placeholders[_placeholder] = parsed_inner_expr
        expr_str = expr_str[:open_index] + _placeholder + expr_str[close_index + 1:]

    if 'AND' in expr_str or 'OR' in expr_str:
        if 'OR' in expr_str:
            parts = expr_str.split('OR')
            operator = Or
        else:
            parts = expr_str.split('AND')
            operator = And
        return operator(*[parse_expression(part.strip(), _placeholders, key_value_pair, is_visual, point_positions)
                          for part in parts])

    # 检查是否包含比较操作符
    for operator, sympy_op in (('<=', Le), ('>=', Ge), ('<', Lt), ('>', Gt)):
        if operator in expr_str:
            lhs, rhs = map(str.strip, expr_str.split(operator))
            return sympy_op(sympify(lhs), sympify(rhs))

    if '=' in expr_str:
        lhs, rhs = map(str.strip, expr_str.split('='))

        if key_value_pair is not None and is_visual:
            lhs_eval = eval_with_none_check(lhs, key_value_pair)
            rhs_eval = eval_with_none_check(rhs, key_value_pair)
            tolerance = None
            for key, value in TOLERANCE.items():
                if key in lhs or key in rhs:
                    tolerance = value
                    break

            if lhs_eval is None or rhs_eval is None:
                return False  # 处理None的情况，返回False或其他适当的值

            return approximately_equal(lhs_eval, rhs_eval, tolerance)
        else:
            return Eq(sympify(lhs), sympify(rhs))

    # 检查是否包含平行符号 ||
    if '||' in expr_str and point_positions is not None:
        lhs, rhs = map(str.strip, expr_str.split('||'))
        _, line_1 = str(lhs).split('_', 1)
        _, line_2 = str(rhs).split('_', 1)
        cross_angle = calc_cross_angle(line_1, line_2, point_positions)
        if (approximately_equal(cross_angle, 180, TOLERANCE.get('angle')) or
                approximately_equal(cross_angle, 0, TOLERANCE.get('angle'))):
            return True
        return False

    return sympify(expr_str)


def replace_variables_str(equation_str, mapping_dict, global_symbols=None):
    if global_symbols is None:
        global_symbols = {}

    # 使用正则表达式匹配可能的变量名（包含字母、数字和下划线）
    pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'

    # 用于替换匹配到的变量的函数
    def replace_variable(match):
        var_name = match.group(0)
        if var_name in mapping_dict:
            return str(mapping_dict[var_name])  # 确保返回的是字符串形式的变量名
        elif var_name in global_symbols:
            return str(global_symbols[var_name])  # 从全局符号表中获取符号
        return var_name  # 如果没有映射，返回原变量名

    new_equation_str = re.sub(pattern, replace_variable, str(equation_str))

    return new_equation_str


def replace_variables_equation(equation, global_symbols):
    new_equation = equation.subs({
        sym: global_symbols.get(sym.name, sym) for sym in equation.atoms() if isinstance(sym, Symbol)
    })

    return new_equation


def evaluate_expression(constraints, global_symbols=None, init_solutions=None):
    placeholders = {}
    parsed_expr = parse_expression(constraints, placeholders, key_value_pair=None, is_visual=False)

    if isinstance(parsed_expr, bool):
        return parsed_expr

    # 循环替换占位符直到没有更多占位符为止
    while any(p in str(parsed_expr) for p in placeholders):
        for placeholder, expr in placeholders.items():
            parsed_expr = parsed_expr.subs(sympify(placeholder), expr)

    if global_symbols is not None:
        parsed_expr = replace_variables_equation(parsed_expr, global_symbols)
    if init_solutions is not None:
        parsed_expr = parsed_expr.subs({k: sympify(v) for k, v in init_solutions.items()})
    simplified_expr = simplify(parsed_expr)

    # 检查是否为布尔值
    is_valid = True if isinstance(simplified_expr, sympy.logic.boolalg.BooleanTrue) else False

    return is_valid


def evaluate_expression_visual(constraints, key_value_pair, point_positions):
    placeholders = {}
    parsed_expr = parse_expression(constraints, placeholders, key_value_pair, is_visual=True,
                                   point_positions=point_positions)

    if isinstance(parsed_expr, bool):
        return parsed_expr

    # 循环替换占位符直到没有更多占位符为止
    while any(p in str(parsed_expr) for p in placeholders):
        for placeholder, expr in placeholders.items():
            parsed_expr = parsed_expr.subs(sympify(placeholder), expr)

    equation_to_check = parsed_expr.subs({k: sympify(v) for k, v in key_value_pair.items()})
    simplified_expr = simplify(equation_to_check)

    # 检查是否为布尔值
    is_valid = True if isinstance(simplified_expr, sympy.logic.boolalg.BooleanTrue) else False

    return is_valid


def extract_variables_and_values(constraints, mapping_dict, global_graph, value_retriever):
    # 提取约束中的所有不重复变量
    vars_in_constraint = re.findall(r'\b(?!pi\b|sin\b|cos\b|tan\b)\w*[a-zA-Z]+\w*\b', constraints)
    variables = {var for var in vars_in_constraint if var not in {'AND', 'OR'}}
    key_value_pair = {}
    for var in variables:
        if var not in mapping_dict:
            return None  # 无映射，返回失败

        node_name = mapping_dict[var]
        try:
            node_value = value_retriever(global_graph, node_name)
            if node_value == 'None':
                if 'OR' in constraints:
                    node_value = node_name
                else:
                    return None
        except KeyError:
            return None  # 节点值未找到，返回失败

        key_value_pair[node_name] = node_value

    return key_value_pair


def verify_constraints(model_graph, global_graph, mapping_dict, global_symbols=None, init_solutions=None):
    new_constraints = replace_variables_str(model_graph.constraints, mapping_dict, global_symbols)

    # 调用 evaluate_expression 函数
    if init_solutions is not None:
        return evaluate_expression(new_constraints, global_symbols, init_solutions)
    else:
        # for testing graph matching
        def safe_get_node_value(graph, node):
            value = graph.get_node_value(node)
            if isinstance(value, list):
                return value[0]
            else:
                return value

        key_value_pair = extract_variables_and_values(model_graph.constraints, mapping_dict, global_graph,
                                                      safe_get_node_value)

        if key_value_pair is None:
            return False
        return evaluate_expression(new_constraints, global_symbols, key_value_pair)


def verify_visual_constraints(model_graph, global_graph, mapping_dict, global_symbols):
    key_value_pair = extract_variables_and_values(
        model_graph.visual_constraints, mapping_dict, global_graph,
        lambda graph, node: graph.get_node_visual_value(node)
    )

    if key_value_pair is None:
        return False

    new_constraints = replace_variables_str(model_graph.visual_constraints, mapping_dict, global_symbols)

    return evaluate_expression_visual(new_constraints, key_value_pair, global_graph.point_positions)


def update_details_with_mapping(details, mapping_dict):
    """
    更新details列表中的元素，使用mapping_dict进行映射替换。
    """
    try:
        if isinstance(details, list) and len(details) >= 1:
            return [mapping_dict.get(d, d) for d in details]
        else:
            return details
    except TypeError:
        return details


def apply_mapping_to_actions(model_graph, mapping_dict):
    """
    根据提供的映射字典，转换模型图的动作列表。
    """

    actions = model_graph.actions
    new_actions = []

    for action in actions:
        # 确保action是字典类型，且包含type键
        if not isinstance(action, dict) or 'type' not in action:
            continue

        new_action = {'type': action.get('type')}
        details = action.get('details', [])

        transformed_details = update_details_with_mapping(details, mapping_dict)

        # 确保transformed_details始终是一个列表
        new_action['details'] = transformed_details if isinstance(transformed_details, list) else []

        new_actions.append(new_action)

    return new_actions


def apply_mapping_to_equations(model_graph, mapping, global_symbols):
    equations = model_graph.equations
    new_equations = []

    for equation in equations:
        # 替换方程中所有匹配到的变量
        new_equation_str = replace_variables_str(str(equation), mapping, global_symbols)
        if '=' in new_equation_str:
            lhs, rhs = map(str.strip, new_equation_str.split('='))
            new_equation = Eq(sympify(lhs), sympify(rhs))
        else:
            new_equation = sympify(new_equation_str)

        # 更新方程，确保所有符号都是全局符号表中的符号
        new_equation = replace_variables_equation(new_equation, global_symbols)
        new_equations.append(new_equation)

    return new_equations


def match_graphs(model_graph, global_graph, global_symbols=None, init_solutions=None):
    """
    使用VF3算法匹配两个图并验证约束条件。
    """

    def is_constraints_valid(_model_graph):
        """
        检查model_graph是否有有效的约束条件。
        """
        return _model_graph.constraints is not None and _model_graph.constraints != ""

    def is_visual_constraints_valid(_model_graph):
        """
        检查model_graph是否有有效的视觉约束。
        """
        return _model_graph.visual_constraints is not None and _model_graph.visual_constraints != ""

    mapping_dict_list = []

    solution = run_vf3(model_graph.grf_data, global_graph.grf_data)
    constraints_valid = is_constraints_valid(model_graph)
    visual_constraints_valid = is_visual_constraints_valid(model_graph)
    if constraints_valid or visual_constraints_valid:
        grouped_list = group_by_id_sets(solution)
        for group in grouped_list:
            for s in group:
                mapping_dict = parse_mapping(model_graph, global_graph, s)

                # 检查视觉约束，如果约束有效则验证它们，否则默认为True
                visual_constraints_flag = (not is_visual_constraints_valid(model_graph) or
                                           verify_visual_constraints(model_graph, global_graph, mapping_dict,
                                                                     global_symbols))

                if visual_constraints_flag:
                    # 检查一般约束，如果约束有效则验证它们，否则默认为True
                    constraints_flag = (not is_constraints_valid(model_graph) or
                                        verify_constraints(model_graph, global_graph, mapping_dict, global_symbols,
                                                           init_solutions))

                    # 如果两个标志都为True，则添加到列表并跳出当前循环
                    if constraints_flag:
                        mapping_dict_list.append(mapping_dict)
                        break  # 退出当前 group 的循环
    else:
        solution = filter_duplicates(solution)
        for s in solution:
            mapping_dict = parse_mapping(model_graph, global_graph, s)
            mapping_dict_list.append(mapping_dict)

    return mapping_dict_list
