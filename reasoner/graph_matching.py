import ctypes
import re
import subprocess
import sys
import sympy

from sympy import sympify, simplify, And, Or, Eq, Symbol, Le, Ge, Lt, Gt
from ctypes import c_char_p, CFUNCTYPE

from reasoner.hologram import GraphModel
from reasoner.utils import filter_duplicates, group_by_id_sets, filter_mappings
from reasoner.config import TOLERANCE
from utils.common_utils import calc_cross_angle
from reasoner.config import logger

# Define callback function types
CALLBACK_FUNC_TYPE = CFUNCTYPE(None, c_char_p)


def run_vf3_in_subprocess(pattern_data, target_data, options=b'-f vfe -u -s'):
    solutions = []
    try:
        result = subprocess.run(
            [sys.executable, "reasoner/run_vf3_subprocess.py", pattern_data, target_data, options],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        solutions = result.stdout.decode('utf-8').strip()
        if len(solutions) > 0:
            solutions = solutions.split('|')
    except subprocess.CalledProcessError as e:
        pass

    return solutions


def run_vf3(pattern_data, target_data, options=b'-f vfe -u -s'):
    solutions = []

    # Define callback function
    @CALLBACK_FUNC_TYPE
    def result_callback(all_solutions_c_str):
        nonlocal solutions
        all_solutions_str = all_solutions_c_str.decode('utf-8')
        if all_solutions_str:
            solutions = all_solutions_str.split('|')  # Split the string based on the delimiter '|'

    try:
        # System judgment to determine whether the loaded shared library is .so or .dll
        if sys.platform.startswith('win'):
            lib_path = 'vf3/bin/vf3.dll'
            kernel32 = ctypes.WinDLL('kernel32.dll')
            handle = kernel32.LoadLibraryW(lib_path)
            libvf3 = ctypes.CDLL(lib_path)
        else:
            lib_path = 'vf3/bin/vf3.so'
            libvf3 = ctypes.CDLL(lib_path)

        # Set the parameter types and return types of run_vf3()
        libvf3.run_vf3.argtypes = [c_char_p, c_char_p, c_char_p, CALLBACK_FUNC_TYPE]

        # run vf3
        libvf3.run_vf3(c_char_p(pattern_data.encode('utf-8')), c_char_p(target_data.encode('utf-8')),
                       c_char_p(options), result_callback)

    except RuntimeError as e:
        logger.error(e)
        return solutions
    except Exception as e:
        logger.error(e)
        return solutions

    return solutions


def load_models_from_json(json_data):
    model_pool = []  # Used to store all instantiated model diagrams
    model_id_map = {}

    # Traverse each model key and its corresponding value in JSON data
    for model_id, (model_name, model_content) in enumerate(json_data.items(), start=0):
        # Instantiate each graph model using from_json() of GraphModel
        model_graph = GraphModel.from_json(model_content, model_name)
        # Add the instantiated graph model to the model pool list
        model_pool.append(model_graph)
        model_id_map[model_id] = model_graph.model_id

    return model_pool, model_id_map


# def get_model(model_pool, model_id_map, model_id):
#     if model_id in model_id_map:
#         for k, v in model_id_map.items():
#             if v == model_id:
#                 return model_pool[k]
#     return None


# TODO Change get_model() in the next training
def get_model(model_pool, model_id_map, model_id):
    if model_id in model_id_map:
        return model_pool[model_id_map[model_id]]
    return None


def get_candidate_models_from_pool(model_pool, global_graph):
    """
    Retrieve candidate models from the model pool, ensuring that the number of node and edge types in the model
    does not exceed the corresponding number in the global graph.
    """
    candidate_models = []

    # Traverse each model graph in the model pool
    for model in model_pool:
        # Check if the number of each node type in the model
        # does not exceed the corresponding number in the global graph
        node_type_check = all(
            model.node_types.get(nt, 0) <= global_graph.node_types.get(nt, 0) for nt in model.node_types)

        # Check if the number of each edge type in the model
        # does not exceed the corresponding number in the global graph
        edge_type_check = all(
            model.edge_types.get(et, 0) <= global_graph.edge_types.get(et, 0) for et in model.edge_types)

        # If the number of types of nodes and edges does not exceed the global graph, add them to the candidate model
        if node_type_check and edge_type_check:
            candidate_models.append(model)

    return candidate_models


def parse_mapping(model_graph, global_graph, mapping_str):
    """
    Parse the mapping string and return a dictionary,
    where the key is the node index in the subgraph and the value is the corresponding node index in the global graph.
    """
    mapping_dict = {}
    mappings = mapping_str.split(':')
    for pair in mappings:
        if pair:
            global_idx, sub_idx = pair.split(',')
            mapping_dict[model_graph.grf_to_id[int(sub_idx)]] = global_graph.grf_to_id[int(global_idx)]
    return mapping_dict


def approximately_equal(value1, value2, tolerance):
    """ Compare whether two values are equal within a specified error range. """
    if tolerance:
        return abs(value1 - value2) <= tolerance
    else:
        return False


def eval_with_none_check(expression, substitutions):
    try:
        """Evaluate the expression by first checking if there is a value of None."""
        if expression == '':
            return None
        if any(value == 'None' or value is None for value in substitutions.values()):
            return None  # If any replacement value is None, return None
        return sympify(expression).evalf(subs=substitutions)
    except Exception as e:
        logger.error(f"Error evaluating expression: {e}")
        return None


def parse_expression(expr_str, _placeholders, key_value_pair=None, is_visual=False, point_positions=None):
    while '(' in expr_str:
        open_index = expr_str.rfind('(')
        close_index = expr_str.find(')', open_index)
        inner_expr = expr_str[open_index + 1:close_index]
        parsed_inner_expr = parse_expression(inner_expr, _placeholders, key_value_pair, is_visual, point_positions)
        _placeholder = f"__placeholder_{len(_placeholders)}__"
        _placeholders[_placeholder] = parsed_inner_expr
        expr_str = expr_str[:open_index] + _placeholder + expr_str[close_index + 1:]

    if ' AND ' in expr_str or ' OR ' in expr_str:
        if ' OR ' in expr_str:
            parts = expr_str.split(' OR ')
            operator = Or
        else:
            parts = expr_str.split(' AND ')
            operator = And
        return operator(*[parse_expression(part.strip(), _placeholders, key_value_pair, is_visual, point_positions)
                          for part in parts])

    # Check if comparison operator is included
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
                return False  # In the case of None, return False

            return approximately_equal(lhs_eval, rhs_eval, tolerance)
        else:
            return Eq(sympify(lhs), sympify(rhs))

    # Check if parallel symbols are included
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

    # Match possible variable names (including letters, numbers, and underscores) using regular expressions
    pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'

    # Function used to replace the matched variable
    def replace_variable(match):
        var_name = match.group(0)
        if var_name in mapping_dict:
            return str(mapping_dict[var_name])  # Ensure that the returned variable name is in string form
        elif var_name in global_symbols:
            return str(global_symbols[var_name])  # Retrieve symbols from the global symbol table
        return var_name  # If there is no mapping, return the original variable name

    new_equation_str = re.sub(pattern, replace_variable, str(equation_str))

    return new_equation_str


def replace_variables_equation(equation, global_symbols):
    new_equation = equation.subs({
        sym: global_symbols.get(sym.name, sym) for sym in equation.atoms() if isinstance(sym, Symbol)
    })

    return new_equation


def evaluate_expression(constraints, global_symbols=None, init_solutions=None):
    placeholders = {}
    try:
        parsed_expr = parse_expression(constraints, placeholders, key_value_pair=None, is_visual=False)
        if isinstance(parsed_expr, bool):
            return parsed_expr

        # Cycle through placeholders until there are no more placeholders left
        while any(p in str(parsed_expr) for p in placeholders):
            for placeholder, expr in placeholders.items():
                parsed_expr = parsed_expr.subs(sympify(placeholder), expr)

        if global_symbols is not None:
            parsed_expr = replace_variables_equation(parsed_expr, global_symbols)
        if init_solutions is not None:
            parsed_expr = parsed_expr.subs({k: sympify(v) for k, v in init_solutions.items()})
    except Exception as e:
        logger.error(f"Error occurred while evaluating expression: {e}")
        return False

    simplified_expr = simplify(parsed_expr)
    # Check if it is a Boolean value
    is_valid = True if isinstance(simplified_expr, sympy.logic.boolalg.BooleanTrue) else False

    return is_valid


def evaluate_expression_visual(constraints, key_value_pair, point_positions):
    placeholders = {}
    try:
        parsed_expr = parse_expression(constraints, placeholders, key_value_pair, is_visual=True,
                                       point_positions=point_positions)
        if isinstance(parsed_expr, bool):
            return parsed_expr

        # Cycle through placeholders until there are no more placeholders left
        while any(p in str(parsed_expr) for p in placeholders):
            for placeholder, expr in placeholders.items():
                parsed_expr = parsed_expr.subs(sympify(placeholder), expr)

        equation_to_check = parsed_expr.subs({k: sympify(v) for k, v in key_value_pair.items()})
    except Exception as e:
        logger.error(f"Error occurred while evaluating visual expression: {e}")
        return False

    simplified_expr = simplify(equation_to_check)
    # Check if it is a Boolean value
    is_valid = True if isinstance(simplified_expr, sympy.logic.boolalg.BooleanTrue) else False

    return is_valid


def extract_variables_and_values(constraints, mapping_dict, global_graph, value_retriever):
    # Extract all non-repeating variables from constraints
    vars_in_constraint = re.findall(r'\b(?!pi\b|sin\b|cos\b|tan\b)\w*[a-zA-Z]+\w*\b', constraints)
    variables = {var for var in vars_in_constraint if var not in {'AND', 'OR'}}
    key_value_pair = {}
    for var in variables:
        if var not in mapping_dict:
            return None

        node_name = mapping_dict[var]
        try:
            node_value = value_retriever(global_graph, node_name)
            if node_value == 'None':
                if 'OR' in constraints:
                    node_value = node_name
                else:
                    return None
        except KeyError:
            return None

        key_value_pair[node_name] = node_value

    return key_value_pair


def verify_constraints(model_graph, global_graph, mapping_dict, global_symbols=None, init_solutions=None):
    new_constraints = replace_variables_str(model_graph.constraints, mapping_dict, global_symbols)

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
    Update the elements in the details list and use mapping_ict for mapping replacement.
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
    Convert the action list of the model diagram based on the provided mapping dictionary.
    """

    actions = model_graph.actions
    new_actions = []

    for action in actions:
        # Ensure that the action is of dictionary type and contains a type key
        if not isinstance(action, dict) or 'type' not in action:
            continue

        new_action = {'type': action.get('type')}
        details = action.get('details', [])

        transformed_details = update_details_with_mapping(details, mapping_dict)

        # Ensure that transformad_details is always a list
        new_action['details'] = transformed_details if isinstance(transformed_details, list) else []

        new_actions.append(new_action)

    return new_actions


def apply_mapping_to_equations(model_graph, mapping, global_symbols):
    equations = model_graph.equations
    new_equations = []

    try:
        for equation in equations:
            # Replace all matched variables in the equation
            new_equation_str = replace_variables_str(str(equation), mapping, global_symbols)
            if '=' in new_equation_str:
                lhs, rhs = map(str.strip, new_equation_str.split('='))
                new_equation = Eq(sympify(lhs), sympify(rhs))
            else:
                new_equation = sympify(new_equation_str)

            # Update the equation to ensure that all symbols are from the global symbol table
            new_equation = replace_variables_equation(new_equation, global_symbols)
            new_equations.append(new_equation)
    except Exception as e:
        logger.error(f"Error occurred while applying mapping to equations: {e}")

    return new_equations


def match_graphs(model_graph, global_graph, global_symbols=None, init_solutions=None):
    """
    Use VF3 algorithm to match two graphs and verify the constraints.
    """

    def is_constraints_valid(_model_graph):
        """
        Check if model_graph has valid mathematical constraints.
        """
        return _model_graph.constraints is not None and _model_graph.constraints != ""

    def is_visual_constraints_valid(_model_graph):
        """
        Check if model_graph has valid visual constraints.
        """
        return _model_graph.visual_constraints is not None and _model_graph.visual_constraints != ""

    mapping_dict_list = []

    try:
        solution = run_vf3_in_subprocess(model_graph.grf_data, global_graph.grf_data)
        # solution = run_vf3(model_graph.grf_data, global_graph.grf_data)
        constraints_valid = is_constraints_valid(model_graph)
        visual_constraints_valid = is_visual_constraints_valid(model_graph)
        if constraints_valid or visual_constraints_valid:
            grouped_list = group_by_id_sets(solution)
            for group in grouped_list:
                parsed_mappings = []
                for s in group:
                    mapping_dict = parse_mapping(model_graph, global_graph, s)
                    parsed_mappings.append(mapping_dict)

                filtered_mappings = filter_mappings(parsed_mappings, model_graph.fixed_nodes) \
                    if len(model_graph.fixed_nodes) > 0 else parsed_mappings
                for mapping_dict in filtered_mappings:
                    # Check visual constraints, validate them if they are valid, otherwise default to True
                    visual_constraints_flag = (not is_visual_constraints_valid(model_graph) or
                                               verify_visual_constraints(model_graph, global_graph, mapping_dict,
                                                                         global_symbols))

                    if visual_constraints_flag:
                        # Check mathematical constraints, validate them if they are valid, otherwise default to True
                        constraints_flag = (not is_constraints_valid(model_graph) or
                                            verify_constraints(model_graph, global_graph, mapping_dict, global_symbols,
                                                               init_solutions))
                        # If both flags are True, add to the list and exit the current loop
                        if constraints_flag:
                            mapping_dict_list.append(mapping_dict)
                            break
        else:
            solution = filter_duplicates(solution)
            for s in solution:
                mapping_dict = parse_mapping(model_graph, global_graph, s)
                mapping_dict_list.append(mapping_dict)
    except Exception as e:
        logger.error(f"An error occurred in graph matching: {e}")

    return mapping_dict_list
