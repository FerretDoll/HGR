import concurrent.futures
import json
from itertools import combinations, islice

from func_timeout import func_timeout, FunctionTimedOut
from sympy import symbols, Eq, sympify, cos, sin, pi, solve, tan, parse_expr, Mul, Rational, N, \
    SympifyError, Symbol, cot, sec, csc, simplify, Interval

from reasoner.graph_matching import get_candidate_models_from_pool, match_graphs, apply_mapping_to_actions, \
    apply_mapping_to_equations, load_models_from_json, replace_variables_equation
from reasoner.config import UPPER_BOUND, logger, max_workers, model_pool_path
from reasoner.utils import is_debugging
from utils.common_utils import isNumber, closest_to_number

with open(model_pool_path, 'r') as model_pool_file:
    model_pool, model_id_map = load_models_from_json(json.load(model_pool_file))


class GraphSolver:
    def __init__(self, global_graph):
        self.rounds = 0
        self.target_node_values = []
        self.answer = None
        self.model_instance_eq_num = [0, 0, 0]
        self.global_graph = global_graph
        self.equations = []
        self.model_equations = []
        self.node_value_equations_dict = {}
        self.known_var = {}
        self.init_solutions = {}
        self.matched_model_list = []
        self.matched_relations = []
        self.is_solved = False
        self.is_updated = True
        self.upper_bound = UPPER_BOUND
        self.symbols = {node_id: symbols(str(node_id), positive=True) for node_id in global_graph.graph.nodes()}
        self.current_new = set()
        self.reasoning_record = []

    @staticmethod
    def is_pure_radian(expr):
        # Ensure that the expression consists only of pi and possible numerical multipliers
        if expr.has(pi):
            # Contains only pi and does not include any other variables or additional symbols
            if isinstance(expr, Mul) and all(isinstance(arg, (Rational, pi.__class__)) for arg in expr.args):
                return True
        return False

    @staticmethod
    def format_value(value):
        try:
            # Attempt to convert the value to a floating-point number and round it to the nearest integer
            if value.is_Float:
                # If the value is an integer, return it in integer form
                if value == round(value):
                    return str(int(value))
                else:
                    formatted = f"{value:.{2}f}"  # Keep two decimal places
                    if '.' in formatted:
                        formatted = formatted.rstrip('0')
                        if formatted[-1] == '.':
                            formatted = formatted[:-1]
                    return formatted
            else:
                # For complex expressions, return a more concise string representation
                return str(value.simplify())  # Simplify the expression and convert it to a string
        except AttributeError:
            pass
        return str(value)  # Directly return the original value in string form, suitable for non-numeric values

    @staticmethod
    def evaluate_if_possible(expr):
        if expr is None:
            return None
        # Check if the expression can be parsed into a specific number
        if expr.is_number:
            return N(expr)  # Use N() to evaluate expressions
        return expr

    @staticmethod
    def equation_variables(eq):
        return eq.free_symbols

    @staticmethod
    def find_equations_and_common_variables(equation_vars, solved_equations, knowns):
        """
        Return a two-dimensional list, each containing a system of equations with the same two variables
        and only these two variables, and return the corresponding list of variables.

        Args:
            equation_vars (list): [(eq, vars_set), ...] Formal equations and sets of variables
            solved_equations (list): Solved equations

        Returns:
            groups (list): A two-dimensional list containing equations
            variables (list): Common variables corresponding to each sub list
        """
        groups = []
        variables = []
        visited = set()

        for eq1, vars_set1 in equation_vars:
            if eq1 not in solved_equations and eq1 not in visited and len(vars_set1 - knowns.keys()) == 2:
                group = [eq1]
                visited.add(eq1)
                common_vars = vars_set1

                for eq2, vars_set2 in equation_vars:
                    if eq2 != eq1 and eq2 not in solved_equations and eq2 not in visited:
                        intersect_vars = vars_set1 & vars_set2 - knowns.keys()
                        if len(intersect_vars) == 2 and len(vars_set2 - knowns.keys()) == 2:
                            group.append(eq2)
                            visited.add(eq2)
                            common_vars = intersect_vars

                if len(group) > 1:
                    groups.append(group)
                    variables.append(list(common_vars))

        return groups, variables

    @staticmethod
    def chunked_iterable(iterable, size):
        it = iter(iterable)
        return iter(lambda: list(islice(it, size)), [])

    def is_within_domain(self, sol):
        for var, value in sol.items():
            if str(var) in self.global_graph.graph.nodes:
                if value.is_number:
                    interval = self.global_graph.get_node_domain(str(var))
                    if isinstance(interval, Interval):
                        if not interval.contains(value):
                            return False

        return True

    def print_node_value(self, print_new=False):
        # Print updated graph node information
        for node, attrs in self.global_graph.graph.nodes(data=True):
            if attrs['value'] == 'None':
                attrs['float_value'] = 'None'
            else:
                attrs['float_value'] = [N(value) for value in attrs['value']]
            if print_new:
                if node in self.current_new:
                    logger.debug(f"*Node {node}: {attrs}")
            else:
                logger.debug(f"Node {node}: {attrs}")

    def remove_solved_equations(self, equations):
        equation_vars = [(eq, self.equation_variables(eq)) for eq in equations]
        for eq, vars_set in equation_vars:
            unknown_vars = vars_set - self.known_var.keys()
            if len(unknown_vars) == 0:
                equations.remove(eq)

    def check_and_evaluate_targets(self):
        targets_values = []
        for node_id in self.global_graph.target:
            node_value = self.global_graph.get_node_value(node_id)
            if node_value != 'None':
                for single_value in node_value:
                    is_evaluate = self.evaluate_if_possible(sympify(single_value))
                    if is_evaluate is not None:
                        if is_evaluate.is_number:
                            targets_values.append({node_id: single_value})
                            break

        if len(targets_values) == len(self.global_graph.target):
            return targets_values
        else:
            return []

    def replace_and_evaluate(self, target_equation):
        # Replace the symbols in target_equation
        for node_id, symbol in self.symbols.items():
            target_equation = target_equation.subs(symbols(node_id), symbol)

        result = target_equation.evalf(subs=self.known_var)

        return result

    def update_graph_node_values(self, solution):
        numeric_values = {}
        for var, value in solution.items():
            if value.has(pi) and self.is_pure_radian(value):
                numeric_values[var] = value
            else:
                evaluated_value = self.evaluate_if_possible(value)  # Attempt to convert the expression to a number
                if evaluated_value.is_number:
                    numeric_values[var] = self.format_value(value)  # Save the converted value

        # Update the corresponding node values in global_graph
        for var_name, value in numeric_values.items():
            var_name_str = str(var_name)
            if var_name_str in self.global_graph.graph.nodes:
                if var_name not in self.known_var.keys():
                    node_value = self.global_graph.get_node_value(var_name_str)
                    if node_value == 'None':
                        node_value = []
                    node_value.append(str(value))
                    self.global_graph.modify_node_value(var_name_str, node_value)  # Update node values
                    self.current_new.add(var_name_str)
                    equations = self.node_value_equations_dict.setdefault(var_name_str, [])
                    equations.append(Eq(var_name, sympify(value), evaluate=False))
                    self.node_value_equations_dict[var_name_str] = equations
                    if not self.is_updated:
                        self.is_updated = True
            for node_id, node_attrs in self.global_graph.graph.nodes(data=True):
                node_value = node_attrs.get('value')
                if node_value != 'None':
                    try:
                        new_node_value = []
                        equations = []
                        for single_value in node_value:
                            if not isNumber(single_value):
                                # Attempt to parse node values into expressions
                                parsed_expr = sympify(single_value)
                                # Retrieve all symbols in the expression
                                symbols_in_expr = parsed_expr.atoms(Symbol)

                                # If the expression contains only one symbol
                                if len(symbols_in_expr) == 1:
                                    single_symbol = next(iter(symbols_in_expr))

                                    # Check if this symbol is in numeric_values
                                    if str(single_symbol) == str(var_name):
                                        # Replace the symbol with its numerical value
                                        new_value = parsed_expr.subs(single_symbol, numeric_values[var_name])

                                        # Check if the new value is a pure number
                                        if new_value.is_number:
                                            # Update node values
                                            new_format_value = self.format_value(new_value)
                                            new_node_value.append(new_format_value)
                                            self.known_var[self.symbols[node_id]] = new_format_value
                                            equations.append(Eq(self.symbols[node_id], sympify(new_format_value,
                                                                                               evaluate=False)))
                                            if not self.is_updated:
                                                self.is_updated = True
                                            continue

                                new_node_value.append(single_value)
                                equations.append(Eq(self.symbols[node_id], sympify(single_value, evaluate=False)))
                            else:
                                new_node_value.append(single_value)
                                equations.append(Eq(self.symbols[node_id], sympify(single_value, evaluate=False)))
                        if set(new_node_value) != set(node_value):
                            self.global_graph.modify_node_value(node_id, list(set(new_node_value)))
                            self.current_new.add(node_id)
                            self.node_value_equations_dict[node_id] = list(set(equations))
                    except SympifyError:
                        # If parsing fails or the node value is not an expression, ignore the node
                        continue

        self.known_var.update(numeric_values)
        self.remove_solved_equations(self.model_equations)

    def solve_equations(self):
        old_init_solutions = self.init_solutions.copy()
        self.equations.extend(self.model_equations)
        self.equations = list(set(self.equations))
        self.current_new = set()
        # Create two lists, one for storing basic equations
        # and one for storing equations containing trigonometric functions
        base_equations = []
        complex_equations = []

        # Traverse the equation list and check if each equation contains trigonometric functions
        for eq in self.equations:
            if eq.has(sin, cos, tan, sec, csc, cot):
                complex_equations.append(eq)
            else:
                base_equations.append(eq)
        logger.debug(f"Base Equations ({len(base_equations)}):\n{base_equations}")
        if len(complex_equations) > 0:
            logger.debug(f"Complex Equations ({len(complex_equations)}):\n{complex_equations}")

        base_solutions = []
        try:
            base_solutions = func_timeout(20, solve, kwargs=dict(f=base_equations, dict=True))
            base_solutions = [sol for sol in base_solutions if self.is_within_domain(sol)]
        except FunctionTimedOut as e:
            logger.debug(f"Failed to solve base equations within the time limit")
        except Exception as e:
            logger.error(f"Failed to solve base equations: {e}")

        base_solution = {}
        estimate = lambda sol: sum([str(expr)[0] != '-' for expr in sol.values()])  # negative value
        if len(base_solutions) > 0:
            base_solution = max(base_solutions, key=estimate)
            self.init_solutions = base_solution
            logger.debug(f"Base Solution:\n{base_solution}")

            self.update_graph_node_values(base_solution)
        else:
            logger.debug("No solution found for base equations")

        if len(complex_equations) > 0:
            # Due to the time-consuming nature of solving nonlinear systems of equations,
            # optimization of the solving algorithm is conducted here
            if len(base_solutions) > 0:
                substituted_equations = list(set([eq.subs(base_solution) for eq in complex_equations]))
                logger.debug(f"Substituted Complex Equations: {substituted_equations}")
                knowns = {}
            else:
                substituted_equations = list(set([eq for eq in complex_equations]))
                knowns = {key: sympify(value) for key, value in self.known_var.items()}

            # Count the number of variables included in each equation
            equation_vars = [(eq, self.equation_variables(eq)) for eq in substituted_equations]
            equation_vars_sorted = sorted(equation_vars, key=lambda x: len(x[1]))
            solved_equations = []
            remaining_equations = substituted_equations.copy()
            logger.debug("Start solving complex equations in step 1")
            # Step 1: First, solve the equation that only contains a single variable
            while True:
                single_unknown_eq = None
                for eq, vars_set in equation_vars_sorted:
                    if eq not in solved_equations:
                        unknown_vars = vars_set - knowns.keys()
                        if len(unknown_vars) == 0:
                            solved_equations.append(eq)
                            remaining_equations.remove(eq)
                            continue
                        if len(unknown_vars) == 1:
                            single_unknown_eq = (eq, list(unknown_vars)[0])
                            break

                if single_unknown_eq:
                    eq, unknown_var = single_unknown_eq
                    solved_equations.append(eq)
                    remaining_equations.remove(eq)
                    try:
                        sol = func_timeout(20, solve, kwargs=dict(f=eq.subs(knowns), symbols=unknown_var,
                                                                  dict=True))
                        sol = [list(s.values())[0] for s in sol if self.is_within_domain(s)]
                        if len(sol) == 1:
                            knowns[unknown_var] = sol[0]
                        elif len(sol) > 1:
                            if str(unknown_var) in self.global_graph.graph.nodes:
                                visual_value = self.global_graph.get_node_visual_value(str(unknown_var))
                                knowns[unknown_var] = closest_to_number(sol, visual_value)
                            else:
                                for node_id, node_attrs in self.global_graph.graph.nodes(data=True):
                                    node_value = node_attrs.get('value')
                                    if node_value != 'None':
                                        try:
                                            for single_value in node_value:
                                                if not isNumber(single_value):
                                                    # Attempt to parse node values into expressions
                                                    parsed_expr = sympify(single_value)
                                                    # Retrieve all symbols in the expression
                                                    symbols_in_expr = parsed_expr.atoms(Symbol)

                                                    # If the expression contains only one symbol
                                                    if len(symbols_in_expr) == 1:
                                                        single_symbol = next(iter(symbols_in_expr))

                                                        # Check if this symbol is in numeric_values
                                                        if str(single_symbol) == str(unknown_var):
                                                            candi_node_values = []
                                                            for s in sol:
                                                                # Replace the symbol with its numerical value
                                                                new_value = parsed_expr.subs(single_symbol, s)

                                                                # Check if the new value is a pure number
                                                                if new_value.is_number:
                                                                    # Update node values
                                                                    candi_node_values.append(new_value)
                                                            if len(candi_node_values) > 0:
                                                                visual_value = (
                                                                    self.global_graph.get_node_visual_value(node_id)
                                                                )
                                                                knowns[unknown_var] = closest_to_number(
                                                                    candi_node_values, visual_value)
                                        except SympifyError:
                                            # If parsing fails or the node value is not an expression, ignore the node
                                            continue

                    except FunctionTimedOut as e:
                        logger.debug(f"Failed to solve complex equations within the time limit in step 1")
                    except Exception as e:
                        logger.error(f"Failed to solve complex equations in step 1: {e}")
                else:
                    break

            if knowns:
                logger.debug(f"Complex Solution in Step 1: {knowns}")
                self.update_graph_node_values(knowns)
            else:
                logger.debug("No solution found for complex equations in step 1")

            if len(remaining_equations) > 0:
                total_added = {}
                step_2_rounds = 0
                logger.debug("Start solving complex equations in step 2")
                while True:
                    knows_added_step_2 = {}
                    groups, variables = self.find_equations_and_common_variables(equation_vars, solved_equations,
                                                                                 knowns)
                    if len(groups) == 0:
                        break
                    step_2_rounds += 1
                    logger.debug(f"Step 2 Round {step_2_rounds}:")
                    for i, (group, common_vars) in enumerate(zip(groups, variables), start=1):
                        logger.debug(f"Group {i} ({common_vars}): {group}")
                        count = 0
                        for eq1, eq2 in combinations(group, 2):
                            if count < 3:
                                try:
                                    pairs = [eq1.subs(knowns), eq2.subs(knowns)]
                                    logger.debug(f"Solving equations: {pairs}")
                                    sol = func_timeout(10, solve, kwargs=dict(f=pairs, dict=True))
                                    sol = [s for s in sol if self.is_within_domain(s)]
                                    if len(sol) > 0:
                                        sol = max(sol,
                                                  key=estimate)  # TODO This may need to be modified because it is more reasonable to select the most similar set of solutions by comparing them with visual values
                                        logger.debug(f"Solution: {sol}")
                                        for var, value in sol.items():
                                            knows_added_step_2[var] = value
                                        break
                                except FunctionTimedOut as e:
                                    logger.debug(
                                        f"Failed to solve complex equations within the time limit in step 2 (group {i})")
                                except Exception as e:
                                    logger.error(f"Failed to solve complex equations in step 2 (group {i}): {e}")
                            else:
                                break
                            count += 1

                        # Regardless of whether the equations in the group can be solved or not,
                        # all equations in the group should be marked as solved to avoid infinite loops
                        for eq in group:
                            solved_equations.append(eq)
                            remaining_equations.remove(eq)

                    if knows_added_step_2:
                        knowns.update(knows_added_step_2)
                        total_added.update(knows_added_step_2)

                if total_added:
                    logger.debug(f"Complex Solution in Step 2: {total_added}")
                    self.update_graph_node_values(total_added)
                else:
                    logger.debug("No solution found for complex equations in step 2")

                # Step 3: Substitute all solutions into the residual equation system and solve them uniformly
                # if len(remaining_equations) > 0:
                #     logger.debug("Start solving complex equations in step 3")
                #     remaining_solutions = []
                #     try:
                #         substitutions = [func_timeout(10, eq.subs, args=(knowns,)) for eq in remaining_equations]
                #         remaining_solutions = func_timeout(20, solve, kwargs=dict(f=substitutions, dict=True))
                #     except FunctionTimedOut as e:
                #         logger.debug(f"Failed to solve complex equations within the time limit in step 3")
                #     except Exception as e:
                #         logger.error(f"Failed to solve complex equations in step 3: {e}")
                #
                #     if len(remaining_solutions) > 0:
                #         remaining_solutions = max(remaining_solutions, key=estimate)
                #         logger.debug(f"Complex Solution in Step 3: {remaining_solutions}")
                #         knowns.update(remaining_solutions)
                #         self.update_graph_node_values(remaining_solutions)
                #     else:
                #         logger.debug("No solution found for complex equations in step 3")
            if knowns:
                substituted_base_equations = list(set([eq.subs(knowns) for eq in base_equations]))
                logger.debug(f"Substituted Base Equations: {substituted_base_equations}")
                base_final_solutions = []
                try:
                    base_final_solutions = func_timeout(20, solve,
                                                        kwargs=dict(f=substituted_base_equations, dict=True))
                    base_final_solutions = [sol for sol in base_final_solutions if self.is_within_domain(sol)]
                except FunctionTimedOut as e:
                    logger.debug(f"Failed to solve substituted base equations within the time limit")
                except Exception as e:
                    logger.error(f"Failed to solve substituted base equations: {e}")
                if len(base_final_solutions) > 0:
                    estimate = lambda sol: sum([str(expr)[0] != '-' for expr in sol.values()])  # negative value
                    base_final_solution = max(base_final_solutions, key=estimate)
                    self.init_solutions = base_final_solution
                    logger.debug(f"Base Final Solution:\n{base_final_solution}")

                    self.update_graph_node_values(base_final_solution)
                else:
                    logger.debug("No solution found for substituted base equations")

        if old_init_solutions != self.init_solutions and not self.is_updated:
            self.is_updated = True
        logger.debug("Equation solving finished!")
        self.print_node_value(print_new=True)

    def execute_actions(self, action_list):
        """Execute preset actions based on the results of model matching"""
        for action in action_list:
            action_type = action['type']
            details = action['details']

            if action_type == 'add_node':
                node_name = details[0]
                node_type = details[1]
                node_value = details[2]
                self.global_graph.add_node(node_name, node_type=node_type, node_value=node_value)

            elif action_type == 'add_edge':
                from_node_name = details[0]
                to_node_name = details[1]
                edge_type = details[2]
                if from_node_name and to_node_name:
                    self.global_graph.add_edge(from_node_name, to_node_name, edge_type=edge_type)

            elif action_type == 'modify_node_value':
                node_name = details[0]
                new_value = details[1]
                if node_name:
                    self.global_graph.modify_node_value(node_name, new_value)

        self.global_graph.node_types = self.global_graph.get_node_types()
        self.global_graph.edge_types = self.global_graph.get_edge_types()
        self.global_graph.update_grf_data()
        self.global_graph.update_grf_to_id()

    def process_one_model(self, model):
        new_actions = []
        new_equations = []
        model_used = False
        instances = []
        try:
            mapping_dict_list = func_timeout(10, match_graphs, args=(model, self.global_graph, self.symbols,
                                                                     self.init_solutions))
        except FunctionTimedOut:
            return new_actions, new_equations
        for mapping_dict in mapping_dict_list:
            relation = model.generate_relation(mapping_dict)
            if relation not in self.matched_relations:
                if not model_used:
                    self.model_instance_eq_num[0] += 1
                    self.matched_model_list.append(model.model_id)
                    model_used = True
                self.matched_relations.append(relation)
                logger.debug(relation)
                instance_info = {'relation': relation, 'actions': [], 'equations': []}
                if len(model.actions) > 0:
                    mapped_actions = apply_mapping_to_actions(model, mapping_dict)
                    new_actions.extend(mapped_actions)
                    instance_info['actions'] = mapped_actions
                if len(model.equations) > 0:
                    mapped_equations = apply_mapping_to_equations(model, mapping_dict, self.symbols)
                    new_equations.extend(mapped_equations)
                    instance_info['equations'] = mapped_equations
                if len(model.actions) > 0 or len(model.equations) > 0:
                    instances.append(instance_info)

        self.reasoning_record.append({'model_name': model.model_name, 'instances': instances})

        return new_actions, new_equations

    def process_model_chunk(self, model_chunk):
        total_actions = []
        total_equations = []

        for model in model_chunk:
            new_actions, new_equations = self.process_one_model(model)
            total_actions.extend(new_actions)
            total_equations.extend(new_equations)

        return total_actions, total_equations

    def init_solve(self):
        if self.global_graph.target is None or len(self.global_graph.target) == 0:
            raise Exception("No target!")
        logger.debug(f"Target Node: {self.global_graph.target}")
        logger.debug(f"Target Equation: {self.global_graph.target_equation}")
        self.rounds = 0

        self.print_node_value()

        for node_id, data in self.global_graph.graph.nodes(data=True):
            node_value = data.get('value')
            if node_value != 'None':  # Check if the value is not empty
                node_var = self.symbols[str(node_id)]
                equations = []
                has_added = False
                try:
                    for single_value in node_value:
                        # If node_value is a string of numbers or floating-point numbers
                        if isinstance(single_value, str):
                            if isNumber(single_value):  # Determine whether it is an integer string
                                node_value_expr = sympify(single_value, evaluate=False)
                                if not has_added:
                                    has_added = True
                                    self.known_var[node_var] = single_value
                            else:
                                # Clean up the string and attempt to parse it into an expression
                                node_value_clean = single_value.replace(" ", "")  # Remove spaces
                                node_value_expr = parse_expr(node_value_clean, evaluate=False)
                                node_value_expr = replace_variables_equation(node_value_expr, self.symbols)
                        else:
                            node_value_expr = sympify(single_value, evaluate=False)
                            # If node_value itself is an integer or floating point number
                            if isinstance(single_value, (int, float)) and not has_added:
                                has_added = True
                                self.known_var[node_var] = single_value
                        equations.append(Eq(node_var, node_value_expr))
                    self.node_value_equations_dict[node_id] = equations
                except Exception as e:
                    logger.error(f"Error parsing value '{node_value}' for node '{node_id}': {e}")

        init_equations = [item for sublist in list(self.node_value_equations_dict.values()) for item in sublist]
        init_equations.extend(self.global_graph.equations)
        try:
            init_solutions = func_timeout(20, solve, kwargs=dict(f=init_equations, dict=True))
            init_solutions = [sol for sol in init_solutions if self.is_within_domain(sol)]
            estimate = lambda sol: sum([str(expr)[0] != '-' for expr in sol.values()])  # negative value
            self.init_solutions = max(init_solutions, key=estimate)
            self.update_graph_node_values(self.init_solutions)
            self.target_node_values = self.check_and_evaluate_targets()
            if len(self.target_node_values) > 0:
                self.answer = self.replace_and_evaluate(self.global_graph.target_equation)
        except FunctionTimedOut:
            logger.error("Timeout when init solving.")

    def solve_with_one_model(self, model):
        self.is_updated = False
        logger.debug(f"Start matching model: {model.model_name}")
        actions, equations = self.process_one_model(model)
        self.model_instance_eq_num[1] = len(self.matched_relations)

        if len(actions) == 0 and len(equations) == 0:
            logger.debug("Model matching failed.")
            return

        if len(actions) > 0:
            self.is_updated = True
            logger.debug(f"Actions List: {actions}")
            self.execute_actions(actions)

        if len(equations) > 0:
            self.is_updated = True
            equations = list(set(equations))
            self.model_instance_eq_num[2] += len(equations)
            logger.debug(f"Equations Added from Model ({len(equations)}):\n{equations}")
            self.model_equations.extend(equations)

            self.equations = [item for sublist in list(self.node_value_equations_dict.values()) for item in sublist]
            self.solve_equations()

            self.target_node_values = self.check_and_evaluate_targets()
            if len(self.target_node_values) > 0:
                self.answer = self.replace_and_evaluate(self.global_graph.target_equation)

    def solve_with_model_sequence(self, model_sequence):
        self.init_solve()
        for model in model_sequence:
            self.solve_with_one_model(model)
            if len(self.target_node_values) > 0:
                self.answer = self.replace_and_evaluate(self.global_graph.target_equation)
                break

    def solve_with_candi_models(self):
        self.init_solve()
        while self.is_updated and self.rounds < self.upper_bound:
            self.rounds += 1
            logger.debug(f"Round {self.rounds}")
            self.is_updated = False

            candidate_models = get_candidate_models_from_pool(model_pool, self.global_graph)
            for model in candidate_models:
                logger.debug(f"Start matching model: {model.model_name}")
                actions, equations = self.process_one_model(model)
                self.model_instance_eq_num[1] = len(self.matched_relations)

                if len(actions) == 0 and len(equations) == 0:
                    logger.debug("Model matching failed.")
                    continue

                if len(actions) > 0:
                    self.is_updated = True
                    logger.debug(f"Actions List: {actions}")
                    self.execute_actions(actions)

                if len(equations) > 0:
                    self.is_updated = True
                    equations = list(set(equations))
                    self.model_instance_eq_num[2] += len(equations)
                    logger.debug(f"Equations Added from Model ({len(equations)}):\n{equations}")
                    self.model_equations.extend(equations)

                    self.equations = [item for sublist in list(self.node_value_equations_dict.values()) for item in
                                      sublist]
                    self.solve_equations()

                    self.target_node_values = self.check_and_evaluate_targets()
                if len(self.target_node_values) > 0:
                    self.answer = self.replace_and_evaluate(self.global_graph.target_equation)
                    return

    def solve(self):
        self.init_solve()
        while self.is_updated and self.rounds < self.upper_bound:
            self.rounds += 1
            logger.debug(f"Round {self.rounds}")
            self.is_updated = False
            self.equations = [item for sublist in list(self.node_value_equations_dict.values()) for item in sublist]
            action_list = []
            added_equations = []

            candidate_models = get_candidate_models_from_pool(model_pool, self.global_graph)
            if is_debugging():
                actions, equations = self.process_model_chunk(candidate_models)
                action_list.extend(actions)
                added_equations.extend(equations)
            else:
                # Split the candidate_models into equal parts as max_workers
                # Calculate the number of models processed by each thread, rounded up to the nearest integer
                chunk_size = (len(candidate_models) + max_workers - 1) // max_workers
                chunks = self.chunked_iterable(candidate_models, chunk_size)

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(self.process_model_chunk, chunk)
                        for chunk in chunks
                    ]

                    for future in concurrent.futures.as_completed(futures):
                        actions, equations = future.result()
                        action_list.extend(actions)
                        added_equations.extend(equations)

            self.model_instance_eq_num[1] = len(self.matched_relations)

            if len(action_list) > 0:
                self.is_updated = True
                logger.debug(f"Actions List: {action_list}")
                self.execute_actions(action_list)

            if len(added_equations) > 0:
                added_equations = list(set(added_equations))
                self.model_instance_eq_num[2] += len(added_equations)
                logger.debug(f"Equations Added from Models ({len(added_equations)}):\n{added_equations}")
                self.model_equations.extend(added_equations)

                self.solve_equations()

                self.target_node_values = self.check_and_evaluate_targets()
                if len(self.target_node_values) > 0:
                    self.answer = self.replace_and_evaluate(self.global_graph.target_equation)
                    return
