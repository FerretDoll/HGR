import concurrent.futures
from itertools import combinations, islice

from func_timeout import func_timeout, FunctionTimedOut
from sympy import symbols, Eq, sympify, cos, sin, pi, solve, tan, parse_expr, Mul, Rational, N, \
    SympifyError, Symbol, cot, sec, csc, simplify

from reasoner.graph_matching import get_candidate_models_from_pool, match_graphs, apply_mapping_to_actions, \
    apply_mapping_to_equations
from reasoner.config import UPPER_BOUND, logger, max_workers
from reasoner.utils import is_debugging
from utils.common_utils import isNumber, closest_to_number


class GraphSolver:
    def __init__(self, global_graph, model_pool):
        self.rounds = 0
        self.target_node_values = []
        self.answer = None
        self.model_instance_eq_num = [0, 0, 0]
        self.global_graph = global_graph
        self.model_pool = model_pool
        self.equations = []
        self.model_equations = []
        self.node_value_equations_dict = {}
        self.known_var = {}
        self.init_solutions = {}
        self.matched_relations = []
        self.is_solved = False
        self.is_updated = True
        self.upper_bound = UPPER_BOUND
        self.symbols = {node_id: symbols(str(node_id), positive=True) for node_id in global_graph.graph.nodes()}
        self.current_new = set()

    @staticmethod
    def is_pure_radian(expr):
        # 确保表达式仅由 pi 和可能的数字乘子组成
        if expr.has(pi):
            # 只包含 pi 且不包含其他变量或额外的符号
            if isinstance(expr, Mul) and all(isinstance(arg, (Rational, pi.__class__)) for arg in expr.args):
                return True
        return False

    @staticmethod
    def format_value(value):
        try:
            # 尝试将值转换为浮点数并四舍五入
            if value.is_Float:
                # 如果数值是整数，则返回整数形式
                if value == round(value):
                    return str(int(value))
                else:
                    formatted = f"{value:.{2}f}"  #保留两位小数
                    if '.' in formatted:
                        formatted = formatted.rstrip('0')
                        if formatted[-1] == '.':
                            formatted = formatted[:-1]
                    return formatted
            else:
                # 对于复杂表达式，返回一个更简洁的字符串表示
                return str(value.simplify())  # 简化表达式后转换为字符串
        except AttributeError:
            pass
        return str(value)  # 直接返回原值的字符串形式，适用于非数字值

    @staticmethod
    def evaluate_if_possible(expr):
        if expr is None:
            return None
        # 检查表达式是否能被解析为一个具体数字
        if expr.is_number:
            return N(expr)  # 使用N()来评估表达式
        return expr

    @staticmethod
    def equation_variables(eq):
        return eq.free_symbols

    @staticmethod
    def find_equations_and_common_variables(equation_vars, solved_equations, knowns):
        """
        返回一个二维列表，每个列表中包含具有相同两个变量且只有这两个变量的方程组，并返回对应的变量列表。

        Args:
            equation_vars (list): [(eq, vars_set), ...] 形式的方程和变量集合
            solved_equations (list): 已经解决的方程

        Returns:
            groups (list): 包含方程的二维列表
            variables (list): 每个子列表对应的公共变量
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

    def print_node_value(self, print_new=False):
        # 打印更新后的图节点信息
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
        # 替换 target_equation 中的符号
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
                evaluated_value = self.evaluate_if_possible(value)  # 尝试将表达式转换为数字
                if evaluated_value.is_number:
                    numeric_values[var] = self.format_value(value)  # 保存转换后的数值

        # 更新 global_graph 中对应的节点值
        for var_name, value in numeric_values.items():
            var_name_str = str(var_name)
            if var_name_str in self.global_graph.graph.nodes:
                if var_name not in self.known_var.keys():
                    node_value = self.global_graph.get_node_value(var_name_str)
                    if node_value == 'None':
                        node_value = []
                    node_value.append(str(value))
                    self.global_graph.modify_node_value(var_name_str, node_value)  # 更新节点值
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
                                # 尝试解析节点值为表达式
                                parsed_expr = sympify(single_value)
                                # 获取表达式中的所有符号
                                symbols_in_expr = parsed_expr.atoms(Symbol)

                                # 如果表达式只包含一个符号
                                if len(symbols_in_expr) == 1:
                                    single_symbol = next(iter(symbols_in_expr))

                                    # 检查这个符号是否在 numeric_values 中
                                    if str(single_symbol) == str(var_name):
                                        # 替换符号为其数值
                                        new_value = parsed_expr.subs(single_symbol, numeric_values[var_name])

                                        # 检查新值是否是一个纯数字
                                        if new_value.is_number:
                                            # 更新节点值
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
                        # 如果解析失败或节点值不是表达式，忽略该节点
                        continue

        self.known_var.update(numeric_values)
        self.remove_solved_equations(self.model_equations)

    def solve_equations(self):
        old_init_solutions = self.init_solutions.copy()
        self.current_new = set()
        self.equations = list(set(self.equations))
        # 创建两个列表，一个用于存放基本方程，一个用于存放包含三角函数的方程
        base_equations = []
        complex_equations = []

        # 遍历方程列表，检查每个方程是否包含三角函数
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
            # 由于求解非线性方程组需要耗费大量时间，因此这里对求解算法进行优化
            # 使用列表推导式应用替换
            if len(base_solutions) > 0:
                substituted_equations = list(set([eq.subs(base_solution) for eq in complex_equations]))
                logger.debug(f"Substituted Complex Equations: {substituted_equations}")
                knowns = {}
            else:
                substituted_equations = list(set([eq for eq in complex_equations]))
                knowns = {key: sympify(value) for key, value in self.known_var.items()}

            # 统计每个方程包含的变量数
            equation_vars = [(eq, self.equation_variables(eq)) for eq in substituted_equations]
            equation_vars_sorted = sorted(equation_vars, key=lambda x: len(x[1]))
            solved_equations = []
            remaining_equations = substituted_equations.copy()
            logger.debug("Start solving complex equations in step 1")
            # 步骤1：先求解只包含单一变量的方程
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
                        sol = func_timeout(20, solve, kwargs=dict(f=eq.subs(knowns), symbols=unknown_var))
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
                                                    # 尝试解析节点值为表达式
                                                    parsed_expr = sympify(single_value)
                                                    # 获取表达式中的所有符号
                                                    symbols_in_expr = parsed_expr.atoms(Symbol)

                                                    # 如果表达式只包含一个符号
                                                    if len(symbols_in_expr) == 1:
                                                        single_symbol = next(iter(symbols_in_expr))

                                                        # 检查这个符号是否在 numeric_values 中
                                                        if str(single_symbol) == str(unknown_var):
                                                            candi_node_values = []
                                                            for s in sol:
                                                                # 替换符号为其数值
                                                                new_value = parsed_expr.subs(single_symbol, s)

                                                                # 检查新值是否是一个纯数字
                                                                if new_value.is_number:
                                                                    # 更新节点值
                                                                    candi_node_values.append(new_value)
                                                            if len(candi_node_values) > 0:
                                                                visual_value = (
                                                                    self.global_graph.get_node_visual_value(node_id)
                                                                )
                                                                knowns[unknown_var] = closest_to_number(
                                                                    candi_node_values, visual_value)
                                        except SympifyError:
                                            # 如果解析失败或节点值不是表达式，忽略该节点
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
                                    sol = [s for s in sol if 0 not in s.values()]  # 暂时不考虑值为0的解
                                    if len(sol) > 0:
                                        sol = max(sol, key=estimate)  # TODO 这里也许需要修改，因为通过与视觉值比较后选出最相近的一组解更合理
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

                        # 不管group中的方程是否能求出解，都要将group中的方程都标记为已求解，避免无限循环
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

                # 步骤3：将所有解代入剩余方程组统一求解
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
        """根据模型匹配的结果执行预设的动作"""
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

    def process_model_chunk(self, model_chunk):
        total_actions = []
        total_equations = []
        for model in model_chunk:
            model_used = False
            mapping_dict_list = []
            try:
                mapping_dict_list = func_timeout(10, match_graphs, args=(model, self.global_graph, self.symbols,
                                                                         self.init_solutions))
            except FunctionTimedOut:
                continue
            for mapping_dict in mapping_dict_list:
                relation = model.generate_relation(mapping_dict)
                if relation not in self.matched_relations:
                    if not model_used:
                        self.model_instance_eq_num[0] += 1
                        model_used = True
                    self.matched_relations.append(relation)
                    logger.debug(relation)
                    if len(model.actions) > 0:
                        new_actions = apply_mapping_to_actions(model, mapping_dict)
                        total_actions.extend(new_actions)
                    if len(model.equations) > 0:
                        new_equations = apply_mapping_to_equations(model, mapping_dict, self.symbols)
                        total_equations.extend(new_equations)
        return total_actions, total_equations

    def solve(self):
        if self.global_graph.target is None or len(self.global_graph.target) == 0:
            raise Exception("No target!")
        logger.debug(f"Target Node: {self.global_graph.target}")
        logger.debug(f"Target Equation: {self.global_graph.target_equation}")
        self.rounds = 0

        self.print_node_value()

        for node_id, data in self.global_graph.graph.nodes(data=True):
            node_value = data.get('value')
            if node_value != 'None':  # 检查value是否非空
                node_var = self.symbols[str(node_id)]
                equations = []
                has_added = False
                try:
                    for single_value in node_value:
                        # 如果node_value是数字字符串或者浮点数表示的字符串
                        if isinstance(single_value, str):
                            if isNumber(single_value):  # 判断是否为整数字符串
                                node_value_expr = sympify(single_value, evaluate=False)
                                if not has_added:
                                    has_added = True
                                    self.known_var[node_var] = single_value
                            else:
                                # 清理字符串并尝试解析为表达式
                                node_value_clean = single_value.replace(" ", "")  # 去除空格
                                node_value_expr = parse_expr(node_value_clean, evaluate=False)
                                for s in self.symbols.values():
                                    if str(s) == str(node_value_expr):
                                        node_value_expr = s
                                        break
                        else:
                            node_value_expr = sympify(single_value, evaluate=False)
                            # 如果node_value本身就是整数或浮点数
                            if isinstance(single_value, (int, float)) and not has_added:
                                has_added = True
                                self.known_var[node_var] = single_value
                        equations.append(Eq(node_var, node_value_expr))
                    self.node_value_equations_dict[node_id] = equations
                except Exception as e:
                    logger.error(f"Error parsing value '{node_value}' for node '{node_id}': {e}")

        init_equations = [item for sublist in list(self.node_value_equations_dict.values()) for item in sublist]
        init_solutions = func_timeout(20, solve, kwargs=dict(f=init_equations, dict=True))
        estimate = lambda sol: sum([str(expr)[0] != '-' for expr in sol.values()])  # negative value
        self.init_solutions = max(init_solutions, key=estimate)

        while self.is_updated and self.rounds < self.upper_bound:
            self.rounds += 1
            logger.debug(f"Round {self.rounds}")
            self.is_updated = False
            self.equations = [item for sublist in list(self.node_value_equations_dict.values()) for item in sublist]
            action_list = []
            added_equations = []

            candidate_models = get_candidate_models_from_pool(self.model_pool, self.global_graph)
            if is_debugging():
                actions, equations = self.process_model_chunk(candidate_models)
                action_list.extend(actions)
                added_equations.extend(equations)
            else:
                # 将 candidate_models 分成 max_workers 等份
                chunk_size = (len(candidate_models) + max_workers - 1) // max_workers  # 计算每个线程处理的模型数，向上取整
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

            if len(action_list) > 0:
                self.is_updated = True
                logger.debug(f"Actions List: {action_list}")
                self.execute_actions(action_list)

            if len(added_equations) > 0:
                added_equations = list(set(added_equations))
                self.model_instance_eq_num[2] += len(added_equations)
                logger.debug(f"Equations Added from Models ({len(added_equations)}):\n{added_equations}")
                self.model_equations.extend(added_equations)

            self.equations.extend(self.model_equations)
            self.equations = list(set(self.equations))
            self.solve_equations()

            self.target_node_values = self.check_and_evaluate_targets()
            self.model_instance_eq_num[1] = len(self.matched_relations)
            if len(self.target_node_values) > 0:
                self.answer = self.replace_and_evaluate(self.global_graph.target_equation)
                return
