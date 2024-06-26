import argparse
import json
import sys

from func_timeout import func_timeout, FunctionTimedOut
from sympy import N

from GeoDRL.converter import Text2Logic, Logic2Graph
from GeoDRL.logic_solver import LogicSolver
from reasoner.graph_matching import load_models_from_json, get_candidate_models_from_pool, match_graphs
from reasoner.logic_graph import GlobalGraph
from reasoner.graph_solver import GraphSolver
from reasoner.utils import json_to_gml, draw_graph_from_gml
from reasoner import config


def get_logic_forms(q_id):
    diagram_logic_forms_json = json.load(open(config.diagram_logic_forms_json_path, 'r'))
    text_logic_forms_json = json.load(open(config.text_logic_forms_json_path, 'r'))
    text = diagram_logic_forms_json[str(q_id)]
    text["logic_forms"] = text.pop("diagram_logic_forms")
    text["logic_forms"].extend(text_logic_forms_json[str(q_id)]["text_logic_forms"])

    return text


def get_global_graph(logic_forms, draw_graph=False):
    parser, target = Text2Logic(logic_forms)
    print("Target: ", target)
    solver = LogicSolver(parser.logic)
    solver.initSearch()
    graph_json = Logic2Graph(solver.logic, target)

    if draw_graph:
        graph_gml = json_to_gml(graph_json, False)
        draw_graph_from_gml(graph_gml)

    return GlobalGraph.from_json(graph_json)


def solve_question(q_id):
    logic_forms = get_logic_forms(q_id)
    global_graph = get_global_graph(logic_forms)
    model_pool = load_models_from_json(json.load(open(config.model_pool_path, 'r')))

    graph_solver = GraphSolver(global_graph, model_pool)
    target_node_values, answer, rounds = graph_solver.solve()
    print("Total Rounds: ", rounds)
    print("Target Node Value(s): ", target_node_values)
    if target_node_values is not None:
        target_node_values_float = [{key: N(value)} for d in target_node_values for key, value in d.items()]
        print("Target Node Value(s) (Float): ", target_node_values_float)
    print("Answer: ", answer)


def test_graph_matching(q_id):
    logic_forms = get_logic_forms(q_id)
    global_graph = get_global_graph(logic_forms)
    model_pool = load_models_from_json(json.load(open(config.model_pool_test_path, 'r')))

    candidate_models = get_candidate_models_from_pool(model_pool, global_graph)
    for model in candidate_models:
        relations = []
        mapping_dict_list = match_graphs(model, global_graph)
        for mapping_dict in mapping_dict_list:
            relation = model.generate_relation(mapping_dict)
            if relation not in relations:
                relations.append(relation)
                print(mapping_dict)
                print(model.generate_relation(mapping_dict))


def test_draw_global_graph(q_id):
    logic_forms = get_logic_forms(q_id)
    global_graph = get_global_graph(logic_forms, True)


def is_debugging():
    return sys.gettrace() is not None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve a specific question by number.")
    parser.add_argument('question_id', type=int, help='The id of the question to solve')
    try:
        args = parser.parse_args()
        q_id = args.question_id

        if is_debugging():
            solve_question(q_id)
        else:
            try:
                # 设置超时时间为60秒
                func_timeout(60, solve_question, args=(q_id,))
            except FunctionTimedOut:
                print(f"Error: solve_question timed out after 60 seconds")

        # 测试模型匹配
        # test_graph_matching(q_id)
        # 绘制全局图
        # test_draw_global_graph(q_id)
    except argparse.ArgumentError:
        print("Error: question_number is required")
        parser.print_help()
        sys.exit(1)
