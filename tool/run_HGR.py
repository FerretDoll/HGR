import argparse
import copy
import json
import os
import gc
import sys
import time
import multiprocessing
import logging
import logging.handlers
from datetime import datetime

import numpy as np
from func_timeout import func_timeout, FunctionTimedOut
from sympy import N
from tqdm import tqdm

from GeoDRL.converter import Text2Logic, Logic2Graph
from GeoDRL.logic_solver import LogicSolver
from reasoner.graph_matching import load_models_from_json, get_candidate_models_from_pool, match_graphs, get_model
from reasoner.hologram import GlobalHologram
from reasoner.graph_solver import GraphSolver
from reasoner.utils import dict_to_gml, draw_graph_from_gml, is_debugging
from reasoner.config import logger, model_pool_path
from reasoner import config


with open(config.diagram_logic_forms_json_path, 'r') as diagram_file:
    diagram_logic_forms_json = json.load(diagram_file)
with open(config.text_logic_forms_json_path, 'r') as text_file:
    text_logic_forms_json = json.load(text_file)
with open(model_pool_path, 'r') as model_pool_file:
    model_pool, model_id_map = load_models_from_json(json.load(model_pool_file))


def get_logic_forms(q_id):
    text = copy.deepcopy(diagram_logic_forms_json[str(q_id)])
    text["logic_forms"] = text.pop("diagram_logic_forms")
    text["logic_forms"].extend(text_logic_forms_json[str(q_id)]["text_logic_forms"])

    return text


def get_global_graph(parser, target, draw_graph=False):
    logger.debug("Target: %s", target)
    solver = LogicSolver(parser.logic)
    solver.initSearch()
    graph_dict = Logic2Graph(solver.logic, target)

    if draw_graph:
        graph_gml = dict_to_gml(graph_dict, False)
        draw_graph_from_gml(graph_gml)

    return GlobalHologram.from_dict(graph_dict)


def get_graph_solver(q_id):
    logic_forms = get_logic_forms(q_id)
    parser, target = Text2Logic(logic_forms)
    global_graph = get_global_graph(parser, target)
    graph_solver = GraphSolver(global_graph)

    return graph_solver, target


def solve_with_model_sequence(q_id, model_id_list):
    res = {"id": q_id, "target": None, "answer": None, "step_lst": None, "model_instance_eq_num": None,
           "correctness": "no", "time": None}
    s_time = time.time()
    try:
        data_path = os.path.join(config.db_dir_single, str(q_id), "data.json")
        with open(data_path, "r") as f:
            data = json.load(f)
        candidate_value_list = data['precise_value']
        gt_id = ord(data['answer']) - 65  # Convert A-D to 0-3

        graph_solver, target = get_graph_solver(q_id)
        models = []
        for model_id in model_id_list:
            model = get_model(model_pool, model_id_map, model_id)
            models.append(model)
        graph_solver.solve_with_model_sequence(models)
        logger.debug("Target Node Value(s): %s", graph_solver.target_node_values)
        if len(graph_solver.target_node_values) > 0:
            target_node_values_float = [{key: N(value)} for d in graph_solver.target_node_values for
                                        key, value in d.items()]
            logger.debug("Target Node Value(s) (Float): %s", target_node_values_float)
        answer = graph_solver.answer

        res["step_lst"] = graph_solver.matched_model_list
        res["model_instance_eq_num"] = graph_solver.model_instance_eq_num
        correctness, answer = check_transformed_answer(answer, candidate_value_list, gt_id)
        if correctness:
            res["correctness"] = "yes"

        res["target"] = target
        res["answer"] = answer
        logger.debug("Answer: %s", answer)
        res['time'] = str(time.time() - s_time)
    except Exception as e:
        logger.error(e)
        res['time'] = str(time.time() - s_time)
        return res

    return res


def solve_question(q_id):
    res = {"id": str(q_id), "target": None, "answer": None, "step_lst": None, "model_instance_eq_num": None,
           "correctness": "no", "time": None}
    s_time = time.time()
    try:
        data_path = os.path.join(config.db_dir_single, str(q_id), "data.json")
        with open(data_path, "r") as f:
            data = json.load(f)
        candidate_value_list = data['precise_value']
        gt_id = ord(data['answer']) - 65  # Convert A-D to 0-3

        graph_solver, target = get_graph_solver(q_id)
        # graph_solver.solve()
        graph_solver.solve_with_candi_models()
        logger.debug("Total Rounds: %s", graph_solver.rounds)
        logger.debug("Target Node Value(s): %s", graph_solver.target_node_values)
        if len(graph_solver.target_node_values) > 0:
            target_node_values_float = [{key: N(value)} for d in graph_solver.target_node_values for
                                        key, value in d.items()]
            logger.debug("Target Node Value(s) (Float): %s", target_node_values_float)
        answer = graph_solver.answer

        res["step_lst"] = graph_solver.matched_model_list
        res["model_instance_eq_num"] = graph_solver.model_instance_eq_num
        if answer is not None:
            correctness, transformed_answer = check_transformed_answer(answer, candidate_value_list, gt_id)
            if correctness:
                res["correctness"] = "yes"
                res["answer"] = transformed_answer
            else:
                res["answer"] = answer

        res["target"] = target
        logger.debug("Answer: %s", answer)
        res['time'] = str(time.time() - s_time)
    except Exception as e:
        logger.error(e)
        res['time'] = str(time.time() - s_time)
        return res

    # Clean up resources
    del graph_solver
    del target
    del candidate_value_list
    gc.collect()

    return res


def log_listener_process(log_queue, log_file):
    """Process to listen to logging messages and write them to a file."""
    logger = logging.getLogger()
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    while True:
        try:
            record = log_queue.get()
            if record is None:
                break
            logger.handle(record)
        except Exception:
            import traceback
            print('Error in log listener:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def worker(q_id, log_queue, time_limit=180):
    """Worker function to process each question."""
    logger = logging.getLogger(f'Worker-{q_id}')
    logger.setLevel(logging.DEBUG)

    try:
        res = func_timeout(time_limit, solve_question, args=(q_id,))

        # Create log records and place them in the log queue
        log_record = logger.makeRecord(
            logger.name, logging.DEBUG, __file__, 0,
            res, None, None
        )
        log_queue.put(log_record)
    except FunctionTimedOut:
        # Handle timeout exceptions and record logs
        log_record = logger.makeRecord(
            logger.name, logging.ERROR, __file__, 0,
            f"Error occurred while solving question {q_id}: FunctionTimedOut.", None, None
        )
        log_queue.put(log_record)
    except Exception as e:
        # Handle other exceptions and record logs
        log_record = logger.makeRecord(
            logger.name, logging.ERROR, __file__, 0,
            f"Error occurred while solving question {q_id}: {e}", None, None
        )
        log_queue.put(log_record)


def evaluate_all_questions(st, ed):
    with open(config.error_ids_path, 'r') as file:
        error_ids = {int(line.strip()) for line in file}  # Ensure that the error ID is an integer

    all_question_ids = set(range(st, ed))
    valid_question_ids = list(all_question_ids - error_ids)
    total = len(valid_question_ids)
    removed_count = len(all_question_ids) - total

    print(f"Removed {removed_count} questions with parsing errors.")

    # Log queue and log listening process
    log_queue = multiprocessing.Queue()
    log_listener = multiprocessing.Process(target=log_listener_process, args=(log_queue, 'output/eval.log'))
    log_listener.start()

    processes = []
    for q_id in tqdm(valid_question_ids):
        p = multiprocessing.Process(target=worker, args=(q_id, log_queue))
        p.start()
        processes.append(p)
        if len(processes) >= 4:
            for proc in processes:
                proc.join()
            processes = []

    for proc in processes:
        proc.join()

    # Stop the log listening process
    log_queue.put_nowait(None)
    log_listener.join()


def check_answer(answer, candidate_value_list, gt_id):
    if answer is None:
        return False
    try:
        if (all([x is not None for x in candidate_value_list]) and
                abs(float(candidate_value_list[gt_id]) - answer) == min([abs(float(x) - answer)
                                                                         for x in candidate_value_list])):
            return True
    except Exception as e:
        logger.error(e)
    return False


def check_transformed_answer(answer, candidate_value_list, gt_id):
    if answer is not None:
        if check_answer(answer, candidate_value_list, gt_id):
            return True, answer
        else:
            # It may be necessary to convert radians to degrees before verifying the answer
            answer_degrees = np.degrees(float(answer))
            if check_answer(answer_degrees, candidate_value_list, gt_id):
                answer = answer_degrees
                return True, answer
            elif check_answer(360 - answer_degrees, candidate_value_list, gt_id):
                answer = 360 - answer_degrees
                return True, answer

            return False, None
    else:
        return False, None


def test_graph_matching(q_id):
    logic_forms = get_logic_forms(q_id)
    parser, target = Text2Logic(logic_forms)
    global_graph = get_global_graph(parser, target)
    with open(config.model_pool_test_path, 'r') as model_pool_file:
        model_pool, _ = load_models_from_json(json.load(model_pool_file))

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
    parser, target = Text2Logic(logic_forms)
    _ = get_global_graph(parser, target, True)


def check_id_in_error_ids(question_id, error_file):
    with open(error_file, 'r') as file:
        error_ids = {line.strip() for line in file}

    if str(question_id) in error_ids:
        return True
    else:
        return False


def test_one_question(q_id, time_limit=180):
    if check_id_in_error_ids(q_id, config.error_ids_path):
        logger.error(f"Error: question id {q_id} is in parsing_error_ids")
        sys.exit(1)

    if is_debugging():
        res = solve_question(q_id)
        logger.debug(res)
    else:
        try:
            res = func_timeout(time_limit, solve_question, args=(q_id,))
            logger.debug(res)
        except FunctionTimedOut:
            logger.error(f"Error: solve_question timed out")


def test_solve_with_model_sequence(q_id, model_id_list, time_limit=300):
    if check_id_in_error_ids(q_id, config.error_ids_path):
        logger.error(f"Error: question id {q_id} is in parsing_error_ids")
        sys.exit(1)

    if is_debugging():
        res = solve_with_model_sequence(q_id, model_id_list)
        logger.debug(res)
    else:
        try:
            res = func_timeout(time_limit, solve_with_model_sequence, args=(q_id, model_id_list,))
            logger.debug(res)
        except FunctionTimedOut:
            logger.error(f"Error: solve_question timed out")


if __name__ == "__main__":
    # Test multiple questions
    # evaluate_all_questions(2401, 3001)

    try:
        q_id = 2456

        # Test and answer single questions
        test_one_question(q_id)

        # Test model matching
        # test_graph_matching(q_id)

        # Draw a global map
        # test_draw_global_graph(q_id)

        # test_solve_with_model_sequence(q_id, [2, 66])
    except argparse.ArgumentError:
        logger.error("Error: question id is required")
        sys.exit(1)
