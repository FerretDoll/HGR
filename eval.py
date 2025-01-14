import json
import argparse
import json
import random
import torch
import torch.multiprocessing as mp
from tqdm import trange

from agent.agent_solver import AgentSolver
from reasoner import config
from reasoner.config import eval_logger
from tool.run_HGR import solve_question, evaluate_all_questions

parser = argparse.ArgumentParser()
parser.add_argument("--use_annotated", action="store_true", help="use annotated data or generated data")
parser.add_argument("--use_agent", action="store_true", help="use model selection agent")
parser.add_argument("--model_path", type=str, help="model weight path")
parser.add_argument("--beam_size", type=int, default=5, help="beam size for search")
parser.add_argument('--question_id', type=int, help='The id of the question to solve')
args = parser.parse_args()

random.seed(0)


def solve_process(agent_solver, return_dict, q_id, res):
    try:
        result = agent_solver.solve(q_id=q_id)
        return_dict['result'] = result
    except Exception as e:
        eval_logger.error(f'q_id: {q_id} - Error in solving: {e}')
        return_dict['result'] = res


def solve_heuristic_process(return_dict, q_id, res):
    try:
        result = solve_question(q_id)
        return_dict['result'] = result
    except Exception as e:
        eval_logger.error(f'q_id: {q_id} - Error in heuristic solving: {e}')
        return_dict['result'] = res


def solve_with_time(agent_solver, q_id, time_limit=180):
    q_id = str(q_id)
    res = {"id": q_id, "target": None, "answer": None, "step_lst": [], "model_instance_eq_num": None,
           "correctness": "no", "time": None}

    if q_id not in diagram_logic_forms_json or q_id not in text_logic_forms_json or q_id in error_ids:
        eval_logger.debug(f'q_id: {q_id} - q_id in parsing_error_ids')
        return res

    manager = mp.Manager()
    return_dict = manager.dict()

    p = mp.Process(target=solve_process, args=(agent_solver, return_dict, q_id, res,))
    p.start()

    p.join(time_limit)
    if p.is_alive():
        p.terminate()
        p.join()
        eval_logger.error(f'q_id: {q_id} - Timeout during agent solving')

    res = return_dict.get('result', res)

    if res["correctness"] == "no":
        eval_logger.error(f'q_id: {q_id} - Agent failed, fallback to heuristic strategy')
        p_fallback = mp.Process(target=solve_heuristic_process, args=(return_dict, q_id, res,))
        p_fallback.start()
        p_fallback.join(time_limit)
        if p_fallback.is_alive():
            p_fallback.terminate()
            p_fallback.join()
            eval_logger.error(f'q_id: {q_id} - Timeout during heuristic solving')

        res = return_dict.get('result', res)

    eval_logger.debug(res)
    return res


def eval(agent_solver, st, ed):
    for q_id in trange(st, ed):
        try:
            solve_with_time(agent_solver, q_id, time_limit=300)
        except Exception as e:
            eval_logger.error(f'q_id: {q_id} - Error: {e}')
            continue


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)

    with open(config.error_ids_path, 'r') as file:
        error_ids = {line.strip() for line in file}

    if args.use_annotated:
        print("Use annotated: True")
        with open(config.diagram_logic_forms_json_path, 'r') as diagram_file:
            diagram_logic_forms_json = json.load(diagram_file)
        with open(config.text_logic_forms_json_path, 'r') as text_file:
            text_logic_forms_json = json.load(text_file)
    else:
        print("Use annotated: False")
        with open(config.pred_diagram_logic_forms_json_path, 'r') as diagram_file:
            diagram_logic_forms_json = json.load(diagram_file)
        with open(config.pred_text_logic_forms_json_path, 'r') as text_file:
            text_logic_forms_json = json.load(text_file)

    agent_solver = AgentSolver(args.model_path)

    if args.question_id:
        if args.use_agent:
            solve_with_time(agent_solver, args.question_id, time_limit=300)
        else:
            res = solve_question(args.question_id)
            eval_logger.debug(res)
    else:
        if args.use_agent:
            eval(agent_solver, st=2401, ed=3002)
        else:
            evaluate_all_questions(st=2401, ed=3002)
