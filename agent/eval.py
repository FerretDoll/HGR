import time
import os
import json
import argparse
import sympy
from tqdm import tqdm, trange
import copy
import numpy as np
import torch

from GeoDRL.logic_solver import LogicSolver
from GeoDRL.extended_definition import ExtendedDefinition
from GeoDRL.logic_parser import LogicParser
from GeoDRL.converter import Logic2Graph
from func_timeout import func_timeout, FunctionTimedOut

from agent.graph_dataset import __preprocess_item
from agent.model.graphtransformer.model import GraphormerEncoder
from agent.model.graphtransformer.model_args import ModelArgs
from agent.gen_vocab import reparse_graph_data

import random
random.seed(0)

EPSILON = 1e-5

invalid_actions = [0] + list(range(24,30))
map_dict = {}

def isLetter(ch):
    return ch.upper() and len(ch) == 1

def parse_logic_forms(parser, diagram_parser, text_parser):
    ## Define diagram primitive elements
    parser.logic.point_positions = diagram_parser['point_positions']
    parser.logic.define_point([p for p in parser.logic.point_positions if isLetter(p)])
    if parser.logic.debug:
        print(parser.logic.point_positions)

    lines = diagram_parser['line_instances']  # ['AB', 'AC', 'AD', 'BC', 'BD', 'CD']
    for line in lines:
        line = line.strip()
        if len(line) == 2 and isLetter(line[0]) and isLetter(line[1]):
            parser.logic.define_line(line[0], line[1])

    circles = diagram_parser['circle_instances']  # ['O']
    for point in circles:
        parser.logic.define_circle(point)

    ## Parse diagram logic forms
    logic_forms = diagram_parser['diagram_logic_forms']
    logic_forms = sorted(logic_forms, key=lambda x: x.find("Perpendicular") != -1)  # put 'Perpendicular' to the end

    for logic_form in logic_forms:
        if logic_form.strip() != "":
            if parser.logic.debug:
                print("The diagram logic form is", logic_form)
            try:
                res = parser.parse(logic_form) # e.g., ['Equals', ['LengthOf', ['Line', 'A', 'C']], '10']
                parser.dfsParseTree(res)
            except Exception as e:
                print(e)
                # print("\033[0;0;41mError:\033[0m", repr(e))
                pass

    ## Parse text logic forms
    target = None
    text_logic_forms = text_parser["text_logic_forms"]
    for text in text_logic_forms:
        if parser.logic.debug:
            print("The text logic form is", text)
        if text.find('Find') != -1:
            target = parser.findTarget(parser.parse(text))  # ['Value', 'A', 'C']
        else:
            res = parser.parse(text)
            parser.dfsParseTree(res)
    
    return parser, target

def check_answer(answer, candidate_value_list, gt_id):
    if isinstance(answer, sympy.Basic):
        answer = answer.evalf()
    try:
        if all([x is not None for x in candidate_value_list]) and \
                    abs(float(candidate_value_list[gt_id]) - answer) == min([abs(float(x) - answer) for x in candidate_value_list]):
            return True
    except:
        pass
    return False

def theorem_pred(solver, target, model, step):
    global map_dict
    graph_data = Logic2Graph(solver.logic, target)
    graph_data, map_dict = reparse_graph_data(graph_data, map_dict)

    node_type_vocab_file = './vocab/node_type_vocab.txt'
    node_attr_vocab_file = './vocab/node_attr_vocab.txt'
    edge_attr_vocab_file = './vocab/edge_attr_vocab.txt'
    node_type_vocab = {line.strip():i for i,line in enumerate(open(node_type_vocab_file,'r').readlines())}
    node_attr_vocab = {line.strip():i for i,line in enumerate(open(node_attr_vocab_file,'r').readlines())}
    edge_attr_vocab = {line.strip():i for i,line in enumerate(open(edge_attr_vocab_file,'r').readlines())}
    single_test_data = __preprocess_item(item = graph_data, node_type_vocab=node_type_vocab, node_attr_vocab = node_attr_vocab, edge_attr_vocab=edge_attr_vocab, spatial_pos_max=1)
    for k,v in single_test_data.items():
        single_test_data[k] = v.unsqueeze(0).cuda()
    output_logits = model(single_test_data)

    # mannual rules
    for i in invalid_actions:
        output_logits[0][i] += -1000.0
    if step == 1 and solver.logic.find_all_circles() != []:
        output_logits[0][1] += (output_logits.max(dim=-1)[0][0] - output_logits.min(dim=-1)[0][0])


    score = torch.softmax(output_logits, dim=-1).squeeze(0)
    sorted_score = torch.sort(score, descending=True)
    sorted_score_dict = {k.cpu().item(): v.cpu().item() for k,v in zip(sorted_score[1], sorted_score[0])}
    # pred = torch.argmax(score, dim=-1).cpu().item()
    return sorted_score_dict

def beam_search(entry, solver, target, model, max_step, beam_size):
    t = 0
    hypotheses = [solver]
    hyp_steps = [[]]
    hyp_scores = [0.]

    while(t < max_step):
        t += 1
        hyp_num = len(hypotheses)
        assert hyp_num <= beam_size, f"hyp_num: {hyp_num}, beam_size: {beam_size}"

        hyp_theorem = []
        conti_hyp_scores = []
        conti_hyp_steps = []
        for hyp_index, hyp in enumerate(hypotheses):
            sorted_score_dict = theorem_pred(hyp, target, model, t)
            # print("step:", t , "past_steps:", hyp_steps[hyp_index], sorted_score_dict)
            for i in range(beam_size):
                cur_score = list(sorted_score_dict.values())[i]
                if cur_score < EPSILON: continue
                hyp_theorem.append([hyp, list(sorted_score_dict.keys())[i]])
                conti_hyp_scores.append(hyp_scores[hyp_index] + np.log(cur_score))
                conti_hyp_steps.append(hyp_steps[hyp_index] + [list(sorted_score_dict.keys())[i]])
        
        conti_hyp_scores = torch.Tensor(conti_hyp_scores)
        top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(conti_hyp_scores, k=min(beam_size,conti_hyp_scores.size(0)))

        new_hypotheses = []
        new_hyp_scores = []
        new_hyp_steps = []

        for cand_hyp_id, cand_hyp_score in zip(top_cand_hyp_pos, top_cand_hyp_scores):
            new_score = cand_hyp_score.detach().item()
            prev_hyp, theorem = hyp_theorem[cand_hyp_id]
            now_steps = conti_hyp_steps[cand_hyp_id]

            Update = False
            now_hyp = copy.deepcopy(prev_hyp)
            now_hyp.equations = []
            changed = now_hyp.function_maps[theorem]()
            if changed is not None and changed:
                Update = True
            Update = Update or len(now_hyp.equations) > 0
            if not Update:
                continue
            now_hyp.Solve_Equations()
            now_answer = now_hyp._getAnswer(target)
            if now_answer is not None:
                entry["step_lst"] = now_steps
                entry["answer"] = now_answer
                return entry

            new_hypotheses.append(now_hyp)
            new_hyp_scores.append(new_score)
            new_hyp_steps.append(now_steps)
            
        hypotheses = new_hypotheses
        hyp_scores = new_hyp_scores
        hyp_steps = new_hyp_steps

    return None

def solve_with_time(id, model, max_step = 10, beam_size = 5):
    try:
        entry = func_timeout(300, solve, kwargs=dict(id = id, model=model, max_step=max_step, beam_size=beam_size))
        return entry
    except:
        return {"id": id, "target": None, "answer": None,  "step_lst": [], "correctness": "no", "time": None}

def solve(id, model, max_step = 10, beam_size = 5):
    entry = {"id": id, "target": None, "answer": None,  "step_lst": [], "correctness": "no", "time": None}
    s_time = time.time()

    id = str(id)
    if id not in diagram_logic_table or id not in text_logic_table:
        return entry
    diagram_parser=diagram_logic_table[id]
    text_parser=text_logic_table[id]

    parser = LogicParser(ExtendedDefinition(debug=False))

    try:
        parser, target = parse_logic_forms(parser, diagram_parser, text_parser)
        entry["target"] = target
    except:
        print(f"{id} parse error!")
        entry['time'] = str(time.time()-s_time)
        return entry

    if int(id) < 2101: split = 'train'
    elif int(id) < 2401: split = 'val'
    elif int(id) < 3003: split = 'test'
    ANSWER_INPUT_PATH = os.path.join(DATA_INPUT_PATH, split, id, "data.json")
    with open(ANSWER_INPUT_PATH, "r") as f:
        data = json.load(f)
    candidate_value_list = data['precise_value']  # ! e.g., [5.0, 12.0, 13.0, 26.0]
    gt_id = ord(data['answer']) - 65  # e.g., 0

    solver = LogicSolver(parser.logic, target)

    try:
        solver.initSearch()
        solver.Solve_Equations()
        now_answer = solver._getAnswer(target)
        if now_answer is not None:
            if check_answer(now_answer, candidate_value_list, gt_id):
                entry["correctness"] = "yes"
            entry['answer'] = now_answer
            entry['time'] = str(time.time()-s_time)
            return entry
    except:
        return entry

    try:
        entry = func_timeout(300, beam_search, kwargs=dict(entry=entry, solver=solver, target=target, model=model, max_step=max_step, beam_size=beam_size))
        if entry:
            if check_answer(entry['answer'], candidate_value_list, gt_id):
                entry["correctness"] = "yes"
            entry['time'] = str(time.time()-s_time)
            return entry
    except:
        entry['time'] = str(time.time()-s_time)
        return entry

parser = argparse.ArgumentParser()
parser.add_argument("--use_annotated", action="store_true", help="use annotated data or generated data")
parser.add_argument("--model_path", type=str, help="model weight path")
parser.add_argument("--output_path", type=str, help="output path of result json file")
parser.add_argument("--beam_size", type=int, help="beam size for search")
args = parser.parse_args()

DATA_INPUT_PATH = '../../data/geometry3k'
if args.use_annotated:
    DIAGRAM_INPUT_PATH = '../../data/geometry3k/logic_forms/diagram_logic_forms_annot.json'
    TEXT_INPUT_PATH = '../../data/geometry3k/logic_forms/text_logic_forms_annot_dissolved.json'
else:
    DIAGRAM_INPUT_PATH = '../parser/diagram_parser/diagram_logic_forms_pred.json'
    TEXT_INPUT_PATH = '../parser/text_parser/text_logic_forms_pred.json'
with open(DIAGRAM_INPUT_PATH, "r") as f1:
    diagram_logic_table = json.load(f1)
with open(TEXT_INPUT_PATH, "r") as f2:
    text_logic_table = json.load(f2)


def eval(model_save_path, json_output_name, beam_size):
    node_type_vocab_file = './vocab/node_type_vocab.txt'
    node_attr_vocab_file = './vocab/node_attr_vocab.txt'
    edge_attr_vocab_file = './vocab/edge_attr_vocab.txt'
    node_type_vocab = {line.strip():i for i,line in enumerate(open(node_type_vocab_file,'r').readlines())}
    node_attr_vocab = {line.strip():i for i,line in enumerate(open(node_attr_vocab_file,'r').readlines())}
    edge_attr_vocab = {line.strip():i for i,line in enumerate(open(edge_attr_vocab_file,'r').readlines())}
    model_args = ModelArgs(num_classes = 30, max_nodes = 256, num_node_type=len(node_type_vocab), num_node_attr=len(node_attr_vocab), num_in_degree=256, num_out_degree=256,
                        num_edges=len(edge_attr_vocab), num_spatial=20, num_edge_dis=256, edge_type="one_hop", multi_hop_max_dist=1)
    model = GraphormerEncoder(model_args).cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    st = 2401
    ed = 3002
    total = ed - st
    correct = 0
    solved = 0
    st_time = time.time()
    search_result_json_dict = {}

    for id in trange(st, ed):
        try:
            entry = func_timeout(300, solve, kwargs=dict(id=id, model=model, max_step=10, beam_size=beam_size))
        except:
            continue
        
        if entry:
            if entry['answer'] != None:
                solved += 1
                for k,v in entry.items():
                    entry[k] = str(v)
                search_result_json_dict[entry["id"]] = entry

                if entry['correctness'] == "yes":
                    correct += 1   
    
    ed_time = time.time()

    print(f"Solved: {solved}, Correctness: {correct}, CorrectRate: {correct*1.0/total}")
    print(f"Time Cost: {ed_time-st_time} seconds.")
    json.dump(search_result_json_dict, open(json_output_name+'correct'+str(correct)+'.json','w'))


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    print("Use annotated: ", args.use_annotated)
    eval(model_save_path = args.model_path,
        json_output_name = args.output_path,
        beam_size = args.beam_size)
    


        