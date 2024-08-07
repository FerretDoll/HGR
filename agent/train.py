import os
import random
import math
import json
import sys
import numpy as np
import argparse
import pickle as pkl
import copy
import torch
import torch.nn as nn
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from collections import namedtuple, deque

from agent.graph_dataset import onehop_collate_fn, __preprocess_item
from agent.gen_vocab import reparse_graph_data
from agent.model.graphtransformer.model import GraphormerEncoder
from agent.model.graphtransformer.model_args import ModelArgs

from GeoDRL.logic_solver import LogicSolver
from GeoDRL.extended_definition import ExtendedDefinition
from GeoDRL.logic_parser import LogicParser
from GeoDRL.converter import Logic2Graph
from func_timeout import func_timeout, FunctionTimedOut

output_path = 'saves/RL'
log_path = 'logs'
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)

writer = SummaryWriter(log_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32, help="number of samples in one batch during RL training.")
parser.add_argument("--gamma", type=float, default=0.5, help="decay factor of RL model.")
parser.add_argument("--beam_size", type=int, default=5, help="beam size for search")
parser.add_argument("--lr", type=float, default=3e-5, help="learning rate for optimizer")
args = parser.parse_args()
BATCH_SIZE = args.batch_size
GAMMA = args.gamma

action_space = list(range(1, 24))
invalid_actions = [0] + list(range(24, 30))

DATA_INPUT_PATH = '../../data/geometry3k'
DIAGRAM_INPUT_PATH = '../../data/geometry3k/logic_forms/diagram_logic_forms_annot.json'
TEXT_INPUT_PATH = '../../data/geometry3k/logic_forms/text_logic_forms_annot_dissolved.json'

with open(DIAGRAM_INPUT_PATH, "r") as f1:
    diagram_logic_table = json.load(f1)
with open(TEXT_INPUT_PATH, "r") as f2:
    text_logic_table = json.load(f2)

node_type_vocab_file = './vocab/node_type_vocab.txt'
node_attr_vocab_file = './vocab/node_attr_vocab.txt'
edge_attr_vocab_file = './vocab/edge_attr_vocab.txt'
node_type_vocab = {line.strip(): i for i, line in enumerate(open(node_type_vocab_file, 'r').readlines())}
node_attr_vocab = {line.strip(): i for i, line in enumerate(open(node_attr_vocab_file, 'r').readlines())}
edge_attr_vocab = {line.strip(): i for i, line in enumerate(open(edge_attr_vocab_file, 'r').readlines())}

model_update_steps = 0
INIT_MODEL = None
model_args = ModelArgs(num_classes=30, max_nodes=256, num_node_type=len(node_type_vocab),
                       num_node_attr=len(node_attr_vocab), num_in_degree=256, num_out_degree=256,
                       num_edges=len(edge_attr_vocab), num_spatial=20, num_edge_dis=256, edge_type="one_hop",
                       multi_hop_max_dist=1)
policy_net = GraphormerEncoder(model_args).cuda()
policy_net = torch.nn.DataParallel(policy_net)
if INIT_MODEL:
    policy_net.load_state_dict(torch.load(INIT_MODEL))

optimizer = torch.optim.AdamW(policy_net.parameters(), args.lr)
memory = ReplayMemory(10000)
map_dict = {}

def beam_search_for_RL(solver, target, model, max_step, beam_size, EPS):
    def theorem_pred(graph_solver, model):
        global map_dict
        graph_data = graph_solver.global_graph.to_dict()
        graph_data, map_dict = reparse_graph_data(graph_data, map_dict)

        single_test_data = __preprocess_item(item=graph_data, node_type_vocab=node_type_vocab,
                                             node_attr_vocab=node_attr_vocab, edge_attr_vocab=edge_attr_vocab,
                                             spatial_pos_max=1)
        for k, v in single_test_data.items():
            single_test_data[k] = v.unsqueeze(0).cuda()
        output_logits = model(single_test_data)
        score = torch.softmax(output_logits, dim=-1).squeeze(0)
        sorted_score = torch.sort(score, descending=True)
        sorted_score_dict = {k.cpu().item(): v.cpu().item() for k, v in zip(sorted_score[1], sorted_score[0])}
        return sorted_score_dict

    tmp_memory = []

    t = 0
    hypotheses = [solver]
    hyp_steps = [[]]
    hyp_scores = [0.]

    while t < max_step:
        t += 1
        hyp_num = len(hypotheses)
        assert hyp_num <= beam_size, f"hyp_num: {hyp_num}, beam_size: {beam_size}"

        hyp_theorem = []
        conti_hyp_scores = []
        conti_hyp_steps = []
        for hyp_index, hyp in enumerate(hypotheses):
            sorted_score_dict = theorem_pred(hyp, model)
            for i in range(beam_size):
                cur_score = list(sorted_score_dict.values())[i]
                cur_theorem = list(sorted_score_dict.keys())[i]
                if random.random() <= EPS:
                    cur_theorem = random.choice(action_space)
                if cur_score < 1e-5:
                    continue
                hyp_theorem.append([hyp, cur_theorem])
                conti_hyp_scores.append(hyp_scores[hyp_index] + np.log(cur_score))
                conti_hyp_steps.append(hyp_steps[hyp_index] + [cur_theorem])

        conti_hyp_scores = torch.Tensor(conti_hyp_scores)
        top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(conti_hyp_scores, k=min(beam_size, conti_hyp_scores.size(0)))

        new_hypotheses = []
        new_hyp_scores = []
        new_hyp_steps = []

        for cand_hyp_id, cand_hyp_score in zip(top_cand_hyp_pos, top_cand_hyp_scores):
            new_score = cand_hyp_score.detach().item()
            prev_hyp, theorem = hyp_theorem[cand_hyp_id]
            now_steps = conti_hyp_steps[cand_hyp_id]

            state = generate_state(prev_hyp)
            Update = False
            now_hyp = copy.deepcopy(prev_hyp)
            now_hyp.equations = []
            try:
                t1 = time.time()
                changed = now_hyp.function_maps[theorem]()
                if changed is not None and changed:
                    Update = True
                Update = Update or len(now_hyp.equations) > 0
                if not Update:
                    reward = -1.0
                    tmp_memory.append((state, theorem, state, reward))
                    continue
                now_hyp.Solve_Equations()
                now_answer = now_hyp._getAnswer(target)
                t2 = time.time()
                time_cost = t2 - t1
            except:
                tmp_memory.append((state, theorem, None, -1.0))
            next_state = generate_state(now_hyp, target)
            reward = - (1 - math.exp(-1. * time_cost / 60.0))
            if now_answer is not None:
                reward += 1.0
                tmp_memory.append((state, theorem, None, reward))
                return now_answer, tmp_memory
            else:
                tmp_memory.append((state, theorem, next_state, reward))

            new_hypotheses.append(now_hyp)
            new_hyp_scores.append(new_score)
            new_hyp_steps.append(now_steps)

        hypotheses = new_hypotheses
        hyp_scores = new_hyp_scores
        hyp_steps = new_hyp_steps

    return None, tmp_memory


def generate_state(graph_solver):
    global map_dict
    graph_data = graph_solver.global_graph.to_dict()
    graph_data, map_dict = reparse_graph_data(graph_data, map_dict)
    state = __preprocess_item(item=graph_data, node_type_vocab=node_type_vocab, node_attr_vocab=node_attr_vocab,
                              edge_attr_vocab=edge_attr_vocab, spatial_pos_max=1)
    for k, v in state.items():
        state[k] = v.unsqueeze(0).cpu()
    del state['edge_input']
    return state


def solve_with_question(id, seq=None, max_steps=10, EPS=0.1):
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
                    res = parser.parse(logic_form)  # e.g., ['Equals', ['LengthOf', ['Line', 'A', 'C']], '10']
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

    id = str(id)
    diagram_parser = diagram_logic_table[id]
    text_parser = text_logic_table[id]

    parser = LogicParser(ExtendedDefinition(debug=False))

    try:
        parser, target = parse_logic_forms(parser, diagram_parser, text_parser)
    except:
        # print(f"{id} parse error!")
        return None

    solver = LogicSolver(parser.logic, target)

    try:
        solver.initSearch()
        solver.Solve_Equations()
        now_answer = solver._getAnswer(target)
        if now_answer is not None:
            return now_answer, []
    except:
        # print(f"{id} init error!")
        return None

    # return None, tmp_memory
    now_answer, tmp_memory = beam_search_for_RL(solver, target, policy_net, max_steps, args.beam_size, EPS)
    return now_answer, tmp_memory


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    policy_net.train()
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = {}
    try:
        for k in batch.state[0].keys():
            if k == 'edge_input': continue
            state_batch[k] = [single_state[k].squeeze(0).cpu() for single_state in batch.state]
    except:
        print("ERROR! Batch:", batch)
        return
    state_batch = onehop_collate_fn([state_batch['x'], state_batch['node_attr'], state_batch['target_attr'], \
                                     state_batch['attn_bias'], state_batch['attn_edge_type'],
                                     state_batch['spatial_pos'], \
                                     state_batch['in_degree'], state_batch['out_degree'], None,
                                     [torch.Tensor(0)] * BATCH_SIZE], zipped=True)
    for k in state_batch.keys():
        state_batch[k] = state_batch[k].cuda()

    action_batch = torch.LongTensor(batch.action).view(BATCH_SIZE, 1).cuda()
    reward_batch = torch.Tensor(batch.reward).view(BATCH_SIZE).cuda()

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool).cuda()
    non_final_next_states = {}
    for k in batch.state[0].keys():
        if k == 'edge_input': continue
        non_final_next_states[k] = [single_state[k].squeeze(0).cpu() for single_state in batch.next_state if
                                    single_state is not None]
    non_final_next_states = onehop_collate_fn(
        [non_final_next_states['x'], non_final_next_states['node_attr'], non_final_next_states['target_attr'], \
         non_final_next_states['attn_bias'], non_final_next_states['attn_edge_type'],
         non_final_next_states['spatial_pos'], \
         non_final_next_states['in_degree'], non_final_next_states['out_degree'], None, [torch.Tensor(0)] * BATCH_SIZE],
        zipped=True)
    for k in non_final_next_states.keys():
        non_final_next_states[k] = non_final_next_states[k].cuda()

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE).cuda()
    with torch.no_grad():
        next_state_values[non_final_mask] = policy_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    nn.utils.clip_grad_norm_(policy_net.parameters(), 1, norm_type=2)
    optimizer.step()
    global model_update_steps
    model_update_steps += 1
    writer.add_scalar('train_loss', loss.item(), global_step=model_update_steps)

    if model_update_steps % 1000 == 0:
        torch.save(policy_net.state_dict(), output_path + "/graph_model_RL_step" + str(model_update_steps) + ".pt")


def pre_store_data():
    num_epoch = 1
    st = 0
    ed = 2101
    data = []
    for epoch in trange(num_epoch):
        for id in trange(st, ed):
            id = str(id)
            if id not in diagram_logic_table or id not in text_logic_table:
                continue
            try:
                answer, tmp_memory = func_timeout(120, solve_with_question, kwargs=dict(id=id))
                for _ in tmp_memory:
                    memory.push(*_)
                print(len(memory))
            except:
                pass

    pkl.dump(memory, open("Memory.pkl", 'wb'))


def train_loop():
    max_steps = 1000000
    while (model_update_steps < max_steps):
        st = 0
        ed = 2101
        for id in trange(st, ed):
            id = str(id)
            if id not in diagram_logic_table or id not in text_logic_table:
                continue

            try:
                res = func_timeout(120, solve_with_question, kwargs=dict(id=id))
                answer, tmp_memory = res
                for _ in tmp_memory:
                    memory.push(*_)
            except:
                continue

            optimize_model()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    pre_store_data()
    # memory = pkl.load(open("Memory.pkl", 'rb'))
    train_loop()
