import re
import argparse
import gc
import copy
import json
import math
import os
import pickle as pkl
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from func_timeout import func_timeout, FunctionTimedOut
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from collections import namedtuple, deque

from agent.gen_vocab import reparse_graph_data
from agent.graph_dataset import onehop_collate_fn, __preprocess_item
from agent.model.graphtransformer.model import GraphormerEncoder
from agent.model.graphtransformer.model_args import ModelArgs
from reasoner import config
from reasoner.config import train_logger
from reasoner.graph_matching import load_models_from_json, get_model
from tool.run_HGR import get_graph_solver

worker_num = 4
output_path = 'saves/RL'
log_path = 'logs'
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)

action_space = list(range(64))

node_type_vocab_file = 'agent/vocab/node_type_vocab.txt'
node_attr_vocab_file = 'agent/vocab/node_attr_vocab.txt'
edge_attr_vocab_file = 'agent/vocab/edge_attr_vocab.txt'
node_type_vocab = {line.strip(): i for i, line in enumerate(open(node_type_vocab_file, 'r').readlines())}
node_attr_vocab = {line.strip(): i for i, line in enumerate(open(node_attr_vocab_file, 'r').readlines())}
edge_attr_vocab = {line.strip(): i for i, line in enumerate(open(edge_attr_vocab_file, 'r').readlines())}

with open(config.diagram_logic_forms_json_path, 'r') as diagram_file:
    diagram_logic_forms_json = json.load(diagram_file)
with open(config.text_logic_forms_json_path, 'r') as text_file:
    text_logic_forms_json = json.load(text_file)
with open(config.error_ids_path, 'r') as file:
    error_ids = {line.strip() for line in file}
with open(config.model_pool_path, 'r') as model_pool_file:
    model_pool, model_id_map = load_models_from_json(json.load(model_pool_file))
with open('db/model_sequence.json', 'r') as model_sequence:
    model_sequence_json = json.load(model_sequence)

parser = argparse.ArgumentParser()
parser.add_argument("--init_memory", action="store_true", help="use annotated data or generated data")
parser.add_argument("--pre_train", action="store_true", help="use annotated data or generated data")
parser.add_argument("--batch_size", type=int, default=16, help="number of samples in one batch during RL training.")
parser.add_argument("--gamma", type=float, default=0.5, help="decay factor of RL model.")
parser.add_argument("--beam_size", type=int, default=5, help="beam size for search")
parser.add_argument("--lr", type=float, default=3e-5, help="learning rate for optimizer")
args = parser.parse_args()
BATCH_SIZE = args.batch_size
GAMMA = args.gamma

model_update_steps = 0
map_dict = {}


def setup_seed(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


setup_seed(0)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


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


memory = ReplayMemory(10000)


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


def beam_search_for_RL(graph_solver, model, max_step, beam_size, eps):
    tmp_memory = []

    t = 0
    hypotheses = [graph_solver]
    hyp_steps = [[]]
    hyp_scores = [0.]
    conti_hyp_steps = []

    while t < max_step and len(hypotheses) > 0:
        t += 1
        assert len(hypotheses) <= beam_size, f"hyp_num: {len(hypotheses)}, beam_size: {beam_size}"

        hyp_theorem = []
        conti_hyp_scores = []
        conti_hyp_steps = []
        for hyp_index, hyp in enumerate(hypotheses):
            sorted_score_dict = theorem_pred(hyp, model)
            # print("step:", t , "past_steps:", hyp_steps[hyp_index], sorted_score_dict)
            for i in range(beam_size):
                cur_score = list(sorted_score_dict.values())[i]
                cur_theorem = list(sorted_score_dict.keys())[i]
                if np.random.random() <= eps:
                    cur_theorem = np.random.choice(action_space)
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
            now_hyp = copy.deepcopy(prev_hyp)
            t1 = time.time()
            try:
                graph_model = get_model(model_pool, model_id_map, theorem)
                now_hyp.solve_with_one_model(graph_model)
                if not now_hyp.is_updated:
                    reward = -1.0
                    # print(f"reward: {reward}")
                    tmp_memory.append((state, theorem, state, reward))
                    continue
                t2 = time.time()
                time_cost = t2 - t1
            except:
                tmp_memory.append((state, theorem, None, -1.0))
                continue
            next_state = generate_state(now_hyp)
            reward = - (1 - math.exp(-1. * time_cost / 60.0))
            if now_hyp.answer is not None:
                reward += 1.0
                # print(f"reward: {reward}")
                tmp_memory.append((state, theorem, None, reward))
                return now_hyp.answer, now_steps, tmp_memory
            else:
                # print(f"reward: {reward}")
                tmp_memory.append((state, theorem, next_state, reward))

            new_hypotheses.append(now_hyp)
            new_hyp_scores.append(new_score)
            new_hyp_steps.append(now_steps)

        hypotheses = new_hypotheses
        hyp_scores = new_hyp_scores
        hyp_steps = new_hyp_steps

    return None, conti_hyp_steps, tmp_memory


def beam_search_for_RL_random(graph_solver, max_step, beam_size):
    tmp_memory = []

    t = 0
    hypotheses = [graph_solver]
    hyp_steps = [[]]

    while t < max_step and len(hypotheses) > 0:
        t += 1
        assert len(hypotheses) <= beam_size, f"hyp_num: {len(hypotheses)}, beam_size: {beam_size}"

        hyp_theorem = []
        conti_hyp_steps = []
        for hyp_index, hyp in enumerate(hypotheses):
            for i in range(beam_size):
                cur_theorem = np.random.choice(action_space)
                hyp_theorem.append([hyp, cur_theorem])
                conti_hyp_steps.append(hyp_steps[hyp_index] + [cur_theorem])

        new_hypotheses = []
        new_hyp_steps = []

        for cand_hyp_id in range(len(hyp_theorem)):
            prev_hyp, theorem = hyp_theorem[cand_hyp_id]
            now_steps = conti_hyp_steps[cand_hyp_id]

            state = generate_state(prev_hyp)
            now_hyp = copy.deepcopy(prev_hyp)
            t1 = time.time()
            try:
                graph_model = get_model(model_pool, model_id_map, theorem)
                now_hyp.solve_with_one_model(graph_model)
                if not now_hyp.is_updated:
                    reward = -1.0
                    tmp_memory.append((state, theorem, state, reward))
                    continue
                t2 = time.time()
                time_cost = t2 - t1
            except:
                tmp_memory.append((state, theorem, None, -1.0))
                continue
            next_state = generate_state(now_hyp)
            reward = - (1 - math.exp(-1. * time_cost / 60.0))
            if len(now_hyp.target_node_values) > 0:
                answer = now_hyp.replace_and_evaluate(now_hyp.global_graph.target_equation)
                if answer is not None:
                    reward += 1.0
                    tmp_memory.append((state, theorem, None, reward))
                    return answer, tmp_memory
            else:
                tmp_memory.append((state, theorem, next_state, reward))

            new_hypotheses.append(now_hyp)
            new_hyp_steps.append(now_steps)

        hypotheses = new_hypotheses
        hyp_steps = new_hyp_steps

    return None, tmp_memory


def beam_search_for_RL_model_sequence(graph_solver, model_sequence):
    tmp_memory = []

    prev_hyp = graph_solver
    for theorem in model_sequence:
        state = generate_state(prev_hyp)
        now_hyp = copy.deepcopy(prev_hyp)
        t1 = time.time()
        try:
            graph_model = get_model(model_pool, model_id_map, theorem)
            now_hyp.solve_with_one_model(graph_model)
            if not now_hyp.is_updated:
                reward = -1.0
                tmp_memory.append((state, theorem, state, reward))
                continue
            t2 = time.time()
            time_cost = t2 - t1
        except:
            tmp_memory.append((state, theorem, None, -1.0))
            continue
        next_state = generate_state(now_hyp)
        reward = - (1 - math.exp(-1. * time_cost / 60.0))
        if len(now_hyp.target_node_values) > 0:
            answer = now_hyp.replace_and_evaluate(now_hyp.global_graph.target_equation)
            if answer is not None:
                reward += 1.0
                tmp_memory.append((state, theorem, None, reward))
                return answer, tmp_memory
        else:
            tmp_memory.append((state, theorem, next_state, reward))
        prev_hyp = now_hyp

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


def solve_with_question(q_id, model, max_steps=10, eps=0.1):
    try:
        graph_solver, target = get_graph_solver(q_id)
        graph_solver.init_solve()

        if graph_solver.answer is not None:
            return graph_solver.answer, [], []

        now_answer, model_id_seq, tmp_memory = beam_search_for_RL(graph_solver, model, max_steps, args.beam_size, eps)
        return now_answer, model_id_seq, tmp_memory
    except Exception as e:
        train_logger.error(f"Solving question error for q_id {q_id}: {e}")
        return None, [], []
    finally:
        gc.collect()


def solve_with_timeout(q_id, model, timeout=120):
    with mp.Pool(processes=1) as pool:
        result = pool.apply_async(solve_with_question, args=(q_id, model))

        try:
            # 设置超时时间
            answer, model_id_seq, tmp_memory = result.get(timeout=timeout)
            return answer, model_id_seq, tmp_memory
        except mp.TimeoutError:
            train_logger.error(f'Solving question error for q_id {q_id} - Timeout')
            return None, [], []
        except Exception as e:
            train_logger.error(f'Solving question error for q_id {q_id} - Error: {e}')
            return None, [], []
        finally:
            gc.collect()  # 确保资源回收


def solve_with_question_random(q_id, max_steps=10):
    try:
        graph_solver, target = get_graph_solver(q_id)
        graph_solver.init_solve()

        if len(graph_solver.target_node_values) > 0:
            answer = graph_solver.answer
            return answer, []

        now_answer, tmp_memory = beam_search_for_RL_random(graph_solver, max_steps, args.beam_size)
        return now_answer, tmp_memory
    except Exception as e:
        train_logger.error(e)
        return None, []


def solve_with_question_model_sequence(q_id):
    try:
        graph_solver, target = get_graph_solver(q_id)
        graph_solver.init_solve()

        if len(graph_solver.target_node_values) > 0:
            answer = graph_solver.answer
            return answer, []

        model_sequence = model_sequence_json[str(q_id)]
        now_answer, tmp_memory = beam_search_for_RL_model_sequence(graph_solver, model_sequence)
        return now_answer, tmp_memory
    except Exception as e:
        train_logger.error(e)
        return None, []


def optimize_model(model, optimizer):
    try:
        if len(memory) < BATCH_SIZE:
            return
        model.train()
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        state_batch = {}
        try:
            for k in batch.state[0].keys():
                if k == 'edge_input':
                    continue
                state_batch[k] = [single_state[k].squeeze(0).cpu() for single_state in batch.state]
        except:
            train_logger.error(f"ERROR! Batch: {batch}")
            return
        state_batch = onehop_collate_fn([state_batch['x'], state_batch['node_attr'], state_batch['target_attr'],
                                         state_batch['attn_bias'], state_batch['attn_edge_type'],
                                         state_batch['spatial_pos'],
                                         state_batch['in_degree'], state_batch['out_degree'], None,
                                         [torch.Tensor(0)] * BATCH_SIZE], zipped=True)
        for k in state_batch.keys():
            state_batch[k] = state_batch[k].cuda()

        action_batch = torch.LongTensor(batch.action).view(BATCH_SIZE, 1).cuda()
        reward_batch = torch.Tensor(batch.reward).view(BATCH_SIZE).cuda()
        # print(f"action_batch: {action_batch}")
        # print(f"reward_batch: {reward_batch}")

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool).cuda()
        non_final_next_states = {}
        for k in batch.state[0].keys():
            if k == 'edge_input':
                continue
            non_final_next_states[k] = [single_state[k].squeeze(0).cpu() for single_state in batch.next_state if
                                        single_state is not None]
        non_final_next_states = onehop_collate_fn(
            [non_final_next_states['x'], non_final_next_states['node_attr'], non_final_next_states['target_attr'],
             non_final_next_states['attn_bias'], non_final_next_states['attn_edge_type'],
             non_final_next_states['spatial_pos'],
             non_final_next_states['in_degree'], non_final_next_states['out_degree'], None,
             [torch.Tensor(0)] * BATCH_SIZE],
            zipped=True)
        for k in non_final_next_states.keys():
            non_final_next_states[k] = non_final_next_states[k].cuda()

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = model(state_batch).gather(1, action_batch)
        # print(state_action_values)
        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE).cuda()
        with torch.no_grad():
            output_logits = model(non_final_next_states)
            # print(f"output_logits: {output_logits}")
            next_state_values[non_final_mask] = output_logits.max(1)[0]

        # scorelists = []
        # for output_logit in output_logits:
        #     score = torch.softmax(output_logit, dim=-1).squeeze(0)
        #     sorted_score = torch.sort(score, descending=True)
        #     sorted_score_dict = {k.cpu().item(): v.cpu().item() for k, v in zip(sorted_score[1], sorted_score[0])}
        #     scorelists.append(sorted_score_dict)

        # for i, scorelist in enumerate(scorelists):
        #     print(f"Scorelist for output_logits[{i}]: {scorelist}")

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        # print(f"state_action_values: {state_action_values}")
        # print(f"expected_state_action_values: {expected_state_action_values.unsqueeze(1)}")
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # print(f"loss: {loss}")
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), 1, norm_type=2)
        optimizer.step()
        global model_update_steps
        model_update_steps += 1
        writer.add_scalar('train_loss', loss.item(), global_step=model_update_steps)

        if model_update_steps % 50 == 0:
            torch.save(model.state_dict(), output_path + "/graph_model_RL_step" + str(model_update_steps) + ".pt")
            torch.cuda.empty_cache()

    except Exception as e:
        train_logger.error(f"optimize model error: {e}")


def pre_store_data():
    st = 0
    ed = 2401
    save_interval = 100
    count = 0
    for q_id in trange(st, ed):
        q_id = str(q_id)
        if q_id not in diagram_logic_forms_json or q_id not in text_logic_forms_json or q_id in error_ids:
            continue
        try:
            answer, tmp_memory = func_timeout(120, solve_with_question_random, kwargs=dict(q_id=q_id))
            for _ in tmp_memory:
                memory.push(*_)
            count += 1

            train_logger.debug(
                f'q_id: {q_id} - total memory: {len(memory)}, added memory: {len(tmp_memory)}, answer: {answer}')
            if count >= save_interval:
                pkl.dump(memory, open("Memory.pkl", 'wb'))
                count = 0
        except FunctionTimedOut:
            train_logger.error(f'q_id: {q_id} - Timeout')
            gc.collect()
            continue
        except Exception as e:
            train_logger.error(f'q_id: {q_id} - Error: {e}')
            gc.collect()
            pass

    if count > 0:
        pkl.dump(memory, open("Memory.pkl", 'wb'))
        gc.collect()


def pre_store_data_processed():
    st = 0
    ed = 2401
    save_interval = 100
    count = 0

    for q_id in trange(st, ed):
        q_id = str(q_id)
        if q_id not in diagram_logic_forms_json or q_id not in text_logic_forms_json or q_id not in model_sequence_json or q_id in error_ids:
            continue

        try:
            answer, tmp_memory = func_timeout(120, solve_with_question_model_sequence, args=(q_id,))
            for _ in tmp_memory:
                memory.push(*_)
            count += 1
            train_logger.debug(
                f'q_id: {q_id} - total memory: {len(memory)}, added memory: {len(tmp_memory)}, answer: {answer}')
        except FunctionTimedOut:
            train_logger.error(f'q_id: {q_id} - Timeout')
        except Exception as e:
            train_logger.error(f'q_id: {q_id} - Error: {e}')

        if count >= save_interval:
            pkl.dump(memory, open("Memory.pkl", 'wb'))
            count = 0
            gc.collect()

    if count > 0:
        pkl.dump(memory, open("Memory.pkl", 'wb'))
        gc.collect()


def train_loop(model, optimizer):
    max_steps = 50000
    total_steps = 0
    save_interval = 100
    count = 0
    global model_update_steps
    while model_update_steps < max_steps and total_steps < max_steps:
        total_steps += 1
        st = 0
        ed = 2401
        for q_id in trange(st, ed):
            model_id_seq = []
            q_id = str(q_id)
            if q_id not in diagram_logic_forms_json or q_id not in text_logic_forms_json or q_id in error_ids:
                train_logger.debug(f'step: {model_update_steps}, q_id: {q_id} - q_id in error_ids')
                continue

            try:
                solve_start_time = time.time()

                answer, model_id_seq, tmp_memory = solve_with_timeout(q_id, model, timeout=60)
                # answer, model_id_seq, tmp_memory = solve_with_question(q_id, model)

                solve_end_time = time.time()
                solve_duration = solve_end_time - solve_start_time
                for _ in tmp_memory:
                    memory.push(*_)
                if len(tmp_memory) > 0:
                    count += 1
                if count >= save_interval:
                    pkl.dump(memory, open("Memory.pkl", 'wb'))
                    count = 0

            except Exception as e:
                train_logger.error(f'step: {model_update_steps}, q_id: {q_id} - Error: {e}')
                gc.collect()
                continue
            # print(f"memory: {len(memory)}")

            optimize_start_time = time.time()
            optimize_model(model, optimizer)
            optimize_end_time = time.time()
            optimize_duration = optimize_end_time - optimize_start_time
            train_logger.info(
                f'step: {model_update_steps}, q_id: {q_id} - answer: {answer}, model_id_seq: {model_id_seq}, solving time: {solve_duration:.4f} seconds, optimizing time: {optimize_duration:.4f} seconds')
            gc.collect()


def pretrain_loop(model, optimizer):
    max_steps = 5000
    for _ in trange(0, max_steps):
        optimize_start_time = time.time()
        optimize_model(model, optimizer)
        optimize_end_time = time.time()
        optimize_duration = optimize_end_time - optimize_start_time
        train_logger.info(f'step: {model_update_steps} - optimizing time: {optimize_duration:.4f} seconds')


if __name__ == "__main__":
    writer = SummaryWriter(log_path)
    model_args = ModelArgs(num_classes=64, max_nodes=256, num_node_type=len(node_type_vocab),
                           num_node_attr=len(node_attr_vocab), num_in_degree=256, num_out_degree=256,
                           num_edges=len(edge_attr_vocab), num_spatial=20, num_edge_dis=256, edge_type="one_hop",
                           multi_hop_max_dist=1)
    policy_net = GraphormerEncoder(model_args).cuda().share_memory()
    INIT_MODEL = None
    # INIT_MODEL = os.path.join(output_path, "graph_model_RL_step" + str(15700) + ".pt")
    if INIT_MODEL:
        match = re.search(r'step(\d+)', INIT_MODEL)
        if match:
            model_update_steps = int(match.group(1))
            print(f"Start training at step: {model_update_steps}")
        policy_net.load_state_dict(torch.load(INIT_MODEL))

    optimizer = torch.optim.AdamW(policy_net.parameters(), args.lr)
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    try:
        memory = pkl.load(open("Memory.pkl", 'rb'))
        print("Memory.pkl loaded successfully.")
    except FileNotFoundError:
        print("Warning: Memory.pkl file not found. Proceeding without loading memory.")
    except Exception as e:
        print(f"Warning: Failed to load Memory.pkl: {e}. Proceeding without loading memory.")

    if args.init_memory:
        pre_store_data()
        pre_store_data_processed()

    if args.pre_train:
        pretrain_loop(policy_net, optimizer)
    else:
        train_loop(policy_net, optimizer)

    # solve_with_question(4)
    # solve_with_question_random(4)
    # solve_with_question_model_sequence(4)
