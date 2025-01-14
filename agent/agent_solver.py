import copy
import json
import os
import time

import numpy as np
import torch

from agent.gen_vocab import reparse_graph_data
from agent.graph_dataset import preprocess_item
from agent.model.graphtransformer.model import GraphormerEncoder
from agent.model.graphtransformer.model_args import ModelArgs
from reasoner import config
from reasoner.config import eval_logger
from reasoner.graph_matching import load_models_from_json, get_model
from tool.run_HGR import get_graph_solver, check_transformed_answer

EPSILON = 1e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AgentSolver:
    def __init__(self, model_path, max_step=10, beam_size=5):
        self.model = None
        self.node_type_vocab = None
        self.node_attr_vocab = None
        self.edge_attr_vocab = None
        self.max_step = max_step
        self.beam_size = beam_size
        self.map_dict = {}
        with open(config.model_pool_path, 'r') as model_pool_file:
            self.model_pool, self.model_id_map = load_models_from_json(json.load(model_pool_file))

        self.init_data_and_model(model_path)

    def init_data_and_model(self, model_path):
        node_type_vocab_file = 'agent/vocab/node_type_vocab.txt'
        node_attr_vocab_file = 'agent/vocab/node_attr_vocab.txt'
        edge_attr_vocab_file = 'agent/vocab/edge_attr_vocab.txt'
        self.node_type_vocab = {line.strip(): i for i, line in enumerate(open(node_type_vocab_file, 'r').readlines())}
        self.node_attr_vocab = {line.strip(): i for i, line in enumerate(open(node_attr_vocab_file, 'r').readlines())}
        self.edge_attr_vocab = {line.strip(): i for i, line in enumerate(open(edge_attr_vocab_file, 'r').readlines())}

        model_args = ModelArgs(num_classes=64, max_nodes=256, num_node_type=len(self.node_type_vocab),
                               num_node_attr=len(self.node_attr_vocab), num_in_degree=256, num_out_degree=256,
                               num_edges=len(self.edge_attr_vocab), num_spatial=20, num_edge_dis=256,
                               edge_type="one_hop",
                               multi_hop_max_dist=1)

        self.model = GraphormerEncoder(model_args).to(device).share_memory()
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def theorem_pred(self, graph_solver):
        graph_data = graph_solver.global_graph.to_dict()
        graph_data, self.map_dict = reparse_graph_data(graph_data, self.map_dict)

        single_test_data = preprocess_item(item=graph_data, node_type_vocab=self.node_type_vocab,
                                           node_attr_vocab=self.node_attr_vocab, edge_attr_vocab=self.edge_attr_vocab,
                                           spatial_pos_max=1)
        for k, v in single_test_data.items():
            single_test_data[k] = v.unsqueeze(0).to(device)
        output_logits = self.model(single_test_data)
        score = torch.softmax(output_logits, dim=-1).squeeze(0)
        sorted_score = torch.sort(score, descending=True)
        sorted_score_dict = {k.cpu().item(): v.cpu().item() for k, v in zip(sorted_score[1], sorted_score[0])}
        return sorted_score_dict

    def beam_search(self, graph_solver):
        t = 0
        hypotheses = [graph_solver]
        hyp_steps = [[]]
        hyp_scores = [0.]
        beam_search_res = {"answer": None, "step_lst": [], "reasoning_record": None, "model_instance_eq_num": None}
        all_conti_hyp_steps = []

        while t < self.max_step:
            t += 1
            hyp_num = len(hypotheses)
            assert hyp_num <= self.beam_size, f"hyp_num: {hyp_num}, beam_size: {self.beam_size}"

            hyp_theorem = []
            conti_hyp_scores = []
            conti_hyp_steps = []
            for hyp_index, hyp in enumerate(hypotheses):
                sorted_score_dict = self.theorem_pred(hyp)
                # print("step:", t, "past_steps:", hyp_steps[hyp_index], sorted_score_dict)
                for i in range(self.beam_size):
                    cur_score = list(sorted_score_dict.values())[i]
                    if cur_score < EPSILON:
                        continue
                    hyp_theorem.append([hyp, list(sorted_score_dict.keys())[i]])
                    conti_hyp_scores.append(hyp_scores[hyp_index] + np.log(cur_score))
                    conti_hyp_steps.append(hyp_steps[hyp_index] + [list(sorted_score_dict.keys())[i]])

            conti_hyp_scores = torch.Tensor(conti_hyp_scores)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(conti_hyp_scores,
                                                               k=min(self.beam_size, conti_hyp_scores.size(0)))
            if len(conti_hyp_steps) > 0:
                all_conti_hyp_steps.append(conti_hyp_steps)

            new_hypotheses = []
            new_hyp_scores = []
            new_hyp_steps = []

            for cand_hyp_id, cand_hyp_score in zip(top_cand_hyp_pos, top_cand_hyp_scores):
                new_score = cand_hyp_score.detach().item()
                prev_hyp, theorem = hyp_theorem[cand_hyp_id]
                now_steps = conti_hyp_steps[cand_hyp_id]

                now_hyp = copy.deepcopy(prev_hyp)

                try:
                    graph_model = get_model(self.model_pool, self.model_id_map, theorem)
                    now_hyp.solve_with_one_model(graph_model)
                    if not now_hyp.is_updated:
                        continue
                except Exception as e:
                    eval_logger.error(e)
                    continue

                if now_hyp.answer is not None:
                    beam_search_res["answer"] = now_hyp.answer
                    beam_search_res["step_lst"] = now_steps
                    beam_search_res["reasoning_record"] = now_hyp.reasoning_record
                    beam_search_res["model_instance_eq_num"] = now_hyp.model_instance_eq_num
                    return beam_search_res

                new_hypotheses.append(now_hyp)
                new_hyp_scores.append(new_score)
                new_hyp_steps.append(now_steps)

            hypotheses = new_hypotheses
            hyp_scores = new_hyp_scores
            hyp_steps = new_hyp_steps

        beam_search_res["step_lst"] = all_conti_hyp_steps
        return beam_search_res

    def solve(self, q_id, record=False):
        q_id = str(q_id)
        if record:
            res = {"id": q_id, "target": None, "answer": None, "step_lst": [], "model_instance_eq_num": None,
                   "reasoning_record": None, "correctness": "no", "time": None}
        else:
            res = {"id": q_id, "target": None, "answer": None, "step_lst": [], "model_instance_eq_num": None,
                   "correctness": "no", "time": None}

        s_time = time.time()

        data_path = os.path.join(config.db_dir_single, str(q_id), "data.json")
        with open(data_path, "r") as f:
            data = json.load(f)
        candidate_value_list = data['precise_value']
        gt_id = ord(data['answer']) - 65  # Convert A-D to 0-3

        graph_solver, target = get_graph_solver(q_id)
        res["target"] = target
        graph_solver.init_solve()

        answer = graph_solver.answer
        if answer is not None:
            correctness, answer = check_transformed_answer(answer, candidate_value_list, gt_id)
            if correctness:
                res["correctness"] = "yes"
                res["answer"] = answer
                res["model_instance_eq_num"] = graph_solver.model_instance_eq_num
                res['time'] = str(time.time() - s_time)
                return res

        beam_search_res = self.beam_search(graph_solver=graph_solver)
        answer = beam_search_res["answer"]
        res["step_lst"] = beam_search_res["step_lst"]
        if record:
            res["reasoning_record"] = beam_search_res["reasoning_record"]
        res["model_instance_eq_num"] = beam_search_res["model_instance_eq_num"]
        res['time'] = str(time.time() - s_time)
        if answer is not None:
            correctness, transformed_answer = check_transformed_answer(answer, candidate_value_list, gt_id)
            if correctness:
                res["correctness"] = "yes"
                res["answer"] = transformed_answer
            else:
                res["answer"] = answer

        return res
