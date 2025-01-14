import json
import random

import torch.multiprocessing as mp
import streamlit as st
from PIL import Image

from agent.agent_solver import AgentSolver
from reasoner.config import eval_logger
from tool.run_HGR import solve_question

random.seed(0)
model_path = 'saves/RL/graph_model_RL_step24200.pt'
agent_solver = AgentSolver(model_path)

def solve_process(agent_solver, return_dict, q_id, res):
    try:
        result = agent_solver.solve(q_id=q_id, record=True)
        return_dict['result'] = result
    except Exception as e:
        eval_logger.error(f'q_id: {q_id} - Error in solving: {e}')
        return_dict['result'] = res


def solve_heuristic_process(return_dict, q_id, res):
    try:
        result = solve_question(q_id, record=True)
        return_dict['result'] = result
    except Exception as e:
        eval_logger.error(f'q_id: {q_id} - Error in heuristic solving: {e}')
        return_dict['result'] = res


def solve_with_time(agent_solver, q_id, time_limit=180):
    q_id = str(q_id)
    res = {"id": q_id, "target": None, "answer": None, "step_lst": [], "model_instance_eq_num": None,
           "correctness": "no", "time": None}

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

st.title('Problem Solving')

index = st.sidebar.text_input(label='Problem ID')
use_agent = st.sidebar.checkbox("Use agent", value=True)

if st.sidebar.button("Preview"):
    if index is not None:
        data = json.load(open("db/Geometry3K/" + str(index) + "/data.json"))
        st.sidebar.markdown(data['annotat_text'])
        choices = data['choices']
        choices_map = ['A', 'B', 'C', 'D']
        for i in range(len(choices)):
            st.sidebar.markdown(choices_map[i] + ". $" + choices[i] + "$")
        img_path = "db/Geometry3K/" + str(index) + "/img_diagram.png"
        image = Image.open(img_path)
        st.sidebar.image(image)

if st.sidebar.button("Solve"):
    if index is not None:

        data = json.load(open("db/Geometry3K/" + str(index) + "/data.json"))
        logic_form = json.load(open("db/Geometry3K/" + str(index) + "/logic_form.json"))
        text_logic_forms = logic_form["text_logic_form"]
        diagram_logic_forms = logic_form['diagram_logic_form']
        st.sidebar.markdown(data['annotat_text'])
        choices = data['choices']
        choices_map = ['A', 'B', 'C', 'D']
        for i in range(len(choices)):
            st.sidebar.markdown(choices_map[i] + ". $" + choices[i] + "$")
        img_path = "db/Geometry3K/" + str(index) + "/img_diagram_point.png"
        image = Image.open(img_path)
        st.sidebar.image(image)

        if len(text_logic_forms) > 1:
            st.subheader('Text Relations')
            for t in text_logic_forms[:-1]:
                st.markdown(t)

        st.subheader('Diagram Relations')
        for d in diagram_logic_forms:
            st.markdown(d)

        if use_agent:
            res = solve_with_time(index, agent_solver)
        else:
            res = solve_question(index, record=True)

        answer = res["answer"]
        reasoning_record = res["reasoning_record"]

        st.subheader('Target')
        st.markdown(res["target"])

        st.subheader('Solving Procedure')
        for i in range(len(reasoning_record)):
            step = reasoning_record[i]
            with st.expander('Step ' + str(i + 1) + ': ' + step["model_name"]):
                instances = step["instances"]
                for instance in instances:
                    relation = instance["relation"]
                    st.markdown(f"**{relation}**")
                    if len(instance["actions"]) > 0:
                        st.markdown('Actions:')
                        for action in instance["actions"]:
                            st.markdown(action)
                    if len(instance["equations"]) > 0:
                        st.markdown('Equations:')
                        for e in instance["equations"]:
                            st.text(e)
                # st.markdown('**Equations:**')
                # for e in equations_list[i]:
                #     st.text(e + " = 0")
                # st.markdown('**Solutions:**')
                # for key, value in solutions_list[i].items():
                #     st.text(key + " = " + value)

        st.subheader('Answer')
        st.markdown(str(res["answer"]))

        if answer:
            candi_answer = []
            for choice in data['precise_value']:
                candi_answer.append(abs(float(choice) - float(answer)))
            st.markdown('Choose **' + choices_map[candi_answer.index(min(candi_answer))] + "**")
