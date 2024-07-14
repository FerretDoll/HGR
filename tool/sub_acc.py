import json
import math
import random
import os
import argparse
import re

from reasoner import config


def quad_alias(diagram_type, QuadAlias):
    alias_diagram_type = []
    for diagram in diagram_type:
        if diagram in QuadAlias:
            diagram = QuadAlias[diagram]
        alias_diagram_type.append(diagram)
    return (alias_diagram_type)


def correct_goal_type(goal_type):
    if "Other" in goal_type and len(goal_type) > 1:
        goal_type.remove("Other")
    return goal_type


def build_dict(data_path, st, ed):
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        if match:
            return int(match.group())
        return 0  # 如果没有数字，返回0

    # 获取文件列表
    file_list = os.listdir(data_path)
    # 按数字大小排序
    file_list = sorted(file_list, key=extract_number)
    if ".DS_Store" in file_list:
        file_list.remove('.DS_Store')

    IdToGoalType = {}
    IdToDiagramType = {}
    GoalTypeToId = {}
    DiagramTypeToId = {}

    QuadAlias = {}
    Quads = ["Rectangle", "Rhombus", "Parallelogram", "Trapezoid", "Square"]
    for quad in Quads:
        QuadAlias[quad] = "Quad"

    for idx, data_id in enumerate(file_list):
        if idx < st or idx >= ed:
            continue
        with open(os.path.join(data_path, data_id, "data.json"), 'r') as f:
            data = json.load(f)

            pid = data["id"]

            # IdToGoalType, GoalTypeToId
            goal_type = data["problem_type_goal"]
            goal_type = correct_goal_type(goal_type)

            IdToGoalType[pid] = goal_type
            for goal in goal_type:
                if goal in GoalTypeToId:
                    GoalTypeToId[goal].append(pid)
                else:
                    GoalTypeToId[goal] = [pid]

            # IdToDiagramType, DiagramTypeToId
            diagram_type = data["problem_type_graph"]
            diagram_type = quad_alias(diagram_type, QuadAlias)

            IdToDiagramType[pid] = diagram_type

            for diagram in diagram_type:
                if diagram in DiagramTypeToId:
                    DiagramTypeToId[diagram].append(pid)
                else:
                    DiagramTypeToId[diagram] = [pid]

    # assert len(IdToGoalType) == 601
    # assert len(IdToDiagramType) == 601

    return IdToGoalType, IdToDiagramType, GoalTypeToId, DiagramTypeToId


def print_type_acc(Result_Acc):
    Acc_GoalTypeToId = {}
    Acc_DiagramTypeToId = {}

    for pid in Result_Acc:
        # goal type acc
        goal_type = IdToGoalType[pid]
        goal_type = correct_goal_type(goal_type)
        for goal in goal_type:
            if goal in Acc_GoalTypeToId:
                Acc_GoalTypeToId[goal].append(pid)
            else:
                Acc_GoalTypeToId[goal] = [pid]

                # diagram type acc
        diagram_type = IdToDiagramType[pid]
        for diagram in diagram_type:
            if diagram in Acc_DiagramTypeToId:
                Acc_DiagramTypeToId[diagram].append(pid)
            else:
                Acc_DiagramTypeToId[diagram] = [pid]

    Angle = 100 * len(Acc_GoalTypeToId["Angle"]) / len(GoalTypeToId["Angle"])
    Length = 100 * len(Acc_GoalTypeToId["Length"]) / len(GoalTypeToId["Length"])
    Area = 100 * len(Acc_GoalTypeToId["Area"]) / len(GoalTypeToId["Area"])
    Ratio = 100 * len(Acc_GoalTypeToId["Ratio"]) / len(GoalTypeToId["Ratio"])

    Line = 100 * len(Acc_DiagramTypeToId["Line"]) / len(DiagramTypeToId["Line"])
    Triangle = 100 * len(Acc_DiagramTypeToId["Triangle"]) / len(DiagramTypeToId["Triangle"])
    Quad = 100 * len(Acc_DiagramTypeToId["Quad"]) / len(DiagramTypeToId["Quad"])
    Circle = 100 * len(Acc_DiagramTypeToId["Circle"]) / len(DiagramTypeToId["Circle"])
    Other = 100 * len(Acc_DiagramTypeToId["Other"]) / len(DiagramTypeToId["Other"])

    # latex type
    print("[Sub Acc]: &{:.3} &{:.3} &{:.3} &{:.3} &{:.3} &{:.3} &{:.3} &{:.3} &{:.3} \\\\".format(
        Angle, Length, Area, Ratio, Line, Triangle, Quad, Circle, Other))


if __name__ == '__main__':
    st = 0
    ed = 596
    result_file = 'correct_data.json'

    DATA_PATH = config.db_dir_single

    IdToGoalType, IdToDiagramType, GoalTypeToId, DiagramTypeToId = build_dict(DATA_PATH, st, ed)

    # read the result json file
    result_data = json.load(open(result_file))

    with open(config.error_ids_path, 'r') as file:
        error_ids = {int(line.strip()) for line in file}  # 确保错误ID是整数

    # 生成所有题目ID并排除错误ID
    all_question_ids = set(range(st, ed))
    valid_question_ids = list(all_question_ids - error_ids)  # 将集合转换为列表
    total = len(valid_question_ids)

    # compute acc
    solved_correct_list = []
    # solved_wrong_list = []
    unsolved_list = []
    for i in range(total):
        if str(i) in result_data:
            if result_data[str(i)]["correctness"] == "yes":
                solved_correct_list.append(i)
            # elif result_data[str(i)]["answer"] != None:
            #     solved_wrong_list.append(i)
            else:
                unsolved_list.append(i)
        else:
            unsolved_list.append(i)
    guess_correct_list = random.sample(unsolved_list, math.ceil(len(unsolved_list) / 4))

    Acc = solved_correct_list + guess_correct_list
    Acc = [int(id) for id in Acc]
    correct = len(Acc)

    print("[File]:\t  ", result_file)
    print("[Acc]:\t   {}/{} = {:.2%}".format(correct, total, correct / total))
    print_type_acc(Acc)

    with open('wrong_data.txt', 'w') as out:
        for id_ in sorted(unsolved_list, key=int):
            out.write(str(id_) + '\n')