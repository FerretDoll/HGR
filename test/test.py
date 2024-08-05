import json
import re

日志数据路径
log_file_path = 'experiment.log'
# 输出JSON文件的路径
correct_data_path = 'correct_data.json'

all_ids = []
correct_ids = []
wrong_half_ids = []
wrong_ids = []

# 用于存储提取出的数据，现在使用字典而不是列表
filtered_data = {}

# 读取并处理日志文件
with open(log_file_path, 'r') as file:
    for line in file:
        # 提取日志中的字典部分
        dict_str = line.split('- ')[1].strip()

        # 移除target部分并保留原来的位置信息
        target_match = re.search(r"'target': (.*?), 'answer'", dict_str)
        if target_match:
            # 提取target字符串
            target_str = target_match.group(1)
            new_target_str = target_str.replace("'", '')
            # new_target_str = json.loads(new_target_str[1:-1])
            # list_str_for_insertion = json.dumps(new_target_str)

            # 从字典字符串中移除target部分
            modified_dict_str = dict_str.replace(f"{target_str}", new_target_str)
            # 将单引号替换为双引号以符合JSON格式
            modified_dict_str = modified_dict_str.replace("'", '"')
        else:
            continue  # 如果没有找到target，跳过这条记录

        try:
            # 解析修改后的字典字符串
            data = json.loads(modified_dict_str)
            all_ids.append(data['id'])
            # 检查correctness字段是否为'yes'
            if data['correctness'] == 'yes':
                # 使用data中的'id'作为键
                filtered_data[data['id']] = data
                correct_ids.append(data['id'])
            else:
                if data['answer'] == 'None' or data['answer'] is None:
                    wrong_ids.append(data['id'])
                else:
                    wrong_half_ids.append(data['id'])
        except json.JSONDecodeError as e:
            wrong_ids.append(data['id'])
            print(f"Error processing line: {modified_dict_str}")
            print(f"Error decoding JSON: {e}")

# 将结果写入JSON文件
with open('all_ids.txt', 'w') as out:
    for id_ in sorted(all_ids, key=int):
        out.write(str(id_) + '\n')

with open(correct_data_path, 'w') as json_file:
    json.dump(filtered_data, json_file, indent=4)

with open('correct_ids.txt', 'w') as out:
    for id_ in sorted(correct_ids, key=int):
        out.write(str(id_) + '\n')

with open('wrong_half_ids.txt', 'w') as out:
    for id_ in sorted(wrong_half_ids, key=int):
        out.write(str(id_) + '\n')

with open('wrong_ids.txt', 'w') as out:
    for id_ in sorted(wrong_ids, key=int):
        out.write(str(id_) + '\n')

# with open('all_ids.txt', 'r') as file:
#     all_ids = {int(line.strip()) for line in file}  # 确保错误ID是整数
# with open('db/error_ids.txt', 'r') as file:
#     error_ids = {int(line.strip()) for line in file}  # 确保错误ID是整数
#
# miss_ids = []
#
# for idx in range(2401, 3001):
#     if idx not in all_ids and idx not in error_ids:
#         miss_ids.append(idx)
#
# with open('miss_ids.txt', 'w') as out:
#     for id_ in sorted(miss_ids, key=int):
#         out.write(str(id_) + '\n')
