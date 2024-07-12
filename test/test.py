import json
import re

# 日志数据路径
log_file_path = 'experiment.log'
# 输出JSON文件的路径
output_json_path = 'correct_data.json'

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

            # 检查correctness字段是否为'yes'
            if data['correctness'] == 'yes':
                # 使用data中的'id'作为键
                filtered_data[data['id']] = data
        except json.JSONDecodeError as e:
            print(f"Error processing line: {modified_dict_str}")
            print(f"Error decoding JSON: {e}")

# 将结果写入JSON文件
with open(output_json_path, 'w') as json_file:
    json.dump(filtered_data, json_file, indent=4)
