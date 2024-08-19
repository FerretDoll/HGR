import json
import re
import os


def process_log_file(log_file_path):
    # Output the path of the JSON file
    output_dir = os.path.dirname(log_file_path)
    correct_data_path = os.path.join(output_dir, 'correct_data.json')
    all_data_path = os.path.join(output_dir, 'all_data.json')

    all_ids = []
    no_target_ids = []
    time_out_ids = []
    error_ids = []
    correct_ids = []
    wrong_half_ids = []
    wrong_ids = []

    filtered_data = {}
    all_data = {}

    # Read and process log files
    with open(log_file_path, 'r') as file:
        for line in file:
            # Extract the dictionary section from the log
            dict_str = line.split('- ')[1].strip()
            dict_str = dict_str.replace("None,", "'None',")
            # Remove the target section and retain the original location information
            target_match = re.search(r"'target': (.*?), 'answer'", dict_str)
            if target_match:
                # Extract target string
                target_str = target_match.group(1)
                if target_str != "'None'":
                    new_target_str = target_str.replace("'", '')
                    new_target_str = "'" + new_target_str + "'"
                    # Replace the target part from the dictionary string
                    modified_dict_str = dict_str.replace(f"{target_str}", new_target_str)
                    # Replace single quotes with double quotes to conform to JSON format
                    modified_dict_str = modified_dict_str.replace("'", '"')
                else:
                    match = re.search(r"'id': (\d+)", dict_str)

                    if match:
                        q_id = match.group(1)
                        no_target_ids.append(q_id)
                    continue  # If the target is not found, skip this record
            else:
                match = re.search(r'question (\d+)', dict_str)

                if match:
                    q_id = match.group(1)
                    if 'FunctionTimedOut' in dict_str:
                        time_out_ids.append(q_id)
                        all_ids.append(q_id)
                    else:
                        error_ids.append(q_id)
                continue

            try:
                # Analyze the modified dictionary string
                data = json.loads(modified_dict_str)
                all_ids.append(data['id'])
                # Check if the correctness field is' yes'
                if data['correctness'] == 'yes':
                    # Use 'id' in the data as the key
                    filtered_data[data['id']] = data
                    correct_ids.append(data['id'])
                else:
                    if data['answer'] == 'None':
                        wrong_ids.append(data['id'])
                    else:
                        wrong_half_ids.append(data['id'])
            except json.JSONDecodeError as e:
                print(f"Error processing line: {modified_dict_str}")
                print(f"Error decoding JSON: {e}")

    # Write the results into JSON files and text files
    with open(os.path.join(output_dir, 'all_ids.txt'), 'w') as out:
        for id_ in sorted(all_ids, key=int):
            out.write(str(id_) + '\n')

    with open(os.path.join(output_dir, 'no_target_ids.txt'), 'w') as out:
        for id_ in sorted(no_target_ids, key=int):
            out.write(str(id_) + '\n')

    with open(os.path.join(output_dir, 'time_out_ids.txt'), 'w') as out:
        for id_ in sorted(time_out_ids, key=int):
            out.write(str(id_) + '\n')

    with open(os.path.join(output_dir, 'error_ids.txt'), 'w') as out:
        for id_ in sorted(error_ids, key=int):
            out.write(str(id_) + '\n')

    with open(correct_data_path, 'w') as json_file:
        json.dump(filtered_data, json_file, indent=4)


    # all_data = dict(sorted(all_data.items(), key=lambda item: int(item[0])))
    # with open(all_data_path, 'w') as json_file:
    #     json.dump(all_data, json_file, indent=4)

    with open(os.path.join(output_dir, 'correct_ids.txt'), 'w') as out:
        for id_ in sorted(correct_ids, key=int):
            out.write(str(id_) + '\n')

    with open(os.path.join(output_dir, 'wrong_half_ids.txt'), 'w') as out:
        for id_ in sorted(wrong_half_ids, key=int):
            out.write(str(id_) + '\n')

    with open(os.path.join(output_dir, 'wrong_ids.txt'), 'w') as out:
        for id_ in sorted(wrong_ids, key=int):
            out.write(str(id_) + '\n')

    # with open('all_ids.txt', 'r') as file:
    #     all_ids = {int(line.strip()) for line in file}
    # with open('db/error_ids.txt', 'r') as file:
    #     error_ids = {int(line.strip()) for line in file}
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
