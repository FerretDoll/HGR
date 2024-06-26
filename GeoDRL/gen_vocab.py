import json
import copy
import os


acute_angle = ['AcuteAngle1', 'AcuteAngle2', 'AcuteAngle3', 'AcuteAngle4', 'AcuteAngle5', 'AcuteAngle6', 'AcuteAngle7',
               'AcuteAngle8', 'AcuteAngle9', 'AcuteAngle10']
obtuse_angle = ['ObtuseAngle1', 'ObtuseAngle2', 'ObtuseAngle3', 'ObtuseAngle4', 'ObtuseAngle5', 'ObtuseAngle6',
                'ObtuseAngle7', 'ObtuseAngle8', 'ObtuseAngle9', 'ObtuseAngle10']
reflex_angle = ['ReflexAngle1', 'ReflexAngle2', 'ReflexAngle3', 'ReflexAngle4', 'ReflexAngle5']
angle_dict = {}
constant_dict = {}
# varX = ['VarX1', 'VarX2', 'VarX3', 'VarX4', 'VarX5', 'VarX6', 'VarX7', 'VarX8', 'VarX9', 'VarX10']
# varY = ['VarY1', 'VarY2', 'VarY3', 'VarY4', 'VarY5', 'VarY6', 'VarY7', 'VarY8', 'VarY9', 'VarY10']
# varZ = ['VarZ1', 'VarZ2', 'VarZ3', 'VarZ4', 'VarZ5', 'VarZ6', 'VarZ7', 'VarZ8', 'VarZ9', 'VarZ10']
# var = []
constant = ['Constant1', 'Constant2', 'Constant3', 'Constant4', 'Constant5', 'Constant6', 'Constant7', 'Constant8',
            'Constant9', 'Constant10']
node_type_vocab = ['[PAD]', '[CLS]'] + ['Point', 'Triangle', 'Circle', 'Arc', 'Polygon', 'Line', 'Angle']
node_attr_vocab = ['None', '30', '45', '60', '90', '120', '135', '150', '180', '210', '240', '270', '300', '330', 'x',
                   'y', 'z', 'h', 'r', 'p', 'd', 's', 'Var'] + acute_angle + obtuse_angle + reflex_angle + constant
edge_attr_vocab = ['None', 'CLS', 'SELF'] + ['Connected', 'Equal', 'Perpendicular', 'Parallel', 'Congruent', 'Similar',
                                             'LiesOnLine', 'LiesOnLine_R', 'LiesOnCircle', 'LiesOnCircle_R', 'Center',
                                             'Center_R', 'Interior', 'Interior_R', 'Endpoint', 'Endpoint_R',
                                             'Sidepoint', 'Sidepoint_R', 'Side', 'Side_R', 'Vertex', 'Vertex_R']


def isNumber(x):
    try:
        x = float(x)
        return True
    except:
        return False


def reparse_graph_data(graph_data):
    output_graph_data = copy.deepcopy(graph_data)
    output_graph_data['node_attr'] = []
    node_num = len(graph_data['node_type'])
    map_dict = {}
    acute_angle_id = obtuse_angle_id = reflex_angle_id = constant_id = 0
    for i in range(node_num):
        node_type = graph_data['node_type'][i]
        node_attr = graph_data['node_attr'][i]
        if node_attr in node_attr_vocab:
            output_graph_data['node_attr'].append(node_attr)
            continue
        if node_attr in map_dict:
            output_graph_data['node_attr'].append(map_dict[node_attr])
            continue
        if isNumber(node_attr):
            if node_type == 'Angle' or node_type == 'Arc':
                if node_attr[0] == '-': node_attr = node_attr[1:]
                angle_dict[node_attr] = 1 if node_attr not in angle_dict else angle_dict[node_attr] + 1
                if float(node_attr) == 90:
                    map_dict[node_attr] = '90'
                elif float(node_attr) < 90:
                    map_dict[node_attr] = acute_angle[acute_angle_id]
                    acute_angle_id += 1
                elif float(node_attr) > 90 and float(node_attr) < 180:
                    map_dict[node_attr] = obtuse_angle[obtuse_angle_id]
                    obtuse_angle_id += 1
                elif float(node_attr) > 180:
                    map_dict[node_attr] = reflex_angle[reflex_angle_id]
                    reflex_angle_id += 1
                else:
                    print(node_attr)
                    raise AssertionError
            else:
                map_dict[node_attr] = constant[constant_id]
                constant_id += 1
        else:
            node_attr = node_attr.replace('a', 'x').replace('b', 'y').replace('c', 'z').replace('t', 'x') \
                .replace('w', 'x').replace('v', 'y').replace('m', 'x').replace('n', 'y')
            if 'x' in node_attr:
                map_dict[node_attr] = 'x'
            elif 'y' in node_attr:
                map_dict[node_attr] = 'y'
            elif 'z' in node_attr:
                map_dict[node_attr] = 'z'
            elif 'r' in node_attr:
                map_dict[node_attr] = 'r'
            elif 'p' in node_attr:
                map_dict[node_attr] = 'p'
            elif 'd' in node_attr:
                map_dict[node_attr] = 'd'
            elif 's' in node_attr:
                map_dict[node_attr] = 's'
            else:
                map_dict[node_attr] = 'Var'

        output_graph_data['node_attr'].append(map_dict[node_attr])

    assert len(output_graph_data['node_attr']) == len(output_graph_data['node'])
    return output_graph_data


def reparse_data(input_data):
    output_data = []
    for item in input_data:
        output_item = copy.deepcopy(item)
        output_item['graph_data'] = reparse_graph_data(item['graph_data'])
        output_data.append(output_item)
    return output_data


if __name__ == "__main__":
    train_data = json.load(open("../results/train/graph_data.json", 'r'))
    val_data = json.load(open("../results/val/graph_data.json", 'r'))
    test_data = json.load(open("../results/test/graph_data.json", 'r'))
    train_data = reparse_data(train_data)
    val_data = reparse_data(val_data)
    test_data = reparse_data(test_data)

    json.dump(train_data, open("../results/train/reparsed_graph_data.json", 'w'))
    json.dump(val_data, open("../results/val/reparsed_graph_data.json", 'w'))
    json.dump(test_data, open("../results/test/reparsed_graph_data.json", 'w'))

    OUTPUT_PATH = '../vocab'
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    with open(f'{OUTPUT_PATH}/node_type_vocab.txt', 'w') as f:
        for w in node_type_vocab:
            f.write(w + '\n')
    with open(f'{OUTPUT_PATH}/node_attr_vocab.txt', 'w') as f:
        for w in node_attr_vocab:
            f.write(w + '\n')
    with open(f'{OUTPUT_PATH}/edge_attr_vocab.txt', 'w') as f:
        for w in edge_attr_vocab:
            f.write(w + '\n')
