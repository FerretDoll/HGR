import copy

import sympy

from reasoner.config import logger
from utils.common_utils import isNumber, findNumber

acute_angle = ['AcuteAngle1', 'AcuteAngle2', 'AcuteAngle3', 'AcuteAngle4', 'AcuteAngle5', 'AcuteAngle6', 'AcuteAngle7',
               'AcuteAngle8', 'AcuteAngle9', 'AcuteAngle10']
obtuse_angle = ['ObtuseAngle1', 'ObtuseAngle2', 'ObtuseAngle3', 'ObtuseAngle4', 'ObtuseAngle5', 'ObtuseAngle6',
                'ObtuseAngle7', 'ObtuseAngle8', 'ObtuseAngle9', 'ObtuseAngle10']
reflex_angle = ['ReflexAngle1', 'ReflexAngle2', 'ReflexAngle3', 'ReflexAngle4', 'ReflexAngle5']
constant = ['Constant1', 'Constant2', 'Constant3', 'Constant4', 'Constant5', 'Constant6', 'Constant7', 'Constant8',
            'Constant9', 'Constant10']
node_attr_vocab = ['None', '30', '45', '60', '90', '120', '135', '150', '180', '210', '240', '270', '300', '330', 'x',
                   'y', 'z', 'h', 'r', 'p', 'd', 's', 'Var'] + acute_angle + obtuse_angle + reflex_angle + constant


def reparse_graph_data(graph_data, map_dict):
    output_graph_data = copy.deepcopy(graph_data)
    output_graph_data['node_attr'] = []
    node_num = len(graph_data['node_type'])
    acute_angle_id = obtuse_angle_id = reflex_angle_id = constant_id = 0
    for i in range(node_num):
        node_type = graph_data['node_type'][i]
        node_attr = graph_data['node_attr'][i]
        if node_attr == 'None':
            output_graph_data['node_attr'].append(node_attr)
            continue
        else:
            number = findNumber(node_attr)
            if number is not None:
                if number in map_dict:
                    output_graph_data['node_attr'].append(map_dict[number])
                    continue
                if node_type == 'Angle' or node_type == 'Arc':
                    if number[0] == '-':
                        number = number[1:]
                    expr = sympy.sympify(number)
                    degrees_expr = sympy.deg(expr)
                    degrees_float = float(degrees_expr.evalf())
                    if degrees_float == 90:
                        map_dict[number] = '90'
                    elif degrees_float < 90:
                        map_dict[number] = acute_angle[acute_angle_id]
                        acute_angle_id += 1
                    elif 90 < degrees_float < 180:
                        map_dict[number] = obtuse_angle[obtuse_angle_id]
                        obtuse_angle_id += 1
                    elif degrees_float > 180:
                        map_dict[number] = reflex_angle[reflex_angle_id]
                        reflex_angle_id += 1
                    else:
                        logger.error(f'reparse angle error: {number}')
                        raise AssertionError
                else:
                    map_dict[number] = constant[constant_id]
                    constant_id += 1
                output_graph_data['node_attr'].append(map_dict[number])
            else:
                has_added = False
                for attr in node_attr:
                    if attr in node_attr_vocab:
                        output_graph_data['node_attr'].append(attr)
                        has_added = True
                        break
                    if attr in map_dict:
                        output_graph_data['node_attr'].append(map_dict[attr])
                        has_added = True
                        break
                    if any(keyword in attr for keyword in ['line', 'angle', 'arc', 'triangle', 'polygon']):
                        continue
                    else:
                        attr = attr.replace('a', 'x').replace('b', 'y').replace('c', 'z').replace('t', 'x') \
                            .replace('w', 'x').replace('v', 'y').replace('m', 'x').replace('n', 'y')
                        if 'x' in attr:
                            map_dict[attr] = 'x'
                        elif 'y' in attr:
                            map_dict[attr] = 'y'
                        elif 'z' in attr:
                            map_dict[attr] = 'z'
                        elif 'r' in attr:
                            map_dict[attr] = 'r'
                        elif 'p' in attr:
                            map_dict[attr] = 'p'
                        elif 'd' in attr:
                            map_dict[attr] = 'd'
                        elif 's' in attr:
                            map_dict[attr] = 's'
                        else:
                            map_dict[attr] = 'Var'
                        output_graph_data['node_attr'].append(map_dict[attr])
                        has_added = True
                        break
                if not has_added:
                    output_graph_data['node_attr'].append('None')

    assert len(output_graph_data['node_attr']) == len(output_graph_data['node'])
    return output_graph_data, map_dict


def reparse_data(input_data):
    output_data = []
    for item in input_data:
        output_item = copy.deepcopy(item)
        output_item['graph_data'] = reparse_graph_data(item['graph_data'])
        output_data.append(output_item)
    return output_data
