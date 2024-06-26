import math
import copy
from functools import reduce
from itertools import combinations

import sympy
from sympy import sympify, N


def convert_to_vec(points, exist_line):
    # 转换为向量
    # use the vector to represent line
    # only for straight line
    vec = []
    for item in exist_line:
        vec.append([points[item[1]][0] - points[item[0]][0], points[item[1]][1] - points[item[0]][1]])
    return vec


def convert_to_r_theta(vec):
    # 转换为角度
    # only for straight line

    center = [0, 0]
    aaa = []
    for item in vec:
        r = math.sqrt(math.pow(item[0] - center[0], 2) + math.pow(item[1] - center[1], 2))
        theta = math.atan2(item[1] - center[1], item[0] - center[0]) / math.pi * 180
        aaa.append([r, theta])
    return aaa


def convert_to_letter(idx_list):
    letter_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q"]
    s = ""
    if len(idx_list) > 1:
        for i in idx_list[0:-1]:
            if i >= 0:
                s = s + letter_list[i]
        return s
    else:
        return letter_list[idx_list[0]]


def convert_multi_to_letter(idx_list, connection):
    s = []
    if len(idx_list) > 0:
        for i in idx_list:
            s.append(convert_to_letter(i))

    result = connection.join([i for i in s])

    return result


def convert_to_shape(idx):
    shape_name = ['Triangle', 'Rectangle', 'Square', 'Parallelogram', 'Trapezoid', 'Sector', 'Circle', 'Other']

    return shape_name[idx]


def from_points_to_line(point_list):
    lines = []
    for i in range(len(point_list) - 1):
        if point_list[i] >= 0 and point_list[i + 1] < 0:
            lines.append([point_list[i], point_list[i + 1], point_list[i + 2]])
        elif point_list[i] < 0:
            pass
        else:
            lines.append([point_list[i], point_list[i + 1]])
    return lines


def check_same(item1, item2):  # 判断两个item是否为同一份封闭图形
    if len(item1) != len(item2):
        return False
    else:
        set1 = from_points_to_line(item1)
        set2 = from_points_to_line(item2)
        cnt = 0
        for i in set1:
            if len(i) == 2:
                if i in set2 or [i[-1], i[0]] in set2:
                    cnt += 1
            else:
                if i in set2 or [i[2], i[1], i[0]] in set2:
                    cnt += 1
        cnt2 = 0
        for e in item1:
            if e >= 0:
                cnt2 += 1
        if cnt == cnt2 - 1:
            return True
        else:
            return False


def check_same_list(list1, list2):
    if len(list1) != len(list2):
        return False

    if (check_element_type(list1, str) and check_element_type(list2, str)) or \
            check_element_type(list1, int) and check_element_type(list2, int):
        diff = list(set(list1).difference(set(list2)))
        diff.extend(list(set(list2).difference(set(list1))))
        if not diff:
            return True
        else:
            return False

    count = 0
    if check_element_type(list1, list) and check_element_type(list2, list):
        for i in range(len(list1)):
            if check_same_list(list1[i], list2[i]):
                count = count + 1
    elif check_element_type(list1, tuple) and check_element_type(list2, tuple):
        num = 0
        for l1 in list1:
            for l2 in list2:
                if check_same_list(l1, l2):
                    num = num + 1
        if num == len(list1):
            return True

    if count == len(list1):
        return True

    return False


def check_contain(_list, _tuple):
    for lis in _list:
        if check_same_list(_tuple, lis):
            return True

    return False


def reduce_same(opt):
    # 去掉重复的封闭图形
    dde = []
    ooo = []

    ccc = copy.deepcopy(opt)
    for i in opt:
        if len(i) <= 3:
            ccc.remove(i)
        else:
            tmp = []
            for e in i:
                if e < 0:
                    tmp.append(e)
            if len(i) - len(tmp) <= 3:
                if len(tmp) > 1:
                    if tmp[0] == tmp[1]:
                        ccc.remove(i)

    cc2 = copy.deepcopy(ccc)

    for m in range(len(ccc) - 1):
        for n in range(m + 1, len(ccc)):
            if check_same(ccc[m], ccc[n]):
                dde.append(m)

    for i in range(len(cc2)):
        if i not in dde:
            ooo.append(cc2[i])
    return ooo


def remove_redundant_element(_list, sorted_func=sorted):
    """
    去重列表里面左右颠倒重复的元素

    Args:
        _list: 待去重列表
        sorted_func: 排序函数
    Return:
        去重后的集合
    {(1, 2), ('a', 'b')}
    """
    if not _list:
        return _list
    return list(set((tuple(sorted_func(i)) for i in _list)))


def remove_redundant_list(_list):
    return [i for n, i in enumerate(_list) if i not in _list[:n]]


def remove_redundant_angle(_list):
    id_list = []
    res = []
    for c1, c2 in combinations(_list, 2):
        if c1[1] == c2[1] and check_same_list([c1[0], c1[2]], [c2[0], c2[2]]):
            id_list.append([_list.index(c1), _list.index(c2)])

    for pair in id_list:
        res.append(_list[pair[0]])

    id_list = [y for x in id_list for y in x]
    if len(_list) > len(id_list):
        for l in _list:
            if l not in id_list:
                res.append(l)

    return res


def list_cluster_gap(data, max_gap):
    """
    对列表进行差值聚类
    """
    data.sort()
    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x - groups[-1][-1]) <= max_gap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups


def list_to_tuple(_list):
    res = []
    for lis in _list:
        res.append(tuple(lis))

    return res


def check_element_type(_list, _type):
    if len(_list) > 0:
        num = 0
        for e in _list:
            if not isinstance(e, _type):
                num = num + 1

        if num == 0:
            return True

    return False


def substitute_by_dict(primitive, map_dict):
    res = []
    for p in primitive:
        res.append(map_dict.get(p))

    return res


def list_composition(*lists):
    # List all possible composition from many list, each item is a tuple
    # Here lists is [list1, list2, list3], return a list of [(item1,item2,item3),...]

    # Length of result list and result list
    total = reduce(lambda x, y: x * y, map(len, lists))
    retList = []

    # Every item of result list
    for i in range(0, total):
        step = total
        tempItem = []
        for l in lists[0]:
            step /= len(l)
            tempItem.append(l[int(i / step % len(l))])
        retList.append(tempItem)

    return retList


def isNumber(number):
    try:
        number = sympify(number)
        if number.is_number:
            return True
        else:
            return False
    except:
        return False


def hasNumber(lst):
    for number in lst:
        if isNumber(number):
            return True
    return False


def findNumber(lst):
    for number in lst:
        if isNumber(number):
            return number
    return None


def isAlgebra(number):
    if isNumber(number):
        return True
    if isinstance(number, sympy.Basic) and '_' not in str(number) and 'angle' not in str(number):
        return True
    else:
        return False


def findAlgebra(lst):
    for number in lst:
        if isAlgebra(number):
            return number
    return None


def sort_angle(angle):
    assert len(angle) == 3
    if angle[0] > angle[2]: return angle[::-1]
    return angle


def sort_points(points):
    min_index = points.index(min(points))
    if points[(min_index - 1 + len(points)) % len(points)] > points[(min_index + 1) % len(points)]:
        sorted_points = points[min_index:] + points[:min_index]
    else:
        sorted_points = points[min_index::-1] + points[:min_index:-1]
    return sorted_points


def heron_triangle_formula(a, b, c):
    s = (a + b + c) / 2
    return math.sqrt(s * (s - a) * (s - b) * (s - c))


def angle_area_formula(a, b, angle):
    return 0.5 * a * b * math.sin(math.pi / 180.0 * angle)


def closest_to_number(sympy_list, target):
    """
    给定一个包含 sympy 表达式的列表和一个目标数字，返回列表中与目标数字最接近的元素。

    Args:
        sympy_list (list): 包含 sympy 表达式的列表
        target (float): 目标数字

    Returns:
        closest_expr: 与目标数字最接近的 sympy 表达式
    """
    # 将 sympy 表达式计算为数值
    numerical_list = [N(expr) for expr in sympy_list]

    # 找到与目标数字最接近的数值
    closest_value = min(numerical_list, key=lambda x: abs(x - target))

    # 找到与目标数字最接近的 sympy 表达式
    closest_expr = sympy_list[numerical_list.index(closest_value)]

    return closest_expr
