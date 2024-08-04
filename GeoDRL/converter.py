from itertools import product, permutations

from sympy import rad, pi, sympify, Mul, Number, Add, Integer, Float, Symbol, cos, tan, cot, sqrt, symbols, sin, \
    Rational
from pyparsing import ParseResults

from GeoDRL.extended_definition import ExtendedDefinition
from GeoDRL.logic_parser import LogicParser
from GeoDRL.base_theorem import BaseTheorem
from utils.common_utils import isNumber, hasNumber, isAlgebra, findAlgebra, sort_points, sort_angle, findNumber

from kanren import run, var

import sympy


def _same(list1, list2):
    return any([pair[0] == pair[1] for pair in product(list1, list2)])


def sympy2latex(x):
    if isinstance(x, sympy.Basic):
        latex = sympy.latex(x)
        return latex
    else:
        return x


def Text2Logic(text, debug_mode=False):
    parser = LogicParser(ExtendedDefinition(debug=debug_mode))

    # Define diagram primitive elements
    parser.logic.point_positions = text['point_positions']

    isLetter = lambda ch: ch.upper() and len(ch) == 1
    parser.logic.define_point([_ for _ in parser.logic.point_positions if isLetter(_)])
    if debug_mode:
        print(parser.logic.point_positions)

    lines = text['line_instances']  # ['AB', 'AC', 'AD', 'BC', 'BD', 'CD']
    for line in lines:
        line = line.strip()
        if len(line) == 2 and isLetter(line[0]) and isLetter(line[1]):
            parser.logic.define_line(line[0], line[1])

    circles = text['circle_instances']  # ['O']
    for point in circles:
        parser.logic.define_circle(point)

    # Parse logic forms
    logic_forms = text['logic_forms']

    def sort_func(x):
        if "Find" in x:
            return 3
        if "AltitudeOf" in x or "HeightOf" in x:
            return 2
        if "Perpendicular" in x:
            return 1
        return -1

    def remove_spaces_from_tree(parse_tree):
        if isinstance(parse_tree, str):
            # 去除字符串中的空格
            return parse_tree.replace(' ', '')
        elif isinstance(parse_tree, list):
            # 递归地处理列表中的每个元素
            return [remove_spaces_from_tree(element) for element in parse_tree]
        elif isinstance(parse_tree, ParseResults):
            # 递归地处理 ParseResults 对象
            return ParseResults([remove_spaces_from_tree(element) for element in parse_tree])
        else:
            return parse_tree

    logic_forms = sorted(logic_forms, key=sort_func)

    target = None
    for logic_form in logic_forms:
        if logic_form.strip() != "":
            if debug_mode:
                print("The logic form is", logic_form)

            parse_tree = parser.parse(logic_form)
            parse_tree = remove_spaces_from_tree(parse_tree)
            if logic_form.find('Find') != -1:
                target = parser.findTarget(parse_tree)  # ['Value', 'A', 'C']
            else:
                parser.dfsParseTree(parse_tree)

    return parser, target


def Logic2Text(logic, reserved_info=None, debug_mode=False):
    ret_text = {}
    if reserved_info != None:
        ret_text['point_positions'] = reserved_info['point_positions']
        ret_text['line_instances'] = reserved_info['line_instances']
        ret_text['circle_instances'] = reserved_info['circle_instances']

    logic_forms = []

    # Circle(O)
    circle_list = logic.find_all_circles()
    logic_forms.extend(["Circle({})".format(circle) for circle in circle_list])
    # Triangle(A,B,C)
    triangle_list = logic.find_all_triangles()
    logic_forms.extend(
        ["Triangle({},{},{})".format(triangle[0], triangle[1], triangle[2]) for triangle in triangle_list])
    # Quadrilateral(A,B,C,D)
    quadrilateral_list = logic.find_all_quadrilaterals()
    logic_forms.extend(
        ["Quadrilateral({},{},{},{})".format(quadrilateral[0], quadrilateral[1], quadrilateral[2], quadrilateral[3]) for
         quadrilateral in quadrilateral_list])
    # Pentagon(A,B,C,D,E)
    pentagon_list = logic.find_all_pentagons()
    logic_forms.extend(
        ["Pentagon({},{},{},{},{})".format(pentagon[0], pentagon[1], pentagon[2], pentagon[3], pentagon[4]) for pentagon
         in pentagon_list])

    # PointLiesOnCircle(A, Circle(O))  Ignored radius
    pointOnCircle_list = logic.find_all_points_on_circles()
    for pointOnCircle in pointOnCircle_list:
        logic_forms.extend(
            ["PointLiesOnCircle({},Circle({}))".format(point, pointOnCircle[1]) for point in pointOnCircle[0]])

    # Equals(LengthOf(Line(A, B)), Value) or Equals(LengthOf(Line(A, B)), LengthOf(Line(C, D)))
    length_list = logic.find_all_irredundant_lines_with_length()
    for length in length_list:
        if f"line_{length[0]}{length[1]}" == str(length[2]) or f"line_{length[1]}{length[0]}" == str(length[2]):
            continue
        else:
            logic_forms.extend(["Equals(line_{}{},{})".format(length[0], length[1], length[2])])
    # Equals(MeasureOf(Angle(A, B, C)), Value) or Equals(MeasureOf(Angle(A, B, C)), MeasureOf(Angle(D, E, F)))
    angleMeasure_list = logic.find_all_irredundant_angle_measures()
    for angleMeasure in angleMeasure_list:
        if f"angle_{angleMeasure[0]}{angleMeasure[1]}{angleMeasure[2]}" == str(
                angleMeasure[3]) or f"angle_{angleMeasure[2]}{angleMeasure[1]}{angleMeasure[0]}" == str(
            angleMeasure[3]):
            continue
        else:
            logic_forms.extend(
                ["Equals(angle_{}{}{},{})".format(angleMeasure[0], angleMeasure[1], angleMeasure[2], angleMeasure[3])])
    # Equals(MeasureOf(Arc(O, A, B)), Value) or Equals(MeasureOf(Arc(O, A, B)), Arc(O, C, D))
    arcMeasure_list = logic.fine_all_arc_measures()
    for arcMeasure in arcMeasure_list:
        if f"arc_{arcMeasure[0]}{arcMeasure[1]}{arcMeasure[2]}" == str(arcMeasure[3]):
            continue
        else:
            logic_forms.extend(
                ["Equals(arc_{}{}{},{})".format(arcMeasure[0], arcMeasure[1], arcMeasure[2], arcMeasure[3])])

    # Parallel(Line(A, B), Line(C, D))
    parallel_list = logic.find_all_parallels()
    logic_forms.extend(
        ["Parallel(Line({},{}),Line({},{}))".format(parallel[0][0], parallel[0][1], parallel[1][0], parallel[1][1]) for
         parallel in parallel_list])

    # Equals(m, n)
    x = var()
    y = var()
    res = run(0, (x, y), logic.Equal(x, y))
    logic_forms.extend(["Equals({},{})".format(sym1, sym2) for sym1, sym2 in res])
    res = run(0, (x, y), logic.Equation(x, y))
    logic_forms.extend(["Equals({},{})".format(sym1, sym2) for sym1, sym2 in res])

    # Similar(Triangle(A,B,C),Triangle(D,E,F))
    a = var()
    b = var()
    c = var()
    d = var()
    e = var()
    f = var()
    # Congruent(Triangle(A,B,C),Triangle(D,E,F))
    congruentTriangle_list = run(0, ((a, b, c), (d, e, f)), logic.CongruentTriangle(a, b, c, d, e, f))
    logic_forms.extend(
        ["Congruent(Triangle({}),Triangle({}))".format(','.join(congruentTriangle[0]), ','.join(congruentTriangle[1]))
         for congruentTriangle in congruentTriangle_list])
    similarTriangle_list = run(0, ((a, b, c), (d, e, f)), logic.SimilarTriangle(a, b, c, d, e, f))
    logic_forms.extend(
        ["Similar(Triangle({}),Triangle({}))".format(','.join(similarTriangle[0]), ','.join(similarTriangle[1])) for
         similarTriangle in similarTriangle_list])

    # Similar(Polygon(),Polygon())
    polygons = logic.find_all_similar_polygons()
    for poly1, poly2 in polygons:
        logic_forms.extend(["Similar(Polygon({}),Polygon({}))".format(','.join(poly1), ','.join(poly2))])

    ret_text['logic_forms'] = logic_forms

    return ret_text


def getTargetObject(logic, target):
    assert target is not None, "target is None"
    if target[0] == 'Value':
        if len(target) == 5:
            # return 'arc_' + ''.join(target[2:])
            return 'angle_' + ''.join(logic.get_same_angle_key((target[3], target[2], target[4])))
        if len(target) == 4:
            return 'angle_' + ''.join(logic.get_same_angle_key(target[1:]))
        if len(target) == 3:
            return 'line_' + ''.join(sorted(target[1:]))
        if len(target) == 2:
            return 'variable_' + target[1]
    if target[0] == 'Area':
        if len(target) == 2:
            return 'circle_' + target[1]
        if len(target) == 4:
            return 'triangle_' + ''.join(sorted(target[1:]))
        if len(target) > 4:
            return 'polygon_' + ''.join(sort_points(target[1:]))
    if target[0] == 'Perimeter':
        if len(target) == 2:
            circle = target[1]
            points_on_circle = sorted(logic.find_points_on_circle(circle))
            return 'line_' + ''.join(sorted(points_on_circle[0] + circle))
        else:
            poly = target[1:]
            return ['line_' + ''.join(sorted([poly[i], poly[(i + 1) % len(poly)]])) for i in range(len(poly))]
    if target[0] == 'Sector':
        O, A, B = target[1:]
        return ['angle_' + sort_angle(A + O + B), 'line_' + ''.join(sorted(O + A))]
    if target[0] in ["SinOf", "CosOf", "TanOf", "CotOf", "HalfOf", "SquareOf", "SqrtOf"]:
        return getTargetObject(logic, target[1])
    if target[0] in ["RatioOf", "Add", "Mul", "SumOf"]:
        return [getTargetObject(logic, target[i]) for i in range(1, len(target))]
    if target[0] == 'ScaleFactorOf':
        if target[1][0] == "Shape" and len(target[1]) == 2:
            line = (target[1][1], target[2][1])
            points = logic.find_all_points_on_line(line)
            O = sorted(set(points) - set(line))[0]
            return ['line_' + ''.join(sorted(O + line[1])), 'line_' + ''.join(sorted(O + line[0]))]
        else:
            shape1 = target[1] if type(target[1][1]) == str else target[1][1]
            shape2 = target[2] if type(target[2][1]) == str else target[2][1]
            return [getTargetObject(logic, ['Area', *shape1[1:]]),
                    getTargetObject(logic, ['Area', *shape2[1:]])]


def create_sympy_equation(logic, target):
    assert target is not None, "target is None"
    if target[0] == 'SinOf':
        inner_expr = create_sympy_equation(logic, target[1])
        return sin(inner_expr)
    elif target[0] == 'CosOf':
        inner_expr = create_sympy_equation(logic, target[1])
        return cos(inner_expr)
    elif target[0] == 'TanOf':
        inner_expr = create_sympy_equation(logic, target[1])
        return tan(inner_expr)
    elif target[0] == 'CotOf':
        inner_expr = create_sympy_equation(logic, target[1])
        return cot(inner_expr)
    elif target[0] == 'SquareOf':
        inner_expr = create_sympy_equation(logic, target[1])
        return inner_expr**2
    elif target[0] == 'SqrtOf':
        inner_expr = create_sympy_equation(logic, target[1])
        return sqrt(inner_expr)
    elif target[0] == 'HalfOf':
        inner_expr = create_sympy_equation(logic, target[1])
        return inner_expr / 2
    elif target[0] == 'RatioOf':
        if len(target[1:]) == 2:
            return create_sympy_equation(logic, target[1]) / create_sympy_equation(logic, target[2])
        else:
            return create_sympy_equation(logic, target[1])
    elif target[0] == 'Add':
        term1 = create_sympy_equation(logic, target[1])
        term2 = create_sympy_equation(logic, target[2])
        return term1 + term2
    elif target[0] == 'Mul':
        factor1 = create_sympy_equation(logic, target[1])
        factor2 = create_sympy_equation(logic, target[2])
        return factor1 * factor2
    elif target[0] == 'SumOf':
        return sum(create_sympy_equation(logic, t) for t in target[1:])
    elif target[0] == 'Value':
        if len(target) == 5:
            return (symbols('angle_' + ''.join(logic.get_same_angle_key((target[3], target[2], target[4])))) *
                    symbols('line_' + ''.join(sorted((target[3], target[2])))))
            # return symbols('arc_' + ''.join(target[2:]))
        if len(target) == 4:
            return symbols('angle_' + ''.join(logic.get_same_angle_key(target[1:])))
        if len(target) == 3:
            return symbols('line_' + ''.join(sorted(target[1:])))
        if len(target) == 2:
            return symbols(target[1])
    elif target[0] == 'Area':
        if len(target) == 2:
            return symbols('circle_' + target[1])
        if len(target) == 4:
            return symbols('triangle_' + ''.join(sorted(target[1:])))
        if len(target) > 4:
            return symbols('polygon_' + ''.join(sort_points(target[1:])))
    elif target[0] == 'Perimeter':
        if len(target) == 2:
            circle = target[1]
            points_on_circle = sorted(logic.find_points_on_circle(circle))
            radius = symbols('line_' + ''.join(sorted(points_on_circle[0] + circle)))
            return 2 * pi * radius
        else:
            poly = target[1:]
            side_lines = [symbols('line_' + ''.join(sorted([poly[i], poly[(i + 1) % len(poly)]])))
                          for i in range(len(poly))]
            return sum(side_lines)
    elif target[0] == 'Sector':
        O, A, B = target[1:]
        angle_measure = symbols('angle_' + ''.join(logic.get_same_angle_key((A, O, B))))
        radius = symbols('line_' + ''.join(sorted(O + A)))
        return radius**2 * angle_measure / 2
    elif target[0] == 'ScaleFactorOf':
        if target[1][0] == "Shape" and len(target[1]) == 2:
            line = (target[1][1], target[2][1])
            points = logic.find_all_points_on_line(line)
            O = sorted(set(points) - set(line))[0]
            line_1 = symbols('line_' + ''.join(sorted(O + line[1])))
            line_2 = symbols('line_' + ''.join(sorted(O + line[0])))
            return line_1 / line_2
        else:
            shape1 = target[1] if type(target[1][1]) == str else target[1][1]
            shape2 = target[2] if type(target[2][1]) == str else target[2][1]
            area_1 = create_sympy_equation(logic, ('Area', *shape1[1:]))
            area_2 = create_sympy_equation(logic, ('Area', *shape2[1:]))
            return area_1 / area_2
    return None


def convert_term(term):
    # 如果项是一个乘法，并且其中包含数字和符号，不转换
    if isinstance(term, Mul) and any(isinstance(arg, Number) for arg in term.args):
        return term  # 保持原样，因为涉及到变量
    elif isinstance(term, Number):
        return term * pi / 180  # 如果项是一个纯数字，转换为弧度
    return term  # 其他情况保持原样


def convert_degrees_to_radians(expr_list):
    if expr_list == 'None':
        return expr_list
    new_expr_list = []
    for expr in expr_list:
        # 将表达式中的数字部分转换为弧度
        if isinstance(expr, (Integer, Float, float)):  # 如果表达式是整数或浮点数
            new_expr_list.append(Rational(expr) * pi / 180)
        else:
            parsed_expr = sympify(expr)
            # 分别处理表达式中的每个项
            if isinstance(parsed_expr, Add):  # 如果是加法表达式
                new_expr_list.append(Add(*[convert_term(term) for term in parsed_expr.args]))
            else:
                new_expr_list.append(convert_term(parsed_expr))  # 如果不是加法表达式，直接转换

    return new_expr_list


def Logic2Graph(logic, target):
    # Node: Point, Line, Angle, Arc, Circle, Triangle, Polygon
    # Relation:
    # <Point, Point>: Connected
    # <Point, Line>: EndPoint, LiesOnLine
    # <Point, Angle>: Vertex, SidePoint
    # <Point, Arc>: Center, EndPoint
    # <Point, Circle>: Center, LiesOnCircle
    # <Point, Triangle> / <Point, Polygon>: Vertex
    # <Line, Line>: Equal, Parallel, Perpendicular
    # <Line, Triangle> / <Line, Polygon>: Side
    # <Angle, Angle>: Equal
    # <Angle, Triangle> / <Angle, Polygon>: Interior
    # <Arc, Arc>: Equal
    # <Triangle, Triangle>: Congruent
    # <Triangle, Triangle>: Similar
    # <Polygon, Polygon>: Similar
    node = []
    node_type = []
    node_attr = []
    node_visual_attr = []
    target_node = []
    edge_st_index = []
    edge_ed_index = []
    edge_attr = []

    base_theorem = BaseTheorem(logic)

    points = sorted(logic.find_all_points())

    lines = sorted(logic.find_all_irredundant_lines())
    length = []
    for line in lines:
        val = logic.find_line_with_length(line, skip_if_has_number=False)
        if len(val) > 0:
            length.append(val)
        else:
            length.append('None')

    angles = []
    angleMeasures = []
    all_angles = sorted(logic.find_all_irredundant_angles())
    for angle in all_angles:
        angle = logic.get_same_angle_key(angle)
        val = logic.find_angle_measure(angle, skip_if_has_number=False)
        if len(val) > 0:
            new_val = []
            added_angle_symbol = []
            for v in val:
                if 'angle' in str(v):
                    if '_' in str(v):
                        _, result = str(v).split('_', 1)
                        if not logic.check_same_angle(angle, result):
                            same_angle = logic.get_same_angle_key(result)
                            if same_angle != angle and same_angle not in added_angle_symbol:
                                v = Symbol("angle_" + ''.join([str(ch) for ch in same_angle]))
                                added_angle_symbol.append(same_angle)
                            new_val.append(v)
                    else:
                        new_val.append(v)
                elif v != 180:
                    new_val.append(v)
            if len(new_val) > 0:
                angles.append(angle)
                angleMeasures.append(new_val)
            else:
                if 180 not in val:
                    angles.append(angle)
                    angleMeasures.append('None')

    arcs = []
    arcMeasures = []
    all_arcs = sorted(logic.find_all_arcs())
    for arc in all_arcs:
        val = logic.find_arc_measure(arc, skip_if_has_number=False)
        if len(val) > 0:
            if 180 not in val:
                arcs.append(arc)
                arcMeasures.append(val)

    circles = logic.find_all_circles()

    triangles = sorted(logic.find_all_triangles())
    tri_lines = []
    tri_angles = []
    triangleAreas = []
    for tri in triangles:
        tri_lines.extend([sorted([tri[i], tri[(i + 1) % 3]]) for i in range(3)])
        tri_angles.extend([(tri[0], tri[2], tri[1]), (tri[0], tri[1], tri[2]), (tri[1], tri[0], tri[2])])
        AreaSymbol = sympy.Symbol("AreaOf(Triangle({}))".format(','.join(sort_points(tri))))
        if AreaSymbol in logic.variables and isAlgebra(logic.variables[AreaSymbol]):
            v = logic.variables[AreaSymbol]
            triangleAreas.append(v)
        else:
            triangleAreas.append('None')
    Vertex = []
    Vertex_R = []
    Interior = []
    Interior_R = []
    Side = []
    Side_R = []
    for i, tri in enumerate(triangles):
        for j in range(3):
            Vertex.append((tri[j], 'triangle_' + ''.join(tri)))
            Vertex_R.append(('triangle_' + ''.join(tri), tri[j]))
            t_line = tri_lines[i * 3 + j]
            if tuple(t_line) not in lines:
                lines.append(t_line)
                length.append('None')
            Side.append(('line_' + ''.join(t_line), 'triangle_' + ''.join(tri)))
            Side_R.append(('triangle_' + ''.join(tri), 'line_' + ''.join(t_line)))

            t_angle = logic.get_same_angle_key(tri_angles[i * 3 + j])
            t_measure = logic.find_angle_measure(t_angle)[0] if hasNumber(logic.find_angle_measure(t_angle)) else 'None'
            NOT_IN = True
            for angle in angles:
                if logic.check_same_angle(t_angle, angle):
                    t_angle = angle
                    t_measure = logic.find_angle_measure(angle)
                    NOT_IN = False
                    break
            if NOT_IN:
                if not isAlgebra(t_measure):
                    angleMeasures.append('None')
                else:
                    angleMeasures.append([t_measure])
                angles.append(t_angle)
            Interior.append(('angle_' + ''.join(t_angle), 'triangle_' + ''.join(tri)))
            Interior_R.append(('triangle_' + ''.join(tri), 'angle_' + ''.join(t_angle)))

    polygons = sorted(logic.find_all_quadrilaterals() + logic.find_all_pentagons())
    poly_lines = []
    poly_angles = []
    polygonAreas = []
    for poly in polygons:
        poly_lines.append([sorted([poly[i], poly[(i + 1) % len(poly)]]) for i in range(len(poly))])
        poly_angles.append([[poly[i], poly[(i + 1) % len(poly)], poly[(i + 2) % len(poly)]] for i in range(len(poly))])
        AreaSymbol = sympy.Symbol("AreaOf(Polygon({}))".format(','.join(sort_points(poly))))
        if AreaSymbol in logic.variables and isAlgebra(logic.variables[AreaSymbol]):
            t_measure = logic.variables[AreaSymbol]
            polygonAreas.append(t_measure)
        else:
            polygonAreas.append('None')

    for i, poly in enumerate(polygons):
        for j in range(len(poly)):
            Vertex.append((poly[j], 'polygon_' + ''.join(poly)))
            Vertex_R.append(('polygon_' + ''.join(poly), poly[j]))
            t_line = poly_lines[i][j]
            if tuple(t_line) not in lines:
                lines.append(t_line)
                length.append('None')
            Side.append(('line_' + ''.join(t_line), 'polygon_' + ''.join(poly)))
            Side_R.append(('polygon_' + ''.join(poly), 'line_' + ''.join(t_line)))

            t_angle = logic.get_same_angle_key(poly_angles[i][j])
            t_measure = logic.find_angle_measure(t_angle)[0] if hasNumber(logic.find_angle_measure(t_angle)) else 'None'
            NOT_IN = True
            for angle in angles:
                if logic.check_same_angle(t_angle, angle):
                    t_angle = angle
                    t_measure = logic.find_angle_measure(angle)
                    NOT_IN = False
                    break
            if NOT_IN:
                if not isAlgebra(t_measure):
                    angleMeasures.append('None')
                else:
                    angleMeasures.append([t_measure])
                angles.append(t_angle)
            Interior.append(('angle_' + ''.join(t_angle), 'polygon_' + ''.join(poly)))
            Interior.append(('polygon_' + ''.join(poly), 'angle_' + ''.join(t_angle)))

    # <Point, Point>
    connected_points = []
    for line in lines:
        connected_points.append((line[0], line[1]))
        connected_points.append((line[1], line[0]))
    # <Point, Line>
    endpoint_line = []
    endpoint_R_line = []
    for line in lines:
        endpoint_line.append((line[0], 'line_' + ''.join(line)))
        endpoint_line.append((line[1], 'line_' + ''.join(line)))
        endpoint_R_line.append(('line_' + ''.join(line), line[0]))
        endpoint_R_line.append(('line_' + ''.join(line), line[1]))
    pointLiesOnLine = []
    pointLiesOnLine_R = []
    p, a, b = var(), var(), var()
    res = run(0, (p, a, b), logic.PointLiesOnLine(p, a, b))
    for p, a, b in list(res):
        if a > b: a, b = b, a
        pointLiesOnLine.append((p, f'line_{a}{b}'))
        pointLiesOnLine_R.append((f'line_{a}{b}', p))
    # <Point, Angle>
    vertex_angle = []
    vertex_R_angle = []
    sidePoint_angle = []
    sidePoint_R_angle = []
    for angle in angles:
        same_angles = logic.get_same_angle_value(angle)
        vertex_angle.append((angle[1], 'angle_' + ''.join(angle)))
        vertex_R_angle.append(('angle_' + ''.join(angle), angle[1]))

        # 初始化一个集合来存储所有相同角的顶点
        side_points = set()

        # 添加当前角的顶点
        side_points.add(angle[0])
        side_points.add(angle[2])

        # 如果存在相同的角，将它们的顶点也添加到集合中
        if same_angles:
            for same_angle in same_angles:
                side_points.add(same_angle[0])
                side_points.add(same_angle[2])

        # 对每个唯一的顶点，添加到sidePoint_angle和sidePoint_R_angle
        for point in side_points:
            sidePoint_angle.append((point, 'angle_' + ''.join(angle)))
            sidePoint_R_angle.append(('angle_' + ''.join(angle), point))
    # <Point, Arc>
    center_arc = []
    center_R_arc = []
    endpoint_arc = []
    endpoint_R_arc = []
    for arc in arcs:
        center_arc.append((arc[0], 'arc_' + ''.join(arc)))
        center_R_arc.append(('arc_' + ''.join(arc), arc[0]))
        endpoint_arc.append((arc[1], 'arc_' + ''.join(arc)))
        endpoint_arc.append((arc[2], 'arc_' + ''.join(arc)))
        endpoint_R_arc.append(('arc_' + ''.join(arc), arc[1]))
        endpoint_R_arc.append(('arc_' + ''.join(arc), arc[2]))
    # <Point, Circle>
    center_cirlce = []
    center_R_cirlce = []
    pointLiesOnCircle = []
    pointLiesOnCircle_R = []
    for circle in circles:
        center_cirlce.append((circle, 'circle_' + circle))
        center_R_cirlce.append(('circle_' + circle, circle))
        for point in logic.find_points_on_circle(circle):
            pointLiesOnCircle.append((point, 'circle_' + circle))
            pointLiesOnCircle_R.append(('circle_' + circle, point))
    # <Line, Line>
    # Equal = []
    Parallel = []
    # Perpendicular = []
    # for s in logic.EqualLineSet:
        # for l1, l2 in permutations(s, 2):
            # Equal.append(('line_'+''.join(l1), 'line_'+''.join(l2)))
    for l1, l2 in logic.find_all_parallels():
        l1 = ''.join(sorted(l1))
        l2 = ''.join(sorted(l2))
        Parallel.append(('line_' + ''.join(l1), 'line_' + ''.join(l2)))
        Parallel.append(('line_' + ''.join(l2), 'line_' + ''.join(l1)))
    # for l1, l2 in logic.find_all_perpendicular():
    #     l1 = ''.join(sorted(l1))
    #     l2 = ''.join(sorted(l2))
    #     Perpendicular.append(('line_' + ''.join(l1), 'line_' + ''.join(l2)))
    #     Perpendicular.append(('line_' + ''.join(l2), 'line_' + ''.join(l1)))
    # <Line, Angle>
    AngleSide = []
    for angle in angles:
        point1 = angle[0]
        vertex = angle[1]
        point2 = angle[2]
        line1_points = logic.find_all_points_on_line([vertex, point1])
        line2_points = logic.find_all_points_on_line([vertex, point2])
        for point in line1_points:
            if point != vertex:
                candidate_line = (vertex, point)
                matching_line = (vertex, point1)
                if base_theorem.calc_cross_angle(matching_line, candidate_line) < 30:
                    for line in lines:
                        if set(line) == set(candidate_line):
                            AngleSide.append((('line_' + ''.join(line)), 'angle_' + angle))
                            break
        for point in line2_points:
            if point != vertex:
                candidate_line = (vertex, point)
                matching_line = (vertex, point2)
                if base_theorem.calc_cross_angle(matching_line, candidate_line) < 30:
                    for line in lines:
                        if set(line) == set(candidate_line):
                            AngleSide.append((('line_' + ''.join(line)), 'angle_' + angle))
                            break
    # <Angle, Angle>
    # for s in logic.EqualAngleSet:
        # for a1, a2 in permutations(s, 2):
        #     a1 = logic.get_same_angle_key(a1)
        #     a2 = logic.get_same_angle_key(a2)
        #     Equal.append(('angle_'+''.join(a1), 'angle_'+''.join(a2)))
    # <Arc, Arc>
    # for s in logic.EqualArcSet:
    #     for a1, a2 in permutations(s, 2):
    #         Equal.append(('arc_'+''.join(a1), 'arc_'+''.join(a2)))
    # <Triangle, Triangle>
    # A small fault: side match
    Congruent = []
    for tri1, tri2 in logic.find_all_congruent_triangles():
        Congruent.append(('triangle_' + ''.join(sorted(tri1)), 'triangle_' + ''.join(sorted(tri2))))
        Congruent.append(('triangle_' + ''.join(sorted(tri2)), 'triangle_' + ''.join(sorted(tri1))))
    Similar = []
    for tri1, tri2 in logic.find_all_similar_triangles():
        Similar.append(('triangle_' + ''.join(sorted(tri1)), 'triangle_' + ''.join(sorted(tri2))))
        Similar.append(('triangle_' + ''.join(sorted(tri2)), 'triangle_' + ''.join(sorted(tri1))))
    for poly1, poly2 in logic.find_all_similar_polygons():
        Similar.append(('polygon_' + ''.join(sort_points(poly1)), 'polygon_' + ''.join(sort_points(poly2))))
        Similar.append(('polygon_' + ''.join(sort_points(poly2)), 'polygon_' + ''.join(sort_points(poly1))))

    node.extend([point for point in points])
    node_type.extend(['Point' for point in points])
    node_attr.extend(['None' for point in points])
    node_visual_attr.extend(['None' for point in points])
    node.extend(['line_' + ''.join(line) for line in lines])
    node_type.extend(['Line' for line in lines])
    node_attr.extend([l for l in length])
    numeric_length = []
    for l in length:
        if l == 'None':
            numeric_length.append('None')
        else:
            numeric_length.append(findNumber(l))
    visual_length = [base_theorem.calc_distance_from_point_to_point(line[0], line[1]) for line in lines]
    # 找到 visual_length 中的最大值
    max_length = max(visual_length)
    # 计算缩放比例
    scale_factor = 100 / max_length
    # 对所有值进行缩放
    scaled_visual_length = [length * scale_factor for length in visual_length]
    node_visual_attr.extend(scaled_visual_length)
    # # 确保所有有效的视觉边长和实际边长都是浮点数
    # valid_pairs = [(float(v), float(l)) for v, l in zip(visual_length, numeric_length) if l != 'None' and l is not None]
    # # 计算所有有效对的缩放比例
    # scaling_factors = [l / v for v, l in valid_pairs]
    # if len(scaling_factors) > 0:
    #     # 计算平均缩放比例
    #     average_scaling_factor = sum(scaling_factors) / len(scaling_factors)
    #     # 使用平均缩放比例处理visual_length中的所有边长
    #     scaled_visual_length = [v * average_scaling_factor for v in visual_length]
    #     node_visual_attr.extend(scaled_visual_length)
    # else:
    #     node_visual_attr.extend(visual_length)
    node.extend(['angle_' + ''.join(angle) for angle in angles])
    node_type.extend(['Angle' for angle in angles])
    for angleMeasure in angleMeasures:
        node_attr.append(convert_degrees_to_radians(angleMeasure))
    node_visual_attr.extend([base_theorem.calc_angle_measure(tuple(char for char in angle), is_rad=True)
                             for angle in angles])
    node.extend(['arc_' + ''.join(arc) for arc in arcs])
    node_type.extend(['Arc' for arc in arcs])
    for arcMeasure in arcMeasures:
        node_attr.append(convert_degrees_to_radians(arcMeasure))
    node_visual_attr.extend(['None' for arc in arcs])
    node.extend(['circle_' + circle for circle in circles])
    node_type.extend(['Circle' for circle in circles])
    node_attr.extend(['None' for circle in circles])
    node_visual_attr.extend(['None' for circle in circles])
    node.extend(['triangle_' + ''.join(tri) for tri in triangles])
    node_type.extend(['Triangle' for tri in triangles])
    node_attr.extend([a for a in triangleAreas])
    node_visual_attr.extend(['None' for tri in triangles])
    node.extend(['polygon_' + ''.join(poly) for poly in polygons])
    node_type.extend(['Polygon' for poly in polygons])
    node_attr.extend([a for a in polygonAreas])
    node_visual_attr.extend(['None' for poly in polygons])

    targetObj = getTargetObject(logic, target)
    if type(targetObj) != list: targetObj = [targetObj]
    for t in targetObj:
        if t.startswith('variable_'):
            variable = t.split('_')[-1]
            for i, attrs in enumerate(node_attr):
                if isinstance(attrs, list):
                    for attr in attrs:
                        if isinstance(attr, sympy.Basic) and sympy.Symbol(variable) in attr.free_symbols:
                            target_node.append(node[i])
        else:
            if t in node:
                target_node.append(t)

    target_equation = create_sympy_equation(logic, target)

    for connected_st, connected_ed in connected_points:
        edge_st_index.append(node.index(connected_st))
        edge_ed_index.append(node.index(connected_ed))
        edge_attr.append('Connected')
    for endpoint, line in endpoint_line:
        edge_st_index.append(node.index(endpoint))
        edge_ed_index.append(node.index(line))
        edge_attr.append('Endpoint')
    for line, endpoint in endpoint_R_line:
        edge_st_index.append(node.index(line))
        edge_ed_index.append(node.index(endpoint))
        edge_attr.append('Endpoint_R')
    for point, line in pointLiesOnLine:
        edge_st_index.append(node.index(point))
        edge_ed_index.append(node.index(line))
        edge_attr.append('LiesOnLine')
    for line, point in pointLiesOnLine_R:
        edge_st_index.append(node.index(line))
        edge_ed_index.append(node.index(point))
        edge_attr.append('LiesOnLine_R')
    for point, angle in vertex_angle:
        edge_st_index.append(node.index(point))
        edge_ed_index.append(node.index(angle))
        edge_attr.append('Vertex')
    for angle, point in vertex_R_angle:
        edge_st_index.append(node.index(angle))
        edge_ed_index.append(node.index(point))
        edge_attr.append('Vertex_R')
    for point, angle in sidePoint_angle:
        edge_st_index.append(node.index(point))
        edge_ed_index.append(node.index(angle))
        edge_attr.append('Sidepoint')
    for angle, point in sidePoint_R_angle:
        edge_st_index.append(node.index(angle))
        edge_ed_index.append(node.index(point))
        edge_attr.append('Sidepoint_R')
    for line, angle in AngleSide:
        edge_st_index.append(node.index(line))
        edge_ed_index.append(node.index(angle))
        edge_attr.append('AngleSide')
    for point, arc in center_arc:
        edge_st_index.append(node.index(point))
        edge_ed_index.append(node.index(arc))
        edge_attr.append('Center')
    for arc, point in center_R_arc:
        edge_st_index.append(node.index(arc))
        edge_ed_index.append(node.index(point))
        edge_attr.append('Center_R')
    for point, arc in endpoint_arc:
        edge_st_index.append(node.index(point))
        edge_ed_index.append(node.index(arc))
        edge_attr.append('Endpoint')
    for arc, point in endpoint_R_arc:
        edge_st_index.append(node.index(arc))
        edge_ed_index.append(node.index(point))
        edge_attr.append('Endpoint_R')
    for point, circle in center_cirlce:
        edge_st_index.append(node.index(point))
        edge_ed_index.append(node.index(circle))
        edge_attr.append('Center')
    for circle, point in center_R_cirlce:
        edge_st_index.append(node.index(circle))
        edge_ed_index.append(node.index(point))
        edge_attr.append('Center_R')
    for point, circle in pointLiesOnCircle:
        edge_st_index.append(node.index(point))
        edge_ed_index.append(node.index(circle))
        edge_attr.append('LiesOnCircle')
    for circle, point in pointLiesOnCircle_R:
        edge_st_index.append(node.index(circle))
        edge_ed_index.append(node.index(point))
        edge_attr.append('LiesOnCircle_R')
    for point, poly in Vertex:
        edge_st_index.append(node.index(point))
        edge_ed_index.append(node.index(poly))
        edge_attr.append('Vertex')
    for poly, point in Vertex_R:
        edge_st_index.append(node.index(poly))
        edge_ed_index.append(node.index(point))
        edge_attr.append('Vertex_R')
    # for st, ed in Equal:
    #     edge_st_index.append(node.index(st))
    #     edge_ed_index.append(node.index(ed))
    #     edge_attr.append('Equal')
    for l1, l2 in Parallel:
        edge_st_index.append(node.index(l1))
        edge_ed_index.append(node.index(l2))
        edge_attr.append('Parallel')
    # for l1, l2 in Perpendicular:
    #     edge_st_index.append(node.index(l1))
    #     edge_ed_index.append(node.index(l2))
    #     edge_attr.append('Perpendicular')
    for line, poly in Side:
        edge_st_index.append(node.index(line))
        edge_ed_index.append(node.index(poly))
        edge_attr.append('Side')
    for poly, line in Side_R:
        edge_st_index.append(node.index(poly))
        edge_ed_index.append(node.index(line))
        edge_attr.append('Side_R')
    for angle, poly in Interior:
        edge_st_index.append(node.index(angle))
        edge_ed_index.append(node.index(poly))
        edge_attr.append('Interior')
    for poly, angle in Interior_R:
        edge_st_index.append(node.index(poly))
        edge_ed_index.append(node.index(angle))
        edge_attr.append('Interior_R')
    for poly1, poly2 in Congruent:
        edge_st_index.append(node.index(poly1))
        edge_ed_index.append(node.index(poly2))
        edge_attr.append('Congruent')
    for poly1, poly2 in Similar:
        edge_st_index.append(node.index(poly1))
        edge_ed_index.append(node.index(poly2))
        edge_attr.append('Similar')

    edge_index = [edge_st_index, edge_ed_index]
    new_node_attr = []
    for i in range(len(node_attr)):
        if not isinstance(node_attr[i], (str, list)):
            node_attr[i] = [node_attr[i]]
    for attr_list in node_attr:
        if attr_list == 'None':
            new_node_attr.append(attr_list)
        else:
            attr_list = [str(float(_)).rstrip("0").rstrip(".")
                         if isNumber(_) and '.' in str(_) else str(_) for _ in attr_list]
            new_node_attr.append(attr_list)

    return {"node": node,
            "node_type": node_type,
            "node_attr": new_node_attr,
            "node_visual_attr": node_visual_attr,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "target_node": target_node,
            "target_equation": target_equation,
            "point_positions": logic.point_positions}
