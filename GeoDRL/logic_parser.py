import os
import sys
from contextlib import contextmanager
from pyparsing import ParseResults
import numpy as np
import sympy
from kanren import facts
from pyparsing import Optional, alphanums, Forward, Group, Word, Literal, ZeroOrMore
from sympy import Symbol
from sympy import cos, sin, tan, cot, pi
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

from GeoDRL.extended_definition import ExtendedDefinition
from utils.common_utils import sort_points


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class LogicParser:
    def __init__(self, logic):
        assert isinstance(logic, ExtendedDefinition)
        self.logic = logic
        self.expression = Forward()

        identifier = Word(alphanums + ' +-*/.\\\{\}^_$\'')
        # integer  = Word( nums )
        lparen = Literal("(").suppress()  # suppress "(" in the result
        rparen = Literal(")").suppress()  # suppress ")" in the result

        arg = Group(self.expression) | identifier  # arg can be a grouping expression or a identifier
        args = arg + ZeroOrMore(Literal(",").suppress() + arg)  # args: arg1, [*arg2, *arg3, ...]

        self.expression <<= identifier + lparen + Optional(args) + rparen  # args is optional

    def parse(self, tree):
        parseResult = self.expression.parseString(tree)
        return parseResult

    @staticmethod
    def EvaluateSymbols(tree):
        a = parse_expr(tree, transformations=(standard_transformations + (implicit_multiplication_application,)))
        return a

    def getValue(self, expr, val=None):
        """
        Give an expression [expr].
        You should output its value if [val = None], otherwise execute [expr := val].
        """
        if type(expr) == str:
            if val is not None:
                self.logic.define_equal(Symbol(expr), val)
            else:
                if expr.find("angle") != -1:  # 'angle_1'
                    return Symbol(expr)
                try:
                    with suppress_stdout():
                        val = sympy.simplify(parse_latex(expr))
                    return val
                except Exception as e:
                    print(e)
                    return self.EvaluateSymbols(expr)  # special case: convert the expression to a symbol

        # Some logic forms may miss these phrases.
        if expr[0] == "Line":
            expr = ["LengthOf", expr]
        if expr[0] in ["Angle", "Arc"]:
            expr = ["MeasureOf", expr]

        if expr[0] == "MeasureOf":
            if type(expr[1]) == str:
                if val is None:
                    return Symbol(expr[1])
                self.logic.define_equal(val, Symbol(expr[1]))

            if expr[1][0] == "Angle":
                if len(expr[1]) == 2 and expr[1][1].isdigit():
                    # MeasureOf(Angle(1)), expr = ['MeasureOf', ['Angle', 1]]
                    angle_value = Symbol("angle " + str(expr[1][1]))  # {Symbol} angle_ABC
                    if val is None:
                        return angle_value
                    self.logic.define_equal(angle_value, val)
                else:
                    # MeasureOf(Angle(A,B,C)), expr = ['MeasureOf', ['Angle', 'A', 'B', 'C']]
                    angle = self.logic.parseAngle(expr[1][1:])  # ['A', 'B', 'C']
                    if val is not None:
                        self.logic.defineAngle(*angle, val)  # val = 2.0*angle_ADC, 83.00, 8.0*y + 2.0
                    else:
                        try:
                            angle_value = self.logic.find_angle_measure(angle)[0]
                            return angle_value
                        except:
                            angle_value = self.logic.newAngleSymbol(angle)  # {Symbol} angle_ABC
                            return angle_value

            if expr[1][0] == "Arc":
                arc = self.logic.parseArc(expr[1][1:])
                if val is not None:
                    self.logic.defineArc(*arc, val)
                    self.logic.defineAngle(arc[1], arc[0], arc[2], val)
                else:
                    try:
                        return self.logic.find_arc_measure(arc)[0]
                    except:
                        return self.logic.newArcSymbol(arc)

        if expr[0] == "LengthOf":
            # ['LengthOf', ['Line', 'B', 'C']]
            if expr[1][0] == "Line":
                line = expr[1][1:]
                if val is not None:
                    self.logic.defineLine(*line, val)
                else:
                    try:
                        length = self.logic.find_line_with_length(line)[0]  # 'line_CA', 'x', 'line_CA+x'
                        return length
                    except:
                        length = self.logic.newLineSymbol(line)
                        return length  # {Symbol} line_CA
            # Assume there is no LengthOf(Arc())

        if expr[0] == 'HypotenuseOf':
            line = self.HypotenuseOf(expr)
            return self.getValue(['LengthOf', ['Line', line[0], line[1]]])

        if expr[0] in ['AltitudeOf', 'HeightOf']:
            line = self.HeightOf(expr)
            return self.getValue(['LengthOf', ['Line', line[0], line[1]]])

        if expr[0] == 'BaseOf':
            line = self.BaseOf(expr)
            return self.getValue(['LengthOf', ['Line', line[0], line[1]]])

        if expr[0] == 'AreaOf':
            if expr[1] == "Shaded":
                # we can't determine shade now...
                pass
            elif expr[1][0] == 'Circle':
                self.logic.define_circle(expr[1][1])
                points = self.logic.find_points_on_circle(expr[1][1])
                if val is not None:
                    for point in points:
                        self.logic.define_length(expr[1][1], point, (val ** 0.5 / pi).subs(self.logic.variables))
                return
            elif type(expr[1][1]) == list:
                # ["AreaOf", ["Regular", ["Polygon", "A", "B", "C", "D"]]]
                self.dfsParseTree(expr[1])
                poly = sort_points(expr[1][1][1:])
                sym = self.logic.newAreaSymbol('Polygon({})'.format(','.join(poly)))
            else:
                self.parseQuad(expr[1])
                poly = sort_points(expr[1][1:])
                sym = self.logic.newAreaSymbol('Polygon({})'.format(','.join(poly)))
            if val is not None:
                self.logic.define_equal(sym, val)
            else:
                return sym

        if expr[0] == 'PerimeterOf':
            self.parseQuad(expr[1])
            points = expr[1][1:]
            lines = [(points[i % len(points)], points[(i + 1) % len(points)]) for i in range(len(points))]
            res = 0
            for line in lines:
                res += self.getValue(['LengthOf', ['Line', line[0], line[1]]])
            if val is not None:
                self.logic.build_equation(res, val)
            else:
                return res

        if expr[0] == 'DiameterOf':
            O = expr[1][1]
            point = self.logic.find_points_on_circle(O)[0]
            if val is not None:
                return self.getValue(['LengthOf', ['Line', O, point]], val / 2)
            else:
                return self.getValue(['LengthOf', ['Line', O, point]]) * 2

        if expr[0] == 'SideOf':
            lines = self.SideOf(expr)
            if val is not None:
                for line in lines:
                    self.getValue(['LengthOf', ['Line', line[0], line[1]]], val)
            else:
                return self.getValue(['LengthOf', ['Line', line[0][0], line[0][1]]])

        if expr[0] in ['RadiusOf', 'CircumferenceOf']:
            O = expr[1][1]
            point = self.logic.find_points_on_circle(O)[0]
            if expr[0] == 'RadiusOf':
                return self.getValue(['LengthOf', ['Line', O, point]])
            elif expr[0] == 'CircumferenceOf':
                return self.getValue(['LengthOf', ['Line', O, point]]) ** 2 * pi

        if expr[0] in ["SinOf", "CosOf", "TanOf", "CotOf"]:
            if expr[1][0] == "Angle":
                # In this case, 'MeasureOf' may be skipped.
                expr[1] = ['MeasureOf', expr[1]]

            mp = {"SinOf": sin, "CosOf": cos, "TanOf": tan, "CotOf": cot}
            return mp[expr[0]](self.getValue(expr[1]))

        if expr[0] in ["Add", "Mul", "SumOf", "HalfOf", "SquareOf", "SqrtOf"]:
            if expr[0] == "HalfOf":
                res = self.getValue(expr[1]) / 2.0
            if expr[0] == "SquareOf":
                res = self.getValue(expr[1]) ** 2
            if expr[0] == "SqrtOf":
                res = self.getValue(expr[1]) ** 0.5
            if expr[0] in ["Add", "SumOf"]:
                res = sum([self.getValue(x) for x in expr[1:]])
            if expr[0] in ['Mul']:
                res = 1  # Mul
                for x in expr[1:]:
                    res *= self.getValue(x)
            if val is not None:
                self.logic.build_equation(res, val)
            return res

    def parseQuad(self, tree):
        # tree = ['Rhombus', 'A', 'B', 'C', 'D']
        identifier = tree[0]
        if len(tree) == 2 and tree[1] == "$":
            polygon = self.logic.find_hidden_polygons(4)
            if len(polygon) != 1:
                return
            tree[1:] = polygon[0]

        if identifier in ["Quadrilateral", "Trapezoid"]:
            self.logic.definePolygon(tree[1:])
        if identifier == "Trapezoid":
            if self.logic.point_positions is not None:
                p = np.array([self.logic.point_positions[x] for x in tree[1:]])
                angle1 = self.calculate_angle_measure((p[1] - p[0]), (p[2] - p[3]))
                angle2 = self.calculate_angle_measure((p[2] - p[0]), (p[1] - p[3]))
                abs_angle1 = min(angle1, 180 - angle1)
                abs_angle2 = min(angle2, 180 - angle2)
                if abs_angle1 < abs_angle2:
                    self.Parallel(['Line', tree[1], tree[2]], ['Line', tree[3], tree[4]])
                else:
                    self.Parallel(['Line', tree[2], tree[3]], ['Line', tree[4], tree[1]])
                # p = [self.logic.point_positions[x] for x in tree[1:]]
                # cross = lambda u, v: u[0] * v[1] - u[1] * v[0]
                # c1 = cross((p[1][0] - p[0][0], p[1][1] - p[0][1]), (p[2][0] - p[3][0], p[2][0] - p[3][1]))
                # c2 = cross((p[3][0] - p[0][0], p[3][1] - p[0][1]), (p[2][0] - p[1][0], p[2][0] - p[1][1]))
                # if abs(c1) < abs(c2):
                #     self.Parallel(['Line', tree[1], tree[2]], ['Line', tree[3], tree[4]])
                # else:
                #     self.Parallel(['Line', tree[2], tree[3]], ['Line', tree[4], tree[1]])
        if identifier == "Rhombus":
            self.logic.definePolygon(tree[1:])
            self.Parallel(['Line', tree[1], tree[2]], ['Line', tree[4], tree[3]])
            self.Parallel(['Line', tree[2], tree[3]], ['Line', tree[1], tree[4]])
            self.Perpendicular(['Line', tree[1], tree[3]], ['Line', tree[2], tree[4]], True)
            for ch in range(1, 4):
                self.logic.lineEqual([tree[ch], tree[ch + 1]], [tree[ch + 1], tree[(ch + 1) % 4 + 1]])
        if identifier == "Parallelogram":
            self.logic.definePolygon(tree[1:])
            self.Parallel(['Line', tree[1], tree[2]], ['Line', tree[4], tree[3]])
            self.Parallel(['Line', tree[2], tree[3]], ['Line', tree[1], tree[4]])
            self.logic.lineEqual([tree[1], tree[2]], [tree[4], tree[3]])
            self.logic.lineEqual([tree[2], tree[3]], [tree[1], tree[4]])
        if identifier in ["Rectangle", "Square"]:
            self.logic.definePolygon(tree[1:])
            self.Parallel(['Line', tree[1], tree[2]], ['Line', tree[4], tree[3]])
            self.Parallel(['Line', tree[2], tree[3]], ['Line', tree[1], tree[4]])
            self.Perpendicular(['Line', tree[1], tree[2]], ['Line', tree[2], tree[3]])
            self.Perpendicular(['Line', tree[2], tree[3]], ['Line', tree[3], tree[4]])
            self.logic.lineEqual([tree[1], tree[2]], [tree[4], tree[3]])
            self.logic.lineEqual([tree[2], tree[3]], [tree[1], tree[4]])
        if identifier == "Square":
            self.logic.definePolygon(tree[1:])
            self.Parallel(['Line', tree[1], tree[2]], ['Line', tree[4], tree[3]])
            self.Parallel(['Line', tree[2], tree[3]], ['Line', tree[1], tree[4]])
            self.Perpendicular(['Line', tree[1], tree[3]], ['Line', tree[2], tree[4]])
            for ch in range(1, 4):
                self.logic.lineEqual([tree[ch], tree[ch + 1]], [tree[ch + 1], tree[(ch + 1) % 4 + 1]])
        if identifier == "Kite":
            self.logic.definePolygon(tree[1:])
            self.Perpendicular(['Line', tree[1], tree[3]], ['Line', tree[2], tree[4]])
            if self.logic.point_positions is not None:
                fp = lambda x: self.logic.point_positions[x]
                p1, p2, p3, p4 = fp(tree[1]), fp(tree[2]), fp(tree[3]), fp(tree[4])
                fdis = lambda x, y: ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5
                if abs(fdis(p1, p2) - fdis(p2, p3)) < abs(fdis(p2, p3) - fdis(p3, p4)):
                    self.logic.lineEqual([tree[1], tree[2]], [tree[2], tree[3]])
                    self.logic.lineEqual([tree[3], tree[4]], [tree[4], tree[1]])
                else:
                    self.logic.lineEqual([tree[2], tree[3]], [tree[3], tree[4]])
                    self.logic.lineEqual([tree[4], tree[1]], [tree[1], tree[2]])

    def PointLiesOnLine(self, point, line):
        line = self.logic.parseLine(line[1:])
        if line is not None:
            self.logic.defineLine(*line)
            self.logic.defineAngle(line[0], point, line[1], 180)
            sorted_line = sorted([line[0], line[1]])
            facts(self.logic.PointLiesOnLine, (point, sorted_line[0], sorted_line[1]))

    def PointLiesOnCircle(self, point, circle):
        assert circle[0] == "Circle", f"Expected 'Circle', but got {circle[0]}"
        if circle[1] == "$":
            circle[1] = self.logic.find_all_circles()[0]
        self.logic.defineCircle(circle[1], point)

    def Perpendicular(self, line1, line2, build_new_point=False):
        # Define the lines
        self.logic.defineLine(line1[1], line1[2])
        self.logic.defineLine(line2[1], line2[2])

        # Find all points on both lines
        s1 = set(self.logic.find_all_points_on_line(line1[1:]))
        s2 = set(self.logic.find_all_points_on_line(line2[1:]))

        # Check if there is exactly one intersection
        if len(s1 & s2) == 1:
            intersection = (s1 & s2).pop()
            # Correctly defining the perpendicular relationship here
            facts(self.logic.Perpendicular, (*line1[1:], *line2[1:]))

            # Define angles at the intersection
            for point1 in s1 - s2:
                for point2 in s2 - s1:
                    self.logic.defineAngle(point1, intersection, point2, 90)
                    self.logic.defineAngle(point2, intersection, point1, 90)
        else:
            # Handle the case where there isn't exactly one intersection
            pass

    def Parallel(self, line1, line2):
        line1 = self.logic.parseLine(line1[1:])
        line2 = self.logic.parseLine(line2[1:])
        if line1 is not None and line2 is not None:
            # print (line1, line2)
            self.logic.defineLine(*line1)
            self.logic.defineLine(*line2)
            self.logic.define_parallel(line1, line2)

    def LegOf(self, expr, refer_point=None):
        if expr[0] == "Isosceles" and expr[1][0] == "Trapezoid":
            expr = expr[1]
        if expr[0] == "Trapezoid":
            # determine parallels
            if self.logic.check_parallel((expr[1], expr[2]), (expr[3], expr[4])):
                LegCand = ((expr[1], expr[4]), (expr[2], expr[3]))
            elif self.logic.check_parallel((expr[2], expr[3]), (expr[1], expr[4])):
                LegCand = ((expr[1], expr[2]), (expr[3], expr[4]))
            else:
                raise RuntimeError

            if refer_point is None:
                return LegCand
            else:
                for Leg in LegCand:
                    if self.logic.check_angle_measure((Leg[0], refer_point, Leg[1], 180)):
                        return Leg
                # given order is wrong
                if self.logic.check_angle_measure((expr[1], refer_point, expr[2], 180)):
                    return expr[1], expr[2]
                if self.logic.check_angle_measure((expr[1], refer_point, expr[3], 180)):
                    return expr[1], expr[3]
                if self.logic.check_angle_measure((expr[1], refer_point, expr[4], 180)):
                    return expr[1], expr[4]
                if self.logic.check_angle_measure((expr[2], refer_point, expr[3], 180)):
                    return expr[2], expr[3]
                if self.logic.check_angle_measure((expr[2], refer_point, expr[4], 180)):
                    return expr[2], expr[4]
                if self.logic.check_angle_measure((expr[3], refer_point, expr[4], 180)):
                    return expr[3], expr[4]

    def BaseOf(self, expr):
        if expr[1][0] == "Trapezoid":
            if self.logic.check_parallel((expr[1][1], expr[1][2]), (expr[1][3], expr[1][4])):
                BaseCand = ((expr[1][1], expr[1][2]), (expr[1][3], expr[1][4]))
            elif self.logic.check_parallel((expr[1][2], expr[1][3]), (expr[1][1], expr[1][4])):
                BaseCand = ((expr[1][2], expr[1][3]), (expr[1][4], expr[1][1]))
            else:
                raise RuntimeError(f"Failed to identify parallel sides in Trapezoid: {expr[1]}")
            return BaseCand
        if expr[1][0] == "Isosceles":
            if self.logic.point_positions is not None:
                points = expr[1][1][1:]
                positions = [self.logic.point_positions[x] for x in points]
                fdis = lambda x, y: ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5
                mindis, minid = 999999, -1
                for i in range(3):
                    nowdis = abs(
                        fdis(positions[i], positions[1 - min(i, 1)]) - fdis(positions[i], positions[3 - max(1, i)]))
                    if nowdis < mindis:
                        mindis, minid = nowdis, i
                o, p, q = points[minid], points[1 - min(minid, 1)], points[3 - max(minid, 1)]
                return p, q
        if expr[1][0] == "Triangle":
            # Height - Base
            points = expr[1][1:]
            o, a = self.HeightOf(expr)
            points.remove(a)
            b, c = points
            return b, c

    def SideOf(self, expr):
        # regular polygon
        self.parseQuad(expr[1])
        if type(expr[1][1]) != str:
            points = expr[1][1][1:]
        else:
            points = expr[1][1:]
        return [(points[i % len(points)], points[(i + 1) % len(points)]) for i in range(len(points))]

    def calculate_angle_measure(self, v1, v2):
        try:
            # input: np.array
            return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) * 180 / np.pi
        except Exception as e:
            # 捕捉并处理异常
            raise RuntimeError(f"Error calculating angle measure between vectors {v1} and {v2}: {e}")

    def HypotenuseOf(self, expr):
        if expr[1][0] == 'Right':
            points = expr[1][1][1:]
        elif expr[1][0] == 'Triangle':
            points = expr[1][1:]
        else:
            raise RuntimeError

        a, b, c = points
        if self.logic.check_angle_measure((b, a, c, 90)):
            return b, c
        elif self.logic.check_angle_measure((a, b, c, 90)):
            return a, c
        elif self.logic.check_angle_measure((a, c, b, 90)):
            return a, b
        else:
            # check point position
            p = np.array([self.logic.point_positions[x] for x in points])
            angle0 = self.calculate_angle_measure(p[1] - p[0], p[2] - p[0])
            angle1 = self.calculate_angle_measure(p[0] - p[1], p[2] - p[1])
            angle2 = self.calculate_angle_measure(p[0] - p[2], p[1] - p[2])
            list = [np.abs(angle0 - 90), np.abs(angle1 - 90), np.abs(angle2 - 90)]
            right_index = np.argmin(list)
            self.logic.define_angle_measure(points[(right_index + 1) % 3], points[right_index],
                                            points[(right_index + 2) % 3], 90)
            return points[(right_index + 1) % 3], points[(right_index + 2) % 3]

    def HeightOf(self, expr):
        if expr[1][0] == 'Triangle':
            points = expr[1][1:]
            rightAngle_list = list(self.logic.find_all_90_angles().keys())
            for a, o, b in rightAngle_list:
                if a in points and b in points:
                    _t = points.copy()
                    _t.remove(a)
                    _t.remove(b)
                    c = _t[0]
                    if c == o:
                        continue
                    if self.logic.check_angle_measure((a, o, c, 180)):
                        return o, b
                    elif self.logic.check_angle_measure((b, o, c, 180)):
                        return o, a
                    else:
                        raise RuntimeError(f"Failed to determine height for triangle {expr[1]} with right angle at {o}")

    def BisectsAngle(self, line, angle):
        if line[1] != angle[2]:
            line[1], line[2] = line[2], line[1]
        self.logic.angleEqual([angle[1], angle[2], line[2]], [angle[3], angle[2], line[2]])

    def Midpoint(self, point, line):
        line = self.logic.parseLine(line, point)
        self.logic.lineEqual([line[0], point], [line[1], point])
        self.logic.defineAngle(line[0], point, line[1], 180)

    def dfsParseTree(self, tree):
        identifier = tree[0]

        ''' 1. Geometric Shapes '''
        if identifier == "Angle":
            if len(tree) == 4:
                self.logic.defineAngle(tree[1], tree[2], tree[3])
        if identifier == "Line":
            self.logic.defineLine(tree[1], tree[2])
        if identifier in ["Quadrilateral", "Parallelogram", "Rhombus", "Rectangle", "Square", "Trapezoid", "Kite"]:
            self.parseQuad(tree)
        if identifier == "Polygon":
            self.logic.definePolygon(tree[1:])
        if identifier == "Triangle":
            self.logic.definePolygon(tree[1:])

        ''' 2. Unary Geometric Attributes '''
        if identifier == "Isosceles" and tree[1][0] == "Triangle":
            if self.logic.point_positions is not None:
                points = tree[1][1:]
                positions = [self.logic.point_positions[x] for x in points]
                fdis = lambda x, y: ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5
                mindis, minid = 999999, -1
                for i in range(3):
                    nowdis = abs(
                        fdis(positions[i], positions[1 - min(i, 1)]) - fdis(positions[i], positions[3 - max(1, i)]))
                    if nowdis < mindis:
                        mindis, minid = nowdis, i
                o, p, q = points[minid], points[1 - min(minid, 1)], points[3 - max(minid, 1)]
                self.logic.lineEqual([o, p], [o, q])
                self.logic.angleEqual([o, p, q], [o, q, p])

        if identifier == "Isosceles" and tree[1][0] == "Trapezoid":
            self.parseQuad(tree[1])
            points = tree[1][1:]
            (a, b), (c, d) = self.BaseOf(['BaseOf', tree[1]])
            self.logic.lineEqual([a, c], [b, d])
            self.logic.angleEqual([a, b, c], [b, a, d])
            self.logic.angleEqual([b, c, d], [c, d, a])

        if identifier == "Regular" or identifier == "Equilateral":
            self.parseQuad(tree[1])
            points = sort_points(tree[1][1:])
            for i in range(1, len(points)):
                self.logic.lineEqual([points[0], points[1]], [points[i % len(points)], points[(i + 1) % len(points)]])
                self.logic.defineAngle(points[i % len(points)], points[(i + 1) % len(points)],
                                       points[(i + 2) % len(points)], 180 * (len(points) - 2) / len(points))

        if identifier == "IsHypotenuseOf":
            # when it occurs in text, it means 'IsAltitudeOf'. Poor annotation!
            pass

        if identifier == "IsAltitudeOf":
            altitude = tree[1][1:]
            triangle_points = tree[2][1:]
            if altitude[0] in triangle_points:
                a, o = altitude
            elif altitude[0] in triangle_points:
                o, a = altitude
            else:
                raise RuntimeError
            triangle_points.remove(a)
            b, c = triangle_points
            self.PointLiesOnLine(o, (b, c))
            self.Perpendicular(('Line', a, o), ('Line', b, c))

        ''' 4. Binary Geometric Relations '''
        if identifier == "PointLiesOnLine":
            self.PointLiesOnLine(tree[1], tree[2])
        if identifier == "PointLiesOnCircle":
            self.PointLiesOnCircle(tree[1], tree[2])
        if identifier == "Parallel":
            self.Parallel(tree[1], tree[2])
        if identifier == "Perpendicular":
            self.Perpendicular(tree[1], tree[2])
        if identifier == "IntersectAt":
            if tree[1][0] == "Line":
                for i in range(1, len(tree) - 1):
                    self.PointLiesOnLine(tree[-1][1], tree[i])
            elif tree[1][0] == "Circle":
                for i in range(1, len(tree) - 1):
                    self.PointLiesOnCircle(tree[-1][1], tree[i])
            else:
                raise RuntimeError("No such format for IntersectAt.")
        if identifier == "BisectsAngle":
            assert tree[1][0] == 'Line', f"Expected 'Line' at tree[1][0], but got {tree[1][0]}"
            if tree[2][0] == 'Angle':
                self.BisectsAngle(tree[1], ['Angle'] + self._find_angle(tree[2])[1:])
            elif tree[2][0] == 'Line':
                line1 = tree[1][1:]
                line2 = tree[2][1:]
                s1 = set(self.logic.find_all_points_on_line(line1))
                s2 = set(self.logic.find_all_points_on_line(line2))
                if len(s1 & s2) != 1:
                    # print ("The perpendicular lines", line1, line2, "should have one intersection.")
                    return
                intersection = (s1 & s2).pop()
                self.logic.lineEqual((line1[0], intersection), (intersection, line1[1]))
                self.logic.lineEqual((line2[0], intersection), (intersection, line2[1]))
            elif tree[2][0] == 'Triangle':
                triangle = tree[2][1:]
                o = set(tree[1][1:]) & set(triangle)
                triangle.remove(o.pop())
                self.BisectsAngle(tree[1], ['Angle'] + self._find_angle(triangle)[1:])
        if identifier == "Congruent":
            if tree[1][0] == "Triangle":
                self.logic.defineCongruentTriangle(tree[1][1:], tree[2][1:])
            # Do not implement the polygon with sides > 3.
        if identifier == "Similar":
            if tree[1][0] == "Triangle":
                self.logic.defineSimilarTriangle(tree[1][1:], tree[2][1:])
            else:
                self.logic.defineSimilarPolygon(tree[1][1:], tree[2][1:])
            # Do not implement the polygon with sides > 3.
        if identifier in ["Tangent", "Secant", "CircumscribedTo", "InscribedIn"]:
            if identifier == "InscribedIn":
                assert tree[2][0] == "Circle", f"Expected 'Circle' at tree[2][0], but got {tree[2][0]}"
                self.dfsParseTree(tree[1])
                if tree[1][0] in ["Triangle", "Quadrilateral", "Square", "Rhombus", "Kite", "Rectangle", "Pentagon",
                                  "Hexagon"]:
                    points = tree[1][1:]
                else:
                    points = tree[1][1][1:]
                for p in points:
                    self.PointLiesOnCircle(p, tree[2])
                if tree[1][0] in ["Square", "Rectangle"]:
                    self.PointLiesOnLine(tree[2][1], ['Line', tree[1][1], tree[1][3]])
                    self.PointLiesOnLine(tree[2][1], ['Line', tree[1][2], tree[1][4]])
            if identifier == "Tangent":
                assert tree[2][0] == "Circle", f"Expected 'Circle' at tree[2][0], but got {tree[2][0]}"
                line = tree[1][1:]
                O = tree[2][1]
                s1 = set(self.logic.find_all_points_on_line(line))
                s2 = set(self.logic.find_points_on_circle(O))
                intersection = s1 & s2
                assert len(intersection) > 0, f"Expected non-empty intersection, but got length {len(intersection)}"
                tangent_point = intersection.pop()
                self.Perpendicular(['Line', *line], ['Line', O, tangent_point])
                # self.logic.define_line(O, line[0])
                # self.logic.define_line(O, line[1])

        ''' 5. A-IsXOf-B  Geometric Relations '''
        if identifier == "IsMidpointOf":
            # IsMidpointOf(Point, Line/LegOf)
            if tree[2][0] == "Line":
                self.Midpoint(tree[1][1], tree[2][1:])
            elif tree[2][0] == "LegOf":
                self.Midpoint(tree[1][1], self.LegOf(tree[2][1], tree[1][1]))
            else:
                raise RuntimeError("No such format for IsMidpointOf.")

        if identifier == "IsCentroidOf":
            o = tree[1][1]
            points = self.logic.find_all_lines_for_point(o)
            if len(tree[2]) == 4:
                # Triangle
                a, b, c = tree[2][1:]
                angles = self.logic.find_all_180_angles()
                for p in points:
                    if (a, p, b) in angles:
                        self.logic.lineEqual([p, a], [p, b])
                        self.getValue(["LengthOf", ["Line", p, o]], self.getValue(["LengthOf", ["Line", c, o]]) / 2)
                    if (b, p, c) in angles:
                        self.logic.lineEqual([p, b], [p, c])
                        self.getValue(["LengthOf", ["Line", p, o]], self.getValue(["LengthOf", ["Line", a, o]]) / 2)
                    if (a, p, c) in angles:
                        self.logic.lineEqual([p, a], [p, c])
                        self.getValue(["LengthOf", ["Line", p, o]], self.getValue(["LengthOf", ["Line", b, o]]) / 2)

        if identifier == "IsIncenterOf":
            o = tree[1][1]
            if len(tree[2]) == 4:
                # Triangle
                a, b, c = tree[2][1:]
                self.logic.angleEqual([b, a, o], [c, a, o])
                self.logic.angleEqual([a, b, o], [c, b, o])
                self.logic.angleEqual([a, c, o], [b, c, o])
        if identifier == "IsRadiusOf":
            if tree[2][1] == tree[1][1]:
                self.PointLiesOnCircle(tree[1][2], tree[2])
            else:
                self.PointLiesOnCircle(tree[1][1], tree[2])
        if identifier == "IsDiameterOf":
            # print (tree[1], tree[2])
            self.PointLiesOnLine(tree[2][1], tree[1])
            self.PointLiesOnCircle(tree[1][1], tree[2])
            self.PointLiesOnCircle(tree[1][2], tree[2])
        if identifier == "IsMidsegmentOf":
            if tree[2][0] == 'Trapezoid':
                self.parseQuad(tree[2])
                Leg1 = self.LegOf(tree[2], tree[1][1])
                Leg2 = self.LegOf(tree[2], tree[1][2])
                self.Midpoint(tree[1][1], Leg1)
                self.Midpoint(tree[1][2], Leg2)
                Base1, Base2 = self.BaseOf(['BaseOf', tree[2]])
                self.Parallel(tree[1][1:], Base1)
                self.Parallel(tree[1][1:], Base2)
            else:
                segs = []
                for i in range(1, 3):
                    for k in range(1, 4):
                        if self.logic.find_angle_measure([tree[2][k], tree[1][i], tree[2][k % 3 + 1], 180]):
                            segs.append([tree[2][k], tree[2][k % 3 + 1]])
                assert len(segs) == 2, "Find Midsegment Error."
                self.Midpoint(tree[1][1], segs[0])
                self.Midpoint(tree[1][2], segs[1])
                baseline = [x for x in tree[2][1:] if x not in segs[0] or x not in segs[1]]
                assert len(baseline) == 2, "Find Midsegment Error."
                self.Parallel(tree[1][1:], baseline)
        if identifier == "IsMedianOf":
            if tree[2][0] == 'Triangle':
                line = tree[1][1:]
                tri = tree[2][1:]
                intersection = set(line) & set(tri)
                if len(intersection) == 0: return
                a = intersection.pop()
                line.remove(a)
                tri.remove(a)
                o = line[0]
                b, c = tri
                self.Midpoint(o, (b, c))

        if identifier == "IsChordOf":
            self.PointLiesOnCircle(tree[1][1], tree[2])
            self.PointLiesOnCircle(tree[1][2], tree[2])
        if identifier == "IsDiagonalOf":
            if len(tree[2]) == 5:
                self.parseQuad(tree[2])

        '''6. Numerical Attributes and Relations'''
        if identifier == "Equals":
            if tree[1][0] == "RadiusOf":
                O = tree[1][1][1]
                radius = self.EvaluateSymbols(tree[2])
                for p in self.logic.find_points_on_circle(O):
                    self.logic.defineLine(O, p, radius)
            elif tree[1][0] == "AltitudeOf" or tree[2][0] == "AltitudeOf":
                if tree[1][0] == "AltitudeOf":
                    triangle = tree[1][1]
                    line = tree[2]
                else:
                    triangle = tree[2][1]
                    line = tree[1]
                new_form = ["IsAltitudeOf", line, triangle]
                self.dfsParseTree(new_form)
            elif tree[1][0] == "RatioOf" and tree[2][0] == 'RatioOf':
                ratio = Symbol('RatioValue')
                val1 = self.getValue(tree[1][2])
                self.getValue(tree[1][1], ratio * val1)
                val2 = self.getValue(tree[2][2])
                self.getValue(tree[2][1], ratio * val2)
            else:
                # self.parseEquals(tree[1], tree[2])
                def _totlength(data):
                    if type(data) == list:
                        return sum([_totlength(x) for x in data])
                    return 1

                if _totlength(tree[1]) < _totlength(tree[2]):
                    # Put more complex expression to the left.
                    tree[1], tree[2] = tree[2], tree[1]
                val = self.getValue(tree[2])
                # print (tree[1], tree[2], val)
                self.getValue(tree[1], val)

                if tree[1][0] == "LengthOf" and tree[1][1][0] == "Line" and tree[2][0] == "LengthOf" and tree[2][1][
                    0] == "Line":
                    line1 = self.logic.parseLine(tree[1][1][1:])
                    line2 = self.logic.parseLine(tree[2][1][1:])
                    self.logic.lineEqual(line1, line2)
                    # self.logic.PutIntoEqualLineSet(tree[1][1][1:], tree[2][1][1:])
                if tree[1][0] == "MeasureOf" and tree[1][1][0] == "Angle" and tree[2][0] == "MeasureOf" and tree[2][1][
                    0] == "Angle":
                    if not (len(tree[1][1]) == 2 and tree[1][1][1].isdigit() or len(tree[2][1]) == 2 and tree[2][1][
                        1].isdigit()):
                        angle1 = self.logic.parseAngle(tree[1][1][1:])
                        angle2 = self.logic.parseAngle(tree[2][1][1:])
                        self.logic.angleEqual(angle1, angle2)
                if tree[1][0] == "MeasureOf" and tree[1][1][0] == "Arc" and tree[2][0] == "MeasureOf" and tree[2][1][
                    0] == "Arc":
                    arc1 = self.logic.parseArc(tree[1][1][1:])
                    arc2 = self.logic.parseArc(tree[2][1][1:])
                    self.logic.arcEqual(arc1, arc2)

    def _find_angle(self, phrase):
        if len(phrase) == 4:
            return ['Value', *phrase[1:4]]
        elif len(phrase) == 2:
            if phrase[1].isupper():
                return ['Value', *self.logic.parseAngle(phrase[1])]
            return ['Value', "angle" + str(phrase[1])]

    def findTarget(self, phrase):
        """
        Generate the target from the 'Find' phrase.
        The format is defined in 'logic_solver.Search()'.
        """
        assert phrase[0] == "Find", f"Expected 'Find' at phrase[0], but got {phrase[0]}"
        phrase = phrase[1]

        if type(phrase) == str:
            return ['Value', phrase]
        phrase = list(phrase)  # ['LengthOf', ['Line', 'O', 'X']]

        if phrase[0] == "LengthOf":
            phrase = phrase[1]
            if phrase[0] == "Arc":
                return ['Value', 'arc_length', *self.logic.parseArc(phrase[1:])]
            elif phrase[0] == "HypotenuseOf":
                return ['Value', *self.HypotenuseOf(phrase)]
            elif phrase[0] in ["AltitudeOf", "HeightOf"]:
                return ['Value', *self.HeightOf(phrase)]
            elif phrase[0] in ["BaseOf"]:
                return ['Value', *self.BaseOf(phrase)]
            elif phrase[0] in ["SideOf"]:
                return ['Value', *(self.SideOf(phrase)[0])]
            assert phrase[0] == "Line" and len(phrase) == 3, \
                f"Expected 'Line' and length 3, got {phrase[0]} and length {len(phrase)}"
            return ['Value', *phrase[1:3]]

        if phrase[0] == "MeasureOf":
            phrase = phrase[1]
            if phrase[0] == "Arc":
                arc = self.logic.parseArc(phrase[1:])
                return ['Value', arc[1], arc[0], arc[2]]
                # return ['Value', 'arc_measure', *self.logic.parseArc(phrase[1:])]
            elif phrase[0] == "Angle":
                return self._find_angle(phrase)
            return None

        if phrase[0] == "RadiusOf":
            O = phrase[1][1]
            return ['Value', O, self.logic.find_points_on_circle(O)[0]]

        if phrase[0] == "PerimeterOf" or phrase[0] == "CircumferenceOf":
            if len(phrase[1]) == 2:
                if isinstance(phrase[1][1], ParseResults):
                    phrase[1] = phrase[1][1]
                if phrase[1][0] == 'Circle' and phrase[1][1] == "$":
                    tmp = self.logic.find_all_circles()
                    if len(tmp) > 0:
                        phrase[1][1:] = list(tmp[0])
                elif phrase[1][1] == "$":
                    for i in range(4, 2, -1):
                        tmp = self.logic.find_hidden_polygons(i)
                        if len(tmp) > 0:
                            phrase[1][1:] = list(tmp[0])
                            break
            if len(phrase[1]) == 5:
                self.parseQuad(phrase[1])
            return ['Perimeter', *phrase[1][1:]]

        if phrase[0] == "AreaOf":
            if phrase[1][0] == 'Shaded' or phrase[1][0] == 'Green':
                # we can't handle it now...
                phrase[1] = phrase[1][1]
            if len(phrase[1]) == 2:
                if isinstance(phrase[1][1], ParseResults):
                    phrase[1] = phrase[1][1]
                if phrase[1][0] == 'Circle' and phrase[1][1] == "$":
                    tmp = self.logic.find_all_circles()
                    if len(tmp) > 0:
                        phrase[1][1:] = list(tmp[0])
                elif phrase[1][1] == "$":
                    for i in range(4, 2, -1):
                        tmp = self.logic.find_hidden_polygons(i)
                        if len(tmp) > 0:
                            phrase[1][1:] = list(tmp[0])
                            break
            if phrase[1][0] == 'Sector':
                return ['Sector', *phrase[1][1:]]
            if len(phrase[1]) == 5:
                self.parseQuad(phrase[1])
            return ['Area', *phrase[1][1:]]

        if phrase[0] == "RatioOf" and len(phrase) == 2:
            return self.findTarget(['Find', phrase[1]])

        if phrase[0] in ["SinOf", "CosOf", "TanOf", "CotOf", "HalfOf", "SquareOf", "SqrtOf"]:
            if phrase[1][0] == "Angle":
                # In this case, 'MeasureOf' may be skipped.
                phrase[1] = ['MeasureOf', phrase[1]]
            return [phrase[0], self.findTarget(['Find', phrase[1]])]

        if phrase[0] in ["RatioOf", "Add", "Mul", "SumOf"]:
            return [phrase[0]] + [self.findTarget(['Find', x]) for x in phrase[1:]]

        if phrase[0] == "ScaleFactorOf":
            return ['ScaleFactorOf', phrase[1], phrase[2]]

        return None
