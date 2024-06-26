import math

from utils.common_utils import isNumber, hasNumber, heron_triangle_formula, angle_area_formula, sort_points, isAlgebra
import sympy
from sympy import Symbol, Number
from sympy import cos, sin, pi, solve, nonlinsolve
from itertools import permutations, product, combinations
from func_timeout import func_timeout, FunctionTimedOut


class LogicSolver:
    def __init__(self, logic, target=None):
        self.logic = logic
        self.can_search = False
        self.hasSolution = False
        self.equations = []
        self.circle_theorem = {
            "low_theorem": {
                1: self.circle_definition,
                2: self.thales_theorem,
                3: self.inscribed_angle_theorem,
            },
            "high_theorem": {
                1: self.tangent_secant_theorem,
                2: self.chord_theorem,
            }
        }
        self.parallel_theorem = {
            "low_theorem": {
                1: self.parallel_lines_theorem,
            },
            "high_theorem": {

            }

        }
        self.single_triangle_theorem = {
            "low_theorem": {
                1: self.triangle_anglesum_theorem,
                2: self.isosceles_triangle_theorem_side,
                3: self.isosceles_triangle_theorem_angle,
                4: self.equilateral_triangle_theorem,
                5: self.pythagoras_theorem,
                6: self.triangle_center_of_gravity_theorem,
            },
            "high_theorem": {
                1: self.angle_bisector_theorem,
            }
        }
        self.double_triangle_theorem = {
            "low_theorem": {
                1: self.congruent_triangles_proving_theorem,
                2: self.congruent_triangles_theorem,
            },
            "high_theorem": {
                1: self.similar_triangle_proving_theorem,
                2: self.similar_triangle_theorem,
            }
        }
        self.polygon_theorem = {
            "low_theorem": {

            },
            "high_theorem": {
                1: self.similar_polygon_theorem,
                2: self.median_line_theorem,
                3: self.area_equation_theorem,
                4: self.polygon_anglesum_theorem,
            }
        }
        self.complex_single_triangle_angle_theorem = {
            "low_theorem": {
                1: self.law_of_sines,
            },
            "high_theorem": {
                1: self.law_of_cosines,
            },
        }
        self.auxiliary_line = {
            "low_theorem": {

            },
            "high_theorem": {
                1: self.connecting_two_points,
            }
        }
        self.function_maps = {}
        self.function_maps.update(
            {k + len(self.function_maps): v for k, v in self.circle_theorem["low_theorem"].items()})
        self.function_maps.update(
            {k + len(self.function_maps): v for k, v in self.parallel_theorem["low_theorem"].items()})
        self.function_maps.update(
            {k + len(self.function_maps): v for k, v in self.single_triangle_theorem["low_theorem"].items()})
        self.function_maps.update(
            {k + len(self.function_maps): v for k, v in self.double_triangle_theorem["low_theorem"].items()})
        self.function_maps.update(
            {k + len(self.function_maps): v for k, v in self.polygon_theorem["low_theorem"].items()})
        self.function_maps.update({k + len(self.function_maps): v for k, v in
                                   self.complex_single_triangle_angle_theorem["low_theorem"].items()})
        self.function_maps.update(
            {k + len(self.function_maps): v for k, v in self.circle_theorem["high_theorem"].items()})
        self.function_maps.update(
            {k + len(self.function_maps): v for k, v in self.parallel_theorem["high_theorem"].items()})
        self.function_maps.update(
            {k + len(self.function_maps): v for k, v in self.single_triangle_theorem["high_theorem"].items()})
        self.function_maps.update(
            {k + len(self.function_maps): v for k, v in self.double_triangle_theorem["high_theorem"].items()})
        self.function_maps.update(
            {k + len(self.function_maps): v for k, v in self.polygon_theorem["high_theorem"].items()})
        self.function_maps.update({k + len(self.function_maps): v for k, v in
                                   self.complex_single_triangle_angle_theorem["high_theorem"].items()})

        self.step_lst = []
        self.target = None

    @staticmethod
    def _triangleEqual(length, angle, original_angle):
        """
        Please consider the order in the parameters
        length[0..5](Boolean)   length[ch]<->length[ch+3]
        angle[0..5] (Boolean)   angle[ch] <-> angle[ch+3]
        The order of original_angle[0..5](list) matters.
        """
        if sum(length) >= 3 or (sum(length) >= 1 and sum(angle) >= 2):
            return True  # SSS or AAS

        if sum(length) >= 2 and sum(angle) == 1:
            if all([angle[0], length[1], length[2]]) or all([angle[1], length[0], length[2]]) or all(
                    [angle[2], length[0], length[1]]):
                return True  # SAS

            for i in range(3):
                if angle[i] == True and (original_angle[i][0] == 90 or original_angle[i + 3][0] == 90) and length[
                    i] == True:
                    return True  # HL
        return False

    @staticmethod
    def _traingleSimilar(angle, ratio):
        if sum(angle) >= 2 or sum(ratio) >= 2:
            return True  # SSS or AAA
        if sum(angle) == 1 and sum(ratio) == 1:
            if (angle[0] and ratio[0]) or (angle[1] and ratio[1]) or (angle[2] and ratio[2]):
                return True
        return False

    @staticmethod
    def _hasSymbol(expr):
        if type(expr) in [int, float]:
            return False
        return len(expr.free_symbols) > 0

    @staticmethod
    def _generateAngles(tri):
        return [(tri[0], tri[2], tri[1]), (tri[0], tri[1], tri[2]), (tri[1], tri[0], tri[2])]

    @staticmethod
    def _generateLines(tri):
        return [(tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])]

    @staticmethod
    def _isComplex(st):
        return st.find("sin") != -1 or st.find("cos") != -1 or st.find("**2") != -1

    @staticmethod
    def _isTrig(st):
        return st.find("sin") != -1 or st.find("cos") != -1

    @staticmethod
    def _same(list1, list2):
        return any([pair[0] == pair[1] for pair in product(list1, list2)])

    @staticmethod
    def _equal(list1, list2):
        return any([pair[0].equals(pair[1]) if isinstance(pair[0], sympy.Basic) else pair[0] == pair[1] for pair in
                    product(list1, list2)])

    def _isPrimitive(self, expr):
        return self._hasSymbol(expr) and all([str(expr).find(t) == -1 for t in ['+', '-', '*']])

    def Solve_Equations(self):
        # add equations for angles, lines and arcs
        for line in self.logic.find_all_lines():
            lst = self.logic.find_line_with_length(line, skip_if_has_number=False)  # [line_CX + line_XD, 24.0]
            for i in range(1, len(lst)):
                # print ("[equations] line equations", line, lst[i-1], lst[i])
                self.equations.append(lst[i] - lst[i - 1])
        for angle in self.logic.find_all_angles():
            lst = self.logic.find_angle_measure(angle, skip_if_has_number=False)
            # [angle_ODC, -angle_COD - angle_ODC + 180, angle_OCD]
            for i in range(1, len(lst)):
                # print ("[equations] angle equations", angle, lst[i-1], lst[i], lst[i] - lst[i-1])
                self.equations.append(lst[i] - lst[i - 1])
        for arc in self.logic.find_all_arcs():  # arc = ('O', 'B', 'C')
            lst = self.logic.find_arc_measure(arc, skip_if_has_number=False)  # [360 - arc_OCB, angle_BOC, arc_OBC]
            for i in range(1, len(lst)):
                # print ("[equations] arc equations", angle, lst[i-1], lst[i])
                self.equations.append(lst[i] - lst[i - 1])

        for equation in self.logic.find_all_equations():
            self.equations.append(equation[0] - equation[1])

        self.equations = list(set(self.equations))  # remove redundant equations quickly
        self.equations, temp_equations = [], self.equations
        mp = []
        for equation in temp_equations:
            if not type(equation) in [float, int] and len(equation.free_symbols) > 0:  # unknown variables
                symbols = set(equation.free_symbols)  # {line_XD, line_CX} # symbols: unknown variables in the equation
                if symbols in mp: continue

                # New Feature: Avoid duplicated equations. # TODO BUG ???

                def discard_zero(t):
                    return 0.0 if abs(t) <= 1e-15 else t

                self.equations.append(equation.xreplace({n: discard_zero(n) for n in equation.atoms(Number)}))
                mp.append(symbols)  # mp = [{line_XD, line_CX}, {angle_BOC, arc_OBC}, ...

        if len(self.equations) == 0:
            return False
        if self.logic.debug:
            print("Try to solve: ", self.equations)

        # solutions1 = solve(self.equations, dict=True, manual=True)  # do not use the polys/matrix method
        try:
            solutions1 = func_timeout(20, solve, kwargs=dict(f=self.equations, dict=True, manual=True))
        except FunctionTimedOut:
            # raise TimeoutError
            solutions1 = []

        complexity = sum([self._isComplex(str(t)) for t in self.equations])
        if complexity <= 3:
            # solutions2 = solve(self.equations, dict = True, manual = False)
            try:
                solutions2 = func_timeout(20, solve, kwargs=dict(f=self.equations, dict=True, manual=False))
            except FunctionTimedOut:
                # raise TimeoutError
                solutions2 = []
            except Exception as e:
                solutions2 = []
        else:
            solutions2 = []

        solutions1_ = [sol for sol in solutions1 if not any(
            [isNumber(t) and t <= 0 for t in sol.values()])]  # [{line_CX: line_XD, angle_BOC: 0.5*arc_ODC}]
        solutions2_ = [sol for sol in solutions2 if not any(
            [isNumber(t) and t <= 0 for t in sol.values()])]  # [{line_CX: line_XD, angle_BOC: 0.5*arc_ODC}]

        solutions = solutions1_ if len(solutions1_) > len(solutions2_) else solutions2_
        # print(solutions)
        if len(solutions) == 0:
            total_symbols = set()
            for e in self.equations:
                if self._hasSymbol(e):
                    free = list(e.free_symbols)
                    for f in free:
                        total_symbols.add(f)
            total_symbols = list(total_symbols)
            if len(total_symbols) > 0:
                # res = list(nonlinsolve(list(self.equations), total_symbols) )
                try:
                    res = list(func_timeout(20, nonlinsolve, args=(list(self.equations), total_symbols)))
                except FunctionTimedOut:
                    res = []
                    # raise TimeoutError
                except Exception as e:
                    res = []

                if len(res) > 0:
                    for j in range(len(res)):
                        sol = dict()
                        for i in range(len(total_symbols)):
                            if total_symbols[i] != list(res[j])[i]:
                                sol[total_symbols[i]] = list(res[j])[i]
                        solutions.append(sol)
        if len(solutions) >= 1:
            # Handle with multiple solution
            estimate = lambda sol: sum([str(expr)[0] != '-' for expr in sol.values()])  # negative value
            solution = max(solutions, key=estimate)  # we like a solution with less negative values :)
            nowdict = {}
            if self.logic.debug:
                print("Solve out: ", solution)
            for key, value in solution.items():
                nowdict[key] = value
            self.hasSolution = True
            self.logic.variables = {key: value if type(value) in [int, float] else value.subs(nowdict)
                                    for key, value in self.logic.variables.items()}
            # We may substitute the key in the previous dict further.
            self.logic.variables.update(nowdict)
            return True
        self.hasSolution = len(self.equations) == 0
        return False

    def triangle_anglesum_theorem(self):
        Update = False
        triangles = self.logic.find_all_triangles()
        for tri in triangles:
            angles = self._generateAngles(tri)
            measures = [self.logic.find_angle_measure(x) for x in angles]
            unknowns = [i for i in range(3) if not hasNumber(measures[i])]
            if 1 <= len(unknowns) <= 2:
                idx = unknowns[0]
                other1, other2 = measures[(idx + 1) % 3][0], measures[(idx + 2) % 3][0]
                Update = self.logic.define_angle_measure(*angles[idx], 180 - other1 - other2) or Update
            if len(unknowns) == 3:
                Update = self.logic.define_angle_measure(*angles[0], 180 - measures[1][0] - measures[2][0]) or Update
        return Update

    def isosceles_triangle_theorem_side(self):
        Update = False
        triangles = self.logic.find_all_triangles()
        for tri in triangles:
            angles = self._generateAngles(tri)
            measures = [self.logic.find_angle_measure(x) for x in angles]
            lines = self._generateLines(tri)
            for ch in permutations([0, 1, 2]):
                if self._same(measures[ch[0]], measures[ch[1]]):
                    Update = self.logic.lineEqual(lines[ch[0]], lines[ch[1]]) or Update
                    self.logic.PutIntoEqualLineSet(lines[ch[0]], lines[ch[1]])
        return Update

    def isosceles_triangle_theorem_angle(self):
        Update = False
        triangles = self.logic.find_all_triangles()
        for tri in triangles:
            angles = self._generateAngles(tri)
            lines = self._generateLines(tri)
            length = [self.logic.find_line_with_length(x) for x in lines]
            for ch in permutations([0, 1, 2]):
                if self._same(length[ch[0]], length[ch[1]]):
                    Update = self.logic.angleEqual(angles[ch[0]], angles[ch[1]]) or Update
                    self.logic.PutIntoEqualAngleSet(angles[ch[0]], angles[ch[1]])
        return Update

    def equilateral_triangle_theorem(self):
        Update = False
        # Equilateral Triangle
        triangles = self.logic.find_all_triangles()
        for tri in triangles:
            angles = self._generateAngles(tri)
            lines = self._generateLines(tri)
            length = [self.logic.find_line_with_length(x) for x in lines]
            for ch in permutations([0, 1, 2]):
                if self._same(length[ch[0]], length[ch[1]]) and self._same(length[ch[0]], length[ch[2]]):
                    Update = self.logic.define_angle_measure(*angles[0], 60) or Update
                    Update = self.logic.define_angle_measure(*angles[1], 60) or Update
                    Update = self.logic.define_angle_measure(*angles[2], 60) or Update
                    break

        return Update

    def congruent_triangles_proving_theorem(self):
        Update = False
        triangles = self.logic.find_all_triangles()
        comb = combinations(triangles, 2)
        for pair in comb:
            for tri1 in permutations(pair[0]):
                tri2 = pair[1]
                if self.logic.check_congruent_triangle(tri1, tri2):
                    continue
                lines = self._generateLines(tri1) + self._generateLines(tri2)
                length = [self.logic.find_line_with_length(x) for x in lines]
                angles = [self.logic.find_angle_measure(x) for x in
                          self._generateAngles(tri1) + self._generateAngles(tri2)]
                s = self._same
                same_length = [s(length[0], length[3]), s(length[1], length[4]), s(length[2], length[5])]
                same_angle = [s(angles[0], angles[3]), s(angles[1], angles[4]), s(angles[2], angles[5])]
                if self._triangleEqual(same_length, same_angle, angles):
                    Update = True
                    self.logic.defineCongruentTriangle(tri1, tri2)
        return Update

    def congruent_triangles_theorem(self):
        Update = False
        for tri1, tri2 in self.logic.find_all_congruent_triangles():
            lines = self._generateLines(tri1) + self._generateLines(tri2)
            for ch in range(3):
                Update = self.logic.lineEqual(lines[ch], lines[ch + 3]) or Update
                self.logic.PutIntoEqualLineSet(lines[ch], lines[ch + 3])
                Update = self.logic.angleEqual(self._generateAngles(tri1)[ch],
                                               self._generateAngles(tri2)[ch]) or Update
                self.logic.PutIntoEqualAngleSet(self._generateAngles(tri1)[ch],
                                                self._generateAngles(tri2)[ch])
        return Update

    def circle_definition(self):
        Update = False
        circles = self.logic.find_all_circles()
        for circle in circles:
            points = self.logic.find_points_on_circle(circle)
            if len(points) > 1:
                # The length of each radius is same.
                for i in range(len(points) - 1):
                    Update = self.logic.lineEqual((circle, points[i]), (circle, points[i + 1])) or Update
                    self.logic.PutIntoEqualLineSet((circle, points[i]), (circle, points[i + 1]))
        return Update

    def thales_theorem(self):
        Update = False
        circles = self.logic.find_all_circles()
        for circle in circles:
            points = self.logic.find_points_on_circle(circle)
            for p1, p2, p3 in permutations(points, 3):
                angle_measure = self.logic.find_angle_measure((p1, circle, p3))
                if hasNumber(angle_measure) and angle_measure[0] == 180 and self.logic.check_angle((p1, p2, p3)):
                    Update = self.logic.define_angle_measure(p1, p2, p3, 90) or Update
        return Update

    def inscribed_angle_theorem(self):
        Update = False
        circles = self.logic.find_all_circles()
        for center in circles:
            points = self.logic.find_points_on_circle(center)
            for x, y in permutations(points, 2):
                if self.logic.cross(center, x, y) > 0:
                    continue
                circumferences = []
                for z in points:
                    if z == x or z == y: continue
                    if all([self.logic.check_line((p, q)) for p, q in [(x, z), (y, z)]]):
                        circumferences.append(x + z + y)
                        center_angle = self.logic.find_arc_measure((center, x, y))[0]
                        # we have to determine which arc z lies on
                        if self.logic.cross(center, x, y) * self.logic.cross(z, x, y) > 0:
                            Update = self.logic.defineAngle(x, z, y, 0.5 * center_angle) or Update
                        else:
                            Update = self.logic.defineAngle(x, z, y, 180 - 0.5 * center_angle) or Update
                for angle in circumferences[1:]:
                    self.logic.PutIntoEqualAngleSet(circumferences[0], angle)
        return Update

    def parallel_lines_theorem(self):
        Update = False
        parallels = self.logic.find_all_parallels()
        for line1, line2 in parallels:
            # If the line is represented by symbol, we should change it to two points.
            line1, line2 = self.logic.parseLine(line1), self.logic.parseLine(line2)

            # We want to figure out the order of the parallel lines.
            if self.logic.point_positions is not None:
                fdis = lambda x, y: (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2
                dismax, idp, idq = -1, None, None
                for p, q in product(line1, line2):
                    nowdis = fdis(self.logic.point_positions[p], self.logic.point_positions[q])
                    if nowdis > dismax:
                        dismax = nowdis
                        idp, idq = p, q
                if line1.index(idp) == line2.index(idq):
                    line1 = line1[::-1]

            # Now start to use Parallel Theorems
            A = self.logic.find_all_points_on_line(line1)
            B = self.logic.find_all_points_on_line(line2)
            for p, q in product(A, B):
                C = self.logic.find_all_points_on_line([p, q])
                if not self.logic.check_line((p, q)): continue
                angles = [(A[0], p, q), (A[-1], p, q), (B[0], q, p), (B[-1], q, p),
                          (A[0], p, C[0]), (A[-1], p, C[0]), (B[0], q, C[-1]), (B[-1], q, C[-1])]
                measures = [self.logic.find_angle_measure(x) for x in angles]
                # It is guaranteed that all the angle has at least one symbol.
                if A[0] != p and B[-1] != q:
                    Update = self.logic.angleEqual(angles[0], angles[3]) or Update
                    self.logic.PutIntoEqualAngleSet(angles[0], angles[3])
                if A[-1] != p and B[0] != q:
                    Update = self.logic.angleEqual(angles[1], angles[2]) or Update
                    self.logic.PutIntoEqualAngleSet(angles[1], angles[2])
                if A[0] != p and B[0] != q:
                    if measures[2] != []:
                        Update = self.logic.define_angle_measure(*angles[0], 180 - measures[2][0]) or Update
                    if C[0] != p:
                        Update = self.logic.angleEqual(angles[2], angles[4]) or Update
                        self.logic.PutIntoEqualAngleSet(angles[2], angles[4])
                    if C[-1] != q:
                        Update = self.logic.angleEqual(angles[0], angles[6]) or Update
                        self.logic.PutIntoEqualAngleSet(angles[0], angles[6])
                if A[-1] != p and B[-1] != q:
                    if measures[3] != []:
                        Update = self.logic.define_angle_measure(*angles[1], 180 - measures[3][0]) or Update
                    if C[0] != p:
                        Update = self.logic.angleEqual(angles[3], angles[5]) or Update
                        self.logic.PutIntoEqualAngleSet(angles[3], angles[5])
                    if C[-1] != q:
                        Update = self.logic.angleEqual(angles[1], angles[7]) or Update
                        self.logic.PutIntoEqualAngleSet(angles[1], angles[7])

        return Update

    def func11_flat_angle_theorem(self):
        # If point O lies on segment (A, B), then AOC + COB = 180.
        Update = False
        angles = self.logic.find_all_angle_measures()
        for angle in angles:
            if angle[3] != 180: continue
            points = set(self.logic.find_all_lines_for_point(angle[1])) - \
                     set(self.logic.find_all_points_on_line((angle[0], angle[2])))
            for point in points:
                val = self.logic.find_angle_measure((point, angle[1], angle[2]))[0]
                Update = self.logic.define_angle_measure(angle[0], angle[1], point, 180 - val) or Update

        angles = self.logic.find_all_angle_measures()
        return Update

    def chord_theorem(self):
        Update = False
        circles = self.logic.find_all_circles()
        for center in circles:
            points = self.logic.find_points_on_circle(center)
            for p1, p2, p3, p4 in permutations(points, 4):
                if self.logic.check_line((p1, p2)) and self.logic.check_line((p3, p4)):
                    intersection = set(self.logic.find_all_points_on_line((p1, p2))) & \
                                   set(self.logic.find_all_points_on_line((p3, p4)))
                    if len(intersection) != 1: continue
                    inter = intersection.pop()
                    l1, l2, l3, l4 = map(self.logic.find_line_with_length,
                                         [(p1, inter), (p2, inter), (p3, inter), (p4, inter)])
                    if l1 != [] and l2 != [] and l3 != [] and l4 != []:
                        expr = l1[0] * l2[0] - l3[0] * l4[0]
                        if len(expr.free_symbols) <= 2:
                            Update = True
                            self.equations.append(expr)
        return Update

    def polygon_anglesum_theorem(self):
        Update = False
        # Equations in Quadrilateral
        Quadrilaterals = self.logic.find_all_quadrilaterals()
        for quad in Quadrilaterals:
            expr = sum(
                [self.logic.find_angle_measure((quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]))[0] for i in range(4)])
            self.equations.append(expr - 180 * (len(quad) - 2))
        # Equations in Pentagons
        Pentagons = self.logic.find_all_pentagons()
        for penta in Pentagons:
            expr = sum(
                [self.logic.find_angle_measure((penta[i], penta[(i + 1) % 5], penta[(i + 2) % 5]))[0] for i in
                 range(5)])
            self.equations.append(expr - 180 * (len(penta) - 2))

    def similar_triangle_proving_theorem(self):
        Update = False

        def _mul(l1, l2):
            return [(pair[0] * pair[1]).subs(self.logic.variables) for pair in product(l1, l2)]

        triangles = self.logic.find_all_triangles()
        comb = combinations(triangles, 2)
        for pair in comb:
            for tri1 in permutations(pair[0]):
                tri2 = pair[1]
                if self.logic.check_similar_triangle(tri1, tri2):
                    continue
                lines = self._generateLines(tri1) + self._generateLines(tri2)
                length = [self.logic.find_line_with_length(x) for x in lines]
                angles = [self.logic.find_angle_measure(x) for x in
                          self._generateAngles(tri1) + self._generateAngles(tri2)]
                s = self._same
                eq = self._equal
                same_angle = [s(angles[0], angles[3]), s(angles[1], angles[4]), s(angles[2], angles[5])]
                same_ratio = [eq(_mul(length[1], length[5]), _mul(length[2], length[4])),
                              eq(_mul(length[0], length[5]), _mul(length[2], length[3])),
                              eq(_mul(length[0], length[4]), _mul(length[1], length[3]))]
                if self._traingleSimilar(same_angle, same_ratio):
                    Update = True
                    self.logic.defineSimilarTriangle(tri1, tri2)
        return Update

    def similar_triangle_theorem(self):
        Update = False
        for tri1, tri2 in self.logic.find_all_similar_triangles():
            lines = self._generateLines(tri1) + self._generateLines(tri2)
            length = [self.logic.find_line_with_length(x) for x in lines]
            angles = [self.logic.find_angle_measure(x) for x in self._generateAngles(tri1) + self._generateAngles(tri2)]
            for ch in permutations(range(3)):
                for i in range(3):
                    expr = angles[i][0] - angles[i + 3][0]
                    if expr != 0:
                        Update = True
                        self.equations.append(expr)
                    self.logic.PutIntoEqualAngleSet(self._generateAngles(tri1)[0], self._generateAngles(tri2)[0])
            for ch1, ch2 in combinations([0, 1, 2], 2):
                if sum([isAlgebra(x[0]) for x in [length[ch1], length[ch1 + 3], length[ch2], length[ch2 + 3]]]) >= 2:
                    Update = True
                    expr = length[ch1][0] * length[ch2 + 3][0] - length[ch2][0] * length[ch1 + 3][0]
                    self.equations.append(expr)
        return Update

    def similar_polygon_theorem(self):
        # Similar Polygon
        polygons = self.logic.find_all_similar_polygons()
        if polygons is None or polygons == []:
            return
        for poly1, poly2 in polygons:
            length1 = [self.logic.find_line_with_length((poly1[i % len(poly1)], poly1[(i + 1) % len(poly1)]))[0] for i
                       in range(len(poly1))]
            length2 = [self.logic.find_line_with_length((poly2[i % len(poly1)], poly2[(i + 1) % len(poly2)]))[0] for i
                       in range(len(poly2))]
            angles1 = [self.logic.find_angle_measure(
                (poly1[i % len(poly1)], poly1[(i + 1) % len(poly1)], poly1[(i + 2) % len(poly1)]))[0] for i in
                       range(len(poly1))]
            angles2 = [self.logic.find_angle_measure(
                (poly2[i % len(poly2)], poly2[(i + 1) % len(poly2)], poly2[(i + 2) % len(poly2)]))[0] for i in
                       range(len(poly2))]
            Area1 = self.logic.newAreaSymbol('Polygon({})'.format(','.join(sort_points(poly1)))).subs(
                self.logic.variables)
            Area2 = self.logic.newAreaSymbol('Polygon({})'.format(','.join(sort_points(poly2)))).subs(
                self.logic.variables)
            for ch1, ch2 in combinations(range(len(poly1)), 2):
                if sum([isAlgebra(x) for x in [length1[ch1], length2[ch1], length1[ch2], length2[ch2]]]) >= 2:
                    expr = length1[ch1] * length2[ch2] - length1[ch2] * length2[ch1]
                    self.equations.append(expr)
                if self.logic.check_line((poly1[ch1], poly1[ch2])) and self.logic.check_line((poly2[ch1], poly2[ch2])):
                    l1 = self.logic.find_line_with_length((poly1[ch1], poly1[ch2]))[0]
                    l2 = self.logic.find_line_with_length((poly2[ch1], poly2[ch2]))[0]
                    if isAlgebra(l1) and isAlgebra(l2):
                        expr = l1 * l1 * Area2 - l2 * l2 * Area1
                        self.equations.append(expr)
            for i in range(len(poly1)):
                expr = angles2[i] - angles1[i]
                self.equations.append(expr)
                self.logic.PutIntoEqualAngleSet(
                    (poly1[i % len(poly1)], poly1[(i + 1) % len(poly1)], poly1[(i + 2) % len(poly1)]),
                    (poly2[i % len(poly2)], poly2[(i + 1) % len(poly2)], poly2[(i + 2) % len(poly2)]))
                if isNumber(length1[i]) or isNumber(length2[i]):
                    expr = length1[i] * length1[i] * Area2 - length2[i] * length2[i] * Area1
                    self.equations.append(expr)

    def angle_bisector_theorem(self):
        triangles = self.logic.find_all_triangles()
        for tri in triangles:
            for z in tri:
                x = tri[1] if z == tri[0] else tri[0]
                y = tri[1] if z == tri[2] else tri[2]
                points = self.logic.find_all_points_on_line((x, y))
                for m in points:
                    if m != x and m != y and self._same(self.logic.find_angle_measure((x, z, m)),
                                                        self.logic.find_angle_measure((y, z, m))):
                        s1, s2, t1, t2 = map(self.logic.find_line_with_length, ((x, z), (y, z), (x, m), (y, m)))
                        know_edges = sum([hasNumber(x) for x in [s1, s2, t1, t2]])
                        # s1 / t1 = s2 / t2
                        if s1 != [] and s2 != [] and t1 != [] and t2 != [] and know_edges >= 2:
                            expr = s1[0] * t2[0] - t1[0] * s2[0]
                            self.equations.append(expr)

    def law_of_cosines(self):
        triangles = self.logic.find_all_triangles()
        for tri in triangles:
            for _ in permutations(tri):
                angleC, angleB, angleA = [self.logic.find_angle_measure(x) for x in self._generateAngles(_)]
                lines = self._generateLines(_)
                lengthAB, lengthAC, lengthBC = [self.logic.find_line_with_length(x) for x in lines]
                known_edges = sum([hasNumber(x) for x in [lengthAB, lengthAC, lengthBC]])
                if hasNumber(angleC) and ((angleC[0] == 90 and known_edges >= 1) or known_edges >= 2):
                    cos_value = cos(angleC[0] * pi / 180.0).evalf()
                    expr = lengthAC[0] * lengthAC[0] + lengthBC[0] * lengthBC[0] - lengthAB[0] * lengthAB[0] \
                           - 2 * lengthAC[0] * lengthBC[0] * cos_value
                elif not hasNumber(angleC) and known_edges == 3:
                    expr = lengthAC[0] * lengthAC[0] + lengthBC[0] * lengthBC[0] - lengthAB[0] * lengthAB[0] \
                           - 2 * lengthAC[0] * lengthBC[0] * cos(angleC[0] * pi / 180.0)
                else:
                    continue
                if self._hasSymbol(expr) and (known_edges != 3 or self._isPrimitive(angleC[0])):
                    self.equations.append(expr)

    def law_of_sines(self):
        triangles = self.logic.find_all_triangles()
        for tri in triangles:
            for _ in permutations(tri):
                angleC, angleB, angleA = [self.logic.find_angle_measure(x) for x in self._generateAngles(_)]
                lines = self._generateLines(_)
                lengthAB, lengthAC, lengthBC = [self.logic.find_line_with_length(x) for x in lines]
                if (hasNumber(angleA) and hasNumber(angleB) and (hasNumber(lengthAC) or hasNumber(lengthBC))) or \
                        (hasNumber(lengthAC) and hasNumber(lengthBC) and (hasNumber(angleA) or hasNumber(angleB))):
                    # use math.cos/sin to calculate the specific value
                    sinA = math.sin if isNumber(angleA[0]) else sin
                    sinB = math.sin if isNumber(angleB[0]) else sin
                    expr = sinA(angleA[0] * pi / 180.0) * lengthAC[0] - sinB(angleB[0] * pi / 180.0) * lengthBC[0]
                    if self._hasSymbol(expr):
                        self.equations.append(expr)

    def pythagoras_theorem(self):
        Update = False
        triangles = self.logic.find_all_triangles()
        for tri in triangles:
            for _ in permutations(tri):
                angleC, angleB, angleA = [self.logic.find_angle_measure(x) for x in self._generateAngles(_)]
                if not (hasNumber(angleC) and angleC[0] == 90):
                    continue
                lines = self._generateLines(_)
                lengthAB, lengthAC, lengthBC = [self.logic.find_line_with_length(x) for x in lines]
                known_edges = sum([hasNumber(x) for x in [lengthAB, lengthAC, lengthBC]])
                if known_edges == 2:
                    if hasNumber(lengthAB) and hasNumber(lengthAC):
                        Update = self.logic.define_length(*lines[2],
                                                          (lengthAB[0] ** 2 - lengthAC[0] ** 2) ** 0.5) or Update
                    if hasNumber(lengthAB) and hasNumber(lengthBC):
                        Update = self.logic.define_length(*lines[1],
                                                          (lengthAB[0] ** 2 - lengthBC[0] ** 2) ** 0.5) or Update
                    if hasNumber(lengthAC) and hasNumber(lengthBC):
                        Update = self.logic.define_length(*lines[0],
                                                          (lengthAC[0] ** 2 + lengthBC[0] ** 2) ** 0.5) or Update
                if known_edges < 2 and sum([isAlgebra(x[0]) for x in [lengthAB, lengthAC, lengthBC]]) >= 2:
                    expr = lengthAB[0] ** 2 - lengthAC[0] ** 2 - lengthBC[0] ** 2
                    if len(expr.free_symbols) == 1:
                        Update = True
                        self.equations.append(expr)
        return Update

    def triangle_center_of_gravity_theorem(self):
        Update = False
        triangles = self.logic.find_all_triangles()
        for tri in triangles:
            center_of_gravity = None
            lines = self._generateLines(tri)
            midpoint_list = [None, None, None]
            for i, line in enumerate(lines):
                for p in self.logic.find_all_points_on_line(line):
                    if p != line[0] and p != line[1] and self._same(self.logic.find_line_with_length((p, line[0])),
                                                                    self.logic.find_line_with_length((p, line[1]))):
                        midpoint_list[i] = p
                        break
            if all([_ is not None for _ in midpoint_list]):
                s0 = set(self.logic.find_all_points_on_line((tri[2], midpoint_list[0])))
                s1 = set(self.logic.find_all_points_on_line((tri[1], midpoint_list[1])))
                s2 = set(self.logic.find_all_points_on_line((tri[0], midpoint_list[2])))
                intersection = s0 & s1 & s2
                if len(intersection) == 0:
                    continue
                center_of_gravity = intersection.pop()
            if center_of_gravity is not None:
                for i in range(3):
                    length1 = self.logic.find_line_with_length((midpoint_list[i], center_of_gravity))
                    length2 = self.logic.find_line_with_length((tri[2 - i], center_of_gravity))
                    if hasNumber(length1) and not hasNumber(length2):
                        Update = self.logic.define_length(tri[2 - i], center_of_gravity, 2 * length1[0])
                    if hasNumber(length2) and not hasNumber(length1):
                        Update = self.logic.define_length(midpoint_list[i], center_of_gravity, length2[0] / 2)
        return Update

    def area_equation_theorem(self):
        Update = False
        if not (self.target != None and 'Area' in self.target or any(
                [('AreaOf' in str(k)) for k in self.logic.variables.keys()])):
            return Update
        quads = self.logic.find_all_quadrilaterals()
        for quad in quads:
            Area = self.logic.newAreaSymbol('Polygon({})'.format(','.join(sort_points(quad)))).subs(
                self.logic.variables)
            parallel_pair = []
            if self.logic.check_parallel(quad[0:2], quad[2:4]):
                parallel_pair.append((quad[0:2], quad[2:4]))
            if self.logic.check_parallel(quad[1:3], [quad[0], quad[3]]):
                parallel_pair.append((quad[1:3], [quad[0], quad[3]]))

            for upper_edge, lower_edge in parallel_pair:
                # The area for trapezoid, parallelogram or rectangle. (upper edge + lower edge) / 2 * height
                upper_length = self.logic.find_line_with_length(upper_edge)
                lower_length = self.logic.find_line_with_length(lower_edge)
                upper_edge_points = self.logic.find_all_points_on_line(upper_edge)
                lower_edge_points = self.logic.find_all_points_on_line(lower_edge)
                angles = self.logic.find_all_90_angles()
                for angle in angles:
                    if (angle[1] in upper_edge_points and angle[0] in upper_edge_points and angle[
                        2] in lower_edge_points) or \
                            (angle[1] in lower_edge_points and angle[0] in lower_edge_points and angle[
                                2] in upper_edge_points):
                        lengthH = self.logic.find_line_with_length([angle[1], angle[2]])
                        expr = (upper_length[0] + lower_length[0]) * lengthH[0] / 2 - Area
                        self.equations.append(expr)
                        Update = True

            # AC ⊥ BD
            if (self.logic.check_perpendicular([quad[0], quad[2]], [quad[1], quad[3]])):
                l1 = self.logic.find_line_with_length([quad[0], quad[2]])
                l2 = self.logic.find_line_with_length([quad[1], quad[3]])
                if l1 != [] and l2 != []:
                    expr = l1[0] * l2[0] / 2 - Area
                    self.equations.append(expr)
                    Update = True

        triangles = self.logic.find_all_triangles()
        for tri in triangles:
            Area = self.logic.newAreaSymbol('Polygon({})'.format(','.join(sort_points(tri)))).subs(self.logic.variables)
            for ch in permutations(range(3)):
                base = [tri[ch[0]], tri[ch[1]]]
                verticle = tri[ch[2]]
                points_on_base = self.logic.find_all_points_on_line(base)
                angles = self.logic.find_all_90_angles()
                for angle in angles:
                    if (angle[1] in points_on_base and angle[0] in points_on_base and angle[2] == verticle):
                        lengthH = self.logic.find_line_with_length([angle[1], angle[2]])
                        lengthB = self.logic.find_line_with_length(base)
                        expr = lengthB[0] * lengthH[0] / 2 - Area
                        self.equations.append(expr)
                        Update = True
        return Update

    def median_line_theorem(self):
        quads = self.logic.find_all_quadrilaterals()
        for quad in quads:
            parallel_pair = []
            leg_pair = []
            if self.logic.check_parallel(quad[0:2], quad[2:4]):
                parallel_pair.append((quad[0:2], quad[2:4]))
                leg_pair.append((quad[1:3], [quad[0], quad[3]]))
            if self.logic.check_parallel(quad[1:3], [quad[0], quad[3]]):
                parallel_pair.append((quad[1:3], [quad[0], quad[3]]))
                leg_pair.append((quad[0:2], quad[2:4]))

            for (upper_edge, lower_edge), (leg1, leg2) in zip(parallel_pair, leg_pair):
                median_line = [None, None]
                for i, leg in enumerate((leg1, leg2)):
                    for p in self.logic.find_all_points_on_line(leg):
                        if p != leg[0] and p != leg[1] and self._same(self.logic.find_line_with_length((p, leg[0])),
                                                                      self.logic.find_line_with_length((p, leg[1]))):
                            median_line[i] = p
                if all([_ is not None for _ in median_line]) and self.logic.check_line(median_line):
                    self.logic.define_parallel(median_line, upper_edge)
                    self.logic.define_parallel(median_line, lower_edge)
                    length = self.logic.find_line_with_length(median_line)
                    upper_length = self.logic.find_line_with_length(upper_edge)
                    lower_length = self.logic.find_line_with_length(lower_edge)
                    expr = length[0] * 2 - upper_length[0] - lower_length[0]
                    self.equations.append(expr)

        triangles = self.logic.find_all_triangles()
        for tri in triangles:
            for ch in permutations(range(3)):
                base = [tri[ch[0]], tri[ch[1]]]
                leg1 = [tri[ch[0]], tri[ch[2]]]
                leg2 = [tri[ch[1]], tri[ch[2]]]
                median_line = [None, None]
                for i, leg in enumerate((leg1, leg2)):
                    for p in self.logic.find_all_points_on_line(leg):
                        if p != leg[0] and p != leg[1] and self._same(self.logic.find_line_with_length((p, leg[0])),
                                                                      self.logic.find_line_with_length((p, leg[1]))):
                            median_line[i] = p
                if all([_ is not None for _ in median_line]) and self.logic.check_line(median_line):
                    self.logic.define_parallel(median_line, base)
                    length = self.logic.find_line_with_length(median_line)
                    base_length = self.logic.find_line_with_length(base)
                    expr = length[0] * 2 - base_length[0]
                    self.equations.append(expr)

    def tangent_secant_theorem(self):
        def _find_tangent_line(circle):
            tangent_line_list = []
            points = self.logic.find_points_on_circle(circle)
            for rightAngle in rightAngles:
                if (circle == rightAngle[0] or circle == rightAngle[2]) and rightAngle[1] in points:
                    far_point = [_ for _ in rightAngle[::2] if _ != circle][0]
                    tangent_line = (rightAngle[1], far_point)
                    tangent_line_list.append(tangent_line)
            return tangent_line_list

        # tangent and secant
        circles = self.logic.find_all_circles()
        rightAngles = self.logic.find_all_90_angles()
        for circle in circles:
            # find tangent line
            tangent_line_list = _find_tangent_line(circle)
            # find secant line
            for tangent_line in tangent_line_list:
                tangent_length = self.logic.find_line_with_length(tangent_line)
                if not isAlgebra(tangent_length[0]):
                    continue
                tangent_point, far_point = tangent_line
                points_on_circle = self.logic.find_points_on_circle(circle)
                for two_points in permutations(points_on_circle, 2):
                    angle_measure = self.logic.find_angle_measure((*two_points, far_point))
                    if angle_measure != [] and angle_measure[0] == 180:
                        l1 = self.logic.find_line_with_length((far_point, two_points[0]))
                        l2 = self.logic.find_line_with_length((far_point, two_points[1]))
                        if isAlgebra(l1[0]) or isAlgebra(l2[0]):
                            expr = l1[0] * l2[0] - tangent_length[0] ** 2
                            self.equations.append(expr)

    def connecting_two_points(self):
        Update = False
        # connect any two points on circle
        circles = self.logic.find_all_circles()
        for circle in circles:
            points_on_circle = self.logic.find_points_on_circle(circle)
            for p1, p2 in combinations(points_on_circle, 2):
                Update = self.logic.define_line(p1, p2) or Update
        return Update

    def _getAnswer(self, target):
        """
        Give the target (The format is defined above) and find its answer if possible.
        """
        if target[0] == 'Value':
            if len(target) == 5:
                tried = self.logic.find_arc_measure(target[2:])
                if hasNumber(tried):
                    if target[1] == 'arc_measure':
                        return tried[0]
                    elif target[1] == "arc_length":
                        points = self.logic.find_points_on_circle(target[2])
                        for i in range(len(points)):
                            length = self.logic.find_line_with_length([target[2], points[i]])
                            if hasNumber(length):
                                return 2 * math.pi * length[0] * tried[0] / 360.0
            elif len(target) == 4:
                tried = self.logic.find_angle_measure(target[1:])
                if hasNumber(tried):
                    return tried[0]
            elif len(target) == 3:
                tried = self.logic.find_line_with_length(target[1:])
                if hasNumber(tried):
                    return tried[0]
            elif len(target) == 2:
                try:
                    return float(target[1])
                except:
                    v = Symbol(target[1])
                    try:
                        return float(self.logic.variables[v])
                    except:
                        pass
            return None

        if target[0] == 'Area':
            l = len(target) - 1
            lengthlst = []
            if l == 1:
                points = self.logic.find_points_on_circle(target[1])
                for i in range(len(points)):
                    length = self.logic.find_line_with_length([target[1], points[i]])
                    if hasNumber(length):
                        return math.pi * length[0] * length[0]

            elif l == 3:
                # The area for triangles
                for i in range(1, l + 1):
                    length = self.logic.find_line_with_length([target[i], target[i % l + 1]])
                    if not hasNumber(length): break
                    lengthlst.append(length[0])
                if len(lengthlst) == 3:
                    return heron_triangle_formula(lengthlst[0], lengthlst[1], lengthlst[2])
                for angle in permutations(target[1:]):
                    angle_measure = self.logic.find_angle_measure(angle)
                    if not hasNumber(angle_measure): break
                    lengthAB = self.logic.find_line_with_length([angle[0], angle[1]])
                    lengthAC = self.logic.find_line_with_length([angle[2], angle[1]])
                    if hasNumber(lengthAB) and hasNumber(lengthAC):
                        return angle_area_formula(lengthAB[0], lengthAC[0], angle_measure[0])
                for ch in permutations(range(1, l + 1)):
                    base = [target[ch[0]], target[ch[1]]]
                    verticle = target[ch[2]]
                    points_on_base = self.logic.find_all_points_on_line(base)
                    angles = self.logic.find_all_90_angles()
                    for angle in angles:
                        if (angle[1] in points_on_base and angle[0] in points_on_base and angle[2] == verticle):
                            lengthH = self.logic.find_line_with_length([angle[1], angle[2]])
                            lengthB = self.logic.find_line_with_length(base)
                            if hasNumber(lengthH) and hasNumber(lengthB):
                                return lengthH[0] * lengthB[0] / 2

            elif l == 4:
                area_symbol = self.logic.newAreaSymbol('Polygon({})'.format(','.join(sort_points(target[1:])))).subs(
                    self.logic.variables)
                if isNumber(area_symbol):
                    return area_symbol
                # The area for trapezoid, parallelogram or rectangle. (upper edge + lower edge) / 2 * height
                alledges = target[1:] + target[1:]
                for i in range(1, 5):
                    upper_edge = alledges[i:i + 2]
                    upper_length = self.logic.find_line_with_length(upper_edge)
                    lower_edge = alledges[i + 2:i + 4]
                    lower_length = self.logic.find_line_with_length(lower_edge)
                    if not hasNumber(upper_length) or not hasNumber(lower_length):
                        continue
                    upper_edge_points = self.logic.find_all_points_on_line(upper_edge)
                    lower_edge_points = self.logic.find_all_points_on_line(lower_edge)
                    angles = self.logic.find_all_90_angles()
                    for angle in angles:
                        if angle[1] in upper_edge_points and angle[0] in upper_edge_points:
                            points = self.logic.find_all_points_on_line([angle[1], angle[2]])
                            intersection = set(points) & set(lower_edge_points)
                            if len(intersection) != 1: continue
                            inter = intersection.pop()
                            if (lower_edge[0], inter, angle[1]) in angles or (lower_edge[1], inter, angle[1]) in angles:
                                lengthH = self.logic.find_line_with_length([angle[1], inter])
                                if hasNumber(lengthH):
                                    return (upper_length[0] + lower_length[0]) * lengthH[0] / 2

                # AC ⊥ BD
                if (self.logic.check_perpendicular([target[1], target[3]], [target[2], target[4]])):
                    l1 = self.logic.find_line_with_length([target[1], target[3]])
                    l2 = self.logic.find_line_with_length([target[2], target[4]])
                    if hasNumber(l1) and hasNumber(l2):
                        return l1[0] * l2[0] / 2

                # Divide into two triangles
                for i in range(1, 5):
                    A, B, C, D = target[i], target[i % 4 + 1], target[(i + 1) % 4 + 1], target[(i + 2) % 4 + 1]
                    area1 = self._getAnswer(['Area', A, B, C])
                    if area1 is None: continue
                    area2 = self._getAnswer(['Area', C, D, A])
                    if area2 is not None: return area1 + area2
                    if self._same(self.logic.find_line_with_length([A, B]),
                                  self.logic.find_line_with_length([C, D])) and \
                            self._same(self.logic.find_line_with_length([B, C]),
                                       self.logic.find_line_with_length([D, A])):
                        return area1 * 2
            return None

        if target[0] == 'Perimeter':
            l = len(target) - 1
            ans = 0
            if l == 1:
                points = self.logic.find_points_on_circle(target[1])
                for i in range(len(points)):
                    length = self.logic.find_line_with_length([target[1], points[i]])
                    if hasNumber(length):
                        return 2 * math.pi * length[0]
                return None
            else:
                for i in range(1, l + 1):
                    length = self.logic.find_line_with_length([target[i], target[i % l + 1]])
                    if length == []: return None
                    ans += length[0]
                if isNumber(ans):
                    return ans
                else:
                    return None

        if target[0] == 'Sector':
            O, A, B = target[1:]
            angle_measure = self.logic.find_angle_measure([A, O, B])
            radius = self.logic.find_line_with_length([O, A])
            if hasNumber(angle_measure) and hasNumber(radius):
                return radius[0] * radius[0] * angle_measure[0] / 180 * math.pi / 2

        if target[0] in ["SinOf", "CosOf", "TanOf", "CotOf", "HalfOf", "SquareOf", "SqrtOf"]:
            try:
                if target[0] == "SinOf": return math.sin(self._getAnswer(target[1]) / 180.0 * math.pi)
                if target[0] == "CosOf": return math.cos(self._getAnswer(target[1]) / 180.0 * math.pi)
                if target[0] == "TanOf": return math.tan(self._getAnswer(target[1]) / 180.0 * math.pi)
                if target[0] == "CotOf": return 1.0 / math.tan(self._getAnswer(target[1]) / 180.0 * math.pi)
                if target[0] == "HalfOf": return self._getAnswer(target[1]) / 2.0
                if target[0] == "SquareOf": return self._getAnswer(target[1]) ** 2
                if target[0] == "SqrtOf": return self._getAnswer(target[1]) ** 0.5
            except:
                return None

        if target[0] in ["RatioOf", "Add", "Mul", "SumOf"]:
            try:
                if target[0] == "RatioOf": return self._getAnswer(target[1]) / self._getAnswer(target[2])
                if target[0] == "Mul": return self._getAnswer(target[1]) * self._getAnswer(target[2])
                if target[0] in ["Add", "SumOf"]: return sum([self._getAnswer(x) for x in target[1:]])
            except:
                return None

        if target[0] == 'ScaleFactorOf':
            if target[1][0] == "Shape" and len(target[1]) == 2:
                line = (target[1][1], target[2][1])
                points = self.logic.find_all_points_on_line(line)
                O = (set(points) - set(line)).pop()
                try:
                    return self._getAnswer(['Value', O, line[1]]) / self._getAnswer(['Value', O, line[0]])
                except:
                    return None
            else:
                shape1 = target[1] if type(target[1][1]) == str else target[1][1]
                shape2 = target[2] if type(target[2][1]) == str else target[2][1]
                return (self._getAnswer(['Area', *shape1[1:]]) / self._getAnswer(['Area', *shape2[1:]])) ** 0.5

    def initSearch(self):
        self.logic.put_equal_into_symbol()
        self.logic.try_delete_unused_points()  # remove no-use points
        self.logic.init_all_uni_lines()  # initialize all the uni-lines (the line do not contain other lines)
        self.logic.find_hidden_polygons(3)  # find all triangles
        self.logic.find_hidden_polygons(4)  # find all quads
        self.logic.find_hidden_polygons(5)  # find all pentas
        self.logic.expand_angles()  # add the rest (hidden) angles into the graph and give each of them a symbol
        # self.logic.set_angle_sum()  # resolve the relation among angles
        # self.logic.set_line_sum()  # resolve the relation among lines (e.g, AB+BC = AC)
        # self.logic.set_arc_sum()  # resolve the relation among arcs

        if self.logic.debug:
            print(self.logic.find_all_angle_measures())
            print(self.logic.find_all_lines_with_length())
            print(self.logic.fine_all_arc_measures())
        self.can_search = True
        self.hasSolution = False
