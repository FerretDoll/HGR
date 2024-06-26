from kanren import Relation, facts
from kanren import run, var, conde
from sympy import Symbol, pi

from itertools import combinations, permutations
from utils.common_utils import isNumber, hasNumber


class BasicDefinition(object):
    def __init__(self, debug):

        # self.Point = Relation()
        self.Point = set()
        self.Line = Relation()
        self.Length = Relation()
        self.UniLine = Relation()
        self.Angle = Relation()
        self.AngleMeasure = Relation()
        self.Arc = Relation()
        self.ArcMeasure = Relation()
        self.Circle = Relation()

        self.PointOnCircle = Relation()
        self.PointLiesOnLine = Relation()
        self.Perpendicular = Relation()
        self.Parallel = Relation()

        self.Triangle = Relation()
        self.Quadrilateral = Relation()
        self.Pentagon = Relation()

        self.Equal = Relation()
        self.Equation = Relation()
        self.CongruentTriangle = Relation()
        self.SimilarTriangle = Relation()
        self.SimilarPolygon = Relation()

        self.relations = [
            # self.Point,
            self.Line,
            self.Length,
            self.UniLine,
            self.Angle,
            self.AngleMeasure,
            self.Arc,
            self.ArcMeasure,
            self.Circle,

            self.PointOnCircle,
            self.PointLiesOnLine,
            self.Perpendicular,
            self.Parallel,

            self.Triangle,
            self.Quadrilateral,
            self.Pentagon,

            self.Equal,
            self.Equation,
            self.CongruentTriangle,
            self.SimilarTriangle,
            self.SimilarPolygon,
        ]

        self.variables = dict()
        self.variables[Symbol('pi')] = pi
        self.points_on_line = {}
        self.lines_for_point = {}
        self.point_positions = {}
        self.initUni = False
        self.debug = debug

        self.EqualLineSet = []
        self.EqualAngleSet = []
        self.EqualArcSet = []

        self.SameAngleDict = {}
        

    ############### Definition of Shapes ###############
    def define_equal(self, para1, para2):
        # This function is used to build a bridge between two expressions.
        # If the logic form is Equals(x, y), then we know self.Equal(x, y).
        facts(self.Equal, (para1, para2))

    def build_equation(self, para1, para2):
        # Build equations when parsing.
        # It will be appended to LogicSolver.equations when solving.
        facts(self.Equation, (para1, para2))

    def PutIntoSameAngleDict(self, angle1, angle2):
        angle1 = ''.join(angle1) if angle1[0] < angle1[2] else ''.join(angle1[::-1])
        angle2 = ''.join(angle2) if angle2[0] < angle2[2] else ''.join(angle2[::-1])
        k1 = k2 = None
        for k,v in self.SameAngleDict.items():
            if angle1 in v:
                k1 = k
            if angle2 in v:
                k2 = k
        if k1 == None: 
            newSet = set()
            newSet.add(angle1)
            self.SameAngleDict[angle1] = newSet
            k1 = angle1
        if k2 == None: 
            newSet = set()
            newSet.add(angle2)
            self.SameAngleDict[angle2] = newSet
            k2 = angle2
        if k1 == k2:
            return
        self.SameAngleDict[k1] = self.SameAngleDict[k1] | self.SameAngleDict[k2]
        del self.SameAngleDict[k2]

    def PutIntoEqualLineSet(self, line1, line2):
        line1 = ''.join(sorted(line1))
        line2 = ''.join(sorted(line2))
        id1 = id2 = -1
        for i,s in enumerate(self.EqualLineSet):
            if line1 in s:
                id1 = i
            if line2 in s:
                id2 = i
        if id1 == -1: 
            newSet = set()
            newSet.add(line1)
            self.EqualLineSet.append(newSet)
            id1 = len(self.EqualLineSet) - 1
        if id2 == -1: 
            newSet = set()
            newSet.add(line2)
            self.EqualLineSet.append(newSet)
            id2 = len(self.EqualLineSet) - 1
        if id1 == id2:
            return
        self.EqualLineSet[id1] = self.EqualLineSet[id1] | self.EqualLineSet[id2]
        del self.EqualLineSet[id2]
    
    def PutIntoEqualAngleSet(self, angle1, angle2):
        angle1 = ''.join(angle1) if angle1[0] < angle1[2] else ''.join(angle1[::-1])
        angle2 = ''.join(angle2) if angle2[0] < angle2[2] else ''.join(angle2[::-1])
        id1 = id2 = -1
        for i,s in enumerate(self.EqualAngleSet):
            if angle1 in s:
                id1 = i
            if angle2 in s:
                id2 = i
        if id1 == -1: 
            newSet = set()
            newSet.add(angle1)
            self.EqualAngleSet.append(newSet)
            id1 = len(self.EqualAngleSet) - 1 
        if id2 == -1: 
            newSet = set()
            newSet.add(angle2)
            self.EqualAngleSet.append(newSet)
            id2 = len(self.EqualAngleSet) - 1
        if id1 == id2:
            return
        self.EqualAngleSet[id1] = self.EqualAngleSet[id1] | self.EqualAngleSet[id2]
        del self.EqualAngleSet[id2]

    def PutIntoEqualArcSet(self, arc1, arc2):
        id1 = id2 = -1
        for i,s in enumerate(self.EqualArcSet):
            if arc1 in s:
                id1 = i
            if arc2 in s:
                id2 = i
        if id1 == -1: 
            newSet = set()
            newSet.add(arc1)
            self.EqualArcSet.append(newSet)
            id1 = len(self.EqualArcSet) - 1
        if id2 == -1: 
            newSet = set()
            newSet.add(arc2)
            self.EqualArcSet.append(newSet)
            id2 = len(self.EqualArcSet) - 1
        if id1 == id2:
            return
        self.EqualArcSet[id1] = self.EqualArcSet[id1] | self.EqualArcSet[id2]
        del self.EqualArcSet[id2]


    def define_circle(self, circle):
        facts(self.Circle, circle)

    def define_point(self, points):
        # This function is to define a list of points.
        # [points] can be either a single-letter string, or a list.
        # Return True if a new point is added.
        if type(points) == str:
            points = [points]
        changed = False
        for point in points:
            if point not in self.Point:
                changed = True
                self.Point.add(point)
        return changed

    def define_line(self, point_A, point_B):
        res = run(1, (), self.Line(point_A, point_B))  # res = () if new p_A and p_B, else ((),)
        if len(res) > 0:  # res = ((),)
            return False
        facts(self.Line, (point_A, point_B))
        facts(self.Line, (point_B, point_A))
        return True

    def define_uni_line(self, point_A, point_B):
        facts(self.UniLine, (point_A, point_B))
        facts(self.UniLine, (point_B, point_A))

    def define_length(self, point_A, point_B, value):
        res = run(1, (), self.Length(point_A, point_B, value))
        if len(res) > 0:
            return False
        facts(self.Length, (point_A, point_B, value))
        facts(self.Length, (point_B, point_A, value))
        return True

    def define_angle(self, point_A, point_B, point_C):
        res = run(1, (), self.Angle(point_A, point_B, point_C))
        if len(res) > 0:
            return False
        facts(self.Angle, (point_A, point_B, point_C))
        facts(self.Angle, (point_C, point_B, point_A))
        return True

    def define_angle_measure(self, point_A, point_B, point_C, value):
        res = run(1, (), self.AngleMeasure(point_A, point_B, point_C, value))
        if len(res) > 0:
            return False
        facts(self.AngleMeasure, (point_A, point_B, point_C, value))
        facts(self.AngleMeasure, (point_C, point_B, point_A, value))
        return True

    def define_arc(self, point_O, point_A, point_B):
        # This arc goes from point_A to point_B in a counter-clockwise order.
        res = run(1, (), self.Arc(point_O, point_A, point_B))
        if len(res) > 0:
            return False
        facts(self.Arc, (point_O, point_A, point_B))
        return True

    def define_arc_measure(self, point_O, point_A, point_B, value):
        # This arc goes from point_A to point_B in a counter-clockwise order.
        res = run(1, (), self.ArcMeasure(point_O, point_A, point_B, value))
        if len(res) > 0:
            return False
        facts(self.ArcMeasure, (point_O, point_A, point_B, value))
        return True

    def define_parallel(self, line1, line2):
        # Define two lines parallel
        facts(self.Parallel, (line1[0], line1[1], line2[0], line2[1]))

    def seem_triangle(self, point_A, point_B, point_C):
        return conde((self.Line(point_A, point_B), self.Line(point_B, point_C), self.Line(point_A, point_C)))

    def seem_quadrilateral(self, point_A, point_B, point_C, point_D):
        return conde((self.Line(point_A, point_B), self.Line(point_B, point_C), self.Line(point_C, point_D),
                      self.Line(point_D, point_A)))

    def seem_pentagon(self, point_A, point_B, point_C, point_D, point_E):
        return conde((self.Line(point_A, point_B), self.Line(point_B, point_C), self.Line(point_C, point_D),
                      self.Line(point_D, point_E), self.Line(point_E, point_A)))

    ############### Finding out Attributes ###############
    def find_all_equations(self):
        x = var()
        y = var()
        res = run(0, (x,y), self.Equation(x,y))
        return list(res)

    def find_all_points(self):
        res = list(self.Point)
        return res

    def find_all_lines(self):
        x = var()
        y = var()
        res = run(0, (x, y), self.Line(x, y))
        return list(res)
    
    def find_all_irredundant_lines(self):
        # Line(B,A) is redundant to Line(A,B). 
        # dictionary order
        lines = [line for line in self.find_all_lines() if line[1] > line[0]]
        return lines

    def check_line(self, line):
        res = run(1, (), self.Line(line[0], line[1]))
        return len(res) > 0

    def check_uni_line(self, line):
        res = run(1, (), self.UniLine(line[0], line[1]))
        return len(res) > 0

    def check_similar_triangle(self, tri1, tri2):
        res = False
        for ch in permutations(range(3)):
            res = res or len(run(1, (), self.SimilarTriangle(tri1[ch[0]], tri1[ch[1]], tri1[ch[2]], tri2[ch[0]], tri2[ch[1]], tri2[ch[2]]))) > 0 or \
                         len(run(1, (), self.SimilarTriangle(tri2[ch[0]], tri2[ch[1]], tri2[ch[2]], tri1[ch[0]], tri1[ch[1]], tri1[ch[2]]))) > 0
        return res

    def check_congruent_triangle(self, tri1, tri2):
        res = False
        for ch in permutations(range(3)):
            res = res or len(run(1, (), self.CongruentTriangle(tri1[ch[0]], tri1[ch[1]], tri1[ch[2]], tri2[ch[0]], tri2[ch[1]], tri2[ch[2]]))) > 0 or \
                         len(run(1, (), self.CongruentTriangle(tri2[ch[0]], tri2[ch[1]], tri2[ch[2]], tri1[ch[0]], tri1[ch[1]], tri1[ch[2]]))) > 0
        return res

    def check_line_with_length(self, l):
        x, y = var(), var()
        line = run(0, (x, y), self.Length(x, y, l))
        return list(line)

    def check_parallel(self, line1, line2):
        res = len(run(1, (), self.Parallel(line1[0], line1[1], line2[0], line2[1]))) > 0 or \
              len(run(1, (), self.Parallel(line1[0], line1[1], line2[1], line2[0]))) > 0 or \
              len(run(1, (), self.Parallel(line1[1], line1[0], line2[0], line2[1]))) > 0 or \
              len(run(1, (), self.Parallel(line1[1], line1[0], line2[1], line2[0]))) > 0 or \
              len(run(1, (), self.Parallel(line2[0], line2[1], line1[0], line1[1]))) > 0 or \
              len(run(1, (), self.Parallel(line2[0], line2[1], line1[1], line1[0]))) > 0 or \
              len(run(1, (), self.Parallel(line2[1], line2[0], line1[0], line1[1]))) > 0 or \
              len(run(1, (), self.Parallel(line2[1], line2[0], line1[1], line1[0]))) > 0
        return res

    def check_perpendicular(self, line1, line2):
        res = len(run(1, (), self.Perpendicular(line1[0], line1[1], line2[0], line2[1]))) > 0 or \
              len(run(1, (), self.Perpendicular(line1[0], line1[1], line2[1], line2[0]))) > 0 or \
              len(run(1, (), self.Perpendicular(line1[1], line1[0], line2[0], line2[1]))) > 0 or \
              len(run(1, (), self.Perpendicular(line1[1], line1[0], line2[1], line2[0]))) > 0 or \
              len(run(1, (), self.Perpendicular(line2[0], line2[1], line1[0], line1[1]))) > 0 or \
              len(run(1, (), self.Perpendicular(line2[0], line2[1], line1[1], line1[0]))) > 0 or \
              len(run(1, (), self.Perpendicular(line2[1], line2[0], line1[0], line1[1]))) > 0 or \
              len(run(1, (), self.Perpendicular(line2[1], line2[0], line1[1], line1[0]))) > 0
        if res: return True
        
        s1 = set(self.find_all_points_on_line(line1))
        s2 = set(self.find_all_points_on_line(line2))
        intersection = s1 & s2
        if len(intersection) == 0:
            return False
        o = intersection.pop()
        for a in s1:
            for b in s2:
                if a == o or b == o: continue
                angle_measure = self.find_angle_measure([a,o,b])
                if hasNumber(angle_measure) and angle_measure[0] == 90:
                    return True
        return False
        

    def find_all_uni_lines(self):
        x = var()
        y = var()
        res = run(0, (x, y), self.UniLine(x, y))
        return list(res)

    def find_line_with_length(self, line, skip_if_has_number=True):
        """Give a line (segment) and try to find its length.
        Args:
            line(point, point): the specific line.
            skip_if_has_number:
                True: If one of its representations is an exact number, only return this number not the whole list.
                False: Return all the representations in a list.
        Returns:
            A list contains representations for the current line.
        """
        z = var()
        res = run(0, z, self.Length(line[0], line[1], z))  # try to find the line length
        final = set()  # use to get the unique result
        for val in res:
            if isNumber(val):
                new_val = val
            else:
                new_val = val.subs(self.variables)
            if skip_if_has_number and isNumber(new_val):
                return [new_val]
            final.add(new_val)
        return list(final)  

    def find_all_lines_with_length(self):
        """Find all lines in the graph with their length."""
        lines = self.find_all_lines()
        res = []
        for line in lines:
            val = self.find_line_with_length(line)
            if len(val) > 0:
                res.append((line[0], line[1], val[0]))
        return res

    def find_all_irredundant_lines_with_length(self):
        """Find all irrendundant lines in the graph with their length."""
        lines = self.find_all_irredundant_lines()
        res = []
        for line in lines:
            val = self.find_line_with_length(line)
            if hasNumber(val):
                res.append((line[0], line[1], val[0]))
            else:
                for v in val:
                    res.append((line[0], line[1], v))
        return res

    def find_points_on_circle(self, circle):
        x = var()
        res = run(0, x, self.PointOnCircle(circle, x))
        return list(res)

    def find_all_circles(self):
        x = var()
        res = run(0, x, self.Circle(x))
        return list(res)

    def find_all_points_on_circles(self):
        res = [[self.find_points_on_circle(circle), circle] for circle in self.find_all_circles()]
        return res

    def check_angle(self, angle):
        res = run(0, (), self.Angle(angle[0], angle[1], angle[2]))
        return len(res) > 0

    def check_angle_measure(self, angle):
        res = run(0, (), self.AngleMeasure(*angle))
        return len(res) > 0
    
    def check_same_angle(self, angle1, angle2):
        angle1 = ''.join(angle1) if angle1[0] < angle1[2] else ''.join(angle1[::-1])
        angle2 = ''.join(angle2) if angle2[0] < angle2[2] else ''.join(angle2[::-1])
        if angle1[1] != angle2[1]:
            return False
        key_angle1 = self.get_same_angle_key(angle1)
        key_angle2 = self.get_same_angle_key(angle2)
        return key_angle1 == key_angle2
    
    def get_same_angle_key(self, angle):
        angle = ''.join(angle) if angle[0] < angle[2] else ''.join(angle[::-1])
        for k,v in self.SameAngleDict.items():
            if angle in v:
                return k
        return angle

    def get_same_angle_value(self, angle):
        angle = ''.join(angle) if angle[0] < angle[2] else ''.join(angle[::-1])

        key = self.get_same_angle_key(angle)

        return self.SameAngleDict.get(key, set())
            
    def find_all_angles(self):
        x = var()
        y = var()
        z = var()
        res = run(0, (x, y, z), self.Angle(x, y, z))
        return list(res)

    def find_all_irredundant_angles(self):
        angles = [angle for angle in self.find_all_angles() if angle[0] < angle[2]]
        res = []
        for angle in angles:
            key_angle = self.get_same_angle_key(angle)
            if key_angle not in res:
                res.append(key_angle)
        return res

    def find_angle_measure(self, angle, skip_if_has_number=True):
        """Give an angle and try to find its measure.
        Args:
            angle(point, point, point): the specific angle.
            skip_if_has_number:
                True: If one of its representations is an exact number, only return this number not the whole list.
                False: Return all the representations in a list.
        Returns:
            A list contains representations for the current angle.
        """
        z = var()
        res = run(0, z, self.AngleMeasure(angle[0], angle[1], angle[2], z))
        final = set()
        for val in res:
            if isNumber(val):
                    new_val = val
            else:
                new_val = val.subs(self.variables)
            if skip_if_has_number and isNumber(new_val):
                return [new_val]
            final.add(new_val)
            # try:
            #     if type(val) in [int, float]:
            #         new_val = float(val)
            #     else:
            #         new_val = float(val.evalf(subs=self.variables))
            #     if skip_if_has_number:
            #         return [new_val]
            #     final.add(new_val)
            # except:
            #     new_val = val.subs(self.variables)
            #     final.add(new_val)
        return list(final)

    def find_all_angle_measures(self):
        res = []
        angles = self.find_all_angles()
        for angle in angles:
            vals = self.find_angle_measure(angle)
            for val in vals:
                res.append((*angle, val))
        return res
    
    def find_all_irredundant_angle_measures(self):
        res = []
        angles = self.find_all_irredundant_angles()
        for angle in angles:
            vals = self.find_angle_measure(angle)
            if hasNumber(vals):
                res.append((*angle, vals[0]))
            else:
                for val in vals:
                    res.append((*angle, val))
        return res

    def find_all_180_angles(self, angles=None):
        if angles is None:
            angles = self.find_all_angle_measures()
        f = {}
        for angle in angles:
            if angle[3] == 180:
                f[(angle[0], angle[1], angle[2])] = True
                # the reverse (2,1,0) will be enumerated, too.
        return f

    def find_all_90_angles(self, angles=None):
        if angles is None:
            angles = self.find_all_angle_measures()
        f = {}
        for angle in angles:
            if angle[3] == 90:
                f[(angle[0], angle[1], angle[2])] = True
                # the reverse (2,1,0) will be enumerated, too.
        return f

    def find_all_arcs(self):
        x, y, z = var(), var(), var()
        res = run(0, (x, y, z), self.Arc(x, y, z))
        return list(res)

    def find_arc_measure(self, arc, skip_if_has_number=True):
        """Give an arc and try to find its measure.
        Args:
            arc(point, point, point): the specific arc.
            skip_if_has_number:
                True: If one of its representations is an exact number, only return this number not the whole list.
                False: Return all the representations in a list.
        Returns:
            A list contains representations for the current angle.
        """
        z = var()
        res = run(0, z, self.ArcMeasure(*arc, z))
        final = set()
        for val in res:
            if isNumber(val):
                    new_val = val
            else:
                new_val = val.subs(self.variables)
            if skip_if_has_number and isNumber(new_val):
                return [new_val]
            final.add(new_val)
        return list(final)

    def fine_all_arc_measures(self):
        x, y, z, w = var(), var(), var(), var()
        res = run(0, (x, y, z, w), self.ArcMeasure(x, y, z, w))
        return list(res)

    def find_all_lines_for_point(self, point):
        """
        Find all points that link to the current point.
        To accelerate this process, the result will be recorded in [self.lines_for_point].
        """
        if point in self.lines_for_point:
            return self.lines_for_point[point]
        lines = self.find_all_lines()
        self.lines_for_point[point] = [line[0] for line in lines if line[1] == point]
        return self.lines_for_point[point]

    @staticmethod
    def is_colinear(pointA, pointB, pointC, f):
        # check that whether pointA, pointB, pointC is colinear.
        # f is a dict which contains all 180 angles.
        assert isinstance(f, dict)
        return (pointA, pointB, pointC) in f or \
               (pointB, pointA, pointC) in f or \
               (pointA, pointC, pointB) in f

    def find_all_points_on_line(self, line):
        """
        Given line, find all the points on this line in the increasing order if [self.initUni = True].
        e.g.    A, B, C, D are four points in a same line.
                Give: line = [D, B]
                Returns: [D, C, B, A]

        If we call this function when [self.initUni = False] (which means we haven't prepared all the uni-lines),
        we can also acquire these points but the order can not be guaranteed.
        """
        line = tuple(line)
        # Accelerate
        if line in self.points_on_line:
            return self.points_on_line[line]

        points = self.find_all_points()
        angles = self.find_all_angle_measures()
        f = self.find_all_180_angles(angles)

        # Try to find all the points on the line
        Update = True
        now_points = [line[0], line[1]]
        while Update:
            Update = False
            for point in set(points) - set(now_points):
                # check whether [point] can be added into the current list.
                for point1, point2 in combinations(now_points, 2):
                    if self.is_colinear(point1, point, point2, f):
                        now_points.append(point)
                        Update = True
                        break
                if Update: break  # avoid adding duplicate points in out list.

        # Try to sort the line with increasing order.
        if self.initUni:
            new_list = [line[0]]
            while len(new_list) < len(now_points):
                changed = False
                for point in now_points:
                    if point not in new_list:
                        if self.check_uni_line((point, new_list[-1])):
                            new_list.append(point)
                            changed = True
                        elif self.check_uni_line((point, new_list[0])):
                            new_list = [point] + new_list
                            changed = True
                if not changed:
                    if self.debug:
                        print("\033[0;0;41mError:\033[0m ", end="")
                        print("the line information is incorrect.")
                        print(now_points, new_list)
                    new_list = now_points
                    break
        else:
            new_list = now_points

        for p1, p2 in combinations(new_list, 2):
            # Change the direction if possible.
            if new_list.index(p1) > new_list.index(p2):
                p1, p2 = p2, p1
            # Store the answer to accelerate
            self.points_on_line[(p1, p2)] = new_list
            self.points_on_line[(p2, p1)] = new_list[::-1]

        return self.points_on_line[line]

    def find_all_triangles(self):
        x = var()
        y = var()
        z = var()

        res = run(0, (x, y, z), self.Triangle(x, y, z))
        return list(res)

    def find_all_quadrilaterals(self):
        x = var()
        y = var()
        z = var()
        w = var()

        res = run(0, (x, y, z, w), self.Quadrilateral(x, y, z, w))
        return list(res)

    def find_all_pentagons(self):
        x = var()
        y = var()
        z = var()
        s = var()
        t = var()

        res = run(0, (x, y, z, s, t), self.Pentagon(x, y, z, s, t))
        return list(res)

    def find_all_parallels(self):
        x = var()
        y = var()
        z = var()
        w = var()

        res = run(0, (x, y, z, w), self.Parallel(x, y, z, w))
        return [((t[0], t[1]), (t[2], t[3])) for t in list(res)]
    
    def find_all_perpendicular(self):
        x = var()
        y = var()
        z = var()
        w = var()

        res = run(0, (x, y, z, w), self.Perpendicular(x, y, z, w))
        return [((t[0], t[1]), (t[2], t[3])) for t in list(res)]
    
    def find_all_similar_triangles(self):
        x1 = var()
        y1 = var()
        z1 = var()
        x2 = var()
        y2 = var()
        z2 = var()
        res = run(0, (x1,y1,z1,x2,y2,z2), self.SimilarTriangle(x1,y1,z1,x2,y2,z2))
        return [(p[0:3],p[3:]) for p in list(res)]

    def find_all_congruent_triangles(self):
        x1 = var()
        y1 = var()
        z1 = var()
        x2 = var()
        y2 = var()
        z2 = var()
        res = run(0, (x1,y1,z1,x2,y2,z2), self.CongruentTriangle(x1,y1,z1,x2,y2,z2))
        return [(p[0:3],p[3:]) for p in list(res)]

    def find_all_similar_polygons(self):
        x1 = var()
        y1 = var()
        z1 = var()
        w1 = var()
        s1 = var()
        t1 = var()
        x2 = var()
        y2 = var()
        z2 = var()
        w2 = var()
        s2 = var()
        t2 = var()        
        res4 = run(0, (x1,y1,z1,w1,x2,y2,z2,w2), self.SimilarPolygon(x1,y1,z1,w1,x2,y2,z2,w2))
        res5 = run(0, (x1,y1,z1,w1,s1,x2,y2,z2,w2,s2), self.SimilarPolygon(x1,y1,z1,w1,s1,x2,y2,z2,w2,s2))
        res6 = run(0, (x1,y1,z1,w1,s1,t1,x2,y2,z2,w2,s2,t2), self.SimilarPolygon(x1,y1,z1,w1,s1,t1,x2,y2,z2,w2,s2,t2))
        ret = [((p[0:4]),(p[4:])) for p in list(res4)] + [((p[0:5]),(p[5:])) for p in list(res5)] + [((p[0:6]),(p[6:])) for p in list(res6)]
        return ret
        
        
