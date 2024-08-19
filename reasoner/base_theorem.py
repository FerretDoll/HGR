import numpy as np


class BaseTheorem:
    def __init__(self, logic):
        self.logic = logic

    @staticmethod
    def get_lines_of_polygon(polygon):
        lines = []
        for i in range(len(polygon)):
            if i + 1 == len(polygon):
                line = (polygon[i], polygon[0])
            else:
                line = (polygon[i], polygon[i + 1])
            lines.append(line)

        return lines

    @staticmethod
    def get_angles_of_polygon(polygon):
        angles = []
        for i in range(len(polygon)):
            if i - 1 < 0:
                angle = (polygon[-1], polygon[i], polygon[i + 1])
            elif i + 1 == len(polygon):
                angle = (polygon[i - 1], polygon[i], polygon[0])
            else:
                angle = (polygon[i - 1], polygon[i], polygon[i + 1])
            angles.append(angle)

        return angles

    def calc_distance_from_point_to_point(self, point1, point2):
        if isinstance(point1, str):
            point1 = np.array(self.logic.point_positions.get(point1))
        if isinstance(point2, str):
            point2 = np.array(self.logic.point_positions.get(point2))
        return np.linalg.norm(point1 - point2)

    def calc_distance_from_point_to_line(self, point, line):
        if isinstance(point, str):
            point = np.array(self.logic.point_positions.get(point))
        line_point1 = np.array(self.logic.point_positions.get(line[0]))
        line_point2 = np.array(self.logic.point_positions.get(line[1]))
        vec1 = line_point1 - point
        vec2 = line_point2 - point
        distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
        return distance

    def calc_cross_angle(self, line1, line2, is_rad=False):
        if set(line1) == set(line2):
            return 0
        line1_point1, line1_point2 = \
            self.logic.point_positions.get(line1[0]), self.logic.point_positions.get(line1[1])
        line2_point1, line2_point2 = \
            self.logic.point_positions.get(line2[0]), self.logic.point_positions.get(line2[1])

        if None in (line1_point1, line1_point2, line2_point1, line2_point2):
            raise ValueError(f"One or more points not found for lines {line1} and {line2}")

        try:
            arr_a = np.array([(line1_point2[0] - line1_point1[0]), (line1_point2[1] - line1_point1[1])])
            arr_b = np.array([(line2_point2[0] - line2_point1[0]), (line2_point2[1] - line2_point1[1])])
            cos_value = (float(arr_a.dot(arr_b)) / (np.sqrt(arr_a.dot(arr_a)) * np.sqrt(arr_b.dot(arr_b))))
            cos_value = max(min(cos_value, 1), -1)
        except Exception as e:
            raise RuntimeError(f"Error calculating cosine value: {e}")

        if is_rad:
            return np.arccos(cos_value)
        else:
            return np.arccos(cos_value) * (180 / np.pi)

    def calc_angle_measure(self, angle, is_rad=False):
        line1 = (angle[1], angle[0])
        line2 = (angle[1], angle[2])
        return self.calc_cross_angle(line1, line2, is_rad)

    def point_inside_rectangle(self, point, line):
        """
        Ensure that the point is within the rectangle formed by the vertices of the line segment,
        and that it is not on the extension line or the reverse extension line of the line segment
        """
        err = 3
        if isinstance(point, str):
            point = np.array(self.logic.point_positions.get(point))
        line_point1 = np.array(self.logic.point_positions.get(line[0]))
        line_point2 = np.array(self.logic.point_positions.get(line[1]))

        if abs(line_point2[1] - line_point1[1]) <= err:
            max_x = max(line_point1[0], line_point2[0])
            min_x = min(line_point1[0], line_point2[0])
            if min_x <= point[0] <= max_x:
                return True
        elif abs(line_point2[0] - line_point1[0]) <= err:
            max_y = max(line_point1[1], line_point2[1])
            min_y = min(line_point1[1], line_point2[1])
            if min_y <= point[1] <= max_y:
                return True
        else:
            max_x = max(line_point1[0], line_point2[0])
            max_y = max(line_point1[1], line_point2[1])
            min_x = min(line_point1[0], line_point2[0])
            min_y = min(line_point1[1], line_point2[1])
            if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
                return True

        return False
