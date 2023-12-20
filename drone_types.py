class Position:
    def __init__(self):
        self.x_m = 0.0
        self.y_m = 0.0
        self.z_m = 0.0

    def __str__(self):
        return f'({self.x_m}, {self.y_m}, {self.z_m})'


class Orientation:
    def __init__(self):
        self.x_rad = 0.0
        self.y_rad = 0.0
        self.z_rad = 0.0

    def __str__(self):
        return f'({self.x_rad}, {self.y_rad}, {self.z_rad})'


class Pose:
    def __init__(self):
        self.pos = Position()
        self.orientation = Orientation()

    def __str__(self):
        return f'pos: {str(self.pos)}\norientation: {str(self.orientation)}'


class PointCloud:
    def __init__(self):
        self.points = []

    def __str__(self):
        return str(self.points)
