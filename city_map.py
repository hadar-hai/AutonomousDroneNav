import math
import time
import numpy as np
import pandas as pd
import matplotlib
import utils
from typing import List, Optional, Tuple
from shapely import LinearRing
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
from drone_client import DroneClient

matplotlib.use('Qt5Agg')

MAX_SPEED = 9                           # [m/s] Max speed of drone
MIN_SPEED = 5                           # [m/s] Min speed of drone
MIN_SAFETY_DISTANCE = 3                 # [m] Minimum distance from obstacles
MAX_SAFETY_DISTANCE = 10                # [m] Maximum distance from obstacles
ACCELERATION = 1.4                      # [k] acceleration of the drone in relation of its distance from closest obstacle
DRONE_SIZE = 4                          # [m] simulate drone size :)
PIXEL_SIZE = 2                          # [m] Minimum distance between two lidar measurement points
UNION_DISTANCE = MIN_SAFETY_DISTANCE * 2 + 1     # [m] Minimum distance between obstacles. will union them if their distance is smaller from one to another
LIDAR_SENSITIVITY = 3                   # [k] Will clean lidar noise, won't insert lidar input if he's not in a group of other lidar points. NOTE: must be 2 or greater
LIDAR_RADIUS = 35                       # [m] Good thing to know the lidar radius :)
HEURISTIC_DISTANCE_COUNTER_MAX = 15     # [k] heuristic distance change counter max - for moving to "boundary following"
BOUNDARY_CLOCKWISE_COUNT_MAX = 50       # [k] clockwise change count - for "boundary following" to go to direction of "motion to goal" without changing its direction
CRISIS_STUCK_COUNT = 5                  # [s] will enter crisis if drone stuck in the same place more than 5 secs
CRISIS_OBSTACLE_TIMEOUT = 15            # [s] will enter crisis if drone is inside the same obstacle more than 15 secs


class Obstacle:
    def __init__(self, line_string: LineString, obs_id):
        """
        Initializes an Obstacle object.

        Args:
        - line_string (LineString): The LineString representing the obstacle.
        - obs_id: The ID of the obstacle.
        """
        self.line_string = line_string
        self.corners = [Point(xy) for xy in list(line_string.coords)]
        self.lidar_input = {Point(xy) for xy in list(line_string.coords)}
        self._simplify_obstacle()
        self.ID = obs_id

    def contains(self, shape) -> bool:
        """
        Checks if the obstacle contains a given shape.

        Args:
        - shape: The shape to check.

        Returns:
        - bool: True if the obstacle contains the shape; False otherwise.
        """
        line_string = self.line_string
        try:
            return Polygon(line_string).contains(shape)
        except Exception:
            return False

    def beside(self, shape, delta: float = PIXEL_SIZE) -> bool:
        """
        Checks if the obstacle is close to a given shape within a specified delta.

        Args:
        - shape: The shape to check.
        - delta (float, optional): The delta value. Defaults to PIXEL_SIZE.

        Returns:
        - bool: True if the obstacle is close to the shape within the specified delta; False otherwise.
        """
        return self.line_string.distance(shape) <= delta

    def intersects(self, line: LineString) -> bool:
        """
        Checks if the obstacle intersects with a given LineString.

        Args:
        - line (LineString): The LineString to check for intersection.

        Returns:
        - bool: True if the obstacle intersects with the given LineString; False otherwise.
        """
        line_string = self.line_string
        try:
            return Polygon(line_string).intersects(line)
        except Exception:
            return self.line_string.intersects(line)

    def distance(self, shape) -> float:
        """
        Calculates the distance between the obstacle and a given shape.

        Args:
        - shape: The shape to calculate the distance to.

        Returns:
        - float: The distance between the obstacle and the given shape.
        """
        if isinstance(shape, Obstacle):
            return self.line_string.distance(shape.line_string)
        else:
            return self.line_string.distance(shape)

    def add(self, shape):
        """
        Adds a shape to the obstacle.

        Args:
        - shape: The shape to add to the obstacle.

        Returns:
        - Obstacle: The updated Obstacle object.
        """
        if isinstance(shape, LineString):
            self.corners += shape.coords
            self.lidar_input.update(Point(xy) for xy in shape.coords)
        elif isinstance(shape, Point):
            self.corners.append(shape)
            self.lidar_input.add(shape)

        self.line_string = LineString(self.corners)
        self._simplify_obstacle()
        return self

    def union(self, other):
        """
        Forms the union of the obstacle with another obstacle.

        Args:
        - other: The other obstacle to form a union with.

        Returns:
        - None: If the obstacle contains the other obstacle; otherwise, updates the obstacle with the union.
        """
        if self.contains(other):
            return

        self.line_string = other.line_string.union(self.line_string)
        self.lidar_input.update(other.lidar_input)
        self._simplify_obstacle()
        return self

    def copy_expanded(self, delta: float = MIN_SAFETY_DISTANCE):
        """
        Creates a copy of the obstacle with an expanded boundary.

        Args:
        - delta (float, optional): The distance by which to expand the boundary. Defaults to MIN_SAFETY_DISTANCE.

        Returns:
        - Obstacle: A copy of the obstacle with an expanded boundary.
        """
        return Obstacle(self.line_string.buffer(delta).exterior.envelope.boundary, self.ID)

    def _simplify_obstacle(self) -> None:
        """
        Simplifies the obstacle's boundary.
        """
        convex_boundary = self.line_string.convex_hull.boundary
        if isinstance(convex_boundary, MultiPoint):
            boundary = list(convex_boundary.geoms)
        elif isinstance(convex_boundary, MultiLineString):
            boundary = list(convex_boundary.geoms)
        elif isinstance(convex_boundary, MultiPolygon):
            boundary = list(convex_boundary.geoms)
        else:
            boundary = list(convex_boundary.coords)

        if len(boundary) > 2:
            del boundary[-1]
        self.line_string = LineString(boundary)
        self.corners = [Point(x) for x in boundary]

    def closest_lidar(self, shape) -> Tuple[Point, float]:
        """
        Finds the closest lidar point to a given shape within the obstacle.

        Args:
        - shape: The shape to which the closest lidar point is determined.

        Returns:
        - tuple: The closest lidar point and its distance to the given shape.
        """
        lidar_tuples = [(lidar, lidar.distance(shape)) for lidar in self.lidar_input]
        return min(lidar_tuples, key=lambda tup: tup[1])

    def find_oi(self, drone) -> Tuple[any]:  # TODO
        """
        Finds potential Oi for a drone.

        Args:
        - drone: The drone object for which Oi are determined.

        Returns:
        - tuple: The reference and destination obstacles with their corresponding heuristics.
        """
        desired_angle = drone.angle_to_destination(drone.goal)
        lidar_datas = [(lid, drone.angle_to_destination(lid) - desired_angle) for lid in self.lidar_input]
        min_angle = min(lidar_datas, key=lambda x: x[1])
        max_angle = max(lidar_datas, key=lambda x: x[1])

        reference_wth_heuristic = (min_angle[0], drone.current_position.distance(min_angle[0]) +
                                   drone.goal.distance(min_angle[0]))
        destination_with_heuristic = (max_angle[0], drone.current_position.distance(max_angle[0]) +
                                      drone.goal.distance(max_angle[0]))
        if reference_wth_heuristic[1] < destination_with_heuristic[1]:
            temp = destination_with_heuristic
            destination_with_heuristic = reference_wth_heuristic
            reference_wth_heuristic = temp

        return reference_wth_heuristic, destination_with_heuristic

    def behind_line(self, path: LineString) -> bool:
        """
        Checks if the obstacle is behind a given line.

        Args:
        - path (LineString): The line to check against.

        Returns:
        - bool: True if the obstacle is behind the line; False otherwise.
        """
        start_point = Point(path.coords[0])
        end_point = Point(path.coords[1])
        closest_lidar = self.closest_lidar(start_point)[0]
        angle_start_to_closest_lidar = math.atan2(closest_lidar.y - start_point.y, closest_lidar.x - start_point.x)
        angle_of_path = math.atan2(end_point.y - start_point.y, end_point.x - start_point.x)

        angle_diff = angle_start_to_closest_lidar - angle_of_path
        return not 0 < angle_diff < math.pi


class CityMap:
    def __init__(self):
        """
        Initializes a CityMap object.
        """
        self.obstacles = []  # Holds all obstacle objects. every obstacle will hold all points
        self.obstacles_dict = dict()
        self.lidar_groups = []  # array of arrays

    def add_point(self, point: Point) -> None:
        """
        Add a lidar point to the CityMap for obstacle detection.

        Args:
        - point: The lidar point to be added.
        """
        sensitivity = 3
        for group in self.lidar_groups:
            distance = point.distance(MultiPoint(group))
            if isinstance(distance, list):
                distance = min(point.distance(MultiPoint(group)))
            if distance <= 0.1:
                return
            if distance <= 1:
                group.append(point)
                if len(group) >= sensitivity:
                    self.add_linestring(LineString(group))
                    return
        self.lidar_groups.append([point])

    def add_linestring(self, line_string: LineString) -> Obstacle:
        """
        Adds a LineString object to the CityMap.

        Args:
        - line_string: The LineString object to be added.
        """
        close_obstacles = [obs for obs in self.obstacles
                           if obs.intersects(line_string) or obs.contains(line_string)
                           or obs.beside(line_string, UNION_DISTANCE)]

        if not close_obstacles:
            self.add_obstacle(line_string)
            return self.obstacles[-1]

        obstacle = close_obstacles[0]
        obstacle.add(line_string)
        for i in range(len(close_obstacles) - 1):
            if i >= len(close_obstacles) - 1:
                break
            obstacle.union(close_obstacles[i+1])
            index = self.obstacles.index(close_obstacles[i+1])
            self.obstacles[-1].ID = close_obstacles[i+1].ID
            self.obstacles[index] = self.obstacles[-1]
            self.obstacles_dict[close_obstacles[i+1].ID] = self.obstacles_dict.pop(self.obstacles[-1].ID)
            del self.obstacles[-1]
        return obstacle

    def add_obstacle(self, line_string: LineString) -> None:
        """
        Adds an obstacle represented by a LineString to the CityMap.

        Args:
        - line_string: The LineString representing the obstacle.
        """
        obstacle = Obstacle(line_string, len(self.obstacles))
        self.obstacles.append(obstacle)
        self.obstacles_dict[obstacle.ID] = obstacle

    def read(self, obs_table_path: str, obs_table_name: str) -> None:
        """
        Read obstacles data from a file and add them to the CityMap.

        Args:
        - obs_table_path (str): The path to the directory containing obstacle data file.
        - obs_table_name (str): The name of the obstacle data file.

        Reads obstacle data from the specified file, processes it, and adds the obstacles to the CityMap.
        """
        csv_obs_raw = pd.read_csv(obs_table_path + '/' + obs_table_name)
        csv_obs = utils.preProcessObs(csv_obs_raw)
        df_obs = utils.point2Obs(csv_obs)
        for obs in df_obs.Polygon:
            if type(obs) is Polygon:
                self.add_linestring(obs.boundary)

    def obstacles_in_path(self, path: LinearRing, radius: Optional[float] = None) -> List[Obstacle]:
        """
        Finds obstacles that intersect or are close to a given path.

        Args:
        - path (LinearRing): The path for which obstacles are checked.
        - radius (float, optional): The radius to consider when searching for obstacles. Defaults to None.

        Returns a list of obstacles that intersect the given path or are within a specified radius of the path.
        """
        if radius:
            obstacles = self.all_obstacles_in_radius(Point(path.coords[0]), radius)
        else:
            obstacles = self.obstacles
        return [x for x in obstacles if x.intersects(path) or (x.line_string.distance(path) < MIN_SAFETY_DISTANCE)]

    def closest_obstacle_in_path(self, path: LinearRing, radius: Optional[float] = None) -> Optional[Obstacle]:
        """
        Finds the closest obstacle to a given path.

        Args:
        - path (LinearRing): The path for which the closest obstacle is determined.
        - radius (float, optional): The radius to consider when searching for obstacles. Defaults to None.

        Returns the closest obstacle to the given path within a specified radius.
        """
        start_point = Point(path.coords[0])
        obstacles = self.obstacles_in_path(path, radius)
        if not obstacles:
            return None
        return min(obstacles, key=lambda obs: start_point.distance(obs.line_string))

    def closest_obstacle(self, shape) -> Optional[Obstacle]:
        """
        Finds the closest obstacle to a given shape.

        Args:
        - shape: The shape for which the closest obstacle is determined.

        Returns the closest obstacle to the given shape.
        """
        if not self.obstacles:
            return None
        return min(self.obstacles, key=lambda obs: shape.distance(obs.line_string))

    def all_obstacles_in_radius(self, point: Point, radius: float = LIDAR_RADIUS) -> List[Obstacle]:
        """
        Finds all obstacles within a specified radius of a given point.

        Args:
        - point (Point): The point around which obstacles are searched.
        - radius (float, optional): The radius within which obstacles are considered. Defaults to LIDAR_RADIUS.

        Returns a list of obstacles within the specified radius of the given point.
        """
        return list(filter(lambda obs: obs.line_string.distance(point) <= radius, self.obstacles))

    def closest_lidar_tuple(self, shape) -> Optional[Tuple[Point, float]]:
        """
        Finds the closest lidar tuple to a given shape.

        Args:
        - shape: The shape for which the closest lidar tuple is determined.

        Returns the closest lidar tuple to the given shape if found, otherwise None.
        """
        closest_obs = self.closest_obstacle(shape)
        if isinstance(closest_obs, Obstacle):
            return closest_obs.closest_lidar(shape)
        return None

    def find_all_oi(self, drone) -> List[any]:
        """
        Finds all potential Oi for a drone.

        Args:
        - drone: The drone object for which Oi are determined.

        Returns a list of all potential Oi for the given drone.
        """
        obstacles = self.all_obstacles_in_radius(drone.current_position)
        all_oi = []
        for obstacle in obstacles:
            reference_with_heuristic, destination_with_heuristic = obstacle.find_oi(drone)
            all_oi.append(reference_with_heuristic)
            all_oi.append(destination_with_heuristic)
        return all_oi


class Drone:
    def __init__(self, client: DroneClient, position: Point):
        """
        Initializes a Drone object.

        Args:
        - client (DroneClient): The drone client used for communication.
        - position (Point): The initial position of the drone.
        """
        self.client = client
        self.current_position = Point(position.x, position.y)
        self.all_positions = [self.current_position]
        self.current_destination = Point(position.x, position.y)
        self.start_position = position
        self.goal = self.current_position
        self.height = position.z
        self.speed = 0

        client.setAtPosition(position.x, position.y, position.z)

    def fly_to_position(self, position: Point) -> None:
        """
        Commands the drone to fly to a specific position.

        Args:
        - position (Point): The destination position for the drone to fly to.
        """
        self.current_destination = position
        self.client.flyToPosition(position.x, position.y, self.height, self.speed)

    def lidar(self, city_map: CityMap):
        """
        Performs lidar scanning and updates the city map.

        Args:
        - city_map (CityMap): The city map to update with lidar data.

        Returns:
        - set: A set of lidar points detected during scanning.
        """
        local_lidar = self.get_lidar_data_ned()

        for local_point in local_lidar:
            city_map.add_point(local_point)

        return local_lidar

    def update_position(self) -> None:
        """
        Updates the drone's current position.
        """
        position = self.client.getPose()
        x_m = position.pos.x_m
        y_m = position.pos.y_m
        self.current_position = Point(x_m, y_m)
        if self.current_position.distance(self.all_positions[-1]) > PIXEL_SIZE:
            self.all_positions.append(self.current_position)

    def destination_reached(self, destination: Point, delta: float = PIXEL_SIZE) -> bool:
        """
        Checks if the drone has reached the destination point.

        Args:
        - destination (Point): The destination point to check.
        - delta (float, optional): The threshold distance for considering the destination reached.
                                   Defaults to PIXEL_SIZE.

        Returns:
        - bool: True if the drone has reached the destination; False otherwise.
        """
        if not destination:
            return False
        return self.current_position.distance(destination) < delta

    def path_to_goal(self):
        """
        Generates a path from the current position to the goal.

        Returns:
        - LineString: A line representing the path to the goal.
        """
        if not self.current_position:
            return None
        return LineString([self.current_position, self.goal])

    def path_to_current_destination(self):
        """
        Generates a path from the current position to the current destination.

        Returns:
        - LineString: A line representing the path to the current destination.
        """
        return LineString([self.current_position, self.current_destination])

    def angle_to_destination(self, destination: Point):
        """
        Calculates the angle from the drone's current position to a destination point.

        Args:
        - destination (Point): The destination point.

        Returns:
        - float: The angle in radians from the drone's current position to the destination.
        """
        return math.atan2(destination.y - self.current_position.y, destination.x - self.current_position.x)

    @staticmethod
    def drone_body_to_ned(position):
        """
        Converts the drone's body orientation to NED (North East Down) transformation matrix.

        Args:
        - position: The position containing orientation information.

        Returns:
        - numpy.ndarray: The transformation matrix for body to NED coordinates.
        """
        # Bad orientations in "utils"!! switched between orientations!!!
        phi = position.orientation.y_rad
        theta = position.orientation.x_rad
        psi = position.orientation.z_rad
        # Rotation matrix - not as we learned
        rx = np.array([[1, 0, 0], [0, math.cos(phi), -math.sin(phi)], [0, math.sin(phi), math.cos(phi)]])
        ry = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
        rz = np.array([[math.cos(psi), -math.sin(psi), 0], [math.sin(psi), math.cos(psi), 0], [0, 0, 1]])

        drone_position = position.pos

        a_p_b_org = np.array([[drone_position.x_m, drone_position.y_m, drone_position.z_m, 1]])
        drone_body_to_ned_rotation_matrix = np.dot(rx, np.dot(ry, rz))
        drone_body_to_ned_rotation_matrix = np.append(drone_body_to_ned_rotation_matrix, [[0, 0, 0]], axis=0)
        drone_body_to_ned_rotation_matrix = np.concatenate((drone_body_to_ned_rotation_matrix, a_p_b_org.T), axis=1)
        return drone_body_to_ned_rotation_matrix

    def get_lidar_data_ned(self):
        """
        Retrieves lidar data in NED (North East Down) coordinates.

        Returns:
        - set: Lidar points detected in NED coordinates.
        """
        points_to_return = set()

        position_orientation = self.client.getPose()
        self.current_position = Point(position_orientation.pos.x_m, position_orientation.pos.y_m)
        lidar_data = self.client.getLidarData().points
        if len(lidar_data) < 3:
            return points_to_return

        lidar_data_points = np.array(lidar_data, dtype=int)
        lidar_data_points = np.reshape(lidar_data_points, (int(lidar_data_points.shape[0] / 3), 3))

        drone_body_to_ned_matrix = self.drone_body_to_ned(position_orientation)

        # Calculation for each point multiply with matrix
        for lidar_point in lidar_data_points:
            lidar_point = np.append(lidar_point, [1], axis=None)
            lidar_data_ned = np.dot(drone_body_to_ned_matrix, lidar_point)
            points_to_return.add(Point(round(lidar_data_ned[0]), round(lidar_data_ned[1])))

        return points_to_return

    def is_in_crisis(self, obstacle: Obstacle):
        """
        Checks if the drone is in a crisis situation based on its position and proximity to an obstacle.

        Args:
        - obstacle (Obstacle): The obstacle to check against.

        Returns:
        - bool: True if the drone is in a crisis; False otherwise.
        """
        return obstacle.contains(self.current_position) and obstacle.closest_lidar(self.current_position)[1] <= 1

    def obstacle_behind_drone(self, obstacle: Obstacle):
        """
        Checks if an obstacle is behind the drone based on its position and orientation.

        Args:
        - obstacle (Obstacle): The obstacle to check against.

        Returns:
        - bool: True if the obstacle is behind the drone; False otherwise.
        """
        closest_lidar = obstacle.closest_lidar(self.current_position)
        angle_to_closest_lidar = self.angle_to_destination(closest_lidar[0])
        angle_to_goal = self.angle_to_destination(self.goal)
        angle_diff = angle_to_closest_lidar - angle_to_goal
        return not math.pi / 6 < angle_diff < math.pi * 5 / 6


class PathPlanner:
    def __init__(self, drone: Drone, city_map: CityMap):
        """
        Initializes a PathPlanner object.

        :param drone: Drone object for path planning.
        :param city_map: CityMap object representing the environment.
        """
        self.drone = drone
        self.city_map = city_map
        self.previous_lidar_in_process = None
        # motion_to_goal
        self.previous_heuristic_distance = 0
        self.heuristic_increased_counter = 0
        self.current_clockwise = False
        self.clockwise_counter = 0
        # following boundary
        self.following_boundary_clockwise = False
        self.d_min = np.inf
        self.previous_oi_in_process = None  # To be continued
        self.previous_obstacle_id = -1
        self.obstacle_in_process = None
        # crisis_mode
        self.stuck_counter = time.time()
        self.drone_previous_position = drone.current_position
        self.drone_previous_local_destination = None
        self.previous_obstacle_processed = None
        self.obstacle_time_out = time.time()
        # state
        self.previous_action_state = State.motion_to_goal  # To be continued
        self.action_state = State.motion_to_goal  # To be continued
        self.safety_distance = MAX_SAFETY_DISTANCE
        # check goal is reachable
        self.first_point_before_obstacle_contains_goal = None
        self.time_in_goal_contained = time.time()

    def tangent_bug(self):
        """
        Implements the Tangent Bug algorithm for path planning and obstacle avoidance.

        Returns:
        -------
        bool:
            True if the drone reaches its destination, otherwise False.
        """
        if self.drone.destination_reached(self.drone.goal):
            print("END!!!!")
            return True
        if not self.check_if_goal_is_reachable():
            print("GOAL UNREACHABLE!!")
            return True

        self.calculate_speed()
        self.calculate_safety_distance()
        self.calculate_obstacle_to_process()

        self.drone_previous_local_destination = self.drone.current_destination

        if self.action_state == State.default:
            self.move_toward_goal()
        elif self.action_state == State.motion_to_goal:
            self.motion_to_goal()
        elif self.action_state == State.boundary_following:
            self.boundary_following()
        elif self.action_state == State.crisis:
            self.crisis_mode()
        return False

    def move_toward_goal(self):
        """
        Moves the drone toward its destination if no obstacles are in the way.
        """
        if self.obstacle_in_process:
            self.change_state_motion_to_goal()
            return

        self.drone.fly_to_position(self.drone.goal)

    def motion_to_goal(self):
        """
        Implements the motion-to-goal behavior considering obstacles in the drone's path.
        """
        # if there's no obstacle in path of drone and goal - Run for it
        if not self.obstacle_in_process:
            self.move_toward_goal()
            return
        if self.crisis_mode_stuck_check():
            self.change_state_crisis()
            return

        # destination, heuristic_distance = self.follow_oi_min_heuristic_distance()
        reference_with_heuristic, destination_with_heuristic = self.obstacle_in_process.find_oi(self.drone)
        heuristic_distance = destination_with_heuristic[1]
        closest_lidar = self.obstacle_in_process.closest_lidar(self.drone.current_position)
        if closest_lidar[0].distance(reference_with_heuristic[0]) > 2 and \
                closest_lidar[0].distance(destination_with_heuristic[0]) > 2:
            reference_with_heuristic = closest_lidar
        self.current_clockwise = self.calculate_direction_clockwise(reference_with_heuristic[0],
                                                                    destination_with_heuristic[0])
        if self.clockwise_changed_count_check(self.current_clockwise):
            self.following_boundary_clockwise = self.current_clockwise

        if self.heuristic_distance_increases_check(heuristic_distance):
            self.change_state_boundary_follow()
            return
        if reference_with_heuristic[0].distance(destination_with_heuristic[0]) > PIXEL_SIZE:
            self.move_toward_oi_min_heuristic_distance(reference_with_heuristic[0], destination_with_heuristic[0])
        else:
            self.boundary_following_next_step(closest_lidar[0])
        if self.obstacle_in_process.intersects(self.drone.path_to_current_destination()):
            self.boundary_following_next_step(closest_lidar[0])

    def boundary_following(self):
        """
        Implements boundary following behavior to navigate around obstacles.
        """
        if not self.obstacle_in_process:
            self.change_state_motion_to_goal()
            return
        if self.crisis_mode_stuck_check():
            self.change_state_crisis()
            return

        blocking_obstacle = self.city_map.closest_obstacle_in_path(self.drone.path_to_goal())
        if blocking_obstacle and not blocking_obstacle.ID == self.obstacle_in_process.ID:
            self.change_state_motion_to_goal()
            return

        closest_lidar = self.obstacle_in_process.closest_lidar(self.drone.current_position)
        # Calculate d_min and d_leave
        current_d_min = min([self.drone.goal.distance(MultiPoint(list(self.obstacle_in_process.lidar_input)))])
        d_leave = current_d_min
        self.d_min = current_d_min if self.d_min == np.inf else self.d_min

        if self.d_min <= d_leave:
            self.boundary_following_next_step(closest_lidar[0])
        else:
            self.change_state_motion_to_goal()
        if self.obstacle_in_process.contains(self.drone.current_position):
            if abs(self.drone.angle_to_destination(self.drone.current_destination) -
                    self.drone.angle_to_destination(self.drone_previous_local_destination)) > 7 * math.pi / 8:
                print("Crisis - drone returns 180 degrees")
                self.change_state_crisis()
        self.d_min = min(current_d_min, self.d_min)

    def crisis_mode(self):
        """
        Handles crisis situations to safely maneuver the drone out of problematic scenarios.
        """
        if not self.drone.all_positions[-1]:
            print("all position reason")
            self.change_state_motion_to_goal()
            self.drone.fly_to_position(self.drone.goal)
            return
        crisis_safety = max(PIXEL_SIZE + MIN_SAFETY_DISTANCE, 6)
        if (not self.obstacle_in_process) or \
                self.obstacle_in_process.distance(self.drone.current_position) > crisis_safety:
            print("another reason")
            self.change_state_motion_to_goal()
            return

        while self.drone.current_position.distance(self.drone.all_positions[-1]) < crisis_safety:
            del self.drone.all_positions[-1]
        self.drone.fly_to_position(self.drone.all_positions[-1])

    def calculate_speed(self):
        """
        Calculates the drone's speed based on obstacles and lidar information.
        """
        new_speed = min(self.drone.speed + 1, MAX_SPEED)

        closest_lidar = self.city_map.closest_lidar_tuple(self.drone.current_position)
        if closest_lidar:
            new_speed = min(new_speed, closest_lidar[1] / ACCELERATION)

        new_speed = min(new_speed, MAX_SPEED)
        if self.action_state == State.crisis:
            new_speed = max(new_speed, 1)
        else:
            new_speed = max(new_speed, MIN_SPEED)
        self.drone.speed = new_speed

    def calculate_safety_distance(self):
        """
        Calculates a safe distance for the drone considering obstacles in its path.
        """
        blocking_obstacle_to_goal = self.city_map.closest_obstacle_in_path(self.drone.path_to_goal())
        blocking_obstacle_to_current = self.city_map.closest_obstacle_in_path(self.drone.path_to_current_destination())
        if (not blocking_obstacle_to_goal) or (not blocking_obstacle_to_current):
            return MAX_SAFETY_DISTANCE
        return max(MIN_SAFETY_DISTANCE, blocking_obstacle_to_goal.distance(blocking_obstacle_to_current) / 2 - 1)

    def calculate_obstacle_to_process(self):
        """
        Determines the obstacle the drone needs to process based on its path.
        """
        self.obstacle_in_process = self.city_map.closest_obstacle_in_path(self.drone.path_to_goal())

    def follow_oi_min_heuristic_distance(self):
        all_oi = self.city_map.find_all_oi(self.drone)
        if not all_oi:
            return None, np.inf
        oi_min_heuristic_distance, heuristic_distance = self.find_oi_min_heuristic_distance(all_oi)
        safety_destination = self.move_toward_oi_min_heuristic_distance(oi_min_heuristic_distance)
        return safety_destination, heuristic_distance

    def find_oi_min_heuristic_distance(self, all_oi: list):
        oi_min_heuristic_distance = min(all_oi, key=lambda x: x[0].distance(self.drone.current_position) +
                                        x[0].distance(self.drone.goal))
        heuristic_distance = oi_min_heuristic_distance[0].distance(self.drone.current_position) + \
            oi_min_heuristic_distance[0].distance(self.drone.goal)
        return oi_min_heuristic_distance, heuristic_distance

    def heuristic_distance_increases_check(self, heuristic):
        if heuristic >= self.previous_heuristic_distance:
            self.heuristic_increased_counter += 1
        else:
            self.heuristic_increased_counter = 0
        self.previous_heuristic_distance = heuristic
        if self.heuristic_increased_counter > HEURISTIC_DISTANCE_COUNTER_MAX:
            return True
        return False

    def crisis_mode_stuck_check(self):
        if self.drone.current_position.distance(self.drone_previous_position) < PIXEL_SIZE \
                and time.time() - self.stuck_counter > CRISIS_STUCK_COUNT:
            print("CRISIS: Stuck - Not moving so much")
            return True

        self.stuck_counter = time.time()
        self.drone_previous_position = self.drone.current_position

        if not self.obstacle_in_process:
            self.obstacle_time_out = time.time()
            return False
        if self.drone.is_in_crisis(self.obstacle_in_process):
            print("CRISIS: Current expanded obstacle contains drone position and closest lidar is too close to drone "
                  "position.")
            return True
        if not self.previous_obstacle_processed:
            self.obstacle_time_out = time.time()
            return False
        if self.previous_obstacle_processed.ID == self.obstacle_in_process \
                and time.time() - self.obstacle_time_out > CRISIS_OBSTACLE_TIMEOUT:
            print("CRISIS: Time out")
            return True

    def clockwise_changed_count_check(self, clockwise):
        if clockwise != self.following_boundary_clockwise:
            self.clockwise_counter += 1
        else:
            self.clockwise_counter = 0

        return self.clockwise_counter > BOUNDARY_CLOCKWISE_COUNT_MAX

    def calculate_direction_clockwise(self, reference: Point, destination: Point):
        desired_angle = self.drone.angle_to_destination(destination)
        angle_to_closest_lidar = self.drone.angle_to_destination(reference)
        angle_diff = angle_to_closest_lidar - desired_angle
        return bool(angle_diff < 0) ^ bool(abs(angle_diff) > math.pi)

    def boundary_following_next_step(self, reference: Point):
        angle_to_closest_lidar = self.drone.angle_to_destination(reference)

        # Choose target direction clockwise or not
        min_fixing_angle = - math.pi / 9
        max_fixing_angle = math.pi / 3
        drone_reference_distance = reference.distance(self.drone.current_position)
        t = min(drone_reference_distance / self.safety_distance, 1)
        angle_for_fix_distance_from_obs = max_fixing_angle - t * (max_fixing_angle - min_fixing_angle)
        direction_angle = math.pi / 2 + angle_for_fix_distance_from_obs
        angle_to_next_step = angle_to_closest_lidar
        if self.current_clockwise:
            angle_to_next_step += direction_angle
        else:            # if angle_diff < 0:
            angle_to_next_step -= direction_angle

        # fixing angle to [-pi, pi]
        if angle_to_next_step > math.pi:
            angle_to_next_step -= 2 * math.pi
        if angle_to_next_step < -math.pi:
            angle_to_next_step += 2 * math.pi

        radius = max(drone_reference_distance, self.safety_distance)
        next_step_x = self.drone.current_position.x + math.cos(angle_to_next_step) * radius
        next_step_y = self.drone.current_position.y + math.sin(angle_to_next_step) * radius
        next_step = Point(next_step_x, next_step_y)
        self.drone.fly_to_position(next_step)
        return next_step

    def move_toward_oi_min_heuristic_distance(self, reference: Point, destination: Point):
        angle_reference_to_destination = math.atan2(destination.y - reference.y, destination.x - reference.x)
        angle_fixing = math.pi / 4 if self.current_clockwise else - math.pi / 4

        safety_destination_angle = angle_reference_to_destination + angle_fixing
        safety_destination_angle += 2 * math.pi if angle_fixing < -math.pi else 0
        safety_destination_angle -= 2 * math.pi if angle_fixing > math.pi else 0

        x_safety_destination = destination.x + math.cos(safety_destination_angle) * self.safety_distance / 0.8
        y_safety_destination = destination.y + math.sin(safety_destination_angle) * self.safety_distance / 0.8
        safety_destination = Point(x_safety_destination, y_safety_destination)
        self.drone.fly_to_position(safety_destination)
        return safety_destination

    def change_state_boundary_follow(self):
        if self.action_state == State.boundary_following:
            return
        self.previous_action_state = self.action_state
        self.action_state = State.boundary_following
        self.current_clockwise = self.following_boundary_clockwise
        self.d_min = np.inf
        print("Changed state: ", self.action_state)

    def change_state_motion_to_goal(self):
        if self.action_state == State.motion_to_goal:
            return
        self.previous_action_state = self.action_state
        self.action_state = State.motion_to_goal
        self.previous_heuristic_distance = 0
        self.heuristic_increased_counter = 0
        print("Changed state: ", self.action_state)

    def change_state_crisis(self):
        if self.action_state == State.crisis:
            return
        self.previous_action_state = self.action_state
        self.action_state = State.crisis
        self.stuck_counter = time.time()
        self.obstacle_time_out = time.time()
        print("Changed state: ", self.action_state)

    def check_if_goal_is_reachable(self):
        if self.obstacle_in_process and self.obstacle_in_process.copy_expanded().contains(self.drone.goal):
            if not self.first_point_before_obstacle_contains_goal:
                self.first_point_before_obstacle_contains_goal = self.drone.current_position
                self.time_in_goal_contained = time.time()
            else:
                return time.time() - self.time_in_goal_contained < 4 \
                    or not self.drone.destination_reached(self.first_point_before_obstacle_contains_goal,
                                                          MAX_SAFETY_DISTANCE + PIXEL_SIZE)
        return True


class State:
    default = "Default"     # -1
    bug0 = "bug0"           # 0
    motion_to_goal = "motion_to_goal"
    boundary_following = "boundary_following"
    crisis = "crisis"
