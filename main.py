from drone_client import DroneClient
import time
import calendar
import city_map
import numpy as np
import matplotlib
import argparse

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from shapely.plotting import plot_points, plot_line
from shapely.geometry import Point, LineString, MultiPoint, MultiLineString


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_x', type=int, default=-500)  # -700   # -346   # -1200
    parser.add_argument('--start_y', type=int, default=-800)  # -1100  # -700   #  -500
    parser.add_argument('--end_x', type=int, default=-10)     # -1216  # -1200  # -1216 # -600
    parser.add_argument('--end_y', type=int, default=-400)    # -372   # -400   # -372 #  -1100
    parser.add_argument('--height', type=int, default=-50)
    args = parser.parse_args()

    start_point = Point(args.start_x, args.start_y)
    end_point = Point(args.end_x, args.end_y)
    client = DroneClient()
    client.connect()

    print(client.isConnected())

    obsTablePath = r'D:\Project\I2R_Sim_WM_800x600'
    obsTableName = 'obstacles_100m_above_sea_level.csv'

    cityMap = city_map.CityMap()

    drone = city_map.Drone(client, Point(args.start_x, args.start_y, args.height))
    drone.goal = Point(args.end_x, args.end_y)
    path_planner = city_map.PathPlanner(drone, cityMap)

    trajectory_motion = [Point(args.start_x, args.start_y)]
    trajectory_boundary = []

    drone_pos_motion = [Point(args.start_x, args.start_y)]
    drone_pos_boundary = []
    drone_pos_default = []
    drone_pos_crisis = []
    previous_print_time = 0
    stuck_count = 0
    previous_drone_position = Point(0, 0)
    lidars_to_print = []
    drone_time = time.time()
    while True:
        drone.update_position()
        local_lidars = drone.lidar(cityMap)
        drone_path = drone.path_to_goal()
        drone_current_path = drone.path_to_current_destination()
        path_planner.tangent_bug()

        if drone.destination_reached(drone.goal):
            print("SUCCESS!!")
            break
        if not path_planner.check_if_goal_is_reachable():
            print("GOAL IS UNREACHABLE")
            break

        # Printers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if path_planner.action_state == city_map.State.motion_to_goal:
            drone_pos_motion.append(drone.current_position)
        elif path_planner.action_state == city_map.State.boundary_following:
            drone_pos_boundary.append(drone.current_position)
        elif path_planner.action_state == city_map.State.default:
            drone_pos_default.append(drone.current_position)
        elif path_planner.action_state == city_map.State.crisis:
            drone_pos_crisis.append(drone.current_position)

        if path_planner.action_state == city_map.State.motion_to_goal:
            trajectory_motion.append(drone.current_destination)
        else:
            trajectory_boundary.append(drone.current_destination)
        lidars_to_print += local_lidars

        if time.time() - previous_print_time > 2:
            previous_print_time = time.time()
            print("Mode: ", path_planner.action_state,
                  "Clockwise: ", "True " if not path_planner.following_boundary_clockwise else "False "
                  "Position: (", round(drone.current_position.x), ", ", round(drone.current_position.y), ")",
                  "Destination: (", round(drone.current_destination.x), ", ", round(drone.current_destination.y), ")",
                  "Speed: ", round(drone.speed),
                  "time: ", round(time.time() - drone_time))
            if drone.destination_reached(previous_drone_position):
                print("STUCK!")
                stuck_count += 1
                if stuck_count == 8:
                    break
            else:
                stuck_count = 0
                previous_drone_position = drone.current_position

    objects_polygons = []
    for obstacle in cityMap.obstacles:
        objects_polygons.append(obstacle.line_string)
    multi_Lines = MultiLineString(objects_polygons)
    trajectory_points_motion = MultiPoint(trajectory_motion)
    trajectory_points_boundary = MultiPoint(trajectory_boundary)
    drone_pos_points_motion = MultiPoint(drone_pos_motion)
    drone_pos_points_boundary = MultiPoint(drone_pos_boundary)
    drone_pos_points_default = MultiPoint(drone_pos_default)
    drone_pos_points_crisis = MultiPoint(drone_pos_crisis)
    lidar_all_points_points = MultiPoint(list(lidars_to_print))

    plt.figure()
    plot_line(LineString([Point(args.start_x, args.start_y), Point(args.end_x, args.end_y)]), color='m', alpha=0.5, label='initTraj')
    plot_line(multi_Lines, color='k', alpha=0.5, label='obs')
    plot_points(trajectory_points_motion, color='r', alpha=0.5, label='traj - motion to goal')
    plot_points(trajectory_points_boundary, color='m', alpha=0.5, label='traj - boundary following')
    plot_points(lidar_all_points_points, color='y', alpha=0.5, label='lidar points')
    plot_points(drone_pos_points_motion, color='b', alpha=0.5, label='drone position - motion to goal')
    plot_points(drone_pos_points_boundary, color='c', alpha=0.5, label='drone position - boundary following')
    plot_points(drone_pos_points_crisis, color='k', alpha=0.5, label='drone position - crisis')
    plt.scatter(args.end_x, args.end_y, color='r', s=200)
    lgd = plt.legend()
    plt.grid(True)
    current_GMT = time.gmtime()
    time_stamp_name = calendar.timegm(current_GMT)
    plt.savefig(f'Examples\\{time_stamp_name}.png')
    plt.show()

    X_max = 100
    Y_max = -400

    X_min = -1250  # -1250
    Y_min = -1300  # -1300

    MAP_WIDTH = abs(X_max) + abs(X_min)
    MAP_HEIGHT = abs(Y_max) + abs(Y_min)

    MAP_TO_PRINT = np.zeros((MAP_HEIGHT, MAP_WIDTH))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for obstacle in cityMap.obstacles:
        for lidar in obstacle.lidar_input:
            print(round(abs(lidar.y - Y_min)), ", ", round(abs(lidar.x - X_min)))
            print(MAP_TO_PRINT[round(abs(lidar.y - Y_min))][round(abs(lidar.x - X_min))])
            MAP_TO_PRINT[round(abs(lidar.y - Y_min))][round(abs(lidar.x - X_min))] = 1
            print(MAP_TO_PRINT[round(abs(lidar.y - Y_min))][round(abs(lidar.x - X_min))])
    print(MAP_TO_PRINT)
    np.savetxt("MAP_TO_PRINT.csv", MAP_TO_PRINT, delimiter=",")

    plt.imshow(MAP_TO_PRINT)
    plt.colorbar()
    # plt.show()
