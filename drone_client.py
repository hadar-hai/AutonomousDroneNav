import airsim
import drone_types


class DroneClient:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.future = None

    def __del__(self):
        if self.future is not None:
            self.future.join()

    def connect(self):
        """
        Connect to simulation

        Args:
            none

        Returns:
            none
        """
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def isConnected(self):
        """
        Check if client is connected to simulation

        Args:
            none

        Returns:
            bool: true if connected. otherwise return false
        """
        return self.client.isApiControlEnabled()

    def getPose(self):
        """
        Get the pose of the drone

        Args:
            none

        Returns:
            DroneTypes.Pose : the pose of the drone
        """
        drone_pose = self.client.simGetVehiclePose()
        res = drone_types.Pose()

        res.pos.x_m = drone_pose.position.x_val
        res.pos.y_m = drone_pose.position.y_val
        res.pos.z_m = drone_pose.position.z_val

        euler = airsim.utils.to_eularian_angles(drone_pose.orientation)

        res.orientation.x_rad = euler[0]
        res.orientation.y_rad = euler[1]
        res.orientation.z_rad = euler[2]

        return res

    def getLidarData(self):
        point_cloud = drone_types.PointCloud()
        lidar_data = self.client.getLidarData()

        point_cloud.points = lidar_data.point_cloud

        return point_cloud

    def flyToPosition(self, x: float, y: float, z: float, v: float):
        """
        Fly the drone to position

        Args:
            x : float - x coordinate
            y : float - y coordinate
            z : float - z coordinate
            v : float - the velocity which the drone fly to position

        Returns:
            none
        """
        self.future = self.client.moveToPositionAsync(x, y, z, v, drivetrain=airsim.DrivetrainType.ForwardOnly,
                                                      yaw_mode=airsim.YawMode(False, 0.0))

    def setAtPosition(self, x: float, y: float, z: float):
        """
        Set the drone at position instantly

        Args:
            x : float - x coordinate
            y : float - y coordinate
            z : float - z coordinate

        Returns:
            none
        """
        pos = airsim.Vector3r(x, y, z)
        q = airsim.Quaternionr(1, 0, 0, 0)
        pose = airsim.Pose(pos, q)

        self.client.simSetVehiclePose(pose, True)
        self.flyToPosition(x, y, z, 1)

    def reset(self):
        """
        Returns the drone to start position

        Args:
            none

        Returns:
            none
        """
        self.client.reset()
