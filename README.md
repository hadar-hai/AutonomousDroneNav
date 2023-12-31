[![Python 3.8.0](https://img.shields.io/badge/python-3.8.10+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3810/)
[![NumPy](https://img.shields.io/badge/numpy-1.26.2+-green?logo=numpy&logoColor=white)](https://pypi.org/project/numpy/1.26.2/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.8.2+-green?logo=plotly&logoColor=white)](https://pypi.org/project/matplotlib/3.8.2)
[![pandas](https://img.shields.io/badge/pandas-2.1.4+-green?logo=pandas&logoColor=white)](https://pandas.org/)

# Autonomous Drone Navigation in an Urban Environment

https://github.com/hadar-hai/AutonomousDroneNav/assets/64587231/dfffb56c-0a91-4ce1-a438-174521250deb

#### [Hadar Hai](https://www.linkedin.com/in/hadar-hai/), Technion - Israel Institute of Technology

The objective of this project is to develop a codebase that enables autonomous navigation for drones within urban settings using a trajectory planning algorithm. The autonomous drone will initiate its flight from a designated start position and maintain a constant height while navigating the most efficient route towards a specified goal position. Upon reaching the goal, the drone will enter a hovering state, signifying its arrival at the destination.

The drone's LIDAR sensor will be utilized to detect obstacles obstructing its flight path, thereby preventing a direct route to the goal position. Upon identifying such obstacles, the drone is required to implement an evasion strategy, circumventing the obstruction until a clear pathway to the goal is achievable.

## Table of Contents

* [Requirements](#requirements)
* [How to Use](#how-to-use)
* [Examples](#examples)
* [Sources](#sources)
* [Acknowledgements](#acknowledgements)

## Requirements

The code is intended to be used on Windows 10 machines, and on a compact version of [AirSim](https://microsoft.github.io/AirSim/#how-to-get-it).  
Therefore, if the normal AirSim installation does not work, please contact us for the slim version.

Install the Python environment.

1. Create an environment.

```batch
python -m venv venv 
venv\Scripts\activate.bat
pip install wheel
```

2. Install the AirSim Python API dependencies.

```batch
pip install numpy msgpack-rpc-python matplotlib pandas shapely pyqt5 
```

3. Install AirSim Python API.

```batch
pip install airsim
```

   1. Install [conda](https://www.anaconda.com/download#downloads).
   2. Create a conda environment: `conda create --name autDrone python=3.8`
   3. Activate the environment: `conda activate autDrone`
   4. Install the AirSim API package: `pip install numpy msgpack-rpc-python matplotlib pandas airsim`

# How to Use

1. Make sure that the AirSim application is up and running.  
2. Run `main.py` using your desired path, e.g.:

```batch
python main.py --start_x -700 --start_y -1100 --end_x -1216 --end_y -372 --height -50
```

# Examples

We provide in the `Examples` directory several navigation examples in video format, using different heights and different paths used, for the following navigation paths:

![1703086169](https://github.com/hadar-hai/AutonomousDroneNav/assets/64587231/0c732209-68ff-43b4-920e-d802bf17f8c5)
![1703085626](https://github.com/hadar-hai/AutonomousDroneNav/assets/64587231/6476737c-e57f-49dc-b3f2-b3605d16fcf5)
![1703085327](https://github.com/hadar-hai/AutonomousDroneNav/assets/64587231/bccfd08b-9d33-4ec2-836b-63b4b436907b)
![1703085112](https://github.com/hadar-hai/AutonomousDroneNav/assets/64587231/5ecc6400-0814-4407-a565-1b4384197da7)

# Sources

* [Kamon, Ishay, Elon Rimon, and Ehud Rivlin. "Tangentbug: A range-sensor-based navigation algorithm." *The International Journal of Robotics Research* 17, no. 9 (1998): 934-953.](https://csaws.cs.technion.ac.il/~ehudr/publications/pdf/KamonRR98a.pdf)

# Acknowledgements

We thank [Hila Manor](https://www.linkedin.com/in/hilamanor/) for contributing the needed resources for running the project.  
The project was created as a part of course CS236927 of Computer Science faculty, Technion.
