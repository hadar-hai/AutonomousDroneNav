[![Python 3.8.0](https://img.shields.io/badge/python-3.8.10+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3810/)
[![NumPy](https://img.shields.io/badge/numpy-1.26.2+-green?logo=numpy&logoColor=white)](https://pypi.org/project/numpy/1.26.2/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.8.2+-green?logo=plotly&logoColor=white)](https://pypi.org/project/matplotlib/3.8.2)
[![pandas](https://img.shields.io/badge/pandas-2.1.4+-green?logo=pandas&logoColor=white)](https://pandas.org/)

# Autonomous Drone Navigation in an Urban Environment



#### [Hadar Hai](https://www.linkedin.com/in/hadar-hai/), Technion - Israel Institute of Technology

The objective of this project is to develop a codebase that enables autonomous navigation for drones within urban settings using a trajectory planning algorithm. The autonomous drone will initiate its flight from a designated start position and maintain a constant height while navigating the most efficient route towards a specified goal position. Upon reaching the goal, the drone will enter a hovering state, signifying its arrival at the destination.

The drone's LIDAR sensor will be utilized to detect obstacles obstructing its flight path, thereby preventing a direct route to the goal position. Upon identifying such obstacles, the drone is required to implement an evasion strategy, circumventing the obstruction until a clear pathway to the goal is achievable.

## Table of Contents

* [Requirements](#requirements)
* [How to Use](#how-to-use)
* [Examples](#examples)
* [Sources](#sources)

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

We provide here several navigation examples, using different heights and different paths used.

## Videos




## Navigation Paths



# Sources

* [Kamon, Ishay, Elon Rimon, and Ehud Rivlin. "Tangentbug: A range-sensor-based navigation algorithm." *The International Journal of Robotics Research* 17, no. 9 (1998): 934-953.](https://csaws.cs.technion.ac.il/~ehudr/publications/pdf/KamonRR98a.pdf)