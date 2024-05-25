# Robot_Warehouse_Sim
Robot_Warehouse_Sim simulates an autonomous scalable framework which uses robots to manage import, storage, and export of packages, in a time-sensitive manner, within a warehouse-type environment.

![robot_warehouse_sim](https://github.com/benfpv/robot_warehouse_sim/assets/55154673/d94b5d34-b65e-41d7-97d8-b29c454b6042)

# Features
- The warehouse environment consists of an import, storage, and export area for packages, where packages "enter" the warehouse via import areas, and "exit" the warehouse via export areas.
- Each package has unique identifiers, real-time position and status, deadline-to-export, and their history in the warehouse from import-to-export is logged.
- Each robot can transport one package at a time, has unique identifiers, real-time position and status, dynamic battery & charging (at charging stations along the warehouse perimiter), and priority-based action queue.
- Robots currently do not have (but are designed to be able to handle) collision-avoidance, optimized trajectories, and ability to carry other robots which are deemed defective.
- Each robot's action queue is managed dynamically by the warehouse to distribute packages across storage and export areas in a time-efficient (i.e., sampling-based search for available space) and deadline-prioritized manner, to charge robots when low on battery, and to avoid package-movement-effort duplication at any time.

# Known Issues
- When a robot is done charging but idle (no tasks assigned), another robot can superimpose to charge at the charger.

# Limitations
- A notable current limitation to the framework is that import, storage, and export areas must be square/rectangular, and divided horizontally (e.g., import areas at the top, storage areas in the middle, export areas at the bottom).
Future implementations/reworks would ideally be able to handle more complex classification/distribution of areas - possibly with use of K-nearest neighbours to auto-classify areas based on how all imports/storage/exports are designated.

# Future Directions
- Ability for warehouse to handle complex classification/distribution of areas.
- Robots collision avoidance.
- Robots optimized trajectories based on acceleration/weight, etc.
- Robot random defects simulation.
- Chargers random defects simulation.
- Packages random defects simulation.
- Ability to carry other robots.
- Data-driven warehouse/robots behaviours.
- Logical optimization: Allow switching higher-priority packages not in exports, with lower-priority packages in exports, to reduce blocking.
- Logical optimization: Have robots attempt to charge at nearest charger, to reduce movements.

# Software Prerequisites (on typical OS)
1. Python 3
2. Python 3 packages: numpy, cv2 (opencv-python)

# Instructions for Use
1. For quick run, just run "py main.py" in command prompt under "logistics2d" directory.
2. Change window resolution(s) in main.py MainGame.__init__() section.
3. Change warehouse characteristics in data/warehouse/warehouse.py Warehouse.__init__() section.

# Legend
Main Window:
- Areas:
	- Dark Grey Areas are just neutral areas (i.e., robots/packages should not park here).
	- Dark Green Areas are import areas.
	- Dark Blue Areas are storage areas.
	- Dark Red Areas are export areas.
- Objects:
	- Yellow items along the perimiter are chargers.
	- White items are robots.
	- Grey items are packages.
	- Green items are packages that are planned to be moved (i.e., allocated a robot and target position).
	- Dark blue items are packages that have a deadline-to-export of <60 seconds.
	- Light blue items are packages that will be exported in <10 seconds.
	- Orange items are packages that have exceeded the deadline, but are planned to be moved.
	- Red items are packages that have exceeded the deadline, and are not yet planned to be moved.

Supporting Windows:
- Top Left window draws chargers only.
- Top Right window draws packages only.
- Bottom Left window draws robots only.
- Bottom Right window draws package planned locations only (i.e., target/planned position).
