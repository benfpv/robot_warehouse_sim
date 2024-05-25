# Robot_Warehouse_Sim
Robot_Warehouse_Sim simulates an autonomous scalable framework which uses robots to manage import, storage, and export of packages, in a time-dependent manner, within a warehouse-type environment.
The warehouse environment consists of an import, storage, and export area for packages, where packages can only "enter" the warehouse via import areas, and "exit" the warehouse via export areas.
Each package has unique identifiers, real-time status, deadline-to-export, and their history in the warehouse from import-to-export is logged.
Each robot can transport one package at a time, has unique identifiers, real-time status, location, trajectory, dynamic battery & charging at charging stations along the warehouse perimiter, and action queue.
Robots currently do not have (but are designed to be able to handle) collision-avoidance, optimized trajectories, and ability to carry other robots which are deemed defective.
Each robot's action queue is managed dynamically by the warehouse to distribute packages across storage and export areas in a space-efficient and deadline-aware manner, charge robots when low on battery, and avoid overlapping package movements or locations at any time.

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
1. Python 3 and packages: numpy, cv2 (opencv-python), 

# Instructions for Use
1. For quick run, just run "py main.py" in command prompt under "logistics2d" directory.
2. Change window resolution(s) in main.py MainGame.__init__() section.
3. Change warehouse characteristics in data/warehouse/warehouse.py Warehouse.__init__() section.

# Legend
The Main window is coloured. Colours are as follows:
Areas:
- Dark Grey Areas are just neutral areas (i.e., robots/packages should not park here).
- Dark Green Areas are import areas.
- Dark Blue Areas are storage areas.
- Dark Red Areas are export areas.
Objects:
- Yellow items along the perimiter are chargers.
- White items are robots.
- Grey items are packages.
- Green items are packages that are planned to be moved (i.e., allocated a robot and target position).
- Dark blue items are packages that have a deadline-to-export of <60 seconds.
- Light blue items are packages that will be exported in <10 seconds.
- Orange items are packages that have exceeded the deadline, but are planned to be moved.
- Red items are packages that have exceeded the deadline, and are not yet planned to be moved.
The supporting windows are as follows:
- Top Left window draws chargers only.
- Top Right window draws packages only.
- Bottom Left window draws robots only.
- Bottom Right window draws package planned locations only (i.e., target/planned position).