'''
Global parameters for the project
'''

# GPIO pins
OUTPUT_PIN = 6

# Controller State
START = "START" # Initial state
RUNNING = "RUNNING" # Conveyor belts running and deepstream processing
YOLOV9_DETECTING = "YOLOV9_DETECTING" # YOLOv9 detecting objects
SORTING = "SORTING" # Sorting objects
STOPPED = "STOPPED" # Conveyor belts stopped
