# Plate detection and tracking

## Introduction
In this project we aim to create a system to detect and track the plates of the cars that enter a parking garage. The camera will read the plate (as if it were a password) and, if the password is in the database is a member of the company and it can enter without restrictions and park. The camera will also follow the car to control everything is working correctly. On the other hand, if the plate is not registred, the car will not be able to enter the garage.

## Implementation
- calibration_script.py: script used to calibrate the camera
- Detect.py: final script to detect plates and track them
- plate_detection_shapes.py: pattern recognition of a plate with shapes
- save_snaps_script.py: script to capture frames from the camera

## Results
- results_pattern_shape_detection: results of the plate_detection_shapes.py
    - Plate_zoomed_in_4: extracted plate
    - Corner_detection_4: corners detected on the extrated plate
    - example4: original image
## Future developments
## Bibliography

