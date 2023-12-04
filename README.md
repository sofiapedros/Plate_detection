# Plate detection and tracking

## Introduction
In this project we aim to create a system to detect and track the plates of the cars that enter a parking garage. The camera will read the plate (as if it were a password) and, if the password is in the database is a member of the company and it can enter without restrictions and park. The camera will also follow the car to control everything is working correctly. On the other hand, if the plate is not registred, the car will not be able to enter the garage.

## Implementation
- calibration_script.py: script used to calibrate the camera
- Detect.py: final script to detect plates and track them
- plate_detection_shapes.py: pattern recognition of a plate with shapes
- save_snaps_script.py: script to capture frames from the camera
- calibration_images: images used for calibration

## Results
- results_pattern_shape_detection: results of the plate_detection_shapes.py
    - Plate_zoomed_in_4: extracted plate
    - Corner_detection_4: corners detected on the extrated plate
    - example4: original image
- corner_calibration: corner calibration results
- Images_detection: intermediate images obtained during the detection
    - binary_image_video: binarized image of the detected plate
    - plate_image_video: interest zone detected
    - Text_recognition_example: example of the text detected display
    - output_example: example of the output
## Future developments
## Bibliography
- Brouton Lab. (s.f.).From: https://broutonlab.com/blog/opencv-object-tracking/
- CircuitDigest. (s.f.). From: https://circuitdigest.com/microcontroller-projects/license-plate-recognition-using-raspberry-pi-and-opencv
- Github. (s.f.). From: tesseract: https://github.com/tesseract-ocr/tesseract
- Github. (s.f.). From: NITC-Automatic-Number-Plate-Recognition-with-Raspberry-Pi: https://github.com/sonianuj287/NITC-Automatic-Number-Plate-Recognition-with-Raspberry-Pi
- Github. (s.f.). From: Motion Detection with RaspberryPi: https://github.com/fninsiima/motion-detection-with-raspberrypi3/blob/master/pi_surveillance.py
- Github. (s.f.). From: Detección de Figuras Geométricas: https://github.com/GabySol/OmesTutorials/blob/master/Detecci%C3%B3n%20de%20Figuras%20Geom%C3%A9tricas/figurasGeometricas.py
- ProgramaFacil. (s.f.).From: https://programarfacil.com/blog/vision-artificial/deteccion-de-movimiento-con-opencv-python/
- PyimageSearch. (s.f.). From: https://pyimagesearch.com/2020/09/21/opencv-automatic-license-number-plate-recognition-anpr-with-python/
- PyimageSearch. (s.f.). From: https://pyimagesearch.com/2020/09/21/opencv-automatic-license-number-plate-recognition-anpr-with-python/
- PyimageSearch. (s.f.). From: https://pyimagesearch.com/2016/02/08/opencv-shape-detection/
- PyShine. (s.f.). From: https://pyshine.com/Object-tracking-in-Python/
- Python geeks. (s.f.). From: Python OpenCV Detect and Recognize Car License Plate: https://pythongeeks.org/python-opencv-detect-and-recognize-car-license-plate/
Science Arena Publications . (s.f.). From: https://sciarena.com/storage/models/article/qsUEQ6r5ctgGUtrfc2jcseg7W7qxqG748SOgFni8Mp9x6JfxwZKu5u5gPvl7/comparison-of-api-trackers-in-opencv-using-raspberry-pi-hardware.pdf
Stackoverflow. (s.f.). From: https://stackoverflow.com/questions/50984205/how-to-find-corners-points-of-a-shape-in-an-image-in-opencv


