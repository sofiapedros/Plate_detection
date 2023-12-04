import cv2
import glob
import copy
import math
import numpy as np
import imageio
import matplotlib.pyplot as plt
from pprint import pprint as pp
import time
import cv2
from picamera2 import Picamera2
import os


def save_snaps(folder):

    picam = Picamera2()
    picam.preview_configuration.main.size=(3460, 1440)
    picam.preview_configuration.main.format="RGB888"
    picam.camera_usb_options="-r 2592x1944 -f10"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    # CONTADOR PARA GUARDAR LOS SNAPS
    i = 1
    while i < 3:
        frame = picam.capture_array()
        cv2.imshow("picam", frame)
        cv2.imwrite(f"car2_image{i}.jpg",frame)
        time.sleep(3)
    
        i += 1
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    save_snaps("prueba_1")