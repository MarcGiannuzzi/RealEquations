import cv2 as cv
import numpy as np
import threading


from utils import *


FRAME_SHAPE = (480, 640, 3)
MAX_TIME_BEFORE_ERASING = 15

class Blackboard:

    def __init__(self):
        self.content = 255 * np.ones(shape=FRAME_SHAPE, dtype=np.uint8)
        self.content = cv.cvtColor(self.content, cv.COLOR_BGR2GRAY)

    def erase(self):
        self.content = 255 * np.ones(shape=FRAME_SHAPE, dtype=np.uint8)
        self.content = cv.cvtColor(self.content, cv.COLOR_BGR2GRAY)
