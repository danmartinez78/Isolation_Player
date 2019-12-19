import numpy as np
import cv2
import glob
import os
import sys
import time
import functools
sys.path.append(os.path.join(os.path.dirname(__file__), './uArm-Python-SDK/'))
from uarm.wrapper import SwiftAPI

class Camera:
    def __init__(self, device = 1, calibration_file = None, position = [0,0,0]):
        self.cap = cv2.VideoCapture(device)
        self.camera_position = position
        if (calibration_file != None):
            # load calibration
            None

    def set_camera_position(self, position):
        self.camera_position = position
    
    def calibrate_camera(self, z=150):
        positions = [[200, 0, 100],
                    [200, 20, 100],
                    [200, -20, 100],
                    [120, 20, 100],
                    [150, -20, 80],
                    [200, 0, 80]]
        swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'})

        swift.waiting_ready(timeout=3)

        device_info = swift.get_device_info()
        print(device_info)
        firmware_version = device_info['firmware_version']
        if firmware_version and not firmware_version.startswith(('0.', '1.', '2.', '3.')):
            swift.set_speed_factor(1)

        swift.set_mode(0)
        swift.reset(wait=True, speed=250)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*6,3), np.float32)
        objp[:,:2] = np.mgrid[0:6,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        for position in positions:
            swift.set_position(x = position[0], y = position[1], z = position[2], speed=250)
            time.sleep(2)
            print(swift.get_position())
            ret, frame = self.cap.read()
            cv2.imshow('frame', frame)
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (6,6), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                frame = cv2.drawChessboardCorners(frame, (6,6), corners2,ret)
                cv2.imshow('frame',frame)
                cv2.waitKey(2000)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
        swift.set_position(x = 200, y = 0, z=5, speed = 200)
        swift.flush_cmd(wait_stop=True)
        swift.flush_cmd()
        time.sleep(5)
        swift.disconnect()

        # # termination criteria
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # objp = np.zeros((6*7,3), np.float32)
        # objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

        # # Arrays to store object points and image points from all the images.
        # objpoints = [] # 3d point in real world space
        # imgpoints = [] # 2d points in image plane.

        # for fname in images:
        #     frame = cv2.imread(fname)
        #     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #     # Find the chess board corners
        #     ret, corners = cv2.findChessboardCorners(gray, (6,6),None)

        #     # If found, add object points, image points (after refining them)
        #     if ret == True:
        #         objpoints.append(objp)

        #         corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        #         imgpoints.append(corners2)

        #         # Draw and display the corners
        #         frame = cv2.drawChessboardCorners(frame, (7,6), corners2,ret)
        #         cv2.imshow('frame',frame)
        #         cv2.waitKey(500)

        # cv2.destroyAllWindows()
   
   
    # localize board 
    # get board state
    # find black play piece