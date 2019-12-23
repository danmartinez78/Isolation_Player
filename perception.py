import numpy as np
import cv2
import glob
import time
from cv2 import aruco
import copy
import random
import os.path
import operator

class Camera:
    def __init__(self, arm, device=1, calibration_file=None, position=[0, 0, 0], resolution=[10000, 10000], offset=[0, 0, 0]):
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.camera_position = position
        self.camera_pts = []
        self.ws_pts = []
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_50)
        if (calibration_file != None):
            # load calibration
            extension = os.path.splitext(calibration_file)[1]
            print(extension)
            if extension == '.npz':
                npzfile = np.load(calibration_file)
                self.ret = npzfile['ret']
                self.mtx = npzfile['mtx']
                self.dist = npzfile['dist']
                rvecs = npzfile['rvecs']
                tvecs = npzfile['tvecs']
                h = npzfile['h']
                w = npzfile['w']
                self.camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
                    self.mtx, self.dist, (w, h), 1, (w, h))
                self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                    self.mtx, self.dist, None, self.camera_matrix, (w, h), 5)
        self.arm = arm

    def get_frame(self, display_window, buffer_size):
        for i in range(buffer_size):
            ret, frame = self.cap.read()
            frame = self.crop(frame)
            cv2.imshow(display_window, frame)
            cv2.waitKey(10)
        frame = self.equalize(frame)
        return frame

    def get_rectified_frame(self, display_window, buffer_size, method=1):
        for i in range(buffer_size):
            ret, frame = self.cap.read()
            frame = self.crop(frame)
            if method == 1:
                frame = self.undistort_1(frame)
            else:
                frame = self.undistort_2(frame)
            cv2.imshow(display_window, frame)
            cv2.waitKey(50)
        return frame

    def undistort_1(self, frame):
        dst = cv2.undistort(frame.copy(), self.mtx,
                            self.dist, None, self.camera_matrix)
        x, y, w, h = self.roi
        dst = dst[y:y+h, x:x+w]
        return dst

    def undistort_2(self, frame):
        dst = cv2.remap(frame.copy(), self.mapx, self.mapy, cv2.INTER_LINEAR)
        x, y, w, h = self.roi
        dst = dst[y:y+h, x:x+w]
        return dst

    def crop(self, image, size=[400, 400]):
        cropped_image = copy.deepcopy(image)
        y, x, c = image.shape
        xc = int(x/2)
        yc = int(y/2)
        return cropped_image[yc-size[0]:yc+size[0], xc-size[1]:xc+size[1]]

    def equalize(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output

    def set_camera_position(self, position):
        self.camera_position = position

    def adjust_gamma(self, image, gamma=1.2):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    def calibrate_camera(self):
        self.arm.set_mode(0)
        self.arm.reset(wait=True, speed=250)
        CHECKERBOARD = (6, 6)
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*6, 3), np.float32)
        objp[:, :2] = np.mgrid[0:6, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = []
        cv2.namedWindow("frame")
        self.go_to_obs_pos()
        while len(images) < 20:
            frame = self.get_frame("frame", 10)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            # If found, add image to image list
            if ret == True:
                images.append(frame)
                frame_w_corners = copy.deepcopy(frame)
                corners2 = cv2.cornerSubPix(
                    gray, corners, (3, 3), (-1, -1), criteria)
                frame_w_corners = cv2.drawChessboardCorners(
                    frame_w_corners, CHECKERBOARD, corners2, ret)
                cv2.imshow('frame', frame_w_corners)
                cv2.waitKey(50)
        self.go_to_safe_pos()

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            if ret == True:
                corners2 = cv2.cornerSubPix(
                    gray, corners, (3, 3), (-1, -1), criteria)
                img_w_corners = copy.deepcopy(img)
                img_w_corners = cv2.drawChessboardCorners(
                    img_w_corners, CHECKERBOARD, corners2, ret)
                cv2.imshow('frame', img_w_corners)
                objpoints.append(objp)
                imgpoints.append(corners2)
                cv2.waitKey(50)

        self.go_to_safe_pos()
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        h,  w = gray.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (w, h), 1, (w, h))
        # undistort
        self.go_to_obs_pos()
        frame = self.get_frame("frame", 10)
        self.go_to_safe_pos()
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imshow('calib1', dst)

        # undistort
        mapx, mapy = cv2.initUndistortRectifyMap(
            mtx, dist, None, newcameramtx, (w, h), 5)
        dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imshow('calib2', dst)
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2,
                             cv2.NORM_L2)/len(imgpoints2)
            mean_error += error

        print("total error: ", mean_error/len(objpoints))
        cv2.waitKey(0)
        write = input('write calibration to file? (y/n)')
        if write == 'y':
            np.savez('.src/calibration/camera_calibration', h=h, w=w,
                     ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        cv2.destroyAllWindows()

    def go_to_obs_pos(self, set_speed=250):
        self.arm.set_position(x=175, y=0, z=150, speed=set_speed)

    def go_to_safe_pos(self, set_speed=50):
        self.arm.set_position(x=175, y=0, z=4, speed=set_speed)

    def go_to(self, x, y, z, set_speed=250):
        self.arm.set_position(x, y, z, speed=set_speed)

    def localize_game_board(self):
        self.go_to_obs_pos()
        cv2.namedWindow('frame')
        cv2.namedWindow('aruco')
        while True:
            frame = self.get_rectified_frame('frame', 10, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            parameters = aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(
                gray, self.aruco_dict, parameters=parameters)
            if ids is not None:
                frame_markers = aruco.drawDetectedMarkers(
                    frame.copy(), corners, ids)
                xc = 0
                yc = 0
                for x, y in corners[0][0]:
                    xc += x
                    yc += y
                xc = int(xc/4)
                yc = int(yc/4)
                # print('center:', xc, yc)
                cv2.imshow('aruco', frame_markers)

    def calibrate_camera_to_workspace(self):
        positions = [[200, 0],
                     [200, 50],
                     [200, -50],
                     [300, 0],
                     [300, 100],
                     [300, -100],
                     [100, 75],
                     [100, -75]]
        for x, y in positions:
            # move to position
            self.go_to(x, y, 3, 100)
            # wait until aruco marker is aligned (user input)
            input('continue?')
            # move to observe point
            self.go_to_obs_pos()
            # detect aruco
            pts = []
            cv2.namedWindow('frame')
            while len(pts) < 10:
                frame = self.get_rectified_frame('frame', 10, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                parameters = aruco.DetectorParameters_create()
                corners, ids, rejectedImgPoints = aruco.detectMarkers(
                    gray, self.aruco_dict, parameters=parameters)
                if ids is not None:
                    frame_markers = aruco.drawDetectedMarkers(
                        frame.copy(), corners, ids)
                    cv2.imshow('frame', frame_markers)
                    cv2.waitKey(100)
                    xc = 0
                    yc = 0
                    for x, y in corners[0][0]:
                        xc += x
                        yc += y
                    xc = int(xc/4)
                    yc = int(yc/4)
                    pts.append([xc, yc])
            x_bar = 0
            y_bar = 0
            for xp, yp in pts:
                x_bar += xp
                y_bar += yp
            x_bar = int(x_bar/10)
            y_bar = int(y_bar/10)
            self.camera_pts.append([y_bar, x_bar])
            self.ws_pts.append([y, x])

            # record aruco coords in camera frame

    def get_test_frames(self, number_of_frames=10):
        self.go_to_obs_pos()
        test_images = []
        cv2.namedWindow('test frames')
        while len(test_images) < 50:
            frame = self.get_rectified_frame('test frames', 25)
            cv2.imshow('test frames', frame)
            cv2.waitKey(0)
            save_image = input('save image? (y/n)')
            if save_image:
                test_images.append(frame.copy())
        for i in range(len(test_images)):
            fn = './test_images/test_image' + str(i) + '.png'
            cv2.imwrite(fn, test_images[i],)

    def locate_game_pieces(self):
        cv2.namedWindow('gray')
        cv2.namedWindow('circles')
        cv2.namedWindow ('detections')
        cv2.namedWindow('edges')
        self.go_to_obs_pos()
        while True:
            frame = self.get_rectified_frame('detections', 5)
            # cv2.waitKey(0)
            # b = input('enter b value')
            # g = input('enter g value')
            # r = input('enter r value')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = median = cv2.medianBlur(gray,5)
            cv2.imshow('gray', gray)
            edges = cv2.Canny(gray, 10, 150)
            cv2.imshow('edges', edges)
            circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT,1,20,param1=30,param2=25, minRadius = 20, maxRadius = 45)
            if circles is not None:
                # convert the (x, y) coordinates and radius of the circles to integers
                circles = np.round(circles[0, :]).astype("int")
            
                # loop over the (x, y) coordinates and radius of the circles
                output = frame.copy()
                for (x, y, r) in circles:
                    # draw the circle in the output image, then draw a rectangle
                    # corresponding to the center of the circle
                    cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                    cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            
                # show the output image
                cv2.imshow("circles", np.hstack([frame, output]))
                cv2.waitKey(10)
    # get board state
    # find black play piece
