import numpy as np
import cv2
import glob
import time


class Camera:
    def __init__(self, arm, device=1, calibration_file=None, position=[0, 0, 0]):
        self.cap = cv2.VideoCapture(device)
        self.camera_position = position
        if (calibration_file != None):
            # load calibration
            None
        self.arm = arm

    def set_camera_position(self, position):
        self.camera_position = position

    def calibrate_camera(self):
        self.arm.set_mode(0)
        self.arm.reset(wait=True, speed=250)
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*6, 3), np.float32)
        objp[:, :2] = np.mgrid[0:6, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        positions = [[250, 0, 170],
                    [250, 75, 170],
                    [250, -75, 170],
                    [275, 0, 125],
                    [275, 75, 125],
                    [275, -75, 125],
                    [250, 0, 125],
                    [250, 100, 125],
                    [250, -100, 125],
                    [200, 100, 100],
                    [200, -100, 100],
                    [150, 25, 125],
                    [150, -25, 125],
                    [175, 0, 125],
                    [170, 50, 125],
                    [170, -50, 125],
                    [130, 0, 100],
                    [130, 75, 100],
                    [130, -75, 100]]
        cv2.namedWindow("frame")
        for position in positions:
            self.arm.set_position(
                x=position[0], y=position[1], z=position[2], speed=200)
            print(self.arm.get_position())
            for i in range(30):
                ret, frame = self.cap.read()
                cv2.imshow('frame', frame)
                cv2.waitKey(50)
            cv2.imshow('frame', frame)
            cv2.waitKey(1000)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (6, 6), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                frame = cv2.drawChessboardCorners(frame, (6, 6), corners2, ret)
                cv2.imshow('frame', frame)
                cv2.waitKey(1000)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        h,  w = gray.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (w, h), 1, (w, h))
        # undistort
        self.arm.set_position(x=250, y=0, z=170, speed=250, wait=True)
        for i in range(10):
            ret, frame = self.cap.read()
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
        cv2.destroyAllWindows()


    def calibrate_fisheye_camera(self):
        self.arm.set_mode(0)
        self.arm.reset(wait=True, speed=250)
        CHECKERBOARD = (6,6)
        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW
        objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        _img_shape = None
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        positions = [[250, 0, 170],
                    [250, 75, 170],
                    [250, -75, 170],
                    [275, 0, 125],
                    [275, 75, 125],
                    [275, -75, 125],
                    [250, 0, 125],
                    [250, 50, 125],
                    [250, -50, 125],
                    [200, 50, 100],
                    [200, -50, 100],
                    [175, 0, 125],
                    [170, 25, 125],
                    [170, -25, 125]]
        cv2.namedWindow("frame")

        for position in positions:
            self.arm.set_position(x = position[0], y = position[1], z = position[2], speed=200)
            print(self.arm.get_position())
            for i in range(30):
                ret, frame = self.cap.read()
                cv2.imshow('frame', frame)
                cv2.waitKey(50)
            cv2.imshow('frame', frame)
            cv2.waitKey(1000)
            _img_shape = frame.shape[:2]
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (6, 6), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                frame = cv2.drawChessboardCorners(frame, (6,6), corners,ret)
                cv2.imshow('frame',frame)
                cv2.waitKey(1000)
        
        N_OK = len(objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
        DIM = _img_shape
        h,  w = gray.shape[:2]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
        # undistort
        self.arm.set_position(x=250, y=0, z=170, speed=250, wait=True)
        for i in range(10):
            ret, frame = self.cap.read()
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        # crop the image
        dst = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cv2.imshow('calib1', dst)
        
    def calibrate_fisheye(self, all_image_points, all_true_points, image_size):
        """ Calibrate a fisheye camera from matching points.
        :param all_image_points: Sequence[Array(N, 2)[float32]] of (x, y) image coordinates of the points.  (see  cv2.findChessboardCorners)
        :param all_true_points: Sequence[Array(N, 3)[float32]] of (x,y,z) points.  (If from a grid, just put (x,y) on a regular grid and z=0)
            Note that each of these sets of points can be in its own reference frame,
        :param image_size: The (size_y, size_x) of the image.
        :return: (rms, mtx, dist, rvecs, tvecs) where
            rms: float - The root-mean-squared error
            mtx: array[3x3] A 3x3 camera intrinsics matrix
            dst: array[4x1] A (4x1) array of distortion coefficients
            rvecs: Sequence[array[N,3,1]] of estimated rotation vectors for each set of true points
            tvecs: Sequence[array[N,3,1]] of estimated translation vectors for each set of true points
        """
        assert len(all_true_points) == len(all_image_points)
        all_true_points = list(all_true_points)  # Because we'll modify it in place
        all_image_points = list(all_image_points)
        while True:
            assert len(all_true_points) > 0, "There are no valid images from which to calibrate."
            try:
                rms, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
                    objectPoints=[p[None, :, :] for p in all_true_points],
                    imagePoints=[p[:, None, :] for p in all_image_points],
                    image_size=image_size,
                    K=np.zeros((3, 3)),
                    D=np.zeros((4, 1)),
                    flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW,
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                print('Found a calibration based on {} well-conditioned images.'.format(len(all_true_points)))
                return rms, mtx, dist, rvecs, tvecs
            except cv2.error as err:
                try:
                    idx = int(err.msg.split('array ')[1][0])  # Parse index of invalid image from error message
                    all_true_points.pop(idx)
                    all_image_points.pop(idx)
                    print("Removed ill-conditioned image {} from the data.  Trying again...".format(idx))
                except IndexError:
                    raise err

   
   
    # localize board 
    # get board state
    # find black play piece
