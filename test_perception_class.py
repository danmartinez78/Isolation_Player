import os
import sys
import time
import functools
sys.path.append(os.path.join(os.path.dirname(__file__), './uArm-Python-SDK/'))
from uarm.wrapper import SwiftAPI
from perception import Camera
import cv2

swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'})

swift.waiting_ready(timeout=3)

device_info = swift.get_device_info()
print(device_info)
firmware_version = device_info['firmware_version']
if firmware_version and not firmware_version.startswith(('0.', '1.', '2.', '3.')):
    swift.set_speed_factor(1)

cam = Camera(swift, device = 2, calibration_file='./calibration/camera_calibration_1.npz')
# cam.calibrate_camera()
# cam.localize_game_board()
# cam.calibrate_camera_to_workspace()
# cam.locate_game_pieces()
# cam.get_test_frames(50)
for i in range(50):
    filename = './test_images/test_image' + str(i) + '.png'
    frame = cv2.imread(filename)
    cam.locate_game_pieces(frame)
swift.set_position(x = 200, y = 0, z=5, speed = 200)
swift.flush_cmd(wait_stop=True)
swift.flush_cmd()
time.sleep(5)
swift.disconnect()
exit() 