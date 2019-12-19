from perception import Camera
import os
import sys
import time
import functools
sys.path.append(os.path.join(os.path.dirname(__file__), './uArm-Python-SDK/'))
from uarm.wrapper import SwiftAPI

swift = SwiftAPI(filters={'hwid': 'USB VID:PID=2341:0042'})

swift.waiting_ready(timeout=3)

device_info = swift.get_device_info()
print(device_info)
firmware_version = device_info['firmware_version']
if firmware_version and not firmware_version.startswith(('0.', '1.', '2.', '3.')):
    swift.set_speed_factor(1)

cam = Camera(swift, device = 1)
cam.calibrate_fisheye_camera()
swift.set_position(x = 200, y = 0, z=5, speed = 200)
swift.flush_cmd(wait_stop=True)
swift.flush_cmd()
time.sleep(5)
swift.disconnect()
exit() 