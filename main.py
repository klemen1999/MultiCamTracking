import cv2
import depthai as dai
from camera import Camera
from typing import List

device_infos = dai.Device.getAllAvailableDevices()
if len(device_infos) == 0:
    raise RuntimeError("No devices found!")
else:
    print("Found", len(device_infos), "devices")

cameras: List[Camera] = []

friendly_id = 0
for device_info in device_infos:
    friendly_id += 1
    cameras.append(Camera(device_info, friendly_id, show_video=True))

selected_camera = None

while True:
    key = cv2.waitKey(1)
    
    # CAMERA SELECTION - use the number keys to select a camera
    if key >= ord('1') and key <= ord('9'):
        i = key - ord('1')
        if i >= len(cameras): continue

        selected_camera = cameras[i]
        print(f"Selected camera {selected_camera.mxid} [{selected_camera.friendly_id}]")

    # CAPTURE CALIBRATION FRAME - press `c` to capture a calibration frame
    if key == ord('c'):
        if selected_camera is None:
            print("No camera selected")
            continue

        selected_camera.capture_calibration_frame()

    # SAVE CALIBRATION - press `s` to calibrate the selected camera and save the calibration
    if key == ord('s'):
        if selected_camera is None:
            print("No camera selected")
            continue

        selected_camera.calibrator.compute_calibration()
        selected_camera.calibrator.save_calibration_to_file()

    # POSE ESTIMATION - press `p` to start pose estimation
    if key == ord('p'):
        if selected_camera is None:
            print("No camera selected")
            continue

        selected_camera.capture_pose_estimation_frame()

    # CAPTURE STILL IMAGE - press `i` to capture a still image
    if key == ord('t'):
        if selected_camera is None:
            print("No camera selected")
            continue

        selected_camera.capture_still(show=True)

    # QUIT - press `q` to quit
    if key == ord('q'):
        break

    for camera in cameras:
        camera.update()


        