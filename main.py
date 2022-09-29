import cv2
import depthai as dai
from camera import Camera
from typing import List
import numpy as np

device_infos = dai.Device.getAllAvailableDevices()
if len(device_infos) == 0:
    raise RuntimeError("No devices found!")
else:
    print("Found", len(device_infos), "devices")

device_infos.sort(key=lambda x: x.getMxId(), reverse=True) # sort the cameras by their mxId

cameras: List[Camera] = []

friendly_id = 0
for device_info in device_infos:
    friendly_id += 1
    cameras.append(Camera(device_info, friendly_id, show_video=True))

selected_camera = cameras[0]

def select_camera(friendly_id: int):
    global selected_camera

    i = friendly_id - 1
    if i >= len(cameras) or i < 0: 
        return None

    selected_camera = cameras[i]
    print(f"Selected camera {friendly_id}")

    return selected_camera

select_camera(1)

while True:
    key = cv2.waitKey(1)
    
    # CAMERA SELECTION - use the number keys to select a camera
    if key >= ord('1') and key <= ord('9'):
        select_camera(key - ord('1') + 1)

    # CAPTURE CALIBRATION FRAME - press `c` to capture a calibration frame
    if key == ord('c'):
        selected_camera.capture_calibration_frame()

    # SAVE CALIBRATION - press `s` to calibrate the selected camera and save the calibration
    if key == ord('s'):
        selected_camera.calibrator.compute_calibration()
        selected_camera.calibrator.save_calibration_to_file()

    # POSE ESTIMATION - press `p` to start pose estimation
    if key == ord('p'):
        selected_camera.capture_pose_estimation_frame()

    # CAPTURE STILL IMAGE - press `i` to capture a still image
    if key == ord('t'):
        selected_camera.capture_still(show=True)

    # SHOW OTHER CAMERAS - press `o` to show the other cameras
    if key == ord('o'):
        still_rgb, still_gray = selected_camera.capture_still(show=False)
        calibrator = selected_camera.calibrator
        p_0 = cv2.projectPoints(
            np.float64([[0, 0, 0]]), calibrator.rot_vec, calibrator.trans_vec, 
            calibrator.intrinsic_mat, calibrator.distortion_coef
        )[0]
        still_rgb = cv2.drawMarker(still_rgb, p_0[0][0].astype(np.int64), (0, 255, 0), cv2.MARKER_CROSS, 50, 4)

        # draw the other cameras
        for camera in cameras:
            if camera is selected_camera:
                continue

            if camera.calibrator.rot_vec is None or camera.calibrator.trans_vec is None:
                continue

            rotM = cv2.Rodrigues(camera.calibrator.rot_vec)[0]
            camera_position = -np.matrix(rotM).T * np.matrix(camera.calibrator.trans_vec)

            print(camera_position)
            p = cv2.projectPoints(
                camera_position, calibrator.rot_vec, calibrator.trans_vec, 
                calibrator.intrinsic_mat, calibrator.distortion_coef
            )[0]

            still_rgb = cv2.drawMarker(still_rgb, p[0][0].astype(np.int64), (255, 0, 255), cv2.MARKER_CROSS, 50, 4)

        cv2.imshow(selected_camera.window_name, still_rgb)
        cv2.waitKey()

    # QUIT - press `q` to quit
    if key == ord('q'):
        break

    for camera in cameras:
        camera.update()
        