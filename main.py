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

    # QUIT - press `q` to quit
    if key == ord('q'):
        break
    
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

    # TOGGLE DEPTH VIEW - press `d` to toggle depth view
    if key == ord('d'):
        selected_camera.show_detph = not selected_camera.show_detph

    # SHOW OTHER CAMERAS - press `o` to show the other cameras
    if key == ord('o'):
        still_rgb = selected_camera.capture_still(show=False)
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

            if camera.calibrator.cam_to_world is None:
                continue

            camera_position = (camera.calibrator.cam_to_world @ np.array([[0,0,0,1]]).T)[:3]

            print(camera_position)
            p = cv2.projectPoints(
                camera_position, calibrator.rot_vec, calibrator.trans_vec, 
                calibrator.intrinsic_mat, calibrator.distortion_coef
            )[0]

            still_rgb = cv2.drawMarker(still_rgb, p[0][0].astype(np.int64), (255, 0, 255), cv2.MARKER_CROSS, 50, 4)

        cv2.imshow(selected_camera.window_name, still_rgb)
        cv2.waitKey()

    # BIRDSEYE VIEW
    width = 512
    height = 512
    scale = 100 # 1m = 100px
    colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0,0,255), (0,255,0), (255,0,0)]
    birds_eye_view = np.zeros([width,height,3], dtype=np.uint8)

    world_to_birds_eye = np.array([
        [scale, 0, 0, width//2],
        [0, scale, 0, height//2],
    ])

    # draw the coordinate system
    p_0 = (world_to_birds_eye @ np.array([0, 0, 0, 1])).astype(np.int64)
    p_x = (world_to_birds_eye @ np.array([0.3, 0, 0, 1])).astype(np.int64)
    p_y = (world_to_birds_eye @ np.array([0, 0.3, 0, 1])).astype(np.int64)
    cv2.line(birds_eye_view, p_0, p_x, (0, 0, 255), 2)
    cv2.line(birds_eye_view, p_0, p_y, (0, 255, 0), 2)

    # make groups
    n = 2 # use only first n components
    distance_threshold = 1.5 # m
    for camera in cameras:
        for det in camera.detected_objects:
            det.corresponding_detections = []
            for other_camera in cameras:
                # find closest detection
                d = np.inf
                closest_det = None
                for other_det in other_camera.detected_objects:
                    if other_det.label != det.label:
                        continue
                    d_ = np.linalg.norm(det.pos[:,:n] - other_det.pos[:,:n])
                    if d_ < d:
                        d = d_
                        closest_det = other_det
                if closest_det is not None and d < distance_threshold:
                    det.corresponding_detections.append(closest_det)
    # keep only double correspondences
    for camera in cameras:
        for det in camera.detected_objects:
            det.corresponding_detections = [other_det for other_det in det.corresponding_detections if det in other_det.corresponding_detections]
    # find groups of correspondences
    groups = []
    for camera in cameras:
        for det in camera.detected_objects:
            # find group
            group = None
            for g in groups:
                if det in g:
                    group = g
                    break
            if group is None:
                group = set()
                groups.append(group)
            # add to group
            group.add(det)
            for other_det in det.corresponding_detections:
                if other_det not in group:
                    group.add(other_det)


    for camera in cameras:
        camera.update()

        try:
            color = colors[camera.friendly_id - 1]
        except:
            color = (255,255,255)

        # draw the camera position
        if camera.calibrator.position is not None:
            p = (world_to_birds_eye @ (camera.calibrator.cam_to_world @ np.array([0,0,0,1]))).astype(np.int64)
            p_l = (world_to_birds_eye @ (camera.calibrator.cam_to_world @ np.array([0.2,0,0.1,1]))).astype(np.int64)
            p_r = (world_to_birds_eye @ (camera.calibrator.cam_to_world @ np.array([-0.2,0,0.1,1]))).astype(np.int64)
            cv2.circle(birds_eye_view, p, 5, color, -1)
            cv2.line(birds_eye_view, p, p_l, color, 1)
            cv2.line(birds_eye_view, p, p_r, color, 1)

    # draw the groups
    for group in groups:
        avg = np.zeros(2)
        label = ""
        for det in group:
            label = det.label
            try: c = colors[det.camera_friendly_id - 1]
            except: c = (255,255,255)
            p = (world_to_birds_eye @ det.pos).flatten().astype(np.int64)
            avg += p
            cv2.circle(birds_eye_view, p, 2, c, -1)

        avg = (avg/len(group)).astype(np.int64)
        cv2.circle(birds_eye_view, avg, int(0.25*scale), (255, 255, 255), 0)
        cv2.putText(birds_eye_view, str(label), avg+np.array([0, int(0.25*scale) + 10]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Bird's Eye View", birds_eye_view)
        