from unittest import skip
import blobconverter
import cv2
import depthai
import numpy as np

# CALIBRATION PARAMETERS

# CHECKERBOARD = (8, 8)
CHECKERBOARD = (10, 7)
# CHECKERBOARD = (11, 8)
# CHECKERBOARD_CELL_SIZE = (0.46 / CHECKERBOARD[0], 0.46 / CHECKERBOARD[1])
CHECKERBOARD_CELL_SIZE = (0.251 / CHECKERBOARD[0], 0.176 / CHECKERBOARD[1])
# CHECKERBOARD_CELL_SIZE = (0.55 / CHECKERBOARD[0], 0.40 / CHECKERBOARD[1])
CHECKERBOARD_INNER = (CHECKERBOARD[0] - 1, CHECKERBOARD[1] - 1)

HSV_LOW = np.array([0, 0, 143])
HSV_HIGH = np.array([179, 61, 252])


corners_world = np.zeros((1, CHECKERBOARD_INNER[0] * CHECKERBOARD_INNER[1], 3), np.float32)
corners_world[0,:,:2] = np.mgrid[0:CHECKERBOARD_INNER[0], 0:CHECKERBOARD_INNER[1]].T.reshape(-1, 2)
corners_world[:,:,0] *= CHECKERBOARD_CELL_SIZE[0]
corners_world[:,:,1] *= CHECKERBOARD_CELL_SIZE[1]


cv2.namedWindow("video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("video", 1280, 720)

cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("result", 1280, 720)



def create_pipeline():
    # define the pipeline
    pipeline = depthai.Pipeline()

    cam_rgb = pipeline.create(depthai.node.ColorCamera)
    # cam_rgb.setPreviewSize(600, 600)
    cam_rgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_4_K)

    controlIn = pipeline.create(depthai.node.XLinkIn)
    controlIn.setStreamName('control')

    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    # cam_rgb.preview.link(xout_rgb.input)
    cam_rgb.video.link(xout_rgb.input)
    controlIn.out.link(cam_rgb.inputControl)

    return pipeline


def drawOrigin(frame, rot_vec, trans_vec, cam_mat, distortion):
    p_0 = cv2.projectPoints(np.float64([[0, 0, 0]]), rot_vec, trans_vec, cam_mat, distortion)[0]
    p_x = cv2.projectPoints(np.float64([[0.1, 0, 0]]), rot_vec, trans_vec, cam_mat, distortion)[0]
    p_y = cv2.projectPoints(np.float64([[0, 0.1, 0]]), rot_vec, trans_vec, cam_mat, distortion)[0]
    p_z = cv2.projectPoints(np.float64([[0, 0, -0.1]]), rot_vec, trans_vec, cam_mat, distortion)[0]

    reprojection = frame.copy()
    reprojection = cv2.line(reprojection, p_0[0][0].astype(np.int64), p_x[0][0].astype(np.int64), (0, 0, 255), 5)
    reprojection = cv2.line(reprojection, p_0[0][0].astype(np.int64), p_y[0][0].astype(np.int64), (0, 255, 0), 5)
    reprojection = cv2.line(reprojection, p_0[0][0].astype(np.int64), p_z[0][0].astype(np.int64), (255, 0, 0), 5)

    return reprojection


def choose_camera():
    devices = depthai.Device.getAllAvailableDevices()
    for i, device in enumerate(devices):
        print(f"[{i+1}] {device.getMxId()} {device.state}")

    selected_device = int(input("Select device: "))
    if selected_device > len(devices) or selected_device < 1:
        print("Invalid device")
        exit(1)

    selected_device_id = devices[selected_device - 1].getMxId()
    device_info = depthai.DeviceInfo(selected_device_id)

    return device_info

def load_camera_calibration(device_info):
    id = device_info.getMxId()
    try:
        distorsion_coef = np.load(f"calibration/distorsion_coef.{id}.npy")
        intrinsic_mat = np.load(f"calibration/intrinsic_mat.{id}.npy")
    except:
        distorsion_coef = None
        intrinsic_mat = None

    return distorsion_coef, intrinsic_mat
    
def save_camera_calibration(device_info, distorsion_coef, intrinsic_mat):
    id = device_info.getMxId()
    np.save(f"calibration/distorsion_coef.{id}.npy", distorsion_coef)
    np.save(f"calibration/intrinsic_mat.{id}.npy", intrinsic_mat)


with depthai.Device(pipeline, device_info) as device:
    q_rgb = device.getOutputQueue("rgb", maxSize=1, blocking=False)
    controlQueue = device.getInputQueue('control')

    corners_list = []
    corners_world_list = []

    while True:
        key = cv2.waitKey(1)
        if key == ord('q'): 
            break

        in_rgb = q_rgb.tryGet()

        if in_rgb is None: 
            continue

        # preprocess the frame
        frame = in_rgb.getCvFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # bw = threshold(frame, gray)

        cv2.imshow("video", gray)

        if key == ord('e'):
            ctrl = depthai.CameraControl()
            ctrl.setLumaDenoise(0)
            ctrl.setChromaDenoise(0)
            ctrl.setSharpness(4)
            ctrl.setManualExposure(33000, 200)
            controlQueue.send(ctrl)

        # CAPTURE - press `c` to capture a calibration frame
        if key == ord('c'):
            found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_INNER, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            if not found: 
                print("No checkerboard found")
                continue

            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            corners_display = cv2.drawChessboardCorners(frame, CHECKERBOARD_INNER, corners, found)
            cv2.imshow("result", corners_display)

            corners_list.append(corners)
            corners_world_list.append(corners_world)

            print(f"Calibration frame added ({len(corners_list)})")

        # POSE ESTIMATION - press `p` to estimate the pose
        if key == ord('p'):
            found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_INNER, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

            if not found: 
                print("No checkerboard found")
                continue

            # corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            corners_display = cv2.drawChessboardCorners(frame, CHECKERBOARD_INNER, corners, found)


            # solve the pose
            ret, rvec, tvec = cv2.solvePnP(corners_world, corners, intrinsic_mat, distorsion_coef, flags=cv2.SOLVEPNP_ITERATIVE)

            reprojection_img = drawOrigin(frame, rvec, tvec, intrinsic_mat, distorsion_coef)
            cv2.imshow("reprojection", reprojection_img)


        # CALIBRATE - press `s` to calibrate the camera and save the parameters
        if key == ord('s'):
            ret, intrinsic_mat, distorsion_coef, rvecs, tvecs = cv2.calibrateCamera(corners_world_list, corners_list, gray.shape[::-1], None, None)

            reprojection_img = drawOrigin(frame, rvecs[0], tvecs[0], intrinsic_mat, distorsion_coef)
            
            cv2.imshow("reprojection", reprojection_img)

            rotM = cv2.Rodrigues(rvecs[0])[0]
            camera_position = -np.matrix(rotM).T * np.matrix(tvecs[0])

            np.save("calibration/intrinsic_mat.npy", intrinsic_mat)
            np.save("calibration/distorsion_coef.npy", distorsion_coef)

            print("CALIBRATION DONE")

            print("ret: ", ret)
            print("Camera intrinsic matrix : \n", intrinsic_mat)
            print("Lens distortion coefficients : \n", distorsion_coef)
            print("Rotation vector : \n", rvecs)
            print("Translation vector : \n", tvecs)
            print("Distance from camera to checkerboard : \n", np.linalg.norm(tvecs))

            print("Camera position : \n", camera_position)


# REFERENCES
# - https://stackoverflow.com/questions/14444433/calculate-camera-world-position-with-opencv-python
# - https://stackoverflow.com/questions/66225558/cv2-findchessboardcorners-fails-to-find-corners
# - https://learnopencv.com/camera-calibration-using-opencv/
# - https://answers.opencv.org/question/83847/extract-camera-position/
# - https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
# - https://stackoverflow.com/questions/71985080/how-do-i-transform-camera-coordinate-to-world-coordinate-use-opencv
