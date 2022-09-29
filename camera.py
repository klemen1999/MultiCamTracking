import depthai as dai
import blobconverter
import cv2
from calibrator import Calibrator

class Camera:
    labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    def __init__(self, device_info: dai.DeviceInfo, friendly_id: int, show_video: bool = True):
        self.show_video = show_video
        self.device_info = device_info
        self.friendly_id = friendly_id
        self.mxid = device_info.getMxId()
        self._createPipeline()
        self.device = dai.Device(self.pipeline, self.device_info)

        self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        self.still_queue = self.device.getOutputQueue(name="still", maxSize=1, blocking=False)
        self.control_queue = self.device.getInputQueue(name="control")

        self.window_name = f"[{self.friendly_id}] Camera {self.mxid}"
        if show_video:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 720)

        self.frame_rgb = None 

        self.still_rgb = None
        self.still_gray = None

        self.calibrator = Calibrator((10, 7), 0.0251, device_info)

        print("=== Connected to " + self.device_info.getMxId())

    def __del__(self):
        self.device.close()
    
    def _createPipeline(self):
        pipeline = dai.Pipeline()

        # RGB cam -> 'rgb'
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.setPreviewSize(1280, 720)
        cam_rgb.preview.link(xout_rgb.input)

        # Still encoder -> 'still'
        still_encoder = pipeline.create(dai.node.VideoEncoder)
        still_encoder.setDefaultProfilePreset(1, dai.VideoEncoderProperties.Profile.MJPEG)
        cam_rgb.still.link(still_encoder.input)
        xout_still = pipeline.createXLinkOut()
        xout_still.setStreamName("still")
        still_encoder.bitstream.link(xout_still.input)

        # Camera control -> 'control'
        control = pipeline.create(dai.node.XLinkIn)
        control.setStreamName('control')
        control.out.link(cam_rgb.inputControl)

        self.pipeline = pipeline

    def update(self):
        in_rgb = self.rgb_queue.tryGet()

        if in_rgb is not None:
            self.frame_rgb = in_rgb.getCvFrame()

            if self.show_video:
                cv2.imshow(self.window_name, self.frame_rgb)
        

    def capture_calibration_frame(self):
        self.capture_still()
        result = self.calibrator.add_calibration_frame(self.still_rgb, self.still_gray)
        if result is not None:
            cv2.imshow(self.window_name, result)
            cv2.waitKey()

    def capture_pose_estimation_frame(self):
        self.capture_still()
        result = self.calibrator.compute_pose_estimation(self.still_rgb, self.still_gray)
        if result is not None:
            cv2.imshow(self.window_name, result)
            cv2.waitKey()

    def capture_still(self, show: bool = False):
        # Empty the queue
        self.still_queue.tryGetAll()

        # Send a capture command
        ctrl = dai.CameraControl()
        ctrl.setCaptureStill(True)
        self.control_queue.send(ctrl)

        # Wait for the still to be captured
        in_still = self.still_queue.get()
        while in_still is None:
            in_still = self.still_queue.get()

        self.still_rgb = cv2.imdecode(in_still.getData(), cv2.IMREAD_UNCHANGED)
        self.still_gray = cv2.cvtColor(self.still_rgb, cv2.COLOR_BGR2GRAY)
        if show:
            cv2.imshow(self.window_name, self.still_rgb)
            cv2.waitKey()

        return self.still_rgb, self.still_gray