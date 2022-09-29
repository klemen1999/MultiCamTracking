import depthai as dai
import blobconverter
import cv2
from calibrator import Calibrator
import time

class Camera:
    label_map = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    def __init__(self, device_info: dai.DeviceInfo, friendly_id: int, show_video: bool = True):
        self.show_video = show_video
        self.device_info = device_info
        self.friendly_id = friendly_id
        self.mxid = device_info.getMxId()
        self._create_pipeline()
        self.device = dai.Device(self.pipeline, self.device_info)

        self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        self.still_queue = self.device.getOutputQueue(name="still", maxSize=1, blocking=False)
        self.control_queue = self.device.getInputQueue(name="control")
        self.mapping_queue = self.device.getOutputQueue(name="mapping", maxSize=1, blocking=False)
        self.nn_queue = self.device.getOutputQueue(name="nn", maxSize=1, blocking=False)
        self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=1, blocking=False)

        self.window_name = f"[{self.friendly_id}] Camera {self.mxid}"
        if show_video:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 720)

        self.frame_rgb = None 

        self.calibrator = Calibrator((10, 7), 0.0251, device_info)

        print("=== Connected to " + self.device_info.getMxId())

    def __del__(self):
        self.device.close()
        print("=== Closed " + self.device_info.getMxId())
    
    def _create_pipeline(self):
        pipeline = dai.Pipeline()

        # RGB cam -> 'rgb'
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        cam_rgb.setPreviewSize(300, 300)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setPreviewKeepAspectRatio(False)
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")

        # Depth cam -> 'depth'
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        cam_stereo = pipeline.create(dai.node.StereoDepth)
        cam_stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        cam_stereo.setDepthAlign(dai.CameraBoardSocket.RGB) # Align depth map to the perspective of RGB camera, on which inference is done
        cam_stereo.setOutputSize(mono_left.getResolutionWidth(), mono_left.getResolutionHeight())
        mono_left.out.link(cam_stereo.left)
        mono_right.out.link(cam_stereo.right)

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")

        # Bounding box maping from depth to RGB -> 'mapping'
        xout_bounding_box_bepth_mapping = pipeline.create(dai.node.XLinkOut)
        xout_bounding_box_bepth_mapping.setStreamName("mapping")

        # Spatial detection network -> 'nn'
        spatial_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        spatial_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
        spatial_nn.setConfidenceThreshold(0.5)
        spatial_nn.input.setBlocking(False)
        spatial_nn.setBoundingBoxScaleFactor(0.5)
        spatial_nn.setDepthLowerThreshold(100)
        spatial_nn.setDepthUpperThreshold(5000)
        xout_nn = pipeline.create(dai.node.XLinkOut)
        xout_nn.setStreamName("nn")

        cam_rgb.preview.link(spatial_nn.input)
        # cam_rgb.preview.link(xout_rgb.input)
        cam_stereo.depth.link(spatial_nn.inputDepth)
        spatial_nn.passthrough.link(xout_rgb.input)
        spatial_nn.passthroughDepth.link(xout_depth.input)
        spatial_nn.boundingBoxMapping.link(xout_bounding_box_bepth_mapping.input)
        spatial_nn.out.link(xout_nn.input)


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
        in_nn = self.nn_queue.tryGet()
        in_depth = self.depth_queue.tryGet()

        if in_rgb is None or in_depth is None:
            return
        
        depth_frame = in_depth.getFrame() # depthFrame values are in millimeters
        depth_frame_color = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depth_frame_color = cv2.equalizeHist(depth_frame_color)
        depth_frame_color = cv2.applyColorMap(depth_frame_color, cv2.COLORMAP_HOT)

        self.frame_rgb = in_rgb.getCvFrame()

        visualization = self.frame_rgb.copy()
        visualization = cv2.resize(visualization, (1280, 720), interpolation = cv2.INTER_NEAREST)

        height = visualization.shape[0]
        width  = visualization.shape[1]

        detections = []
        if in_nn is not None:
            detections = in_nn.detections

        if len(detections) > 0:
            roi_datas = self.mapping_queue.get().getConfigData()

            for roi_data, detection in zip(roi_datas, detections):
                roi = roi_data.roi
                roi = roi.denormalize(width, height)
                top_left = roi.topLeft()
                bottom_right = roi.bottomRight()
                xmin = int(top_left.x)
                ymin = int(top_left.y)
                xmax = int(bottom_right.x)
                ymax = int(bottom_right.y)

                x1 = int(detection.xmin * width)
                x2 = int(detection.xmax * width)
                y1 = int(detection.ymin * height)
                y2 = int(detection.ymax * height)

                try:
                    label = self.label_map[detection.label]
                except:
                    label = detection.label

                cv2.rectangle(visualization, (xmin, ymin), (xmax, ymax), (100, 0, 0), 2)
                cv2.rectangle(visualization, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(visualization, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(visualization, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(visualization, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(visualization, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(visualization, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)


        if self.show_video:
            cv2.imshow(self.window_name, visualization)
        

    def capture_calibration_frame(self):
        still_rgb = self.capture_still()
        if still_rgb is None:
            return

        still_gray = cv2.cvtColor(still_rgb, cv2.COLOR_BGR2GRAY)
        result = self.calibrator.add_calibration_frame(still_rgb, still_gray)
        if result is not None:
            cv2.imshow(self.window_name, result)
            cv2.waitKey()

    def capture_pose_estimation_frame(self):
        still_rgb = self.capture_still()
        if still_rgb is None:
            return

        still_gray = cv2.cvtColor(still_rgb, cv2.COLOR_BGR2GRAY)
        result = self.calibrator.compute_pose_estimation(still_rgb, still_gray)
        if result is not None:
            cv2.imshow(self.window_name, result)
            cv2.waitKey()

    def capture_still(self, show: bool = False, timeout_ms: int = 1000):
        print("capturing still")
        # Empty the queue
        self.still_queue.tryGetAll()

        # Send a capture command
        ctrl = dai.CameraControl()
        ctrl.setCaptureStill(True)
        self.control_queue.send(ctrl)

        # Wait for the still to be captured
        in_still = None
        start_time = time.time()*1000
        while in_still is None:
            print(".", end="")
            in_still = self.still_queue.tryGet()
            if time.time()*1000 - start_time > timeout_ms:
                print("did not recieve still - timeout")
                return None

        still_rgb = cv2.imdecode(in_still.getData(), cv2.IMREAD_UNCHANGED)
        if show:
            cv2.imshow(self.window_name, still_rgb)
            cv2.waitKey()

        return still_rgb