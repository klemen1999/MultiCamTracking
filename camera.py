import depthai as dai
import blobconverter
import cv2
import json
import time
import numpy as np
from typing import List
# from depthai_sdk.fps import FPSHandler 

from calibration import Calibration
from detection import Detection
from multi_msg_sync import TwoStageHostSeqSync

class Camera:
    label_map = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


    def __init__(self, device_info: dai.DeviceInfo, friendly_id: int, show_video: bool = True):
        self.show_video = show_video
        self.show_detph = False
        self.device_info = device_info
        self.friendly_id = friendly_id
        self.mxid = device_info.getMxId()
        self._create_pipeline()
        self.device = dai.Device(self.pipeline, self.device_info)

        self.rgb_queue = self.device.getOutputQueue(name="color", maxSize=1, blocking=False)
        self.still_queue = self.device.getOutputQueue(name="still", maxSize=1, blocking=False)
        self.control_queue = self.device.getInputQueue(name="control")
        self.mapping_queue = self.device.getOutputQueue(name="mapping", maxSize=1, blocking=False)
        self.nn_queue = self.device.getOutputQueue(name="detection", maxSize=1, blocking=False)
        self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=1, blocking=False)
        # self.emb_queue = self.device.getOutputQueue(name="embedding", maxSize=1, blocking=False)

        self.window_name = f"[{self.friendly_id}] Camera - mxid: {self.mxid}"
        self.viz_height, self.viz_width = 360, 640
        if show_video:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.viz_width, self.viz_height)

        self.frame_color = None
        self.frame_depth = None
        self.detected_objects: List[Detection] = []

        self.calibration = Calibration((10, 7), 0.0251, self.device)
        # self.sync = TwoStageHostSeqSync()
        # self.fps_handler = FPSHandler()

        print("=== Connected to " + self.device_info.getMxId())

    def __del__(self):
        self.device.close()
        print("=== Closed " + self.device_info.getMxId())
    
    def _create_pipeline(self):
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

        # RGB cam -> 'color'
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K) # use THE_4_K for calibration
        # cam_rgb.setPreviewSize(640, 640)
        cam_rgb.setPreviewSize(300, 300)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setPreviewKeepAspectRatio(False)
        cam_rgb.setPreviewNumFramesPool(30)
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("color")

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
        
        # Spatial detection network -> 'detection'
        # with open("models/yolov6n.json", "r") as f:
        #     data = json.load(f)
        #     config = data["nn_config"]["NN_specific_metadata"]
        #     self.label_map = data["mappings"]["labels"]

        # spatial_nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        # spatial_nn.setBlobPath("models/yolov6n.blob")
        # spatial_nn.setConfidenceThreshold(config.get("confidence_threshold", {}))
        # spatial_nn.setNumClasses(config.get("classes", {}))
        # spatial_nn.setCoordinateSize(config.get("coordinates", {}))
        # spatial_nn.setIouThreshold(config.get("iou_threshold", {}))
        # spatial_nn.setAnchors(config.get("anchors", {}))
        # spatial_nn.setAnchorMasks(config.get("anchor_masks", {}))
        # spatial_nn.input.setBlocking(False)
        # spatial_nn.setBoundingBoxScaleFactor(0.2)
        # spatial_nn.setDepthLowerThreshold(100) # Min 10 centimeters
        # spatial_nn.setDepthUpperThreshold(5000) # Max 5 meters
        spatial_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        spatial_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
        spatial_nn.setConfidenceThreshold(0.6)
        spatial_nn.input.setBlocking(False)
        spatial_nn.setBoundingBoxScaleFactor(0.2)
        spatial_nn.setDepthLowerThreshold(100)
        spatial_nn.setDepthUpperThreshold(5000)
        # xout_nn = pipeline.create(dai.node.XLinkOut)
        # xout_nn.setStreamName("nn")

        xout_nn = pipeline.create(dai.node.XLinkOut)
        xout_nn.setStreamName("detection")

        cam_rgb.preview.link(spatial_nn.input)
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
        #queues = [self.rgb_queue, self.nn_queue, self.depth_queue, self.emb_queue]
        queues = [self.rgb_queue, self.nn_queue, self.depth_queue]
        msgs = {}
        for q in queues:
            data = q.tryGet()
            if data:
                msgs[q.getName()] = data
                # self.sync.add_msg(data, q.getName())

        # msgs = self.sync.get_msgs()
        # if msgs == None:
        #     self.frame_color = None
        #     self.frame_depth = None
        #     # self.fps_handler.nextIter()
        #     return
        if len(msgs) != 3:
            return

        self.frame_color = msgs["color"]
        self.frame_depth = msgs["depth"]
        
        detections = msgs["detection"].detections
        embeddings = self.get_embeddings(self.frame_color.getCvFrame(), detections)
        #embeddings = [np.ones(256) for _ in detections] #msgs["embedding"]

        self.mapping = self.mapping_queue.tryGet()
        self.detected_objects = []

        if len(detections) > 0 and self.mapping is not None:
            for detection, embedding in zip(detections, embeddings):
                # embedding = np.array(embedding.getFirstLayerFp16())
                if np.all((embedding==0)):
                    embedding[0] = 1
                try:
                    label = self.label_map[detection.label]
                except:
                    label = detection.label

                if self.calibration.cam_to_world is not None:
                    pos_camera_frame = np.array([[detection.spatialCoordinates.x / 1000, -detection.spatialCoordinates.y / 1000, detection.spatialCoordinates.z / 1000, 1]]).T
                    pos_world_frame = self.calibration.cam_to_world @ pos_camera_frame

                    self.detected_objects.append(
                        Detection(
                            bbox = np.array([detection.xmin, detection.ymin, detection.xmax, detection.ymax]),
                            confidence = detection.confidence,
                            label = label,
                            pos = pos_world_frame,
                            embedding = embedding,
                            spatial_coords = np.array([detection.spatialCoordinates.x, detection.spatialCoordinates.y, detection.spatialCoordinates.z]),
                            camera_friendly_id = self.friendly_id
                        )
                    )
        # self.fps_handler.nextIter()

    def render_tracks(self, tracks):
        if self.show_detph:
            if not self.frame_depth:
                return
            depth_frame = self.frame_depth.getFrame() # depthFrame values are in millimeters
            depth_frame_color = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depth_frame_color = cv2.equalizeHist(depth_frame_color)
            depth_frame_color = cv2.applyColorMap(depth_frame_color, cv2.COLORMAP_HOT)
            visualization = depth_frame_color.copy()
        else:
            if not self.frame_color:
                return
            visualization = self.frame_color.getCvFrame().copy()

        visualization = cv2.resize(visualization, (self.viz_width, self.viz_height), interpolation = cv2.INTER_NEAREST)

        if len(tracks) > 0 and self.mapping is not None:
            roi_datas = self.mapping.getConfigData()
            for roi_data, det in zip(roi_datas, tracks):
                roi = roi_data.roi
                roi = roi.denormalize(self.viz_width, self.viz_height)
                top_left = roi.topLeft()
                bottom_right = roi.bottomRight()
                xmin = int(top_left.x)
                ymin = int(top_left.y)
                xmax = int(bottom_right.x)
                ymax = int(bottom_right.y)

                x1 = int(det["bbox"][0] * self.viz_width)
                x2 = int(det["bbox"][2] * self.viz_width)
                y1 = int(det["bbox"][1] * self.viz_height)
                y2 = int(det["bbox"][3] * self.viz_height)

                cv2.rectangle(visualization, (xmin, ymin), (xmax, ymax), (100, 0, 0), 2)
                cv2.rectangle(visualization, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(visualization, str(det["label"])+f"_{det['object_id']}", (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(visualization, "{:.2f}".format(det["confidence"]*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(visualization, f"X: {int(det['spatial_coords'][0])} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(visualization, f"Y: {int(det['spatial_coords'][1])} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(visualization, f"Z: {int(det['spatial_coords'][2])} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        # fps = self.fps_handler.fps()
        # cv2.putText(visualization, f"Fps: {fps}", (5,15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        if self.show_video:
            cv2.imshow(self.window_name, visualization)

    def capture_pose_estimation_frame(self):
        still_rgb = self.capture_still()
        if still_rgb is None:
            return

        still_gray = cv2.cvtColor(still_rgb, cv2.COLOR_BGR2GRAY)
        result = self.calibration.compute_pose_estimation(still_rgb, still_gray)
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
            # print(".", end="")
            time.sleep(0.1)
            in_still = self.still_queue.tryGet()
            if time.time()*1000 - start_time > timeout_ms:
                print("did not recieve still image - retrying")
                return self.capture_still(show, timeout_ms)

        still_rgb = cv2.imdecode(in_still.getData(), cv2.IMREAD_UNCHANGED)
        if show:
            cv2.imshow(self.window_name, still_rgb)
            cv2.waitKey()

        return still_rgb

    def get_embeddings(self, frame, detections):
        embeddings = []
        W, H = frame.shape[:2]
        for det in detections:
            x1 = int(det.xmin * W)
            x2 = int(det.xmax * W)
            y1 = int(det.ymin * H)
            y2 = int(det.ymax * H)
            # print(x1, x2, y1, y2)
            box = frame[x1:x2, y1:y2, :]
            embedding = []
            for i in range(3):
                hist,bins = np.histogram(box[:,:,i], bins=range(0,256))
                embedding.extend(hist)

            embeddings.append(np.array(embedding))
        return embeddings