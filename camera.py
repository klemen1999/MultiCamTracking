import depthai as dai
import blobconverter
import cv2
import json
from calibration import Calibration
import time
import numpy as np
from typing import List
from detection import Detection
from MultiMsgSync import TwoStageHostSeqSync


class Camera:

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
        self.rec_queue = self.device.getOutputQueue(name="recognition", maxSize=1, blocking=False)

        self.window_name = f"[{self.friendly_id}] Camera - mxid: {self.mxid}"
        self.viz_height, self.viz_width = 360, 640
        if show_video:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.viz_width, self.viz_height)

        self.frame_color = None
        self.frame_depth = None
        self.detected_objects: List[Detection] = []

        self.calibration = Calibration((10, 7), 0.0251, self.device)
        self.sync = TwoStageHostSeqSync()

        print("=== Connected to " + self.device_info.getMxId())

    def __del__(self):
        self.device.close()
        print("=== Closed " + self.device_info.getMxId())
    
    def _create_pipeline(self):
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

        # RGB cam -> 'color'
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setPreviewSize(640, 640)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setPreviewKeepAspectRatio(False)
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
        with open("models/yolov6n.json", "r") as f:
            data = json.load(f)
            config = data["nn_config"]["NN_specific_metadata"]
            self.label_map = data["mappings"]["labels"]

        spatial_nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
        spatial_nn.setBlobPath("models/yolov6n.blob")
        spatial_nn.setConfidenceThreshold(config.get("confidence_threshold", {}))
        spatial_nn.setNumClasses(config.get("classes", {}))
        spatial_nn.setCoordinateSize(config.get("coordinates", {}))
        spatial_nn.setIouThreshold(config.get("iou_threshold", {}))
        spatial_nn.setAnchors(config.get("anchors", {}))
        spatial_nn.setAnchorMasks(config.get("anchor_masks", {}))
        spatial_nn.input.setBlocking(False)
        spatial_nn.setBoundingBoxScaleFactor(0.2)
        spatial_nn.setDepthLowerThreshold(100) # Min 10 centimeters
        spatial_nn.setDepthUpperThreshold(5000) # Max 5 meters
        
        xout_nn = pipeline.create(dai.node.XLinkOut)
        xout_nn.setStreamName("detection")

        cam_rgb.preview.link(spatial_nn.input)
        cam_stereo.depth.link(spatial_nn.inputDepth)
        spatial_nn.passthrough.link(xout_rgb.input)
        spatial_nn.passthroughDepth.link(xout_depth.input)
        spatial_nn.boundingBoxMapping.link(xout_bounding_box_bepth_mapping.input)
        spatial_nn.out.link(xout_nn.input)

        # Image manip -> crop detections
        image_manip_script = pipeline.create(dai.node.Script)
        spatial_nn.passthrough.link(image_manip_script.inputs["passthrough"])
        spatial_nn.passthrough.link(image_manip_script.inputs["preview"])
        spatial_nn.out.link(image_manip_script.inputs["dets_in"])

        image_manip_script.setScript("""
        import time
        msgs = dict()
        def add_msg(msg, name, seq = None):
            global msgs
            if seq is None:
                seq = msg.getSequenceNum()
            seq = str(seq)
            # node.warn(f"New msg {name}, seq {seq}")
            # Each seq number has it's own dict of msgs
            if seq not in msgs:
                msgs[seq] = dict()
            msgs[seq][name] = msg
            # To avoid freezing (not necessary for this ObjDet model)
            if 15 < len(msgs):
                node.warn(f"Removing first element! len {len(msgs)}")
                msgs.popitem() # Remove first element
        def get_msgs():
            global msgs
            seq_remove = [] # Arr of sequence numbers to get deleted
            for seq, syncMsgs in msgs.items():
                seq_remove.append(seq) # Will get removed from dict if we find synced msgs pair
                # node.warn(f"Checking sync {seq}")
                # Check if we have both detections and color frame with this sequence number
                if len(syncMsgs) == 2: # 1 frame, 1 detection
                    for rm in seq_remove:
                        del msgs[rm]
                    # node.warn(f"synced {seq}. Removed older sync values. len {len(msgs)}")
                    return syncMsgs # Returned synced msgs
            return None
        def correct_bb(bb):
            if bb.xmin < 0: bb.xmin = 0.001
            if bb.ymin < 0: bb.ymin = 0.001
            if bb.xmax > 1: bb.xmax = 0.999
            if bb.ymax > 1: bb.ymax = 0.999
            return bb
        while True:
            time.sleep(0.001) # Avoid lazy looping
            preview = node.io['preview'].tryGet()
            if preview is not None:
                add_msg(preview, 'preview')
            dets = node.io['dets_in'].tryGet()
            if dets is not None:
                # TODO: in 2.18.0.0 use dets.getSequenceNum()
                passthrough = node.io['passthrough'].get()
                seq = passthrough.getSequenceNum()
                add_msg(dets, 'dets', seq)
            sync_msgs = get_msgs()
            if sync_msgs is not None:
                img = sync_msgs['preview']
                dets = sync_msgs['dets']
                for i, det in enumerate(dets.detections):
                    cfg = ImageManipConfig()
                    correct_bb(det)
                    cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
                    # node.warn(f"Sending {i + 1}. age/gender det. Seq {seq}. Det {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
                    cfg.setResize(128, 256)
                    cfg.setKeepAspectRatio(False)
                    node.io['manip_cfg'].send(cfg)
                    node.io['manip_img'].send(img)
        """)

        # Recognition manip -> resize for embedding
        recognition_manip = pipeline.create(dai.node.ImageManip)
        recognition_manip.initialConfig.setResize(128, 256)
        recognition_manip.setWaitForConfigInput(True)
        image_manip_script.outputs['manip_cfg'].link(recognition_manip.inputConfig)
        image_manip_script.outputs['manip_img'].link(recognition_manip.inputImage)

        # Recogniton nn -> 'recognition'
        recognition_nn = pipeline.create(dai.node.NeuralNetwork)
        recognition_nn.setBlobPath(blobconverter.from_zoo(name="person-reidentification-retail-0288", shaves=6))
        recognition_manip.out.link(recognition_nn.input)
        recognition_nn_xout = pipeline.create(dai.node.XLinkOut)
        recognition_nn_xout.setStreamName("recognition")
        recognition_nn.out.link(recognition_nn_xout.input)

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
        queues = [self.rgb_queue, self.nn_queue, self.depth_queue, self.rec_queue]
        for q in queues:
            data = q.tryGet()
            if data:
                self.sync.add_msg(data, q.getName())

        msgs = self.sync.get_msgs()
        if msgs == None:
            return

        self.frame_color = msgs["color"]
        self.frame_depth = msgs["depth"]
        
        detections = msgs["detection"].detections
        recognitions = msgs["recognition"]

        self.mapping = self.mapping_queue.tryGet()
        self.detected_objects = []

        if len(detections) > 0 and self.mapping is not None:
            for detection, recognition in zip(detections, recognitions):
                embedding = np.array(recognition.getFirstLayerFp16())

                try:
                    label = self.label_map[detection.label]
                except:
                    label = detection.label

                if self.calibration.cam_to_world is not None:
                    pos_camera_frame = np.array([[detection.spatialCoordinates.x / 1000, -detection.spatialCoordinates.y / 1000, detection.spatialCoordinates.z / 1000, 1]]).T
                    pos_world_frame = self.calibration.cam_to_world @ pos_camera_frame

                    self.detected_objects.append(
                        Detection(
                            dai_det = detection, # need bounding box and confidence data
                            label = label,
                            pos = pos_world_frame,
                            embedding= embedding,
                            camera_friendly_id = self.friendly_id
                        )
                    )


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
            for roi_data, detection in zip(roi_datas, tracks):
                roi = roi_data.roi
                roi = roi.denormalize(self.viz_width, self.viz_height)
                top_left = roi.topLeft()
                bottom_right = roi.bottomRight()
                xmin = int(top_left.x)
                ymin = int(top_left.y)
                xmax = int(bottom_right.x)
                ymax = int(bottom_right.y)

                dai_det = detection["dai_det"]

                x1 = int(dai_det.xmin * self.viz_width)
                x2 = int(dai_det.xmax * self.viz_width)
                y1 = int(dai_det.ymin * self.viz_height)
                y2 = int(dai_det.ymax * self.viz_height)

                cv2.rectangle(visualization, (xmin, ymin), (xmax, ymax), (100, 0, 0), 2)
                cv2.rectangle(visualization, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(visualization, str(detection["label"])+f"_{detection['object_id']}", (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(visualization, "{:.2f}".format(dai_det.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(visualization, f"X: {int(dai_det.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(visualization, f"Y: {int(dai_det.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(visualization, f"Z: {int(dai_det.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

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