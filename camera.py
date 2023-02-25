import depthai as dai
import blobconverter
import cv2
import json
import time
import numpy as np
from typing import List

from calibration import Calibration
from tracklet import Tracklet
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
        self.tracks_queue = self.device.getOutputQueue(name="tracks", maxSize=1, blocking=False)
        self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=1, blocking=False)
        self.emb_queue = self.device.getOutputQueue(name="embedding", maxSize=1, blocking=False)

        self.window_name = f"[{self.friendly_id}] Camera - mxid: {self.mxid}"
        self.viz_height, self.viz_width = 360, 640
        if show_video:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, self.viz_width, self.viz_height)

        self.frame_color = None
        self.frame_depth = None
        self.curr_tracklets: List[Tracklet] = []

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
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P) # use THE_4_K for calibration
        # cam_rgb.setPreviewSize(640, 640)
        cam_rgb.setPreviewSize(300, 300)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setPreviewKeepAspectRatio(False)
        cam_rgb.setPreviewNumFramesPool(30)

        # Depth cam -> 'depth'
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        cam_stereo = pipeline.create(dai.node.StereoDepth)
        # cam_stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        cam_stereo.setDepthAlign(dai.CameraBoardSocket.RGB) # Align depth map to the perspective of RGB camera, on which inference is done
        cam_stereo.setOutputSize(mono_left.getResolutionWidth(), mono_left.getResolutionHeight())
        mono_left.out.link(cam_stereo.left)
        mono_right.out.link(cam_stereo.right)


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

        cam_rgb.preview.link(spatial_nn.input)
        cam_stereo.depth.link(spatial_nn.inputDepth)
        
        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        spatial_nn.passthroughDepth.link(xout_depth.input)

        # Object Tracker
        object_tracker = pipeline.create(dai.node.ObjectTracker)
        object_tracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
        object_tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
        spatial_nn.passthrough.link(object_tracker.inputTrackerFrame)
        spatial_nn.passthrough.link(object_tracker.inputDetectionFrame)
        spatial_nn.out.link(object_tracker.inputDetections)
        
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("color")
        object_tracker.passthroughTrackerFrame.link(xout_rgb.input)

        xout_nn = pipeline.create(dai.node.XLinkOut)
        xout_nn.setStreamName("tracks")
        object_tracker.out.link(xout_nn.input)

        # Image manip -> crop tracks
        image_manip_script = pipeline.create(dai.node.Script)
        object_tracker.passthroughTrackerFrame.link(image_manip_script.inputs["preview"])
        object_tracker.out.link(image_manip_script.inputs["tracks_in"])
        
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
            if 30 < len(msgs):
                node.warn(f"Removing first element! len {len(msgs)}")
                msgs.popitem() # Remove first element
        def get_msgs():
            global msgs
            seq_remove = [] # Arr of sequence numbers to get deleted
            for seq, syncMsgs in msgs.items():
                seq_remove.append(seq) # Will get removed from dict if we find synced msgs pair
                # node.warn(f"Checking sync {seq}")
                # Check if we have both tracks and color frame with this sequence number
                if len(syncMsgs) == 2: # 1 frame, 1 track
                    for rm in seq_remove:
                        del msgs[rm]
                    # node.warn(f"synced {seq}. Removed older sync values. len {len(msgs)}")
                    return syncMsgs # Returned synced msgs
            return None
        def correct_bb(bb):
            if bb.topLeft().x < 0: bb.topLeft().x = 0.001
            if bb.topLeft().y < 0: bb.topLeft().y = 0.001
            if bb.bottomRight().x > 1: bb.bottomRight().x = 0.999
            if bb.bottomRight().y > 1: bb.bottomRight().y = 0.999
            return bb
        while True:
            time.sleep(0.001) # Avoid lazy looping
            preview = node.io['preview'].tryGet()
            if preview is not None:
                add_msg(preview, 'preview')
            tracks = node.io['tracks_in'].tryGet()
            if tracks is not None:
                seq = tracks.getSequenceNum()
                add_msg(tracks, 'tracks', seq)
            sync_msgs = get_msgs()
            if sync_msgs is not None:
                img = sync_msgs['preview']
                tracks = sync_msgs['tracks']
                for tracklet in tracks.tracklets:
                    cfg = ImageManipConfig()
                    # correct_bb(tracklet.roi)
                    cfg.setCropRect(tracklet.roi.topLeft().x, tracklet.roi.topLeft().y, 
                        tracklet.roi.bottomRight().x, tracklet.roi.bottomRight().y)
                    cfg.setResize(224, 224)
                    # cfg.setResize(128, 256)
                    cfg.setKeepAspectRatio(False)
                    node.io['manip_cfg'].send(cfg)
                    node.io['manip_img'].send(img)
        """)

        # Embedding manip -> resize for embedding
        embedding_manip = pipeline.create(dai.node.ImageManip)
        embedding_manip.initialConfig.setResize(224, 224)
        # embedding_manip.initialConfig.setResize(128, 256)
        embedding_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
        embedding_manip.setWaitForConfigInput(True)
        image_manip_script.outputs['manip_cfg'].link(embedding_manip.inputConfig)
        image_manip_script.outputs['manip_img'].link(embedding_manip.inputImage)

        # Embedding nn -> 'embedding'
        embedding_nn = pipeline.create(dai.node.NeuralNetwork)
        # embedding_nn.setBlobPath(blobconverter.from_zoo(name="person-reidentification-retail-0288", shaves=4))
        embedding_nn.setBlobPath(blobconverter.from_zoo(name="mobilenetv2_imagenet_embedder_224x224", zoo_type="depthai", shaves=6))
        embedding_manip.out.link(embedding_nn.input)
        embedding_nn_xout = pipeline.create(dai.node.XLinkOut)
        embedding_nn_xout.setStreamName("embedding")
        embedding_nn.out.link(embedding_nn_xout.input)

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
        queues = [self.rgb_queue, self.tracks_queue, self.depth_queue, self.emb_queue]
        for q in queues:
            data = q.tryGet()
            if data:
                self.sync.add_msg(data, q.getName())

        msgs = self.sync.get_msgs()
        if msgs == None:
            self.frame_color = None
            self.frame_depth = None
            return

        self.frame_color = msgs["color"]
        self.frame_depth = msgs["depth"]
        
        tracklets = msgs["tracks"].tracklets
        embeddings = msgs["embedding"]

        self.curr_tracklets = []

        if len(tracklets):
            for tracklet, embedding in zip(tracklets, embeddings):
                if tracklet.status.name not in ["TRACKED", "NEW"]:
                    continue

                embedding = np.array(embedding.getFirstLayerFp16())

                if self.calibration.cam_to_world is not None:
                    pos_camera_frame = np.array([[tracklet.spatialCoordinates.x / 1000, -tracklet.spatialCoordinates.y / 1000, tracklet.spatialCoordinates.z / 1000, 1]]).T
                    pos_world_frame = self.calibration.cam_to_world @ pos_camera_frame

                    self.curr_tracklets.append(
                        Tracklet(
                            dai_tracklet=tracklet,
                            embedding=embedding,
                            pos=pos_world_frame,
                            device_id=self.friendly_id
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

        for track in tracks:
            t = track["dai_tracklet"]
            roi = t.roi.denormalize(visualization.shape[1], visualization.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            label_str = self.label_map[t.label]

            cv2.rectangle(visualization, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(visualization, f"{label_str}_{track['object_id']}", (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(visualization, f"X: {int(t.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(visualization, f"Y: {int(t.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(visualization, f"Z: {int(t.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

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