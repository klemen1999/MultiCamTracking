import numpy as np
from typing import List
import depthai as dai

class Detection:
    def __init__(self, dai_det: dai.ImgDetections, label: str, pos: np.ndarray, embedding: np.ndarray, camera_friendly_id: int):
        self.dai_det = dai_det
        self.label = label
        self.pos = pos
        self.embedding = embedding
        self.camera_friendly_id = camera_friendly_id

    def bbox_xyxy(self):
        return np.array([self.dai_det.xmin, self.dai_det.ymin, self.dai_det.xmax, self.dai_det.ymax])

    def bbox_to_ltwh(self): # [left, top, width, height]
        bbox = self.bbox_xyxy()
        return np.array([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]])