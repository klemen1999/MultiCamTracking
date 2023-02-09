import numpy as np
from typing import List
import depthai as dai

class Detection:
    def __init__(self, bbox: np.ndarray, confidence: float, label: str, pos: np.ndarray,
        frame: np.ndarray, spatial_coords: np.ndarray, camera_friendly_id: int):
        
        self.bbox = bbox # [x_min, y_min, x_max, y_max]
        self.confidence = confidence
        self.label = label
        self.pos = pos
        self.frame = frame
        self.spatial_coords = spatial_coords # [x, y, z]
        self.camera_friendly_id = camera_friendly_id

    def bbox_to_ltwh(self): # [left, top, width, height]
        return np.array([self.bbox[0], self.bbox[1], self.bbox[2]-self.bbox[0], self.bbox[3]-self.bbox[1]])
