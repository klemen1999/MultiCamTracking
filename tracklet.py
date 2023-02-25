import numpy as np
import depthai as dai

class Tracklet:
    def __init__(self, dai_tracklet: dai.Tracklet, embedding: np.ndarray,
        pos: np.ndarray, device_id: int):
        self.dai_tracklet = dai_tracklet         
        self.embedding = embedding
        self.pos = pos # [x, y, z]
        self.device_id = device_id