from camera import Camera
from typing import List
import numpy as np
import cv2
import collections

class BirdsEyeView:
    colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0,0,255), (0,255,0), (255,0,0)]

    def __init__(self, cameras: List[Camera], width, height, scale, trail_length=300):
        self.cameras = cameras
        self.width = width
        self.height = height
        self.scale = scale
        self.history = collections.deque(maxlen=trail_length)

        self.img = np.zeros((height, width, 3), np.uint8)
        self.world_to_birds_eye = np.array([
            [scale, 0, 0, width//2],
            [0, scale, 0, height//2],
        ])

    def draw_coordinate_system(self):
        p_0 = (self.world_to_birds_eye @ np.array([0, 0, 0, 1])).astype(np.int64)
        p_x = (self.world_to_birds_eye @ np.array([0.3, 0, 0, 1])).astype(np.int64)
        p_y = (self.world_to_birds_eye @ np.array([0, 0.3, 0, 1])).astype(np.int64)
        cv2.line(self.img, p_0, p_x, (0, 0, 255), 2)
        cv2.line(self.img, p_0, p_y, (0, 255, 0), 2)

    def draw_cameras(self):
        for camera in self.cameras:
            try:
                color = self.colors[camera.friendly_id - 1]
            except:
                color = (255,255,255)

            # draw the camera position
            if camera.calibration.position is not None:
                p = (self.world_to_birds_eye @ (camera.calibration.cam_to_world @ np.array([0,0,0,1]))).astype(np.int64)
                p_l = (self.world_to_birds_eye @ (camera.calibration.cam_to_world @ np.array([0.2,0,0.1,1]))).astype(np.int64)
                p_r = (self.world_to_birds_eye @ (camera.calibration.cam_to_world @ np.array([-0.2,0,0.1,1]))).astype(np.int64)
                cv2.circle(self.img, p, 5, color, -1)
                cv2.line(self.img, p, p_l, color, 1)
                cv2.line(self.img, p, p_r, color, 1)
    
    def to_birdseye(self, pos):
        return (self.world_to_birds_eye @ pos).flatten().astype(np.int64)
    
    def draw_history(self):
        for i, group in enumerate(self.history):
            for obj in group:
                c = int(i/self.history.maxlen*50)
                p = self.to_birdseye(obj.pos)
                cv2.circle(self.img, p, int(i/self.history.maxlen*10), (c, c, c), -1)

    def draw_objects(self, tracks, objects):
        # draw individual tracks
        for device_id in tracks:
            for track in tracks[device_id]:
                p = self.to_birdseye(track["pos"])
                try: c = self.colors[device_id - 1]
                except: c = (255,255,255)
                cv2.circle(self.img, p, 2, c, -1)
        # draw objects
        for obj in objects:
            p = self.to_birdseye(obj.pos)
            label = f"{obj.label}_{obj.id}"
            cv2.circle(self.img, p, int(0.25*self.scale), (255, 255, 255), 0)
            cv2.putText(self.img, label, p+np.array([0, int(0.25*self.scale) + 10]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    def render(self, tracks, objects):
        self.img = np.zeros((self.height, self.width, 3), np.uint8)
        self.draw_coordinate_system()
        self.draw_cameras()
        self.draw_history()
        self.history.append(objects)
        self.draw_objects(tracks, objects)

        cv2.imshow("Bird's Eye View", self.img)
        