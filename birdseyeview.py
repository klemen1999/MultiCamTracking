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


    def make_groups(self, tracks):
        groups = {}
        for device_id in tracks:
            for track in tracks[device_id]:
                obj_id = track["object_id"]
                if obj_id not in groups:
                    groups[obj_id] = [track]
                else:
                    groups[obj_id].append(track)
        return groups

    
    def draw_groups(self, groups):
        for obj_id in groups:
            avg = np.zeros(2)
            label = ""
            for track in groups[obj_id]:
                label = track["label"]+f"_{track['object_id']}"
                try: c = self.colors[track["device_id"] - 1]
                except: c = (255,255,255)
                p = (self.world_to_birds_eye @ track["pos"]).flatten().astype(np.int64)
                avg += p
                cv2.circle(self.img, p, 2, c, -1)

            avg = (avg/len(groups[obj_id])).astype(np.int64)
            cv2.circle(self.img, avg, int(0.25*self.scale), (255, 255, 255), 0)
            cv2.putText(self.img, str(label), avg+np.array([0, int(0.25*self.scale) + 10]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def draw_history(self):
        for i, groups in enumerate(self.history):
            for obj_id in groups:
                avg = np.zeros(2)
                for track in groups[obj_id]:
                    p = (self.world_to_birds_eye @ track["pos"]).flatten().astype(np.int64)
                    avg += p

                avg = (avg/len(groups[obj_id])).astype(np.int64)
                c = int(i/self.history.maxlen*50)
                cv2.circle(self.img, avg, int(i/self.history.maxlen*10), (c, c, c), -1)

    def render(self, tracks):
        self.img = np.zeros((self.height, self.width, 3), np.uint8)
        self.draw_coordinate_system()
        self.draw_cameras()
        groups = self.make_groups(tracks)
        self.draw_history()
        self.history.append(groups)
        self.draw_groups(groups)

        cv2.imshow("Bird's Eye View", self.img)
        