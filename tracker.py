import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self, devices, multi_cam_max_dist=2, multi_cam_assoc_coef=0.5, multi_cam_assoc_thresh=0.7, **kwargs):
        self.devices = devices
        self.multi_cam_max_dist = multi_cam_max_dist
        self.multi_cam_assoc_coef = multi_cam_assoc_coef
        self.multi_cam_assoc_thresh = multi_cam_assoc_thresh

        self.trackers = [(device_id, DeepSort(**kwargs)) for device_id in devices] # one DeepSort tracker per device

    def update(self, detections):
        tracks = []
        for device_id, tracker in self.trackers:
            curr_dets, curr_embeds, others = self.decode_detections(detections[device_id]["detections"])
            curr_tracks = tracker.update_tracks(curr_dets, embeds=curr_embeds, others=others)
            tracks.extend(curr_tracks)

        # filter out non active tracks
        active_tracks = [t for t in tracks if t.is_confirmed() and t.time_since_update == 0]

        output = {device_id: [] for device_id in self.devices}
        if len(active_tracks) == 0:
            return output
        
        probs = self.find_probs_of_same_ids(active_tracks)
        probs[probs < self.multi_cam_assoc_coef] = 0

        # for each track check associated tracks
        assoc_tracks = {}
        for i, prob in enumerate(probs):
            curr_max = prob.max()
            if curr_max != 0:
                assoc_tracks[i] = set(np.where(prob == curr_max)[0])
                assoc_tracks[i].add(i)
            else: # if curr_max == 0 then this track is not associated with any other track from other cameras
                assoc_tracks[i] = set([i])

        # merge associated tracks into groups (1 group == 1 real life object)
        groups = []
        for i in range(len(active_tracks)):
            curr_set = assoc_tracks[i]
            joined = False
            for g in groups:
                if len(g.intersection(curr_set)):
                    g.union(curr_set)
                    joined = True
                    break
            if not joined:
                groups.append(curr_set)

        # return merged active tracks
        for i, g in enumerate(groups):
            for track_id in g:
                curr_track = active_tracks[track_id]
                curr_device = curr_track.others["device_id"]
                output[curr_device].append({
                    "object_id": curr_track.track_id,
                    "device_id": curr_device,
                    "bbox": curr_track.to_ltrb(orig=True),
                    "confidence": curr_track.det_conf,
                    "spatial_coords": curr_track.others["spatial_coords"],
                    "label": curr_track.det_class,
                    "pos": curr_track.others["pos"],
                })
        
        return output

    def decode_detections(self, detections):
        dets = []
        embeds = []
        others = []
        for detection in detections:
            dets.append([detection.bbox_to_ltwh(), detection.confidence, detection.label])
            embeds.append(detection.embedding)
            others.append({
                "device_id": detection.camera_friendly_id,
                "pos": detection.pos,
                "spatial_coords": detection.spatial_coords
            })
        return dets, embeds, others

    def find_probs_of_same_ids(self, tracks):
        n = len(tracks)
        cos_sim_mat = np.zeros((n,n)) # cosine similarity between objects features
        dist_mat = np.zeros((n,n)) # distance between tracks in real space
        cam_coef_mat = np.ones((n,n)) # 0 means different object

        for i in range(n):
            for j in range(n):
                cos_sim_mat[i,j] = self.cos_similarity(tracks[i].get_feature(), tracks[j].get_feature())
                dist_mat[i,j] = np.linalg.norm(tracks[i].others["pos"] - tracks[j].others["pos"])
                if tracks[i].others["device_id"] == tracks[j].others["device_id"]: # if from same device than different object
                    cam_coef_mat[i,j] = 0
                if tracks[i].det_class != tracks[j].det_class: # if different class than different object
                    cam_coef_mat[i,j] = 0

        # normalize distances between 0 and max value
        dist_mat[dist_mat > self.multi_cam_max_dist] = self.multi_cam_max_dist
        denominator = (self.multi_cam_max_dist - dist_mat.min())
        if denominator != 0:
            dist_mat = (dist_mat - dist_mat.min()) / denominator

        # remap so closest tracks have higest values
        dist_mat = np.where(dist_mat == 0, 1, 1-dist_mat)

        result = cam_coef_mat * np.add(self.multi_cam_assoc_coef * cos_sim_mat, 
            (1-self.multi_cam_assoc_coef)*dist_mat)
        return result

    @staticmethod
    def cos_similarity(a, b):
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator == 0:
            return np.nan
        return np.dot(a, b) / denominator