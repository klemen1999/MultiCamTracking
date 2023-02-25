from dataclasses import dataclass
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

@dataclass
class Object:
    id: int
    label: str
    pos: np.ndarray
    embedding: np.ndarray
    age: int

class ObjectList:
    def __init__(self, max_age, alpha, max_dist=2, assoc_thresh=0.7):
        self.objects = []
        self.max_age = max_age
        self.alpha = alpha # factor for blending between old and new features
        self.max_dist = max_dist
        self.assoc_thresh = assoc_thresh
        self.new_id = 0 # id to use on the newest object

    def match_group_to_object(self, group_pos, group_embedding, group_label, active_objects):
        if len(self.objects) == 0:
            return self.create_obj(group_pos, group_embedding, group_label)
        
        probs = []
        for obj in self.objects:
            if obj.label != group_label:
                probs.append(0)
                continue

            curr_cos_sim = cos_similarity(obj.embedding, group_embedding)
            curr_dist = euclidian_dist(obj.pos, group_pos)
            curr_prob = curr_cos_sim + (1 - curr_dist/self.max_dist)
            if curr_prob < self.assoc_thresh: curr_prob = 0
            probs.append(curr_prob)

        max_id = np.argmax(probs)
        if probs[max_id] == 0: # create obj if no matches
            obj = self.create_obj(group_pos, group_embedding, group_label)
        else: 
            obj = self.objects[max_id]
            if obj in active_objects: # if matched obj already active then create new one
                obj = self.create_obj(group_pos, group_embedding, group_label)
            else:
                self.update_obj(max_id, group_pos, group_embedding)

        return obj

    def update_obj(self, id, new_pos, new_embedding):
        obj = self.objects[id]
        obj.pos = new_pos
        obj.embedding = (1-self.alpha)*obj.embedding + self.alpha*new_embedding

    def create_obj(self, pos, embedding, label):
        new_obj = Object(id=self.new_id, label=label, pos=pos, embedding=embedding, age=self.max_age)
        print(f"New object created: {new_obj.label}_{new_obj.id}")
        self.objects.append(new_obj)
        self.new_id += 1
        return new_obj

    def update(self, active_objs):
        to_remove = []
        for obj in self.objects:
            if obj in active_objs:
                obj.age = self.max_age
            else:
                obj.age -= 1
                if obj.age <= 0: to_remove.append(obj)
        for obj in to_remove:
            print(f"Object removed: {obj.label}_{obj.id}")
            self.objects.remove(obj)
             

class Tracker:
    def __init__(self, devices, multi_cam_max_dist=2, multi_cam_assoc_coef=0.5, multi_cam_assoc_thresh=0.7, max_age=100, alpha=0.1, **kwargs):
        self.devices = devices
        self.multi_cam_max_dist = multi_cam_max_dist # maximum euclidian distance between objects to be still considered same
        self.multi_cam_assoc_coef = multi_cam_assoc_coef # ratio between cosine similarity and euclidian distance (1=only cos sim, 0=only euclidian dist)
        self.multi_cam_assoc_thresh = multi_cam_assoc_thresh # threshold for min probability for objects to be still considered same

        self.obj_list = ObjectList(max_age, alpha, multi_cam_max_dist, multi_cam_assoc_thresh)
        self.counter = 0

    def update(self, tracks):
        tracks_flat = []
        for device_id in tracks:
            tracks_flat.extend(tracks[device_id]["tracks"])

        track_output = {device_id: [] for device_id in self.devices}
        if len(tracks_flat) == 0:
            self.obj_list.update([]) # still update age of objects
            return track_output, []
        
        # 1 group == 1 real life object
        groups = self.group_tracks(tracks_flat)

        # create output for each device
        active_objects = []
        for group in groups:
            group_tracks = [tracks_flat[i] for i in group]
            group_label = group_tracks[0].dai_tracklet.label
            # test = [t.det_class for t in group_tracks]
            # print("TESTING LABELS", all(x==test[0] for x in test))
            avg_pos, avg_embedding = self.calc_group_average(group_tracks)
            curr_obj = self.obj_list.match_group_to_object(avg_pos, avg_embedding, group_label, active_objects)
            active_objects.append(curr_obj)
            for track in group_tracks:
                track_output[track.device_id].append({
                    "object_id": curr_obj.id,
                    "device_id": track.device_id,
                    "pos": track.pos,
                })
        
        # update objs ages and remove non active ones
        self.obj_list.update(active_objects)
        return track_output, active_objects

    def calc_group_average(self, tracks):
        avg_pos = np.zeros_like(tracks[0].pos)
        avg_embedding = np.zeros_like(tracks[0].embedding)         
        for t in tracks:
            avg_pos += t.pos
            avg_embedding += t.embedding
        avg_pos /= len(tracks)
        avg_embedding /= len(tracks)
        return avg_pos, avg_embedding


    def group_tracks(self, tracks):
        probs = self.find_probs_of_same_ids(tracks)
        probs[probs < self.multi_cam_assoc_thresh] = 0

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
        for i in range(len(tracks)):
            curr_set = assoc_tracks[i]
            joined = False
            for g in groups:
                if len(g.intersection(curr_set)):
                    g.union(curr_set)
                    joined = True
                    break
            if not joined:
                groups.append(curr_set) 
        
        return groups

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
                cos_sim_mat[i,j] = cos_similarity(tracks[i].embedding, tracks[j].embedding)
                dist_mat[i,j] = euclidian_dist(tracks[i].pos, tracks[j].pos)
                if tracks[i].device_id == tracks[j].device_id: # if from same device than different object
                    cam_coef_mat[i,j] = 0
                if tracks[i].dai_tracklet.label != tracks[j].dai_tracklet.label: # if different class than different object
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


def cos_similarity(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return np.nan
    return np.dot(a, b) / denominator

def euclidian_dist(a, b):
    return np.linalg.norm(a-b)