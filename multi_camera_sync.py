from typing import List

# sync msgs accross multiple cameras by timestamp
# sync process: pick first device, go over all it's messages and for each check if there is
#               one message in every other device with small enough time difference           

class MultiCameraSync:
    devices: List[str] # ids of all devices
    max_dt: float # maximum difference between msgs from different cameras in seconds
    msgs: dict # each msg has detections and timestamp

    def __init__(self, devices, max_dt):
        self.devices = devices
        self.max_dt = max_dt
        self.msgs = {id:[] for id in devices}

    def add_msg(self, msg, device_id):
        self.msgs[device_id].append(msg)

    def delete_previous(self, tstamp):
        # go over all msgs and remove ones that happened before tstamp
        for device_id in self.msgs:
            to_keep = []
            for msg in self.msgs[device_id]:
                curr_tstamp = msg["timestamp"]
                if curr_tstamp > tstamp:
                    to_keep.append(msg)
            self.msgs[device_id] = to_keep

    def get_msgs(self):
        # if we only have one device then return first msg
        if len(self.devices) == 1:
            if len(self.msgs[self.devices[0]]) == 0:
                return None
            synced_msg = {self.devices[0]: self.msgs[self.devices[0]][0]}
            self.msgs[self.devices[0]].pop(0)
            return synced_msg

        device1 = self.devices[0]
        synced = {device:None for device in self.devices}
        for msg1 in self.msgs[device1]:
            tstamp1 = msg1["timestamp"]
            for device2 in self.devices:
                if device1 == device2:
                    continue
                found = False
                for msg2 in self.msgs[device2]:
                    tstamp2 = msg2["timestamp"]
                    if abs((tstamp1-tstamp2).total_seconds()) < self.max_dt:
                        found = True
                        synced[device2] = msg2
                        if synced[device1] is None:
                            synced[device1] = msg1
                        break
                if not found:
                    synced[device2] = None
                    break
            
            # check if we have synced msg = in synced dict there must be no None values
            if len([x for x in synced if synced[x] is not None]) == len(synced):
                self.delete_previous(tstamp1) # delete old messages
                return synced
        
        return None
