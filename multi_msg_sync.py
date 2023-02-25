# Color frames (ImgFrame), object tracks (Tracklets) and embedding (NNData)
# messages arrive to the host all with some additional delay.
# For each ImgFrame there's one Tracklets msg, which has multiple tracklets, and for each
# tracklet there's a NNData msg which contains embedding results.

# How it works:
# Every ImgFrame, Tracklets and NNData message has it's own sequence number, by which we can sync messages.

class TwoStageHostSeqSync:
    def __init__(self):
        self.msgs = {}
    # name: color, tracks, or embedding
    def add_msg(self, msg, name):
        seq = str(msg.getSequenceNum())
        if seq not in self.msgs:
            self.msgs[seq] = {} # Create directory for msgs
        if "embedding" not in self.msgs[seq]:
            self.msgs[seq]["embedding"] = [] # Create embedding array

        if name == "embedding":
            # Append embedding msgs to an array
            self.msgs[seq]["embedding"].append(msg)
            # print(f'Added embedding seq {seq}, total len {len(self.msgs[seq]["embedding"])}')

        elif name == "tracks":
            # Save tracks msg in the directory
            self.msgs[seq][name] = msg
            self.msgs[seq]["len"] = len(msg.tracklets)
            # print(f'Added tracks seq {seq}')

        elif name == "color" or name == "depth": # color
            # Save frame in the directory
            self.msgs[seq][name] = msg
            # print(f'Added frame seq {seq}')
            


    def get_msgs(self):
        seq_remove = [] # Arr of sequence numbers to get deleted

        for seq, msgs in self.msgs.items():
            seq_remove.append(seq) # Will get removed from dict if we find synced msgs pair

            # Check if we have tracks, color and depth frame with this sequence number
            if "color" in msgs and "depth" in msgs and "len" in msgs:

                # Check if all detected objects (faces) have finished embedding inference
                if msgs["len"] == len(msgs["embedding"]):
                    # print(f"Synced msgs with sequence number {seq}", msgs)

                    # We have synced msgs, remove previous msgs (memory cleaning)
                    for rm in seq_remove:
                        del self.msgs[rm]

                    return msgs # Returned synced msgs

        return None # No synced msgs