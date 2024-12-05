import cv2
import numpy as np

class Tracker:
    def __init__(self, max_lost = 30, activation_frames=60):
        self.next_id=0
        self.tracks={}
        self.max_lost = max_lost
        self.activation_frames = activation_frames

    def add_track(self, bbox):
        self.tracks[self.next_id] = {
            "bbox": bbox,
            "lost": 0,
            "stable_frames": 0,
            "active": False,
        }
        self.next_id+=1

    def update_track(self, detections):
        update_tracks = {}
        for track_id, track in self.tracks.items():
            track["lost"] += 1
            if track["lost"] <= self.max_lost:
                update_tracks[track_id] = track

        self.tracks = update_tracks

        for det in detections:
            assigned = False
            for track_id, track in self.tracks.items():
                if self.is_close(track["bbox"], det):
                    track["bbox"] = det
                    track["lost"] = 0
                    track["stable_frames"] += 1  # Erhöhe stabile Frames
                    if track["stable_frames"] >= self.activation_frames:
                        track["active"] = True  # Aktiviere den Track
                    assigned = True
                    break
            if not assigned:
                self.add_track(det)

    def is_close(self, bbox1, bbox2, distance_threshold=200):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
        cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
        return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) < distance_threshold

    def get_active_tracks(self):
        return {tid: t for tid, t in self.tracks.items() if t["active"]}