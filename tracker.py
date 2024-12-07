import cv2
import numpy as np

class Tracker:
    def __init__(self, max_lost=30, activation_frames=60, max_tracks=1, height_smoothing=20, smoothing_window=5):
        self.next_id = 0
        self.tracks = {}
        self.max_lost = max_lost
        self.activation_frames = activation_frames
        self.max_tracks = max_tracks
        self.height_smoothing = height_smoothing  # Puffer für die Höhe
        self.smoothing_window = smoothing_window  # Anzahl der letzten Frames, die für den Durchschnitt verwendet werden
        self.smoothing_buffers = {}

    def add_track(self, bbox):
        track_id = self.next_id
        self.tracks[self.next_id] = {
            "bbox": tuple(map(int, bbox)),
            "lost": 0,
            "stable_frames": 0,
            "active": False,
            "height_history": []
        }
        self.smoothing_buffers[track_id] = []
        self.next_id += 1

    def apply_smoothing(self, track_id, bbox):
        # Puffer der Bounding-Boxen des Tracks abrufen
        if track_id not in self.smoothing_buffers:
            return bbox

        self.smoothing_buffers[track_id].append(bbox)

        if len(self.smoothing_buffers[track_id]) > self.smoothing_window:
            self.smoothing_buffers[track_id].pop(0)

        # Durchschnitt der Bounding-Boxen
        avg_bbox = np.mean(self.smoothing_buffers[track_id], axis=0).astype(int)

        return tuple(avg_bbox)

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
            if not assigned and len(self.tracks) < self.max_tracks:
                self.add_track(det)


    def is_close(self, bbox1, bbox2, distance_threshold=200):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
        cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
        return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) < distance_threshold

    def get_active_tracks(self):
        return {tid: t for tid, t in self.tracks.items() if t["active"]}

    def refine_tracks_with_optical_flow(self, frame, old_frame):
        if len(self.get_active_tracks()) == 0:
            return

        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            p0 = np.array([[[track["bbox"][0] + track["bbox"][2] // 2,
                             track["bbox"][1] + track["bbox"][3] // 2]]
                           for track in self.get_active_tracks().values()
                           if "bbox" in track], dtype=np.float32)
        except ValueError as e:
            print("Fehler bei der Verarbeitung der Bounding-Boxen:", e)
            print("Datenstruktur:", self.get_active_tracks())
            return

        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        for i, (track_id, track) in enumerate(self.get_active_tracks().items()):
            if st[i] == 1:
                new_cx, new_cy = p1[i][0]
                old_cx, old_cy = p0[i][0]
                dx, dy = int(new_cx - old_cx), int(new_cy - old_cy)
                x, y, w, h = track["bbox"]

                # Update bounding box
                track["bbox"] = (x + dx, y + dy, w, h)

                #Glätten
                smoothed_bbox = self.apply_smoothing(track_id, track["bbox"])
                track["bbox"] = smoothed_bbox