import cv2
import numpy as np
import KalmanFilter

class Tracker:
    def __init__(self, max_lost=30, activation_frames=30, max_tracks=1, height_smoothing=20, smoothing_window=4):
        self.next_id = 0
        self.tracks = {}
        self.max_lost = max_lost
        self.activation_frames = activation_frames
        self.max_tracks = max_tracks
        self.height_smoothing = height_smoothing  # Puffer für die Höhe
        self.smoothing_window = smoothing_window  # Anzahl der letzten Frames, die für den Durchschnitt verwendet werden
        self.smoothing_buffers = {}

    def add_track(self, bbox,hist):
        track_id = self.next_id
        self.tracks[self.next_id] = {
            "bbox": tuple(map(int, bbox)),
            "lost": 0,
            "stable_frames": 0,
            "active": False,
            "height_history": [],
            "center_predicter":KalmanFilter.KalmanFilter(),
            "prediction" : tuple([0,0,0,0]),
            "previous_center":tuple([0,0]),
            "histogram":[hist]
        }
        self.smoothing_buffers[track_id] = []
        self.next_id += 1

    def apply_smoothing(self, track_id, bbox):
        if track_id not in self.smoothing_buffers:
            return bbox

        self.smoothing_buffers[track_id].append(bbox)

        if len(self.smoothing_buffers[track_id]) > self.smoothing_window:
            self.smoothing_buffers[track_id].pop(0)

        avg_bbox = np.mean(self.smoothing_buffers[track_id], axis=0).astype(int)

        return tuple(avg_bbox)

    def update_track(self, detections,detection_areas):
        update_tracks = {}
        for track_id, track in self.tracks.items():
            track["lost"] += 1
            if track["lost"] <= self.max_lost:
                update_tracks[track_id] = track

        self.tracks = update_tracks

        for track_id, track in self.tracks.items():
            possible = []
            for det,det_area in zip(detections,detection_areas):
                if self.is_close(track["bbox"], det) or self.is_close(track["prediction"], det):
                    det_hist = self.get_hist(det_area)
                    cmp = self.compare_histogramm(det_hist,track)
                    bundle = det[2]>det[3]/1.5
                    if cmp > 0.5 or bundle:
                        possible.append((det,cmp,bundle,det_hist))

            if len(possible) > 0:
                print("match")
                pos = max(possible,key=lambda x: x[1])
                det,cmp,bundle,det_hist = pos
                track["bbox"] = det
                track["lost"] = 0
                track["stable_frames"] += 1
                if not bundle and cmp > 0.8:
                    self.add_histogramm(det_hist, track)
                if track["stable_frames"] >= self.activation_frames:
                    track["active"] = True
            self.predict_future_bbox(track)
        if len(self.tracks) < self.max_tracks and len(detections) > 0:
            self.add_track(detections[0],self.get_hist(detection_areas[0]))

    def sortSecond(self,val):
        return val[1]

    def add_histogramm(self,hist, track):
        track["histogram"].append(hist)
        if len(track["histogram"])>=20:
            track["histogram"].pop(0)

    def compare_histogramm(self,det_area_hist,track):      #0.67 - 68
        val = []
        for det_hist,track_hist in zip(det_area_hist,np.mean(track["histogram"],axis=0)):
            val.append(cv2.compareHist(det_hist,track_hist,cv2.HISTCMP_CORREL))
        return np.mean(val)

    def get_hist(self,segment):
        hist_b = cv2.calcHist([segment], [0], None, [127], [1, 256])
        hist_g = cv2.calcHist([segment], [1], None, [127], [1, 256])
        hist_r = cv2.calcHist([segment], [2], None, [127], [1, 256])

        return [hist_b, hist_g, hist_r]

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
            self.predict_future_bbox(track)


    def predict_future_bbox(self,track):
        if(track["lost"] < 1):
            track_bbox = track["bbox"]

            bbox_width = track_bbox[2]
            bbox_height = track_bbox[3]

            predicted_center = track["center_predicter"].predict(
                track_bbox[0] + track_bbox[2] // 2,
                track_bbox[1] + track_bbox[3] // 2
            )

            new_bbox_x = int(predicted_center[0] - bbox_width // 2)
            new_bbox_y = int(predicted_center[1] - bbox_height // 2)
            new_bbox_width = bbox_width
            new_bbox_height = bbox_height

            track["prediction"] = (new_bbox_x, new_bbox_y, new_bbox_width, new_bbox_height)
        else:
            track_prediction = track["prediction"]
            track_last_bbox_size = track["bbox"]

            bbox_width = track_last_bbox_size[2]
            bbox_height = track_last_bbox_size[3]

            predicted_center = track["center_predicter"].predict(
                track_prediction[0] + bbox_width // 2,
                track_prediction[1] + bbox_height // 2
            )

            new_bbox_x = int(predicted_center[0] - bbox_width // 2)
            new_bbox_y = int(track_prediction[1] - bbox_height // 2)

            track["prediction"] = (new_bbox_x, track_prediction[1], bbox_width, bbox_height)


    def simple_future_box_prediction(self,track):
        if (track["lost"] < 1):
            track_bbox = track["bbox"]
        else:
            track_bbox = track["prediction"]

        bbox_width = track_bbox[2]
        bbox_height = track_bbox[3]
        bbox_center_x = track_bbox[0] + bbox_width // 2
        bbox_center_y = track_bbox[1] + bbox_height // 2

        predicted_center = [bbox_center_x + (bbox_center_x - track["previous_center"][0]),bbox_center_y]

        new_bbox_x = int(predicted_center[0] - bbox_width // 2)
        new_bbox_y = int(predicted_center[1] - bbox_height // 2)
        new_bbox_width = bbox_width
        new_bbox_height = bbox_height

        track["previous_center"] = [bbox_center_x,bbox_center_y]
        track["prediction"] = (new_bbox_x, new_bbox_y, new_bbox_width, new_bbox_height)


