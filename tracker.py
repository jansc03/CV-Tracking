import cv2
import numpy as np
import kalman_filter

class Tracker:
    def __init__(self, max_lost=30, activation_frames=0, max_tracks=1, height_smoothing=20, smoothing_window=4):
        self.next_id = 0
        self.tracks = {}
        self.max_lost = max_lost
        self.activation_frames = activation_frames
        self.max_tracks = max_tracks
        self.height_smoothing = height_smoothing  # Puffer für die Höhe
        self.smoothing_window = smoothing_window  # Anzahl der letzten Frames, die für den Durchschnitt verwendet werden
        self.smoothing_buffers = {}

    """Die Tracker Klasse enthält eine Liste aller Tracks(bei Abgabe zwei ist dies maximal einer),
    In dieser Methode wird eine Boundingbox und ein Histogramm mitgegeben und daraus ein Track erstellt der dann zur
    Liste hinzugefügt werden kann"""
    def add_track(self, bbox,hist):
        track_id = self.next_id
        self.tracks[self.next_id] = {
            "bbox": tuple(map(int, bbox)),
            "lost": 0,
            "stable_frames": 0,
            "active": False,
            "height_history": [],
            "center_predicter":kalman_filter.KalmanFilter(),
            "prediction" : tuple([0,0,0,0]),
            "previous_center":tuple([0,0]),
            "histogram":[hist]
        }
        self.smoothing_buffers[track_id] = []
        self.next_id += 1

    """Diese Methode hat für alle tracks eine Liste der Vergangenen Boundingboxen verwaltet und daraus
    eine durchschnits Box ersttellt, welche genutzt werden konnte  um die Box flüssiger zu bewegen"""
    def apply_smoothing(self, track_id, bbox):
        if track_id not in self.smoothing_buffers:
            return bbox

        self.smoothing_buffers[track_id].append(bbox)

        if len(self.smoothing_buffers[track_id]) > self.smoothing_window:
            self.smoothing_buffers[track_id].pop(0)

        avg_bbox = np.mean(self.smoothing_buffers[track_id], axis=0).astype(int)

        return tuple(avg_bbox)

    """Diese Methode itteriert Anfangs durch alle Tracks und filter die heraus die seit längeren nicht mehr zugeordnet werden konnten.
    Im nächsten abschnitt wird dann für jeden Track alle erkannten Personen durchsucht und alle passenden
    (in der Nähe der letzten gefunden Bounding Box oder der Prediction und passendes Histogramm) abgespeichert.
    Anschließend werden alle abgespeicherten Personen für den jeweiligen Track nach der am besten passenden Sortiert,
    und dann die Boundingbox übernommen
    Anschließend wird dann für jeden Track die Prediction noch geupdated.
    Das histogramm eines Tracks wird nur angepasst wenn die änderungen zum durchschnitts Histogramm nicht zu groß sind.
    (da sonst von einer Störung ausgegangen wird"""

    def update_track(self, detections,detection_areas_histogram):
        update_tracks = {}
        for track_id, track in self.tracks.items():
            track["lost"] += 1
            if track["lost"] <= self.max_lost:
                update_tracks[track_id] = track

        self.tracks = update_tracks

        for track_id, track in self.tracks.items():
            possible = []
            for det,det_area_hist in zip(detections,detection_areas_histogram):
                if self.is_close_or_overlap(track["bbox"], det) or self.is_close_or_overlap(track["prediction"], det):
                    cmp = self.compare_histogramm(det_area_hist,track)
                    bundle = det[2]>det[3]/2
                    if cmp > 0.6 or bundle:                  # Magic Number Wann eine Box als gleiche person erkannt wird
                        possible.append((det,cmp,bundle,det_area_hist))
            if len(possible) > 0:
                pos = max(possible,key=lambda x: x[1])
                det,cmp,bundle,det_hist = pos
                track["bbox"] = det
                track["lost"] = 0
                track["stable_frames"] += 1
                if not bundle and cmp > 0.75:                 # Magic Number Wann man Histogramm hinzufügt
                    self.add_histogramm(det_hist, track)
                if track["stable_frames"] >= self.activation_frames:
                    track["active"] = True
            self.predict_future_bbox(track)
        if len(self.tracks) < self.max_tracks and len(detections) > 0:
            self.add_track(detections[0],detection_areas_histogram[0])

    """Fügt der Liste der gespeicherten Histogramme für einen Track ein weiters hinzu und entfernt falls nötig eines"""
    def add_histogramm(self,hist, track):
        track["histogram"].append(hist)
        if len(track["histogram"])>=20:                       # Magic Number Menge der genutzten Histogramme
            track["histogram"].pop(0)

    """Vergleicht die drei Farbchannel der Histogramme und gibt die Durchschnittliche übereinstimmung zurück"""
    def compare_histogramm(self,det_area_hist,track):
        val = []
        for det_hist,track_hist in zip(det_area_hist,np.mean(track["histogram"],axis=0)):
            val.append(cv2.compareHist(det_hist,track_hist,cv2.HISTCMP_CORREL))
        return np.mean(val)

    """Erstellt ein Histogramm für eine Person
    hierbei wird für jeden Channel (Blau,Grün,Rot) ein eigenes erstellt.
    Um möglichst wenig Hintergrundpixel mit aufzunehmen, wird eine Maske der Person,#
     welche aus der Backgroundsubstraction entzogen wird genutzt """
    def get_hist(self,segment,mask):
        hist_b = cv2.calcHist([segment], [0], mask, [127], [1, 256])
        hist_g = cv2.calcHist([segment], [1], mask, [127], [1, 256])
        hist_r = cv2.calcHist([segment], [2], mask, [127], [1, 256])
        cv2.normalize(hist_b,hist_b,0,255,cv2.NORM_MINMAX)
        cv2.normalize(hist_g,hist_g,0,255,cv2.NORM_MINMAX)
        cv2.normalize(hist_r,hist_r,0,255,cv2.NORM_MINMAX)

        return [hist_b, hist_g, hist_r]

    """Liefert alle Tracks aus der Liste der Klasse zurück die als Activ mackiert worden sind zurück,
    dies Stellt die getrackten Personen dar"""
    def get_active_tracks(self):
        return {tid: t for tid, t in self.tracks.items() if t["active"]}

    """wurde genutzt um die BOundingboxen flüßiger zwischen frames zu bewegen"""
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

    """Errechnet Center der Boundingbox und übergibt dies den Kalmanfilter,
     um mithilfe dessen eine Zukünftige Boundingbox zu Predicten,
     als Breite und höhe für diese Boundingbox werden die Werte der letzten gefundenen Boundingbox genutzt.
     Sollte Im letzten frame keine Passende Person Detectiert worden sein (Track["Lost"]>0) wird die Letzte prediction weiter genutzt
     """
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


    """Dient als Alternative zum Kalmanfilter.
    Hierbei wird einfach nur die Distanz der zentren genutzt um eine Richtung zu errechnen
     und die neue Boundingbox in diese zu verschieben"""
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


    """Gleiche Methode wie beim detector"""
    def is_close_or_overlap(self,bbox1, bbox2, threshold=50):           #Treshhold ist Magic Number für nähe der Boxen
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Rechtecke erweitern (Threshold) für "Nähe"
        extended_bbox1 = (x1 - threshold, y1 - 2*threshold, w1 + 2 * threshold, h1 + 4 * threshold)

        # Prüfen, ob bbox2 innerhalb der erweiterten bbox1 liegt
        return not (
                x2 + w2 < extended_bbox1[0] or  # bbox2 rechts von bbox1
                x2 > extended_bbox1[0] + extended_bbox1[2] or  # bbox2 links von bbox1
                y2 + h2 < extended_bbox1[1] or  # bbox2 unterhalb von bbox1
                y2 > extended_bbox1[1] + extended_bbox1[3]  # bbox2 oberhalb von bbox1
        )
