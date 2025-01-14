import cv2
import numpy as np
from kalman_filter import KalmanFilter

class Tracker:
    def __init__(self, max_lost=60, activation_frames=0, max_tracks=2, height_smoothing=20, smoothing_window=4):
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
            "center_predicter":KalmanFilter(track_id),
            "prediction" : tuple([0,0,0,0]),
            "previous_center":tuple([0,0]),
            "histogram":[hist]
        }
        self.smoothing_buffers[track_id] = []
        self.next_id += 1

    """Diese Methode itteriert Anfangs durch alle Tracks und filter die heraus die seit längeren nicht mehr zugeordnet werden konnten.
    Im nächsten abschnitt wird dann für jeden Track alle erkannten Personen durchsucht und alle passenden
    (in der Nähe der letzten gefunden Bounding Box oder der Prediction und passendes Histogramm) abgespeichert.
    Anschließend werden alle abgespeicherten Personen für den jeweiligen Track nach der am besten passenden Sortiert,
    und dann die Boundingbox übernommen
    Anschließend wird dann für jeden Track die Prediction noch geupdated.
    Das histogramm eines Tracks wird nur angepasst wenn die änderungen zum durchschnitts Histogramm nicht zu groß sind.
    (da sonst von einer Störung ausgegangen wird"""

    def update_track(self, detections, detection_areas_histogram):
        update_tracks = {}
        for track_id, track in self.tracks.items():
            track["lost"] += 1
            if track["lost"] <= self.max_lost:
                update_tracks[track_id] = track

        self.tracks = update_tracks

        for track_id, track in self.tracks.items():
            possible = []
            for det, det_area_hist in zip(detections, detection_areas_histogram):
                if self.is_close_or_overlap(track["bbox"], det) or self.is_close_or_overlap(track["prediction"], det):
                    cmp = self.compare_histogramm(det_area_hist, track)
                    if cmp > 0.7:  # Threshold for matching
                        possible.append((det, cmp, det_area_hist))

            if len(possible) > 0:
                pos = max(possible, key=lambda x: x[1])
                det, cmp, det_hist = pos
                track["bbox"] = det
                track["lost"] = 0
                track["stable_frames"] += 1
                if cmp > 0.75:
                    self.add_histogramm(det_hist, track)
                if track["stable_frames"] >= self.activation_frames:
                    track["active"] = True

                i = detections.index(det)
                del detections[i]
                del detection_areas_histogram[i]

            self.predict_future_bbox(track)

        if len(self.tracks) < self.max_tracks and len(detections) > 0:
            to_add = self.get_addable_detection_id_from_leftover(detections)
            if to_add != -1:
                self.add_track(detections[to_add], detection_areas_histogram[to_add])

    def get_addable_detection_id_from_leftover(self, detections):
        possible = []
        for det_id, det in enumerate(detections):
            if det[2] < det[3]/2 and (det[0] < 200 or det[0] > 1000):  #keine sehr beiten detectionen werden hinzugefügt
                possible.append(det_id)                                #+ nur am linken und rechtren rand werden neue akzeptiert

        for id in possible:
            for x,track in self.tracks.items():
                if self.is_close_or_overlap(track["bbox"], detections[id],0,200):
                    possible.remove(id)
                    break

        if(len(possible) > 0):
            return possible[0]
        else:
            return -1

    """Fügt der Liste der gespeicherten Histogramme für einen Track ein weiters hinzu und entfernt falls nötig eines"""
    def add_histogramm(self,hist, track):
        track["histogram"].append(hist)
        if len(track["histogram"])>=20:                       # Magic Number Menge der genutzten Histogramme
            track["histogram"].pop(0)

    """Vergleicht die drei Farbchannel der Histogramme und gibt die Durchschnittliche übereinstimmung zurück"""
    def compare_histogramm(self, det_area_hist, track):
        upper_hist_det, lower_hist_det = det_area_hist
        upper_hist_track, lower_hist_track = np.mean(track["histogram"], axis=0)

        upper_similarity = np.mean([
            cv2.compareHist(upper_hist_det[i], upper_hist_track[i], cv2.HISTCMP_CORREL)
            for i in range(3)
        ])

        lower_similarity = np.mean([
            cv2.compareHist(lower_hist_det[i], lower_hist_track[i], cv2.HISTCMP_CORREL)
            for i in range(3)
        ])

        # Combine similarities with a weighted average
        return 0.8 * upper_similarity + 0.4 * lower_similarity


    """Erstellt ein Histogramm für eine Person
    hierbei wird für jeden Channel (Blau,Grün,Rot) ein eigenes erstellt.
    Um möglichst wenig Hintergrundpixel mit aufzunehmen, wird eine Maske der Person,#
     welche aus der Backgroundsubstraction entzogen wird genutzt """
    def get_hist(self, segment, mask):
        # Split the segment into upper and lower
        height, width = segment.shape[:2]
        mid_y = height // 2

        upper_segment = segment[:mid_y, :]
        lower_segment = segment[mid_y:, :]

        upper_mask = mask[:mid_y, :]
        lower_mask = mask[mid_y:, :]

        # Calculate histograms for upper and lower segments
        upper_hist = self.calc_hist(upper_segment, upper_mask)
        lower_hist = self.calc_hist(lower_segment, lower_mask)

        return [upper_hist, lower_hist]

    def calc_hist(self, segment, mask):
        hist_b = cv2.calcHist([segment], [0], mask, [127], [1, 256])
        hist_g = cv2.calcHist([segment], [1], mask, [127], [1, 256])
        hist_r = cv2.calcHist([segment], [2], mask, [127], [1, 256])

        cv2.normalize(hist_b, hist_b, 0, 255, cv2.NORM_MINMAX)
        cv2.normalize(hist_g, hist_g, 0, 255, cv2.NORM_MINMAX)
        cv2.normalize(hist_r, hist_r, 0, 255, cv2.NORM_MINMAX)

        return [hist_b, hist_g, hist_r]

    """Liefert alle Tracks aus der Liste der Klasse zurück die als Activ mackiert worden sind zurück,
    dies Stellt die getrackten Personen dar"""
    def get_active_tracks(self):
        return {tid: t for tid, t in self.tracks.items() if t["active"]}

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
            track_x = max(min(track_prediction[0], 10000),-1000)
            track_y = max(min(track_prediction[1], 10000),-1000)
            track_last_bbox_size = track["bbox"]
            bbox_width = track_last_bbox_size[2]
            bbox_height = track_last_bbox_size[3]
            predicted_center = track["center_predicter"].predict(
                track_x + bbox_width // 2,
                track_y + bbox_height // 2
            )
            new_bbox_x = int(predicted_center[0] - bbox_width // 2)
            new_bbox_y = int(track_prediction[1] - bbox_height // 2)

            track["prediction"] = (new_bbox_x, track_y, bbox_width, bbox_height)


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
    def is_close_or_overlap(self,bbox1, bbox2, threshold1=50,threshold2 = 100):           #Treshhold ist Magic Number für nähe der Boxen
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Rechtecke erweitern (Threshold) für "Nähe"
        extended_bbox1 = (x1 - threshold1, y1 - threshold2, w1 + 2 * threshold1, h1 + 2 * threshold2)

        # Prüfen, ob bbox2 innerhalb der erweiterten bbox1 liegt
        return not (
                x2 + w2 < extended_bbox1[0] or  # bbox2 rechts von bbox1
                x2 > extended_bbox1[0] + extended_bbox1[2] or  # bbox2 links von bbox1
                y2 + h2 < extended_bbox1[1] or  # bbox2 unterhalb von bbox1
                y2 > extended_bbox1[1] + extended_bbox1[3]  # bbox2 oberhalb von bbox1
        )
