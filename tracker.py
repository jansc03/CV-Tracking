import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from kalman_filter import KalmanFilter

from iou import calculate_iou

class Tracker:
    def __init__(self, max_lost=60, activation_frames=0, max_tracks=3, height_smoothing=20, smoothing_window=4):
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
            "group_bbox": tuple([0,0,0,0]),
            "in_group": False,
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

    def update_track(self, detections, detection_areas_histogram):

        wide_detections = []
        wide_detections_hist = []
        filtered_detections = []
        filtered_histograms = []

        """Filtern von Wahrscheinlich mehreren Personen"""
        for det, det_hist in zip(detections, detection_areas_histogram):
            if det[2] > det[3] / 1.5:  # Breite > Teil der Höhe
                wide_detections.append(det)
                wide_detections_hist.append(det_hist)
            else:
                filtered_detections.append(det)
                filtered_histograms.append(det_hist)

        detections = filtered_detections
        detection_areas_histogram = filtered_histograms

        # Tracks als Liste
        track_list = list(self.tracks.values())
        track_ids = list(self.tracks.keys())


        """Hier werden die Tracks den Detektionen zugeordnet """
        matches, unmatched_tracks, unmatched_detections = self.associate_detections_to_tracks(
            detections, detection_areas_histogram, track_list
        )

        """Die gefundenen Matches zwischen Detektion und Tracks werden verarbeiten => Track Daten werden aktualisiert"""
        # Matches verarbeiten
        for track_idx, detection_idx in matches:
            track_id = track_ids[track_idx]
            detection = detections[detection_idx]
            detection_hist = detection_areas_histogram[detection_idx]

            track = self.tracks[track_id]
            track["bbox"] = detection
            track["in_group"] = False
            track["lost"] = 0
            track["stable_frames"] += 1

            # Histogramm hinzufügen, wenn Schwellenwert überschritten
            if self.compare_histogramm(detection_hist, track) > 0.75:  # Magic Number
                self.add_histogramm(detection_hist, track)

            # Track aktivieren
            if track["stable_frames"] >= self.activation_frames:
                track["active"] = True

        """Sollten Tracks nicht zugeordnet werden können wird geschaut ob sich in ihrer nähe eine Gruppe Befindet"""
        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            track = self.tracks[track_id]

            for det in wide_detections:
                if (self.is_close_or_overlap(track["bbox"], det,0,0) or
                        self.is_close_or_overlap(track["group_bbox"],det,0,0)):  # Überprüfen auf iou
                    track["group_bbox"] = det
                    track["lost"] = 0
                    track["in_group"] = True
                    track["stable_frames"] += 1
                    unmatched_tracks.remove(track_idx)
                    break

        """SOllten abschließend immernoch tracks nicht zugeordnet werden können wird hier nochmal geschaut ob dieser gerade in einer gruppe war
         und wenn dies der Fall ist und nun an der Stelle ein Tracks ist wird vermutet das diese gruppe sich verkleinert hat/ einer vor den anderen gegangen ist"""
        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            track = self.tracks[track_id]

            if(track["lost"]<2):
                for det in detections:
                    if (calculate_iou(track["group_bbox"],det) > 0.3):  # Überprüfen auf iou
                        track["group_bbox"] = det
                        track["lost"] = 0
                        track["in_group"] = True
                        track["stable_frames"] += 1
                        unmatched_tracks.remove(track_idx)
                        break


        # Unmatched Tracks verarbeiten (als verloren markieren)
        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            self.tracks[track_id]["lost"] += 1
            self.tracks[track_id]["in_group"] = False

        # Verlorene Tracks löschen
        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if track["lost"] <= self.max_lost
        }

        # Unmatched Detektionen verarbeiten (neue Tracks hinzufügen)
        if len(self.tracks) < self.max_tracks and len(unmatched_detections) > 0:
            to_add = self.get_addable_detection_id_from_leftover(
                [detections[idx] for idx in unmatched_detections]
            )
            if to_add != -1:
                actual_index = unmatched_detections[to_add]
                self.add_track(detections[actual_index], detection_areas_histogram[actual_index])

        for track in self.tracks.values():
            self.predict_future_bbox(track)

    """Ordnet Detektionen den bestehenden Tracks zu basierend auf Hungarian Algorithmus"""
    def associate_detections_to_tracks(self,detections, detection_areas_histogram, tracks, cost_threshold=0.29): #magic number
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Kostenmatrix berechnen
        cost_matrix = np.zeros((len(tracks), len(detections)))

        for t, track in enumerate(tracks):
            for d, (detection, detection_hist) in enumerate(zip(detections, detection_areas_histogram)):
                cost_matrix[t, d] = self.calculate_matching_value(track, detection, detection_hist)

        # Hungarian Algorithmus anwenden
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_detections = list(range(len(detections)))

        for t, d in zip(track_indices, detection_indices):
            # Zusätzliche Überprüfung: Nähe von Track und Detektion
            track_bbox = tracks[t]["bbox"]
            detection_bbox = detections[d]

            # Überprüfung der Kosten und Nähe
            if cost_matrix[t, d] < cost_threshold:
                matches.append((t, d))
                unmatched_tracks.remove(t)
                unmatched_detections.remove(d)

        return matches, unmatched_tracks, unmatched_detections

    def calculate_matching_value(self,track, detection, detection_hist):

        histogram_similarity = self.compare_histogramm(detection_hist, track)

        bbox_overlap = self.is_close_or_overlap(track["bbox"], detection,0,0)
        prediction_overlap = self.is_close_or_overlap(track["prediction"], detection,0,0)
        if(track["in_group"]):
            bbox_overlap = 0.5*self.is_close_or_overlap(track["group_bbox"], detection,0,0)

        overlap = prediction_overlap
        if(bbox_overlap > prediction_overlap):
            overlap = bbox_overlap

        print(histogram_similarity)

        cost = 1 - (
                0.7 * histogram_similarity +  # Histogramm: 70%
                0.3 * overlap
        )
        return cost

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
        # Konvertiere Segment auf HSV
        hsv_segment = cv2.cvtColor(segment, cv2.COLOR_BGR2HSV)

        # Erstelle Histogramme für H, S und V
        hist_h = cv2.calcHist([hsv_segment], [0], mask, [127], [0, 180])
        hist_s = cv2.calcHist([hsv_segment], [1], mask, [127], [0, 256])
        hist_v = cv2.calcHist([hsv_segment], [2], mask, [127], [0, 256])

        # Normalisiere Histogramme
        cv2.normalize(hist_h, hist_h, 0, 255, cv2.NORM_MINMAX)
        cv2.normalize(hist_s, hist_s, 0, 255, cv2.NORM_MINMAX)
        cv2.normalize(hist_v, hist_v, 0, 255, cv2.NORM_MINMAX)

        return [hist_h, hist_s, hist_v]

    """Vergleicht die drei Farbchannel der Histogramme und gibt die Durchschnittliche übereinstimmung zurück"""

    def compare_histogramm(self, det_area_hist, track):
        upper_hist_det, lower_hist_det = det_area_hist
        upper_hist_track, lower_hist_track = np.mean(track["histogram"], axis=0)

        # Berechne Ähnlichkeit für H, S und V
        def weighted_similarity(det, track, weights):
            return sum(
                weights[i] * cv2.compareHist(det[i], track[i], cv2.HISTCMP_CORREL)
                for i in range(3)
            )

        # Gewichtung: H, dann S, dann V
        weights = [0.7, 0.3, 0.2]

        upper_similarity = weighted_similarity(upper_hist_det, upper_hist_track, weights)
        lower_similarity = weighted_similarity(lower_hist_det, lower_hist_track, weights)

        # Kombiniere obere und untere Ähnlichkeit
        return 0.7 * upper_similarity + 0.3 * lower_similarity

    """Liefert alle Tracks aus der Liste der Klasse zurück die als Activ mackiert worden sind zurück,
    dies Stellt die getrackten Personen dar"""

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
