import numpy as np
import cv2



class Detector:
    def __init__(self):
        return

    """Erhält ein Backgroundimage und gibt Bounding Boxen für alle erkannten Personen zurück
    und alle Bounding Boxen von allen anfangs gefundenen Konturen, zum Visualisieren des Ablaufs"""
    def detect(self, bgImg):
        contours, hierarchy = cv2.findContours(bgImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        people = self._filter(contours)
        return people

    """In dieser Methode werden die Konturen zunächst nach ihren Flächeninhalt Sortiert.
    Dann wird geprüft ob kleinere Kunturen in der nähe von größeren sind und dann zusammengefügt,
    dies passiert mithilfe von Boundingboxen.
    Anschließend werden die Übrigen Boundingenboxen nocheinmal nach iherer Form und tatsächlichen Inhalt gefiltert.
    """
    def _filter(self, contours, padding=5):
        low_potential_contour_area = 1000   #Magic Numbers für Sortierung
        min_contour_area = 10000
        max_contour_area = 500000

        all_contours = [cv2.boundingRect(cnt) for cnt in contours if
                        max_contour_area > cv2.contourArea(cnt) > low_potential_contour_area]

        contour = [(cv2.boundingRect(cnt),cv2.contourArea(cnt)) for cnt in contours if
                   max_contour_area > cv2.contourArea(cnt) > min_contour_area]

        potentialParts = [(cv2.boundingRect(cnt),cv2.contourArea(cnt)) for cnt in contours if
                          min_contour_area > cv2.contourArea(cnt) > low_potential_contour_area]

        for pp,pa in potentialParts:
            for cp,ca in contour:
                if self.is_close_or_overlap(pp, cp):
                    merge = (np.array(pp), np.array(cp))
                    contour.append((self.merge_bounding_boxes(merge),pa+ca))
                    contour.remove((cp,ca))
                    break
        potentialPerson = []

        for cnt,cnt_area in contour:
            x, y, w, h = cnt
            aspect_ratio = float(w) / h
            if 0.2 < aspect_ratio < 1.3 and cnt_area > min_contour_area:            # Magic Number für Seitenverhältniss
                potentialPerson.append((x - padding, y - padding, w + 2 * padding, h + 2 * padding))

        return potentialPerson, all_contours

    """Diese Methode kombienirt 2 Boundingboxen zu einer großen zusammen"""
    def merge_bounding_boxes(self, bboxes):
        x_min = min([bbox[0] for bbox in bboxes])
        y_min = min([bbox[1] for bbox in bboxes])
        x_max = max([bbox[0] + bbox[2] for bbox in bboxes])
        y_max = max([bbox[1] + bbox[3] for bbox in bboxes])
        return x_min, y_min, x_max - x_min, y_max - y_min


    """Diese Methode überprüft ob zwei Boundingboxen nahe beieinander sind oder überlappen durch das Erweitern der
    Boundingboxen und dass anschließende vergleichen auf überlappungen.
    Hierbei werden die Boundingboxen in der Vertikalen doppelt so weit gestereckt wie ind der Vertikalen,
     sie dürfen also horizontal weiter auseinander liegen als vertikal"""
    def is_close_or_overlap(self,bbox1, bbox2, threshold=50):
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

    """Diese Methode gibt den Bildausschnitt zurück, der durch eine Boundingbox
     welche zu einer Person gehört, beschrieben wird"""
    def extract_person_areas(self,frame,background, people):
        person_areas = []
        height, width, _ = frame.shape

        for x, y, w, h in people:
            x_end = min(x + w, width)
            y_end = min(y + h, height)
            x_start = max(x, 0)
            y_start = max(y, 0)

            if (x_end > x_start) and (y_end > y_start):
                person_area = frame[y_start:y_end, x_start:x_end, :]
                background_frame = background[y_start:y_end, x_start:x_end]
                if person_area.size > 0:
                    person_areas.append((person_area,background_frame))
        return person_areas
