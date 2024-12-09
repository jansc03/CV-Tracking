import numpy as np
import cv2



class Detector:
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        return

    def detect(self, bgImg):

        contours, hierarchy = cv2.findContours(bgImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        people = self._filter(contours)
        return people

    def _filter(self, contours, padding=5):
        low_potential_contour_area = 1000
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
            if 0.2 < aspect_ratio < 1.3 and cnt_area > min_contour_area:
                potentialPerson.append((x - padding, y - padding, w + 2 * padding, h + 2 * padding))

        return potentialPerson, all_contours


    # Kombiniert 2 BoundingBoxen
    def merge_bounding_boxes(self, bboxes):
        x_min = min([bbox[0] for bbox in bboxes])
        y_min = min([bbox[1] for bbox in bboxes])
        x_max = max([bbox[0] + bbox[2] for bbox in bboxes])
        y_max = max([bbox[1] + bbox[3] for bbox in bboxes])
        return x_min, y_min, x_max - x_min, y_max - y_min



    # Funktion zur Prüfung, ob zwei BoundingBoxen überlappen oder nahe beieinanderliegen
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

    def extract_person_areas(self,frame,backframe, people):
        person_areas = []
        height, width, _ = frame.shape

        for x, y, w, h in people:
            x_end = min(x + w, width)
            y_end = min(y + h, height)
            x_start = max(x, 0)
            y_start = max(y, 0)

            if (x_end > x_start) and (y_end > y_start):
                person_area = frame[y_start:y_end, x_start:x_end, :]
                person_background_area = backframe[y_start:y_end, x_start:x_end]
                if person_area.size > 0:
                    person_background_area = cv2.cvtColor(person_background_area, cv2.COLOR_GRAY2BGR)
                    person = cv2.bitwise_and(person_area,person_background_area)
                    person_areas.append(person)
        return person_areas
