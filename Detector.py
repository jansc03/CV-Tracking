import numpy as np
import cv2



class Detector:
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return

    def detect(self, bgImg):
        bgImg = cv2.GaussianBlur(bgImg, (5, 5), 0)
        mask_eroded = cv2.morphologyEx(bgImg, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        mask_eroded = cv2.morphologyEx(mask_eroded, cv2.MORPH_OPEN, self.kernel, iterations=2)

        contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        people = self._filter(contours)
        return people

    def _filter(self, contours, padding=20):
        low_potential_contour_area = 1000
        min_contour_area = 10000
        max_contour_area = 500000

        all_contours = [cv2.boundingRect(cnt) for cnt in contours if
                        max_contour_area > cv2.contourArea(cnt) > low_potential_contour_area]

        contour = [cv2.boundingRect(cnt) for cnt in contours if
                   max_contour_area > cv2.contourArea(cnt) > min_contour_area]

        potentialParts = [cv2.boundingRect(cnt) for cnt in contours if
                          min_contour_area > cv2.contourArea(cnt) > low_potential_contour_area]

        for pp in potentialParts:
            for cp in contour:
                if self.is_close_or_overlap(pp, cp):
                    merge = (np.array(pp), np.array(cp))
                    print(merge)
                    contour.append(self.merge_bounding_boxes(merge))
                    contour.remove(cp)
        potentialPerson = []

        for x, y, w, h in contour:
            if w < h :
                # Padding hinzufügen
                x_padded = max(x - padding, 0)
                y_padded = max(y - padding, 0)
                w_padded = w + 2 * padding
                h_padded = h + 2 * padding

                potentialPerson.append((x_padded, y_padded, w_padded, h_padded))

        return potentialPerson, all_contours

    # Kombiniert 2 BoundingBoxen
    def merge_bounding_boxes(self,bboxes):
        print(bboxes)
        x_min = min([bbox[0] for bbox in bboxes])
        y_min = min([bbox[1] for bbox in bboxes])
        x_max = max([bbox[0] + bbox[2] for bbox in bboxes])
        y_max = max([bbox[1] + bbox[3] for bbox in bboxes])
        return x_min, y_min, x_max - x_min, y_max - y_min

    # Funktion zur Prüfung, ob zwei BoundingBoxen überlappen oder nahe beieinanderliegen
    def is_close_or_overlap(self,bbox1, bbox2, threshold=200):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Rechtecke erweitern (Threshold) für "Nähe"
        extended_bbox1 = (x1 - threshold, y1 - threshold, w1 + 2 * threshold, h1 + 2 * threshold)

        # Prüfen, ob bbox2 innerhalb der erweiterten bbox1 liegt
        return not (
                x2 + w2 < extended_bbox1[0] or  # bbox2 rechts von bbox1
                x2 > extended_bbox1[0] + extended_bbox1[2] or  # bbox2 links von bbox1
                y2 + h2 < extended_bbox1[1] or  # bbox2 unterhalb von bbox1
                y2 > extended_bbox1[1] + extended_bbox1[3]  # bbox2 oberhalb von bbox1
        )
