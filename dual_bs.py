import cv2
import numpy as np
import video_supplier as vs
class DualBackgroundSubtraction:
    # Video Parameter
    target_width = 1280
    target_height = 720

    # Zwei Hintergrundsubtraktoren
    def __init__(self):
        self.videoSupplier = vs.VideoSupplier()
        self.tracked_objects = []  # Liste der getrackten Objekte (Bounding Boxes)
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=400, detectShadows=True, varThreshold=150)
        self.backSub.setNMixtures(5)
        self.backSub.setBackgroundRatio(0.75)
        self.backSub.setComplexityReductionThreshold(0.05)
        self.backSub.setShadowThreshold(0.8)
        self.backSub.setShadowValue(0)
        self.backSub.setVarInit(15)

    def init_video(self, vidNum=0):
        self.videoSupplier.getSingleVideo(vidNum)

    def get_mask(self):
        """Erzeugt die kombinierte Maske aus beiden Hintergrundsubtraktionen."""
        frame = self.videoSupplier.getNextFrame()
        frame = cv2.resize(frame, (self.target_width, self.target_height))

        # Hintergrundsubtraktion 1 (normale Subtraktion)
        mask1 = self.backSub.apply(frame, learningRate=0)

        # Hintergrundsubtraktion 2 (mit ignorierten Bereichen)
        mask = np.ones_like(mask1, dtype=np.uint8) * 255
        for (x, y, w, h) in self.tracked_objects:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)  # Maske über getrackten Bereichen
        mask2 = self.backSub.apply(frame)
        mask2 = cv2.bitwise_and(mask2, mask2, mask=mask)  # Ignorierte Bereiche entfernen

        # Kombinierte Maske erstellen
        combined_mask = cv2.bitwise_or(mask1, mask2)
        return combined_mask, frame

    def update_tracked_objects(self, objects):
        """Aktualisiert die Liste der getrackten Objekte."""
        self.tracked_objects = objects

    def close(self):
        self.videoSupplier.clear_caps()