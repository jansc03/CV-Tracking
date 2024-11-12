import numpy as np
import cv2

class VideoSupplier:
    def __init__(self):
        # Liste der Videopfad-Dateien
        self.vids = [
            "Vid/Links-Rechts bewegung mit Helligkeitsunterschied.mp4",
            "Vid/Schwer_weißer Hintergund mit weißen klamotten.mp4",
            "Vid/Vorne-Hinten.mp4",
            "Vid/film.mov"
        ]
        self.caps = []  # Liste der VideoCapture-Objekte
        self.target_width = 1280  # Zielbreite
        self.target_height = 720   # Zielhöhe

    def clear_caps(self):
        """Beendet und löscht alle VideoCapture-Objekte."""
        for cap in self.caps:
            if cap.isOpened():
                cap.release()
        self.caps.clear()

    def getSingleVideo(self, vidNum=0):
        """Öffnet ein einzelnes Video und speichert es in der caps-Liste."""
        self.clear_caps()  # Schließt vorherige Videos
        cap = cv2.VideoCapture(self.vids[vidNum])
        self.caps.append(cap)

    def getMultiVideo(self):
        """Öffnet alle Videos in der vids-Liste und speichert sie in der caps-Liste."""
        self.clear_caps()  # Schließt vorherige Videos
        for vid in self.vids:
            cap = cv2.VideoCapture(vid)
            self.caps.append(cap)

    def getNextFrame(self):
        """Holt den nächsten Frame von jedem Video, passt ihn auf 1280x720 an und arrangiert ihn in einem 2x2-Grid."""
        frames = []
        for cap in self.caps:
            ret, frame = cap.read()
            if ret:
                # Frame auf Zielgröße 1280x720 anpassen
                resized_frame = cv2.resize(frame, (self.target_width, self.target_height))
                frames.append(resized_frame)
            else:
                # Wenn kein Frame mehr verfügbar ist, wird ein schwarzer Frame hinzugefügt
                frames.append(np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8))

        # Sicherstellen, dass wir genau 4 Frames für das 2x2-Grid haben
        while len(frames) < 4:
            frames.append(np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8))

        # Frames in einem 2x2-Grid ohne zusätzliche Größenanpassung arrangieren
        top_row = np.hstack((frames[0], frames[1]))
        bottom_row = np.hstack((frames[2], frames[3]))
        grid_frame = np.vstack((top_row, bottom_row))

        return grid_frame