import numpy as np
import cv2

class VideoSupplier:
    def __init__(self):
        self.vids = [
            "Vid/Links-Rechts bewegung mit Helligkeitsunterschied.mp4",
            "Vid/Schwer_weißer Hintergund mit weißen klamotten.mp4",
            "Vid/WIN_20241113_14_28_28_Pro.mp4",
            "Vid/film.mov"
        ]
        self.caps = []  # Liste der VideoCapture-Objekte
        self.target_width = 1280  # Zielbreite
        self.target_height = 720   # Zielhöhe

    def clear_caps(self):
        for cap in self.caps:
            if cap.isOpened():
                cap.release()
        self.caps.clear()

    """Startet ein einzelnes Video Capture"""
    def getSingleVideo(self, vidNum=0):
        self.clear_caps()  # Schließt vorherige Videos
        cap = cv2.VideoCapture(self.vids[vidNum])
        self.caps.append(cap)

    """Liefert mehrere Video Captures"""
    def getMultiVideo(self):
        self.clear_caps()  # Schließt vorherige Videos
        for vid in self.vids:
            cap = cv2.VideoCapture(vid)
            self.caps.append(cap)

    """Liefert den nächsten Frame aus den einzelnen oder von den mehreren in einen 2x2 Frame
    sollte ein Video schneller enden als andere wird für dieses ein schwarzes Bild erstellt"""
    def getNextFrame(self):
        frames = []
        if(len(self.caps) > 1):
            for cap in self.caps:
                ret, frame = cap.read()
                if ret:
                    resized_frame = cv2.resize(frame, (self.target_width, self.target_height))
                    frames.append(resized_frame)
                else:
                    frames.append(np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8))

            while len(frames) < 4:
                frames.append(np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8))

            top_row = np.hstack((frames[0], frames[1]))
            bottom_row = np.hstack((frames[2], frames[3]))
            grid_frame = np.vstack((top_row, bottom_row))
        else:
            for cap in self.caps:
                ret, frame = cap.read()
                if ret:
                    grid_frame = cv2.resize(frame, (self.target_width, self.target_height))
        return grid_frame