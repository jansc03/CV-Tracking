import numpy as np
import cv2

class VideoSupplier:
    def __init__(self):
        self.vids = [
            "Vid/daraussen.mp4",                      #0 Perfekt
            "Vid/gegenlicht.mp4",                     #1
            "Vid/gegenlicht_und_spiegel.mp4",         #2
            "Vid/gehend_zwei.mp4",                    #3
            "Vid/schnell.mp4",                        #4
            "Vid/spiegel.mp4",                        #5
            "Vid/winkel.mp4",                         #6
            "Vid/stoerung.mp4",                       #7
            "Vid/2personen.mp4",                      #8
            "Vid/2personen_verdeckt.mp4",             #9 Hindernis
            "Vid/autos.mov",                          #10 Bewegungs im Hintergrund
            "Vid/autos2.mp4",                          #11
            "Vid/Schwer_weißer Hintergund mit weißen klamotten.mp4", #12 Mehrer Personen
            "Vid/Links-Rechts bewegung mit Helligkeitsunterschied.mp4", #13
            "Vid/Vorne-Hinten.mp4", #14
            "Vid/zuhause.mp4", #15
            "Vid/4_Personen.mp4",#16
            "Vid/3_Personen.mp4",#17
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
        if len(self.caps) > 1:
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
            grid_frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)  # Standardwert
            for cap in self.caps:
                ret, frame = cap.read()
                if ret:
                    grid_frame = cv2.resize(frame, (self.target_width, self.target_height))

        return grid_frame
