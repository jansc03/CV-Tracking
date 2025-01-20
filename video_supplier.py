import numpy as np
import cv2

class VideoSupplier:
    def __init__(self):
        self.vids = [
            "Vid/2_Personen.mp4",                               #0
            "Vid/2_Personen_init_erschweren.mp4",               #1
            "Vid/2_Personen_spiel_sim.mp4",                     #2
            "Vid/3_Personen.mp4",                               #3
            "Vid/3_Personen_Hindernis.mp4",                     #4
            "Vid/4_Personen.mp4",                               #5
            "Vid/4_Personen_rein_raus.mp4",                     #6
            "Vid/4_Personen_v2.mp4",                            #7
            "Vid/5_Personen.mp4",                               #8
            "Vid/6_Personen.mp4",                               #9
            "Vid/6_Personen_v2.mp4",                            #10
            "Vid/6_Personen_v3.mp4",                            #11
            "Vid/7_Personen.mp4",                               #12
            "Vid/Eingang_2_Personen_nah.mp4",                   #13
            "Vid/Eingang_2_Personen_nah_v2.mp4",                #14
            "Vid/Eingang_2_Personen_nah_v3.mp4",                #15
            "Vid/Eingang_2_Personen_ueberholen.mp4",            #16
            "Vid/Eingang_2_Personen_viele_Personen_im_hg.mp4",  #17
            "Vid/Eingang_4_Personen_Polonaise.mp4",             #18
            "Vid/Eingang_mir_wird_schwindelig_huiiiiiii.mp4",   #19
            "Vid/Eingang_Nacheinander.mp4",                     #20 ZEIGEN
            "Vid/Session2",                                     #21
            "Vid/Session2_2_Personen_hinderniss.mp4",           #22
            "Vid/Session2_2_Personen_ueberlappen.mp4",          #23
            "Vid/Session2_2_Personen_verdecken.mp4",            #24
            "Vid/Session2_3_Personen_cross.mp4",                #25
            "Vid/Session2_3_Personen_ueberlappen.mp4",          #26
            "Vid/Session2_3_Personen_ueberlappen_v2.mp4",       #27
            "Vid/Session2_3_Personen_wechseln.mp4",             #28
            "Vid/Session2_4_Personen_ueberlappen.mp4",          #29
            "Vid/Session2_4_Personen_ueberlappen (2).mp4",      #30
            "Vid/",
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
