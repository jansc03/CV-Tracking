import video_supplier as vs
import numpy as np
import cv2




class BackgroundSubtraction:
    # Video Parameter
    target_width = 1280  # Zielbreite
    target_height = 720  # Zielhöhe

    backSubMOG2 = cv2.createBackgroundSubtractorMOG2(history=400, detectShadows=True, varThreshold=150)
    backSubMOG2.setNMixtures(5)
    backSubMOG2.setBackgroundRatio(0.75)
    backSubMOG2.setComplexityReductionThreshold(0.05)
    backSubMOG2.setShadowThreshold(0.8)
    backSubMOG2.setShadowValue(0)
    backSubMOG2.setVarInit(15)

    backSubKNN = cv2.createBackgroundSubtractorKNN(history=500, detectShadows=False,
                                                   dist2Threshold=300.0)  # unten Rechts
    backSubKNN.setNSamples(120)
    backSubKNN.setkNNSamples(7)
    backSubKNN.setShadowThreshold(60)


    # Hintergrundsubtraktor-Liste
    backSubtractor = [backSubMOG2, backSubKNN]
    method_names = ["MOG2", "KNN", "CNT", "GSOC"]

    # Globale Variablen
    backSub = None
    videoSupplier = vs.VideoSupplier()

    """Gibt den Nackground Subtractor Informationen welches vefahren getestet werden soll oder welches video genutzt werden soll bzw mehrere videos"""
    def initBackgroundSubtractor(self,backSubNum=0, multi=False,vidNum = 0):
        self.backSub = self.backSubtractor[backSubNum]
        if multi:
            self.videoSupplier.getMultiVideo()
        else:
            self.videoSupplier.getSingleVideo(vidNum)

    """Eine Maske auf ein Video"""
    def getNextSingleBackground(self):
        cameraFrame = self.videoSupplier.getNextFrame()
        fgMask = self.backSub.apply(cameraFrame)
        return fgMask,cameraFrame


    """Mehrere Masken auf ein Video + Beschriftung"""
    def getNextCombinedBackground(self):
        cameraframe = self.videoSupplier.getNextFrame()

        frame = cv2.resize(cameraframe, (self.target_width, self.target_height))

        masks = []

        for i in range(4):
            mask = self.backSubtractor[i].apply(frame)
            cv2.putText(mask, self.method_names[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            mask = cv2.flip(mask, 1)
            masks.append(mask)

        top_row = np.hstack((masks[0], masks[1]))
        bottom_row = np.hstack((masks[2], masks[3]))
        combined_grid = np.vstack((top_row, bottom_row))

        combined_grid = cv2.resize(combined_grid, (self.target_width, self.target_height))

        return combined_grid

    """Ein Verfahren auf mehreren Videos"""
    def getNextMultipleBackground(self):
        cameraFrame = self.videoSupplier.getNextFrame()
        fgMask = self.backSub.apply(cameraFrame)
        fgMask = cv2.resize(fgMask, (self.target_width, self.target_height))
        return fgMask
    def closeAll(self):
        self.videoSupplier.clear_caps()