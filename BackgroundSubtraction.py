import VideoSupplier as vs
import numpy as np
import cv2




class BackgroundSubtraction:
    # Video Parameter
    target_width = 1280  # Zielbreite
    target_height = 720  # Zielhöhe

    backSubMOG2 = cv2.createBackgroundSubtractorMOG2(history=400, detectShadows=True, varThreshold=100)
    backSubMOG2.setNMixtures(5)
    backSubMOG2.setBackgroundRatio(0.7)
    backSubMOG2.setComplexityReductionThreshold(0.05)
    backSubMOG2.setShadowThreshold(0.8)
    backSubMOG2.setShadowValue(127)
    backSubMOG2.setVarInit(15)

    backSubKNN = cv2.createBackgroundSubtractorKNN(history=200, detectShadows=False, dist2Threshold=300.0)
    backSubCNT = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=10, useHistory=True,
                                                          maxPixelStability=15 * 15)
    backSubGMG = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=60, decisionThreshold=0.9)
    backSubGSOC = cv2.bgsegm.createBackgroundSubtractorGSOC(mc=10, nSamples=10, replaceRate=0.005, propagationRate=0.01,
                                                            hitsThreshold=32, alpha=0.01, beta=0.01,
                                                            blinkingSupressionDecay=0.1,
                                                            blinkingSupressionMultiplier=0.1,
                                                            noiseRemovalThresholdFacBG=0.0004,
                                                            noiseRemovalThresholdFacFG=0.008)
    backSubLSBP = cv2.bgsegm.createBackgroundSubtractorLSBP(nSamples=10, mc=10)

    # Hintergrundsubtraktor-Liste
    backSubtractor = [backSubMOG2, backSubKNN, backSubCNT, backSubGMG, backSubGSOC, backSubLSBP]

    # Globale Variablen
    backSub = None
    videoSupplier = vs.VideoSupplier()

    def initBackgroundSubtractor(self,backSubNum=0, multi=False):
        self.backSub = self.backSubtractor[backSubNum]
        if multi:
            self.videoSupplier.getMultiVideo()
        else:
            self.videoSupplier.getSingleVideo()

    def getNextSingleBackground(self):
        cameraFrame = self.videoSupplier.getNextFrame()
        fgMask = self.backSub.apply(cameraFrame)
        return fgMask

    def getNextMultipleBackground(self):
        cameraFrame = self.videoSupplier.getNextFrame()
        fgMask = self.backSub.apply(cameraFrame)
        fgMask = cv2.resize(fgMask, (self.target_width, self.target_height))
        return fgMask
    def closeAll(self):
        self.videoSupplier.clear_caps()