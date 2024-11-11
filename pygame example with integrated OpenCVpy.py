"""
-----------------------------------------------------------------------------
Vorlesung: Computer Vision (Wintersemester 2024/25)
Thema: pygame example with integrated OpenCV

-----------------------------------------------------------------------------
"""

import numpy as np
import cv2
import pygame


SCREEN_WIDTH  = 1280
SCREEN_HEIGHT = 720
SCREEN 	      = [SCREEN_WIDTH,SCREEN_HEIGHT]


# --------------------------------------------------------------------------
# -- player class
# --------------------------------------------------------------------------
class Player(pygame.sprite.Sprite):
    # -----------------------------------------------------------
    # init class
    def __init__(self, posX, posY):        
        super(Player, self).__init__()
        self.surf = pygame.Surface((100, 30))
        # fill with color
        self.surf.fill((0, 0, 255))
        self.rect = self.surf.get_rect()
        # start at screen center
        self.rect.x = posX
        self.rect.y = posY
        
    # -----------------------------------------------------------
	# update player rectangle
    def update(self, keys):
        if keys[pygame.K_UP]:
            self.rect.y -=5
        if keys[pygame.K_DOWN]:
            self.rect.y +=5
        if keys[pygame.K_LEFT]:
            self.rect.x -=5
        if keys[pygame.K_RIGHT]:
            self.rect.x +=5        



# --------------------------------------------------------------------------
# -- game
# --------------------------------------------------------------------------

# init pygame
pygame.init()

# set display size and caption
screen = pygame.display.set_mode(SCREEN)
pygame.display.set_caption("Computer Vision Game")

# init game clock
fps = 300
frames = []
clock = pygame.time.Clock()


# opencv - init webcam capture
#cap = cv2.VideoCapture(0)

#cap = cv2.VideoCapture("854204-hd_1920_1080_30fps.mp4")
cap = cv2.VideoCapture("Vid/Vorne-Hinten.mp4")



#cap = cv2.bgsegm.SyntheticSequenceGenerator()
#cap.getNextFrame()
# set width & height to screen size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen.get_width())
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen.get_height())


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))


backSubMOG2 = cv2.createBackgroundSubtractorMOG2(history=200,detectShadows=False,varThreshold=32) # besseres MOG/mehr Parameter #Minimalistisch langsamer
backSubMOG2.setNMixtures(5)                                                                      # oben Rechts

backSubKNN = cv2.createBackgroundSubtractorKNN(history=200,detectShadows=False,dist2Threshold=300.0) # unten Rechts
backSubCNT = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=10,useHistory=True,maxPixelStability=15*15)  #Besseres MOG # oben Links
backSubGMG = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=60,decisionThreshold=0.8) #unten Links

"""backSub = cv2.bgsegm.createBackgroundSubtractorGSOC(mc=10,nSamples=10,replaceRate=0.003,propagationRate=0.01,hitsThreshold=32,     #besseres LSBP
                                                    alpha=0.01,beta=0.01,blinkingSupressionDecay=0.1,blinkingSupressionMultiplier=0.1,
                                                   noiseRemovalThresholdFacBG=0.0004,noiseRemovalThresholdFacFG=0.008)"""
#backSub = cv2.bgsegm.createBackgroundSubtractorLSBP(nSamples=10,mc=10)

# init player
player = Player(screen.get_width()/2, screen.get_height()/2)

# example variable for game score
gameScore = 0

# -------------
# -- main loop
running = True
ksize=5
blursize = 5 # nicht größer als 5 => zu langsam
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # press 'esc' to quit
        if event.type==pygame.KEYDOWN:
            if event.key==pygame.K_ESCAPE:
                running = False


    #bilateral blur == slooooooooooooow


    # -- opencv & viz image
    ret, cameraFrame = cap.read()

    #cameraFrame = cv2.GaussianBlur(cameraFrame,(ksize,ksize),0)
    #cameraFrame = cv2.medianBlur(cameraFrame, blursize)

    if(cameraFrame is None):
        print(np.array(frames).mean())
    fgMaskMOG2 = backSubMOG2.apply(cameraFrame)
    fgMaskKNN = backSubKNN.apply(cameraFrame)
    fgMaskCNT = backSubCNT.apply(cameraFrame)
    fgMaskGMG = backSubGMG.apply(cameraFrame)
    vert1 = cv2.vconcat([fgMaskMOG2, fgMaskKNN])
    vert2 = cv2.vconcat([fgMaskCNT, fgMaskGMG])
    fgMask = cv2.hconcat([vert1, vert2])
    fgMask = cv2.resize(fgMask, (screen.get_width(), screen.get_height()))
    imgRGB = cv2.cvtColor(fgMask, cv2.COLOR_BGR2RGB)

    """fgMask = backSubCNT.apply(cameraFrame)
    #ret, fgMask = cv2.threshold(cameraFrame,127,255,cv2.THRESH_BINARY)  #deleting shadows
    #fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=4)
    foreground = cv2.bitwise_and(cameraFrame, cameraFrame, mask=fgMask)
    imgRGB = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)"""

    #imgRGB = cv2.cvtColor(fgMask, cv2.COLOR_BGR2RGB)
    # image needs to be rotated for pygame
    imgRGB = np.rot90(imgRGB)


	# convert image to pygame and visualize
    gameFrame = pygame.surfarray.make_surface(imgRGB).convert()    
    screen.blit(gameFrame, (0, 0))
    

	# -- update & draw object on screen
    player.update(pygame.key.get_pressed())
    screen.blit(player.surf, player.rect)


	# -- add Text on screen (e.g. score)
    textFont = pygame.font.SysFont("arial", 26)
    textExample = textFont.render(f'Score: {gameScore}', True, (255, 0, 0))
    screen.blit(textExample, (20, 20))


    # update entire screen
    pygame.display.update()
    frames.append(clock.get_fps())
    # set clock
    clock.tick(fps)

    

# quit game
pygame.quit()


# release capture
cap.release()
print(np.array(frames).mean())


