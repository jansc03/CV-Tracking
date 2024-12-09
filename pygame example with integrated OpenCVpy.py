"""
-----------------------------------------------------------------------------
Vorlesung: Computer Vision (Wintersemester 2024/25)
Thema: pygame example with integrated OpenCV

-----------------------------------------------------------------------------
"""

import numpy as np
import cv2
import pygame

import BackgroundSubtraction as bs
import Detector as dt
import tracker as tr

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
fps = 30
clock = pygame.time.Clock()

# init player
player = Player(screen.get_width()/2, screen.get_height()/2)

# example variable for game score
gameScore = 0

# -------------
# -- main loop
running = True
paused = False
ksize=5
blursize = 5 # nicht größer als 5 => zu langsam
backgroundSubtraction = bs.BackgroundSubtraction()
#backgroundSubtraction.initBackgroundSubtractor(backSubNum=0,multi=True)
backgroundSubtraction.initBackgroundSubtractor(backSubNum=0,multi=False,vidNum=4)

detector = dt.Detector()
tracker = tr.Tracker(max_lost=90)



while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # press 'esc' to quit
        if event.type==pygame.KEYDOWN:
            if event.key==pygame.K_ESCAPE:
                running = False
            elif event.key==pygame.K_SPACE:
                paused = not paused


    if not paused:
        #bilateral blur == slooooooooooooow
        background,original_vid = backgroundSubtraction.getNextSingleBackground()

        people,all_contours = detector.detect(background)


        frame_out = original_vid.copy()
        for x,y,w,h in people:
            frame_out = cv2.rectangle(original_vid, (x, y), (x + w, y + h), (0, 0, 200), 3)

        for x,y,w,h in all_contours:
            frame_out = cv2.rectangle(original_vid, (x, y), (x + w, y + h), (200, 0, 0), 3)

        #tracker
        tracker.update_track(people)
        for track_id, track in tracker.get_active_tracks().items():
            x, y, w, h = track["bbox"]
            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_out, f'ID: {track_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        #imgRGB = cv2.cvtColor(fgMask, cv2.COLOR_BGR2RGB)
        # image needs to be rotated for pygame

        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)

        imgRGB = np.rot90(frame_out)
        # convert image to pygame and visualize
        game_frame = pygame.surfarray.make_surface(imgRGB).convert()

        screen.blit(game_frame, (0, 0))

        '''
        # -- update & draw object on screen
        player.update(pygame.key.get_pressed())
        screen.blit(player.surf, player.rect)

        
        # -- add Text on screen (e.g. score)
        textFont = pygame.font.SysFont("arial", 26)
        textExample = textFont.render(f'Score: {gameScore}', True, (255, 0, 0))
        screen.blit(textExample, (20, 20))
        '''

        # update entire screen
        pygame.display.update()
        # set clock
        clock.tick(fps)

    

# quit game
pygame.quit()
backgroundSubtraction.closeAll()


