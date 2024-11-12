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
backgroundSubtraction.initBackgroundSubtractor(backSubNum=0,multi=True)
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
        background = backgroundSubtraction.getNextMultipleBackground()

        imgRGB = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

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
        # set clock
        clock.tick(fps)

    

# quit game
pygame.quit()
backgroundSubtraction.closeAll()


