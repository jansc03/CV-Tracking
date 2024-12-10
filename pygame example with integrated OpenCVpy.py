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
previous_frame = None
backgroundSubtraction = bs.BackgroundSubtraction()
#backgroundSubtraction.initBackgroundSubtractor(backSubNum=0,multi=True)
backgroundSubtraction.initBackgroundSubtractor(backSubNum=0,multi=False,vidNum=0)

detector = dt.Detector()
tracker = tr.Tracker(max_lost=90)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))



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
        #bgImg = cv2.GaussianBlur(background, (5, 5), 2)
        mask_eroded = cv2.morphologyEx(background, cv2.MORPH_CLOSE, kernel, iterations=2)
        background = cv2.morphologyEx(mask_eroded, cv2.MORPH_OPEN, kernel, iterations=2)

        cv2.imshow("Background", background)

        people,all_contours = detector.detect(background)

        frame_out = original_vid.copy()

        person_areas = detector.extract_person_areas(original_vid,background, people)

        word = tr.Tracker()
        person_hist = []
        for person,background in person_areas:
            person_hist.append(word.get_hist(person,background))

        """if len(person_areas)>0:
            cv2.imshow("Frame", person_areas[0])"""


        """if len(person_areas) > 0 and person_areas[0].size > 0:
            cv2.imshow("person", person_areas[0])"""


        for x,y,w,h in people:
            frame_out = cv2.rectangle(original_vid, (x, y), (x + w, y + h), (200, 0, 200), 5)

        
        """for x,y,w,h in all_contours:
            frame_out = cv2.rectangle(original_vid, (x, y), (x + w, y + h), (200, 0, 0), 3)"""


        #tracker
        tracker.update_track(people,person_hist)

        # Optischen Fluss anwenden, wenn vorheriger Frame existiert
        #if previous_frame is not None:
            #tracker.refine_tracks_with_optical_flow(frame_out, previous_frame)

        # Speichere den aktuellen Frame als vorherigen
        previous_frame = frame_out.copy()

        # Tracks visualisieren
        for track_id, track in tracker.get_active_tracks().items():
            if(track["lost"]>0):
                x, y, w, h = track["prediction"]
            else:
                x, y, w, h = track["bbox"]
            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_out, f'ID: {track_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



        #imgRGB = cv2.cvtColor(fgMask, cv2.COLOR_BGR2RGB)
        # image needs to be rotated for pygame

        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)

        img_rgb = np.rot90(frame_out)

        # convert image to pygame and visualize

        game_frame = pygame.surfarray.make_surface(img_rgb).convert()

        screen.blit(game_frame, (0, 0))

        """
        # Potenzielle Personenbereiche
        for i, person_area in enumerate(person_areas):

            person_area = cv2.cvtColor(person_area, cv2.COLOR_BGR2RGB)
            person_img = np.rot90(person_area)
            person_surface = pygame.surfarray.make_surface(person_img).convert()

            screen.blit(person_surface, (10 + i * 110, 10))  # Verschieben für mehrere Personen
            
        """

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


