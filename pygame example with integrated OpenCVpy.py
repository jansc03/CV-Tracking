"""
-----------------------------------------------------------------------------
Vorlesung: Computer Vision (Wintersemester 2024/25)
Thema: pygame example with integrated OpenCV

-----------------------------------------------------------------------------
"""

import numpy as np
import cv2
import pygame


import background_subtraction as bs
import detector as dt
import tracker as tr
import entity
import player as Player
import iou
from yolo_tracker_integration import YOLOTracker

SCREEN_WIDTH  = 1280
SCREEN_HEIGHT = 720
SCREEN 	      = [SCREEN_WIDTH,SCREEN_HEIGHT]

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
player = Player.Player(0,0, screen_w=SCREEN_HEIGHT)
bullet = Player.Projektil(screen_height=SCREEN_HEIGHT)

moving_entities = [
    entity.MovingEntity(x=0, y=0, width=300, height=30, speed=5, row_height=50, SCREEN_WIDTH=SCREEN_WIDTH, SCREEN_HEIGHT=SCREEN_HEIGHT)
]

# example variable for game score
gameScore = 0

# -------------
# -- main loop
running = True
paused = False
ksize=5
blursize = 5 # nicht größer als 5 => zu langsam
previous_frame = None

collision_check_timer = 0
collision_check_interval = 800
fire_interval_ms = 500
last_fire_time = 0
cooldown_duration = 1000
last_collision_time = 0

n = 2

backgroundSubtraction = bs.BackgroundSubtraction()
#backgroundSubtraction.initBackgroundSubtractor(backSubNum=0,multi=True)
backgroundSubtraction.initBackgroundSubtractor(backSubNum=0,multi=False,vidNum=8)

detector = dt.Detector()
custom_tracker = tr.Tracker(max_lost=90)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
word = tr.Tracker()
# Initialize YOLO tracker
yolo_tracker = YOLOTracker(fps=fps)

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
        bgImg = cv2.GaussianBlur(background, (5, 5), 2)
        mask_eroded = cv2.morphologyEx(bgImg, cv2.MORPH_CLOSE, kernel, iterations=2)
        background = cv2.morphologyEx(mask_eroded, cv2.MORPH_OPEN, kernel, iterations=2).astype(np.uint8)

        #cv2.imshow("Background", background)

        people,all_contours = detector.detect(background)
        frame_out = original_vid.copy()
        person_areas = detector.extract_person_areas(original_vid,background, people)

        person_hist = []
        for person,background in person_areas:
            person_hist.append(word.get_hist(person,background))

        """if len(person_areas)>0:
            cv2.imshow("Frame", person_areas[0])"""

        """if len(person_areas) > 0 and person_areas[0].size > 0:
            cv2.imshow("person", person_areas[0])"""

        for x,y,w,h in people:
            frame_out = cv2.rectangle(original_vid, (x, y), (x + w, y + h), (200, 0, 200), 5)


        for x,y,w,h in all_contours:
            frame_out = cv2.rectangle(original_vid, (x, y), (x + w, y + h), (200, 0, 0), 3)

        #tracker
        custom_tracker.update_track(people,person_hist)
        # YOLO detection and tracking
        yolo_tracks = yolo_tracker.process_frame(original_vid)
        mock_detections = [(track.box[0], track.box[1], track.box[2] - track.box[0], track.box[3] - track.box[1]) for
                           track in yolo_tracks]

        # Speichere den aktuellen Frame als vorherigen
        previous_frame = frame_out.copy()

        # Visualize YOLO tracker results
        for track in yolo_tracks:
            x1, y1, x2, y2 = map(int, track.box)
            cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_out, f"ID {track.id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Visualize custom tracker results
        for track_id, track in custom_tracker.get_active_tracks().items():
            x, y, w, h = track["bbox"]
            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame_out, f"Custom ID {track_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            #player.update_position(x, y, w, h)

        #imgRGB = cv2.cvtColor(fgMask, cv2.COLOR_BGR2RGB)
        # image needs to be rotated for pygame

        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)

        img_rgb = np.rot90(frame_out)

        # convert image to pygame and visualize

        img_rgb = np.flip(img_rgb,axis=0)
        game_frame = pygame.surfarray.make_surface(img_rgb).convert()
        screen.blit(game_frame, (0, 0))
        iou.process_frame(yolo_tracker, custom_tracker, original_vid)



        """
        'GAME'
        current_time = pygame.time.get_ticks()
        if current_time - collision_check_timer >= collision_check_interval:
            collision_check_timer = current_time
            if frame_out is not None:
                #player.lose_life()
                print(f"Spieler getroffen! Verbleibende Leben: {player.lives}")

        if current_time - last_fire_time >= fire_interval_ms:
            last_fire_time = current_time
            bullet.fire(player.rect.x, player.rect.y, player.rect.width)


        for moving_entity in moving_entities[:]:
            if bullet.rect.colliderect(moving_entity.rect):
                if current_time - last_collision_time >= cooldown_duration:
                    print("Kollision erkannt!")
                    last_collision_time = current_time
                    moving_entities.remove(moving_entity)

                    # Füge zwei neue Entitäten hinzu
                    distance = 0
                    if len(moving_entities) == 0:
                        for _ in range(n):
                            new_entity = entity.MovingEntity(
                                x=0 + distance ,y=0 , width=100, height=30, speed=5, row_height=50,
                                SCREEN_WIDTH=SCREEN_WIDTH, SCREEN_HEIGHT=SCREEN_HEIGHT
                            )
                            distance += 150
                            moving_entities.append(new_entity)
                        n += 1

            moving_entity.update()
            moving_entity.draw(screen)

        bullet.update()
        bullet.draw(screen)
        # Lebensanzeige oben links
        player.draw_lives(screen, player.lives)

        # Prüfen, ob der Spieler noch lebt
        if not player.is_alive():
            print("Spieler 1 hat alle Leben verloren! Spiel beendet.")
            running = False

        player.draw(screen)
        """
        # update entire screen
        pygame.display.update()
        # set clock
        clock.tick(fps)

# quit game
pygame.quit()
backgroundSubtraction.closeAll()


