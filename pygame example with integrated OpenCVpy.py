"""
-----------------------------------------------------------------------------
Vorlesung: Computer Vision (Wintersemester 2024/25)
Thema: pygame example with integrated OpenCV

-----------------------------------------------------------------------------
"""

import numpy as np
import cv2
import pygame
#from torch.ao.nn.quantized.functional import threshold

import background_subtraction as bs
import detector as dt
import tracker as tr
import entity
import player as Player
from dual_bs import DualBackgroundSubtraction

#import iou
#from yolo_tracker_integration import YOLOTracker

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
detector = dt.Detector()
custom_tracker = tr.Tracker(max_lost=90)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
word = tr.Tracker()

# init game clock
fps = 30
clock = pygame.time.Clock()
# Dynamische Spielererstellung basierend auf max_tracks
players = [Player.Player(0, 0, screen_w=SCREEN_HEIGHT) for _ in range(custom_tracker.max_tracks)]
bullets = [Player.Projektil(screen_height=SCREEN_HEIGHT) for _ in range(custom_tracker.max_tracks)]

moving_entities = []

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
collision_check_interval = 1000
fire_interval_ms = 500
last_fire_time = 0
cooldown_duration = 1000
last_collision_time = 0

n = 1
background_subtraction = DualBackgroundSubtraction()
background_subtraction.init_video(vidNum=20) # 17: Phillip bleibt lange Id=3 alles andere ist katastrophe
                                                                                   #18: K.a. wie aber es läuft extrem gut
                                                                                   #19: Katastrophe
                                                                              #20:Tracks die rechts aus den Bildschirm gehen tauchen links wieder auf

"""
# Initialize YOLO tracker
yolo_tracker = YOLOTracker(fps=fps)
"""

def handle_player_collisions():
    """Überprüft Kollisionen der Spieler mit beweglichen Entitäten."""
    global collision_check_timer
    current_time = pygame.time.get_ticks()
    if current_time - collision_check_timer >= collision_check_interval:
        collision_check_timer = current_time
        for moving_entity in moving_entities[:]:
            for player in players:
                if player.rect.colliderect(moving_entity.rect):
                    player.lose_life()
                    print(f"Spieler {players.index(player) + 1} getroffen! Verbleibende Leben: {player.lives}")
                    moving_entities.remove(moving_entity)


def handle_bullet_firing():
    """Feuert ein Projektil für jeden Spieler, wenn das Intervall überschritten wurde."""
    global last_fire_time
    current_time = pygame.time.get_ticks()
    if current_time - last_fire_time >= fire_interval_ms:
        last_fire_time = current_time
        for player, bullet in zip(players, bullets):
            bullet.fire(player.rect.x, player.rect.y, player.rect.width)
            print(player.rect.x, player.rect.y, player.rect.width)


def handle_ufo_collisions():
    """Überprüft Kollisionen von UFOs mit den Projektilen aller Spieler."""
    global last_collision_time
    current_time = pygame.time.get_ticks()
    for moving_entity in moving_entities[:]:
        for bullet in bullets:
            if bullet.rect.colliderect(moving_entity.rect):
                if current_time - last_collision_time >= cooldown_duration:
                    print("Kollision erkannt!")
                    last_collision_time = current_time
                    moving_entities.remove(moving_entity)


def spawn_entities():
    """Erstellt neue Entitäten, wenn keine vorhanden sind."""
    global n
    if len(moving_entities) == 0:
        distance = 0
        for _ in range(n):
            new_entity = entity.MovingEntity(
                x=distance, y=0, width=120, height=80, speed=9, row_height=50,
                SCREEN_WIDTH=SCREEN_WIDTH, SCREEN_HEIGHT=SCREEN_HEIGHT
            )
            distance += 150
            moving_entities.append(new_entity)
        n += 1


def draw_game_objects():
    """Zeichnet alle Spielobjekte auf den Bildschirm."""
    for bullet in bullets:
        bullet.update()
        bullet.draw(screen)

    for moving_entity in moving_entities:
        moving_entity.update()
        moving_entity.draw(screen)

    for player in players:
        player.draw_lives(screen, player.lives)
        player.draw(screen)

def check_player_status():
    """Überprüft, ob ein Spieler noch lebt."""
    for i, player in enumerate(players):
        if not player.is_alive():
            print(f"Spieler {i + 1} hat alle Leben verloren! Spiel beendet.")
            return False
    return True


# Hauptspiel-Logik
def game_logic():
    handle_player_collisions()
    handle_bullet_firing()
    handle_ufo_collisions()
    spawn_entities()
    draw_game_objects()
    return check_player_status()


all_iou_values = []

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
        # Hintergrundmaske abrufen
        combined_mask, original_vid = background_subtraction.get_mask()
        mask_eroded = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        combined_mask = cv2.morphologyEx(mask_eroded, cv2.MORPH_OPEN, kernel, iterations=1).astype(np.uint8)
        cv2.imshow("combined_mask", combined_mask)
        people,all_contours = detector.detect(combined_mask)

        """Alle Personen im Frame werden Detektiert"""
        frame_out = original_vid.copy()
        person_areas = detector.extract_person_areas(original_vid,combined_mask, people)

        for x,y,w,h in people:
            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (255, 255, 0), 4)



        """Histogramme für die Detektierten Personen werden erstellt"""
        person_hist = []
        for person,background in person_areas:
            person_hist.append(word.get_hist(person,background))


        """
        Zeichnet BBoxen
        for x,y,w,h in people:
            frame_out = cv2.rectangle(original_vid, (x, y), (x + w, y + h), (200, 0, 200), 5)


        for x,y,w,h in all_contours:
            frame_out = cv2.rectangle(original_vid, (x, y), (x + w, y + h), (200, 0, 0), 3)
        """

        #tracker
        custom_tracker.update_track(people,person_hist)
        tracked_objects = [(x, y, w, h) for x, y, w, h in people]
        background_subtraction.update_tracked_objects(tracked_objects)
        """
        # YOLO detection und tracking
        yolo_tracks = yolo_tracker.process_frame(original_vid)
        mock_detections = [(track.box[0], track.box[1], track.box[2] - track.box[0], track.box[3] - track.box[1]) for
                           track in yolo_tracks]
        """

        # Speichere den aktuellen Frame als vorherigen
        previous_frame = frame_out.copy()

        # Visualisiere custom tracker
        for track_id, track in custom_tracker.get_active_tracks().items():
            if track["in_group"]:
                x, y, w, h = track["group_bbox"]
            elif track["lost"] > 0:
                x, y, w, h = track["prediction"]
            else:
                x, y, w, h = track["bbox"]

            # Boundingbox anzeigen
            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (track["color"]), 2)
            cv2.putText(frame_out, f'ID: {track_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (track["color"]), 2)

            # Spielerposition aktualisieren (Track-ID zu Spieler-ID)
            if track_id < len(players):  # Nur Spieler innerhalb der Begrenzung aktualisieren
                players[track_id].update_position(x, y, w, h)

            """#IOU
            # Definiere den erweiterten Bereich für den YOLO-Tracker
            threshold = 100
            extended_x1 = max(0, x - threshold)
            extended_y1 = max(0, y - threshold)
            extended_x2 = min(SCREEN_WIDTH, x + w + threshold)
            extended_y2 = min(SCREEN_HEIGHT, y + h + threshold)"""

            """
            # Filtere YOLO-Tracker-Ergebnisse basierend auf dem erweiterten Bereich
            filtered_yolo_tracks = [
                track for track in yolo_tracks
                if track.box[0] >= extended_x1 and track.box[1] >= extended_y1 and
                   track.box[2] <= extended_x2 and track.box[3] <= extended_y2
            ]
            
             
            # Visualisiere die gefilterten YOLO-Tracker-Ergebnisse
            for track in filtered_yolo_tracks:
                x1, y1, x2, y2 = map(int, track.box)
                cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_out, f"YOLO ID {track.id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            2)
            """

        # Frame konvertieren
        frame_out = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        img_rgb = np.rot90(frame_out)
        img_rgb = np.flip(img_rgb,axis=0)
        game_frame = pygame.surfarray.make_surface(img_rgb).convert()
        screen.blit(game_frame, (0, 0))

        """
        # Berechne IoU-Werte für den Frame und speichere sie
        frame_iou_values = iou.process_frame(yolo_tracker, custom_tracker, original_vid)
        all_iou_values.append(frame_iou_values)
        """

        'GAME'
        """game_logic()"""

        # update screen
        pygame.display.update()
        clock.tick(fps)

# quit game

"""
#IoU ausgabe
final_results = iou.aggregate_iou_results(all_iou_values)
print(final_results)
"""

pygame.quit()