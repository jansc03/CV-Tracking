import pygame


class MovingEntity:
    def __init__(self, x, y, width, height, speed, row_height, SCREEN_WIDTH, SCREEN_HEIGHT, image_path="sprite/ufo.png"):
        super(MovingEntity, self).__init__()
        self.surf = None
        if image_path:
            self.surf = pygame.image.load(image_path).convert_alpha()
            self.surf = pygame.transform.scale(self.surf, (width,height))
        else:
            self.surf = pygame.Surface((width, height), pygame.SRCALPHA)
            self.surf.fill((0, 0, 0, 0))  # Unsichtbares Rechteck
        self.rect = self.surf.get_rect(topleft=(x, y))
        self.speed = speed
        self.row_height = row_height
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT
        self.move_right = True
        self.show = True

    def update(self):
        # Überprüfen, ob die Entität den rechten Rand berührt
        if self.rect.right >= self.screen_width:
            self.move_right = False
            self.rect.y += self.row_height  # Eine Zeile tiefer

        # Überprüfen, ob die Entität den linken Rand berührt
        if self.rect.left <= 0:
            self.move_right = True
            self.rect.y += self.row_height  # Eine Zeile tiefer

        # Bewegung basierend auf der aktuellen Richtung
        if self.move_right:
            self.rect.x += self.speed
        else:
            self.rect.x -= self.speed

    def draw(self, surface):
        if self.show and self.surf:
            surface.blit(self.surf, self.rect)

    def get_position(self):
        return self.rect.x, self.rect.y
