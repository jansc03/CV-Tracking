import pygame

class Player(pygame.sprite.Sprite):

    def __init__(self, posX, posY, width=0, height=0, screen_w=0, lives=3):
        super(Player, self).__init__()
        self.screen_width = screen_w
        self.width = width
        self.height = height
        self.surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.surf.fill((0, 0, 255, 100))
        self.rect = self.surf.get_rect()
        self.rect.x = posX
        self.rect.y = posY
        self.lives = lives

    def update_position(self, x, y, w, h):
        """Setze Position und Größe basierend auf der getrackten Person."""
        self.rect.y = y
        self.rect.width = w
        self.rect.height = h
        self.rect.x = x
        self.surf = pygame.Surface((w, h), pygame.SRCALPHA)
        self.surf.fill((0, 0, 255, 100))


    def draw(self, surface):
        surface.blit(self.surf, self.rect)

    def lose_life(self):
        if self.lives > 0:
            self.lives -= 1

    def is_alive(self):
        return self.lives > 0

    def draw_lives(self ,screen, lives, x=10, y=10):
        font = pygame.font.SysFont(None, 36)
        text = font.render(f'Leben: {lives}', True, (0, 0, 255))
        screen.blit(text, (x, y))


class Projektil(pygame.sprite.Sprite):
    def __init__(self, screen_height):
        super(Projektil, self).__init__()
        self.active = False
        self.hit = False
        self.width = 10
        self.height = 40
        self.surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.surf.fill((255, 0, 0, 100))
        self.rect = self.surf.get_rect()
        self.rect.x = 0
        self.rect.y = 0
        self.screen_height = screen_height

    def fire(self, x, y, w):
        """Startet das Projektil in der Mitte des Trackers."""
        if not self.active:
            self.rect.x = x + w // 2 - self.width // 2
            self.rect.y = y
            self.active = True

    def update(self):
        """Aktualisiert die Position des Projektils."""
        if self.active and not self.hit:
            self.rect.y -= 5
            if self.rect.y + self.height <= 0:
                self.active = False

    def draw(self, surface):
        """Zeichnet das Projektil, wenn aktiv."""
        if self.active:
            surface.blit(self.surf, self.rect)



