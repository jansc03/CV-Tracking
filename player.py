import pygame

class Player(pygame.sprite.Sprite):
    def __init__(self, posX, posY, width=0, height=0, screen_w=1280, lives=3):
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
        self.rect.x = self.screen_width - (x - w)
        self.surf = pygame.Surface((w, h), pygame.SRCALPHA)
        self.surf.fill((0, 0, 255, 100))

    def flip_horizontally(self):
        """Flippt den Spieler horizontal."""
        self.rect.x = self.screen_width - self.rect.x - self.rect.width

    def draw(self, surface):
        surface.blit(self.surf, self.rect)

    def lose_life(self):
        """Reduziert die Anzahl der Leben um eins."""
        if self.lives > 0:
            self.lives -= 1

    def is_alive(self):
        """Prüft, ob der Spieler noch Leben hat."""
        return self.lives > 0

    def draw_lives(self ,screen, lives, x=10, y=10):
        font = pygame.font.SysFont(None, 36)
        text = font.render(f'Leben: {lives}', True, (255, 255, 255))
        screen.blit(text, (x, y))
