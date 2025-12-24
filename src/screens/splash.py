import pygame
import math
from core.config import *

class SplashScreen:
    def __init__(self, screen):
        self.screen = screen
        self.font_big = pygame.font.SysFont("arial", 120)
        self.font_small = pygame.font.SysFont("arial", 28)
        self.time = 0

    def update(self, dt):
        self.time += dt

    def draw(self, width=None, height=None):
        if width is None:
            width = SCREEN_WIDTH
        if height is None:
            height = SCREEN_HEIGHT
        
        self.screen.fill(BG_COLOR)

        pulse = 1 + 0.02 * math.sin(self.time * 0.004)
        title = self.font_big.render("MLMaster", True, TEXT_COLOR)
        title = pygame.transform.rotozoom(title, 0, pulse)

        rect = title.get_rect(center=(width // 2, height // 2))
        self.screen.blit(title, rect)

        hint = self.font_small.render("Press any key to begin", True, ACCENT_COLOR)
        hint_rect = hint.get_rect(center=(width // 2, height // 2 + 120))
        self.screen.blit(hint, hint_rect)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            return True
        return False
