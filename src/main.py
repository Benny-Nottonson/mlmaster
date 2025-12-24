import pygame
from core.config import *
from screens.splash import SplashScreen
from screens.map import MapScreen
from screens.level import LevelScreen

def main():
    pygame.init()
    screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    pygame.display.set_caption("MLMaster")
    clock = pygame.time.Clock()

    splash = SplashScreen(screen)
    map_screen = MapScreen(screen)
    level_screen = None
    running = True
    current_screen = "splash"

    def start_level(level_id):
        nonlocal level_screen, current_screen
        level_screen = LevelScreen(screen, level_id)
        level_screen.on_level_complete = lambda: switch_to_map(level_id)
        current_screen = "level"

    def switch_to_map(level_id=None):
        nonlocal current_screen, level_screen
        if level_id:
            map_screen.complete_level(level_id)
        current_screen = "map"
        level_screen = None

    map_screen.on_level_start = start_level

    while running:
        dt = clock.tick(FPS)
        screen_width, screen_height = screen.get_size()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11 or (event.key == pygame.K_RETURN and event.mod & pygame.KMOD_ALT):
                    continue

            if current_screen == "splash" and splash.handle_event(event):
                current_screen = "map"
            elif current_screen == "map":
                map_screen.handle_event(event)
            elif current_screen == "level" and level_screen:
                level_screen.handle_event(event)

        if current_screen == "splash":
            splash.update(dt)
            splash.draw(screen_width, screen_height)
        elif current_screen == "map":
            map_screen.update(dt)
            map_screen.draw(screen_width, screen_height)
        elif current_screen == "level" and level_screen:
            level_screen.update(dt)
            level_screen.draw(screen_width, screen_height)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
