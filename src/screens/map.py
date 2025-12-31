import pygame
from core.config import *
from data.levels import LEVELS

class MapScreen:
    def __init__(self, screen):
        self.screen = screen
        self._initialize_fonts()
        self._initialize_state()
        self._initialize_pan_and_drag()
        self.on_level_start = None

    def _initialize_fonts(self):
        self.font_title = pygame.font.SysFont("arial", 48, bold=True)
        self.font_label = pygame.font.SysFont("arial", 22)
        self.font_small = pygame.font.SysFont("arial", 16)

    def _initialize_state(self):
        self.selected_node = None
        self.unlocklist = []
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT

    def _initialize_pan_and_drag(self):
        self.pan_x = 0
        self.pan_y = 0
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0

    def update(self, dt):
        if self.dragging:
            self._update_pan_position()

    def _update_pan_position(self):
        mouse_x, mouse_y = pygame.mouse.get_pos()
        self.pan_x = self.drag_start_x + (mouse_x - self.drag_x_start)
        self.pan_y = self.drag_start_y + (mouse_y - self.drag_y_start)

    def draw(self, width=None, height=None):
        if width is None:
            width = SCREEN_WIDTH
        if height is None:
            height = SCREEN_HEIGHT
        
        self.screen_width = width
        self.screen_height = height
        self.screen.fill(BG_COLOR)

        self._draw_all_elements()

    def _draw_all_elements(self):
        self.draw_title()
        self.draw_connections()
        self.draw_nodes()

    def draw_title(self):
        title = self.font_title.render("Campaign Map", True, TEXT_COLOR)
        self.screen.blit(title, (50, 30))

    def draw_connections(self):
        levels_list = list(LEVELS.values())
        for i in range(len(levels_list) - 1):
            current = levels_list[i]
            next_level = levels_list[i + 1]
            
            if not current.unlocked:
                continue
            
            x1, y1 = current.position
            x1 += self.pan_x
            y1 += self.pan_y
            
            x2, y2 = next_level.position
            x2 += self.pan_x
            y2 += self.pan_y
            
            line_color = (100, 100, 100) if next_level.unlocked else (60, 60, 60)
            pygame.draw.line(
                self.screen,
                line_color,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                3
            )

    def draw_nodes(self):
        for level_key, level in LEVELS.items():
            self.draw_level_node(level)

    def draw_level_node(self, level):
        x, y = level.position
        x += self.pan_x
        y += self.pan_y
        radius = 60

        color = self._get_level_node_color(level)

        pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)
        self._draw_level_selection_ring(level, x, y, radius)
        self._draw_level_label(level, x, y)

    def _get_level_node_color(self, level):
        if not level.unlocked:
            return DISABLED_COLOR
        elif level.completed:
            return SUCCESS_COLOR
        else:
            return ACCENT_COLOR

    def _draw_level_selection_ring(self, level, x, y, radius):
        if self.selected_node == level.name:
            pygame.draw.circle(self.screen, TEXT_COLOR, (int(x), int(y)), radius + 8, 4)

    def _draw_level_label(self, level, x, y):
        label_color = (0, 0, 0) if not level.unlocked else TEXT_COLOR
        words = level.name.split()
        total_height = len(words) * 22
        start_y = y - total_height // 2 + 11
        
        for i, word in enumerate(words):
            label = self.font_label.render(word, True, label_color)
            label_rect = label.get_rect(center=(int(x), int(start_y + i * 22)))
            self.screen.blit(label, label_rect)



    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self._handle_mouse_down(event)
        elif event.type == pygame.MOUSEBUTTONUP:
            self._handle_mouse_up(event)
        elif event.type == pygame.KEYDOWN:
            self._handle_key_down(event)

    def _handle_mouse_down(self, event):
        mouse_pos = pygame.mouse.get_pos()
        if event.button == 1:
            if not self.check_node_click(mouse_pos):
                self._start_dragging(mouse_pos)
        elif event.button == 3:
            self._start_dragging(mouse_pos)

    def _start_dragging(self, mouse_pos):
        self.dragging = True
        self.drag_x_start = mouse_pos[0]
        self.drag_y_start = mouse_pos[1]
        self.drag_start_x = self.pan_x
        self.drag_start_y = self.pan_y

    def _handle_mouse_up(self, event):
        if event.button == 1 or event.button == 3:
            self.dragging = False

    def _handle_key_down(self, event):
        if event.key == pygame.K_ESCAPE:
            self.selected_node = None

    def check_node_click(self, pos):
        if self._check_level_click(pos):
            return True
        return False

    def _check_level_click(self, pos):
        for level_key, level in LEVELS.items():
            if not level.unlocked:
                continue
            x, y = level.position
            x += self.pan_x
            y += self.pan_y
            if (pos[0] - x) ** 2 + (pos[1] - y) ** 2 <= 60 ** 2:
                self._try_start_level(level_key)
                return True
        return False

    def _try_start_level(self, level_id):
        if self.on_level_start:
            self.on_level_start(level_id)

    def complete_level(self, level_id):
        if level_id not in LEVELS:
            return
        
        LEVELS[level_id].completed = True
        self._unlock_next_level(level_id)

    def _unlock_next_level(self, level_id):
        levels_list = list(LEVELS.keys())
        try:
            current_index = levels_list.index(level_id)
            if current_index + 1 < len(levels_list):
                next_level_id = levels_list[current_index + 1]
                LEVELS[next_level_id].unlocked = True
        except ValueError:
            pass
