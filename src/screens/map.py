import pygame
from core.config import *
from data.levels import LEVELS, CHECKPOINTS, ActivationLayer

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
        self.show_details = False
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
        self.draw_details()

    def draw_title(self):
        title = self.font_title.render("Campaign Map", True, TEXT_COLOR)
        self.screen.blit(title, (50, 30))

    def draw_connections(self):
        self._draw_level_to_checkpoint_connections()
        self._draw_checkpoint_to_level_connections()

    def _draw_level_to_checkpoint_connections(self):
        for level_key, level in LEVELS.items():
            if not level.is_unlocked:
                continue
            
            level_x, level_y = level.position
            level_x += self.pan_x
            level_y += self.pan_y
            
            for checkpoint_key, checkpoint in CHECKPOINTS.items():
                if checkpoint["required_level"] == level.id:
                    self._draw_single_connection(level_x, level_y, checkpoint)

    def _draw_single_connection(self, level_x, level_y, checkpoint):
        cp_x, cp_y = checkpoint["position"]
        cp_x += self.pan_x
        cp_y += self.pan_y
        
        line_color = (100, 100, 100) if checkpoint["is_unlocked"] else (60, 60, 60)
        pygame.draw.line(
            self.screen,
            line_color,
            (int(level_x), int(level_y)),
            (int(cp_x), int(cp_y)),
            3
        )

    def _draw_checkpoint_to_level_connections(self):
        for _, checkpoint in CHECKPOINTS.items():
            cp_x, cp_y = checkpoint["position"]
            cp_x += self.pan_x
            cp_y += self.pan_y
            
            next_level = self._find_next_level(checkpoint)
            
            if next_level:
                self._draw_checkpoint_connection_to_level(cp_x, cp_y, next_level, checkpoint)

    def _find_next_level(self, checkpoint):
        next_level = None
        min_distance = float('inf')
        for _, level in LEVELS.items():
            if level.is_unlocked and level.position[0] > checkpoint["position"][0]:
                distance = abs(level.position[0] - checkpoint["position"][0])
                if distance < min_distance:
                    min_distance = distance
                    next_level = level
        return next_level

    def _draw_checkpoint_connection_to_level(self, cp_x, cp_y, next_level, checkpoint):
        next_x, next_y = next_level.position
        next_x += self.pan_x
        next_y += self.pan_y
        
        line_color = (100, 100, 100) if checkpoint["is_unlocked"] else (60, 60, 60)
        pygame.draw.line(
            self.screen,
            line_color,
            (int(cp_x), int(cp_y)),
            (int(next_x), int(next_y)),
            3
        )

    def draw_nodes(self):
        for level_key, level in LEVELS.items():
            self.draw_level_node(level)

        for checkpoint_key, checkpoint in CHECKPOINTS.items():
            self.draw_checkpoint_node(checkpoint)

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
        if not level.is_unlocked:
            return DISABLED_COLOR
        elif level.is_completed:
            return SUCCESS_COLOR
        else:
            return ACCENT_COLOR

    def _draw_level_selection_ring(self, level, x, y, radius):
        if self.selected_node == level.id:
            pygame.draw.circle(self.screen, TEXT_COLOR, (int(x), int(y)), radius + 8, 4)

    def _draw_level_label(self, level, x, y):
        label_color = (0, 0, 0) if not level.is_unlocked else TEXT_COLOR
        words = level.name.split()
        total_height = len(words) * 22
        start_y = y - total_height // 2 + 11
        
        for i, word in enumerate(words):
            label = self.font_label.render(word, True, label_color)
            label_rect = label.get_rect(center=(int(x), int(start_y + i * 22)))
            self.screen.blit(label, label_rect)

    def draw_checkpoint_node(self, checkpoint):
        x, y = checkpoint["position"]
        x += self.pan_x
        y += self.pan_y
        size = 45

        color = self._get_checkpoint_node_color(checkpoint)

        pygame.draw.rect(self.screen, color, (int(x - size), int(y - size), size * 2, size * 2))
        self._draw_checkpoint_selection_ring(x, y, size, checkpoint)
        self._draw_checkpoint_label(checkpoint, x, y)

    def _get_checkpoint_node_color(self, checkpoint):
        if not checkpoint["is_unlocked"]:
            return DISABLED_COLOR
        else:
            return SECONDARY_COLOR

    def _draw_checkpoint_selection_ring(self, x, y, size, checkpoint):
        if self.selected_node == checkpoint["id"]:
            pygame.draw.rect(
                self.screen,
                TEXT_COLOR,
                (int(x - size - 4), int(y - size - 4), size * 2 + 8, size * 2 + 8),
                4
            )

    def _draw_checkpoint_label(self, checkpoint, x, y):
        label_color = (0, 0, 0) if not checkpoint["is_unlocked"] else TEXT_COLOR
        
        if checkpoint["unlocks"]:
            unlock_text = checkpoint["unlocks"][0].value
        else:
            unlock_text = "Checkpoint"
        
        label = self.font_small.render(unlock_text, True, label_color)
        label_rect = label.get_rect(center=(int(x), int(y)))
        self.screen.blit(label, label_rect)

    def draw_details(self):
        if not self.show_details or not self.selected_node:
            return

        panel_x, panel_y, panel_width, panel_height = self._calculate_detail_panel_dimensions()
        
        self._draw_detail_panel_background(panel_x, panel_y, panel_width, panel_height)
        self._draw_detail_panel_content(panel_x, panel_y, panel_width, panel_height)

    def _calculate_detail_panel_dimensions(self):
        panel_width = 400
        panel_height = min(800, self.screen_height - 100)
        panel_x = max(20, self.screen_width - panel_width - 20)
        panel_y = max(100, (self.screen_height - panel_height) // 2)
        return panel_x, panel_y, panel_width, panel_height

    def _draw_detail_panel_background(self, panel_x, panel_y, panel_width, panel_height):
        pygame.draw.rect(self.screen, (30, 30, 40), (int(panel_x), int(panel_y), int(panel_width), int(panel_height)))
        pygame.draw.rect(self.screen, ACCENT_COLOR, (int(panel_x), int(panel_y), int(panel_width), int(panel_height)), 3)

    def _draw_detail_panel_content(self, panel_x, panel_y, panel_width, panel_height):
        if self.selected_node in LEVELS:
            self.draw_level_details(LEVELS[self.selected_node], int(panel_x), int(panel_y), int(panel_width), int(panel_height))
        elif self.selected_node in CHECKPOINTS:
            self.draw_checkpoint_details(CHECKPOINTS[self.selected_node], int(panel_x), int(panel_y), int(panel_width), int(panel_height))

    def draw_level_details(self, level, px, py, pw, ph):
        y_offset = py + 25

        y_offset = self._draw_level_name(level, px, y_offset)
        y_offset = self._draw_level_status(level, px, y_offset)
        y_offset = self._draw_level_description(level, px, y_offset)
        y_offset = self._draw_level_dataset_info(level, px, y_offset)
        y_offset = self._draw_level_goals_info(level, px, y_offset)

    def _draw_level_name(self, level, px, y_offset):
        name = self.font_title.render(level.name, True, TEXT_COLOR)
        self.screen.blit(name, (px + 25, y_offset))
        return y_offset + 60

    def _draw_level_status(self, level, px, y_offset):
        status = "COMPLETED" if level.is_completed else "IN PROGRESS"
        status_color = SUCCESS_COLOR if level.is_completed else ACCENT_COLOR
        status_text = self.font_label.render(status, True, status_color)
        self.screen.blit(status_text, (px + 25, y_offset))
        return y_offset + 50

    def _draw_level_description(self, level, px, y_offset):
        desc = self.font_small.render(level.description, True, (200, 200, 200))
        self.screen.blit(desc, (px + 25, y_offset))
        return y_offset + 40

    def _draw_level_dataset_info(self, level, px, y_offset):
        dataset_title = self.font_label.render("Dataset", True, ACCENT_COLOR)
        self.screen.blit(dataset_title, (px + 25, y_offset))
        y_offset += 35

        dataset_info = [
            f"Type: {level.dataset.name}",
            f"Train: {level.dataset.train_samples} samples",
            f"Test: {level.dataset.test_samples} samples",
            f"Features: {level.dataset.input_features}",
            f"Classes: {level.dataset.output_classes}"
        ]

        for info in dataset_info:
            text = self.font_small.render(info, True, (180, 180, 180))
            self.screen.blit(text, (px + 45, y_offset))
            y_offset += 28

        return y_offset + 15

    def _draw_level_goals_info(self, level, px, y_offset):
        goals_title = self.font_label.render("Goals", True, ACCENT_COLOR)
        self.screen.blit(goals_title, (px + 25, y_offset))
        y_offset += 35

        goals_info = [
            f"Accuracy: {level.goals.accuracy_target * 100:.1f}%",
            f"Time: {level.goals.time_limit_seconds}s",
            f"Cost: ${level.goals.cost_limit:.1f}"
        ]

        for info in goals_info:
            text = self.font_small.render(info, True, (180, 180, 180))
            self.screen.blit(text, (px + 45, y_offset))
            y_offset += 28

    def draw_checkpoint_details(self, checkpoint, px, py, pw, ph):
        y_offset = py + 25

        y_offset = self._draw_checkpoint_name(checkpoint, px, y_offset)
        y_offset = self._draw_checkpoint_description(checkpoint, px, y_offset)
        y_offset = self._draw_checkpoint_unlocks(checkpoint, px, y_offset)

    def _draw_checkpoint_name(self, checkpoint, px, y_offset):
        name = self.font_title.render(checkpoint["name"], True, TEXT_COLOR)
        self.screen.blit(name, (px + 25, y_offset))
        return y_offset + 60

    def _draw_checkpoint_description(self, checkpoint, px, y_offset):
        desc = self.font_small.render(checkpoint["description"], True, (200, 200, 200))
        self.screen.blit(desc, (px + 25, y_offset))
        return y_offset + 40

    def _draw_checkpoint_unlocks(self, checkpoint, px, y_offset):
        unlocks_title = self.font_label.render("Unlocks", True, SECONDARY_COLOR)
        self.screen.blit(unlocks_title, (px + 25, y_offset))
        y_offset += 35

        for unlock in checkpoint["unlocks"]:
            unlock_text = self.font_small.render(f"  ✓ {unlock.value}", True, SUCCESS_COLOR)
            self.screen.blit(unlock_text, (px + 45, y_offset))
            y_offset += 28

        return y_offset

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
            self.show_details = False
            self.selected_node = None

    def check_node_click(self, pos):
        if self._check_level_click(pos):
            return True
        if self._check_checkpoint_click(pos):
            return True
        return False

    def _check_level_click(self, pos):
        for level_key, level in LEVELS.items():
            if not level.is_unlocked:
                continue
            x, y = level.position
            x += self.pan_x
            y += self.pan_y
            if (pos[0] - x) ** 2 + (pos[1] - y) ** 2 <= 60 ** 2:
                self._try_start_level(level.id)
                return True
        return False

    def _try_start_level(self, level_id):
        if self.on_level_start:
            self.on_level_start(level_id)

    def _check_checkpoint_click(self, pos):
        for checkpoint_key, checkpoint in CHECKPOINTS.items():
            if not checkpoint["is_unlocked"]:
                continue
            x, y = checkpoint["position"]
            x += self.pan_x
            y += self.pan_y
            size = 45
            if (x - size <= pos[0] <= x + size) and (y - size <= pos[1] <= y + size):
                return True
        return False

    def complete_level(self, level_id):
        if level_id not in LEVELS:
            return
        
        LEVELS[level_id].is_completed = True
        self._unlock_checkpoints_for_level(level_id)

    def _unlock_checkpoints_for_level(self, level_id):
        for checkpoint_key, checkpoint in CHECKPOINTS.items():
            if checkpoint["required_level"] == level_id:
                checkpoint["is_unlocked"] = True
                self._unlock_next_level(checkpoint)
                self._add_unlocked_activations(checkpoint)

    def _unlock_next_level(self, checkpoint):
        next_level_id = None
        for level_key, level in LEVELS.items():
            if level.position[0] > checkpoint["position"][0]:
                next_level_id = level_key
                break
        if next_level_id:
            LEVELS[next_level_id].is_unlocked = True

    def _add_unlocked_activations(self, checkpoint):
        for activation in checkpoint["unlocks"]:
            if activation not in self.unlocklist:
                self.unlocklist.append(activation)

    def get_available_activations(self):
        available = []
        for level in LEVELS.values():
            if level.is_unlocked:
                available.extend(level.available_activations)
        return list(set(available))
