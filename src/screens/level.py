import pygame
import numpy as np
from core.config import *
from data.levels import LEVELS, ActivationLayer, OperationType

class Node:
    def __init__(self, node_id, x, y, node_type, input_size=None):
        self.id = node_id
        self.x = x
        self.y = y
        self.type = node_type
        self.radius = 25
        self.input_size = input_size
        self.output_size = input_size
        self.connections = []
        self.is_selected = False

    def contains_point(self, px, py):
        return (px - self.x) ** 2 + (py - self.y) ** 2 <= self.radius ** 2

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)


class Layer:
    def __init__(self, layer_id, layer_type, x, y, input_size, output_size, activation=None):
        self.id = layer_id
        self.type = layer_type
        self.x = x
        self.y = y
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.width = 100
        self.height = 60
        self.is_selected = False
        self.is_dragging = False
        self.input_node = Node(f"{layer_id}_in", x - self.width // 2, y, "input", input_size)
        self.output_node = Node(f"{layer_id}_out", x + self.width // 2, y, "output", output_size)

    def contains_point(self, px, py):
        return (self.x - self.width // 2 <= px <= self.x + self.width // 2 and
                self.y - self.height // 2 <= py <= self.y + self.height // 2)

    def get_rect(self):
        return pygame.Rect(self.x - self.width // 2, self.y - self.height // 2, self.width, self.height)

    def update_position(self, x, y):
        self.x = x
        self.y = y
        self.input_node.x = x - self.width // 2
        self.input_node.y = y
        self.output_node.x = x + self.width // 2
        self.output_node.y = y

    def get_display_name(self):
        if self.activation:
            return self.activation.value
        return self.type.value


class OperationBlock:
    def __init__(self, block_id, operation_type, x, y):
        self.id = block_id
        self.operation = operation_type
        self.x = x
        self.y = y
        self.width = 100
        self.height = 60
        self.is_selected = False
        self.is_dragging = False
        self.input_nodes = [
            Node(f"{block_id}_in1", x - self.width // 2, y - 15, "input", None),
            Node(f"{block_id}_in2", x - self.width // 2, y + 15, "input", None)
        ]
        self.output_node = Node(f"{block_id}_out", x + self.width // 2, y, "output", None)

    def contains_point(self, px, py):
        return (self.x - self.width // 2 <= px <= self.x + self.width // 2 and
                self.y - self.height // 2 <= py <= self.y + self.height // 2)

    def get_rect(self):
        return pygame.Rect(self.x - self.width // 2, self.y - self.height // 2, self.width, self.height)

    def update_position(self, x, y):
        self.x = x
        self.y = y
        self.input_nodes[0].x = x - self.width // 2
        self.input_nodes[0].y = y - 15
        self.input_nodes[1].x = x - self.width // 2
        self.input_nodes[1].y = y + 15
        self.output_node.x = x + self.width // 2
        self.output_node.y = y

    def get_display_name(self):
        return self.operation.value


class Connection:
    def __init__(self, from_node, to_node):
        self.from_node = from_node
        self.to_node = to_node

    def draw(self, screen, color=(100, 100, 100)):
        x1 = self.from_node.x
        y1 = self.from_node.y
        x2 = self.to_node.x
        y2 = self.to_node.y
        
        if self.from_node.id == "input" or self.from_node.type == "input":
            x1 = self.from_node.x + self.from_node.radius
        
        if self.to_node.id == "output" or self.to_node.type == "output":
            x2 = self.to_node.x - self.to_node.radius
        
        self._draw_bezier_curve(screen, x1, y1, x2, y2, color)

    def _draw_bezier_curve(self, screen, x1, y1, x2, y2, color, segments=50):
        mid_x = (x1 + x2) / 2
        control_x1 = x1 + (x2 - x1) * 0.25
        control_x2 = x1 + (x2 - x1) * 0.75
        
        points = []
        for i in range(segments + 1):
            t = i / segments
            t_inv = 1 - t
            
            x = (t_inv ** 3 * x1 + 
                 3 * t_inv ** 2 * t * control_x1 + 
                 3 * t_inv * t ** 2 * control_x2 + 
                 t ** 3 * x2)
            y = (t_inv ** 3 * y1 + 
                 3 * t_inv ** 2 * t * y1 + 
                 3 * t_inv * t ** 2 * y2 + 
                 t ** 3 * y2)
            points.append((x, y))
        
        for i in range(len(points) - 1):
            pygame.draw.line(screen, color, points[i], points[i + 1], 2)


class LevelScreen:
    def __init__(self, screen, level_id):
        self.screen = screen
        self.level = LEVELS[level_id]
        self.level_id = level_id
        self.font_title = pygame.font.SysFont("arial", 32, bold=True)
        self.font_label = pygame.font.SysFont("arial", 16)
        self.font_small = pygame.font.SysFont("arial", 12)

        self._initialize_layout()
        self._initialize_nodes_and_layers()
        self._initialize_state()
        self._generate_training_data()
        self.on_level_complete = None

    def _initialize_layout(self):
        self.inventory_width = 200
        self.progress_height = 100
        self.workspace_x = self.inventory_width
        self.workspace_y = self.progress_height

    def _initialize_nodes_and_layers(self):
        self.input_node = Node("input", 0, 0, "output", self.level.dataset.input_features)
        self.output_node = Node("output", 0, 0, "input", self.level.dataset.output_classes)
        self.layers = []
        self.operations = []
        self.connections = []
        self.layer_counter = 0
        self.operation_counter = 0

    def _initialize_state(self):
        self.selected_layer = None
        self.dragging_layer = None
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.connecting_from_node = None
        self.total_cost = 0
        self.accuracy = 0.0
        self.model_tested = False
        self.model_run = False
        self.layer_counter = 0
        self.training = False
        self.training_progress = 0
        self.loss_history = []
        self.l1_history = []
        self.ce_history = []
        self.epoch = 0
        self.max_epochs = 100
        self.last_width = 0
        self.last_height = 0

    def _generate_training_data(self):
        np.random.seed(42)
        
        if self.level_id == "level_1":
            self.X_train = np.random.uniform(-np.pi, np.pi, (1000, 1))
            self.y_train = np.sin(self.X_train)
            
            self.X_test = np.random.uniform(-np.pi, np.pi, (200, 1))
            self.y_test = np.sin(self.X_test)
        
        elif self.level_id == "level_2":
            n_train = 800
            n_test = 200
            n_classes = 3
            
            self.X_train = np.random.randn(n_train, 2)
            centers = np.array([[2, 2], [-2, -2], [2, -2]])
            y_labels = np.random.randint(0, n_classes, n_train)
            
            for i in range(n_train):
                self.X_train[i] += centers[y_labels[i]] + np.random.randn(2) * 0.5
            
            self.y_train = np.zeros((n_train, n_classes))
            self.y_train[np.arange(n_train), y_labels] = 1
            
            self.X_test = np.random.randn(n_test, 2)
            y_test_labels = np.random.randint(0, n_classes, n_test)
            
            for i in range(n_test):
                self.X_test[i] += centers[y_test_labels[i]] + np.random.randn(2) * 0.5
            
            self.y_test = np.zeros((n_test, n_classes))
            self.y_test[np.arange(n_test), y_test_labels] = 1
        
        elif self.level_id == "level_3":
            self.X_train = np.random.uniform(-3, 3, (1200, 1))
            self.y_train = self.X_train * np.sin(self.X_train ** 2) + np.cos(self.X_train * 2) * 0.5
            
            self.X_test = np.random.uniform(-3, 3, (300, 1))
            self.y_test = self.X_test * np.sin(self.X_test ** 2) + np.cos(self.X_test * 2) * 0.5
        
        else:
            self.X_train = np.random.uniform(-np.pi, np.pi, (1000, 1))
            self.y_train = np.sin(self.X_train)
            
            self.X_test = np.random.uniform(-np.pi, np.pi, (200, 1))
            self.y_test = np.sin(self.X_test)

    def update(self, dt):
        if self.training and self.epoch < self.max_epochs:
            mse_loss, l1_loss, ce_loss = self._train_step()
            self.loss_history.append(mse_loss)
            self.l1_history.append(l1_loss)
            self.ce_history.append(ce_loss)
            self.epoch += 1
            self.training_progress = self.epoch / self.max_epochs
            
            if self.epoch >= self.max_epochs:
                self.training = False
                self._evaluate_model()
                self._update_cost()
                if not self.model_tested:
                    self.model_run = True

    def draw(self, width=None, height=None):
        if width is None:
            width = 1280
        if height is None:
            height = 720

        self.screen.fill(BG_COLOR)
        
        self._ensure_node_positions(width, height)
        
        self._draw_progress_bar(width, height)
        self._draw_inventory(width, height)
        self._draw_workspace(width, height)

    def _ensure_node_positions(self, width, height):
        if self.input_node.x == 0 or width != self.last_width or height != self.last_height:
            input_x = self.workspace_x + 50
            input_y = height // 2
            self.input_node.x = input_x
            self.input_node.y = input_y
            
            output_x = width - 100
            output_y = height // 2
            self.output_node.x = output_x
            self.output_node.y = output_y
            
            self.last_width = width
            self.last_height = height

    def _draw_progress_bar(self, width, height):
        pygame.draw.rect(self.screen, (20, 20, 30), (0, 0, width, self.progress_height))
        pygame.draw.line(self.screen, ACCENT_COLOR, (0, self.progress_height), (width, self.progress_height), 2)

        title = self.font_title.render(self.level.name, True, TEXT_COLOR)
        self.screen.blit(title, (10, 15))
        
        back_button_width = 80
        back_button_height = 30
        back_button_x = width - back_button_width - 10
        back_button_y = 10
        pygame.draw.rect(self.screen, ACCENT_COLOR, (back_button_x, back_button_y, back_button_width, back_button_height), 2)
        back_text = self.font_small.render("← Back", True, TEXT_COLOR)
        back_rect = back_text.get_rect(center=(back_button_x + back_button_width // 2, back_button_y + back_button_height // 2))
        self.screen.blit(back_text, back_rect)

        self._draw_progress_metrics(width)

    def _draw_progress_metrics(self, width):
        y_offset = 60
        
        cost_text = f"Cost: ${self.total_cost:.1f} / ${self.level.goals.cost_limit:.1f}"
        text = self.font_small.render(cost_text, True, TEXT_COLOR)
        self.screen.blit(text, (10, y_offset))
        
        y_offset = 80
        accuracy_text = f"Accuracy: {self.accuracy * 100:.1f}% / {self.level.goals.accuracy_target * 100:.1f}%"
        text = self.font_small.render(accuracy_text, True, TEXT_COLOR)
        self.screen.blit(text, (10, y_offset))

        if self._is_level_passed():
            button_width = 120
            button_height = 30
            button_x = width - 100 - button_width - 10
            button_y = 10
            
            pygame.draw.rect(self.screen, SUCCESS_COLOR, (button_x, button_y, button_width, button_height))
            
            complete_text = self.font_small.render("COMPLETE", True, (0, 0, 0))
            text_rect = complete_text.get_rect(center=(button_x + button_width // 2, button_y + button_height // 2))
            self.screen.blit(complete_text, text_rect)
        else:
            button_width = 120
            button_height = 30
            button_x = width - 100 - button_width - 10
            button_y = 10
            
            pygame.draw.rect(self.screen, ACCENT_COLOR, (button_x, button_y, button_width, button_height), 2)
            status_text = self.font_small.render("IN PROGRESS", True, ACCENT_COLOR)
            text_rect = status_text.get_rect(center=(button_x + button_width // 2, button_y + button_height // 2))
            self.screen.blit(status_text, text_rect)

    def _is_level_passed(self):
        return (self.accuracy >= self.level.goals.accuracy_target and
                self.total_cost <= self.level.goals.cost_limit and
                self.model_run)

    def _draw_inventory(self, width, height):
        pygame.draw.rect(self.screen, (20, 20, 30), (0, self.progress_height, self.inventory_width, height - self.progress_height))
        pygame.draw.line(self.screen, ACCENT_COLOR, (self.inventory_width, self.progress_height),
                        (self.inventory_width, height), 2)

        title = self.font_label.render("Inventory", True, TEXT_COLOR)
        self.screen.blit(title, (10, self.progress_height + 15))
        
        goal_y = self.progress_height + 45
        
        if self.level_id == "level_1":
            goal_text = self.font_small.render("Goal: Build a network", True, (150, 150, 150))
            self.screen.blit(goal_text, (10, goal_y))
            goal_text2 = self.font_small.render("to estimate sin(x)", True, (150, 150, 150))
            self.screen.blit(goal_text2, (10, goal_y + 15))
        elif self.level_id == "level_2":
            goal_text = self.font_small.render("Goal: Classify 3 classes", True, (150, 150, 150))
            self.screen.blit(goal_text, (10, goal_y))
            goal_text2 = self.font_small.render("using probabilities", True, (150, 150, 150))
            self.screen.blit(goal_text2, (10, goal_y + 15))
        elif self.level_id == "level_3":
            goal_text = self.font_small.render("Goal: Approximate", True, (150, 150, 150))
            self.screen.blit(goal_text, (10, goal_y))
            goal_text2 = self.font_small.render("complex function", True, (150, 150, 150))
            self.screen.blit(goal_text2, (10, goal_y + 15))
        else:
            goal_text = self.font_small.render("Goal: Complete the", True, (150, 150, 150))
            self.screen.blit(goal_text, (10, goal_y))
            goal_text2 = self.font_small.render("task", True, (150, 150, 150))
            self.screen.blit(goal_text2, (10, goal_y + 15))

        self._draw_available_layers(height)
        self._draw_inventory_buttons(width, height)

    def _draw_available_layers(self, height):
        y_offset = self.progress_height + 80
        for activation in self.level.available_activations:
            self._draw_inventory_item(activation.value, 10, y_offset)
            y_offset += 35
        
        for operation in self.level.available_operations:
            self._draw_inventory_item(operation.value, 10, y_offset)
            y_offset += 35

    def _draw_inventory_item(self, name, x, y):
        pygame.draw.rect(self.screen, SECONDARY_COLOR, (x, y, self.inventory_width - 20, 30), 2)
        text = self.font_small.render(name, True, TEXT_COLOR)
        text_rect = text.get_rect(center=(x + (self.inventory_width - 20) // 2, y + 15))
        self.screen.blit(text, text_rect)

    def _draw_inventory_buttons(self, width, height):
        button_y = height - 120
        button_height = 35
        button_width = self.inventory_width - 20

        self._draw_button("Test Model", 10, button_y, button_width, button_height)
        self._draw_button("Run", 10, button_y + 45, button_width, button_height)

    def _draw_button(self, text, x, y, w, h):
        pygame.draw.rect(self.screen, ACCENT_COLOR, (x, y, w, h), 2)
        text_surf = self.font_small.render(text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=(x + w // 2, y + h // 2))
        self.screen.blit(text_surf, text_rect)

    def _draw_workspace(self, width, height):
        workspace_rect = pygame.Rect(self.workspace_x, self.workspace_y,
                                     width - self.workspace_x, height - self.workspace_y)
        pygame.draw.rect(self.screen, (15, 15, 25), workspace_rect)
        pygame.draw.rect(self.screen, (40, 40, 50), workspace_rect, 1)

        self._draw_workspace_nodes()
        self._draw_workspace_connections()
        self._draw_workspace_layers()
        
        if self.training or self.loss_history:
            self._draw_training_graph(width, height)
        
    def _draw_workspace_nodes(self):
        self._draw_node(self.input_node, "INPUT")
        self._draw_node(self.output_node, "OUTPUT")

    def _draw_node(self, node, label):
        color = ACCENT_COLOR if node.is_selected else (80, 100, 120)
        pygame.draw.circle(self.screen, color, (int(node.x), int(node.y)), node.radius)
        pygame.draw.circle(self.screen, TEXT_COLOR, (int(node.x), int(node.y)), node.radius + 2, 2)

        text = self.font_small.render(label, True, (0, 0, 0) if node.is_selected else TEXT_COLOR)
        text_rect = text.get_rect(center=(int(node.x), int(node.y)))
        self.screen.blit(text, text_rect)

        size_text = self.font_small.render(f"[{node.input_size}]", True, (150, 150, 150))
        self.screen.blit(size_text, (node.x + 35, node.y - 10))
        
        if self.connecting_from_node == node:
            pygame.draw.circle(self.screen, SUCCESS_COLOR, (int(node.x), int(node.y)), node.radius + 5, 3)

    def _draw_workspace_connections(self):
        for connection in self.connections:
            connection.draw(self.screen)

        if self.connecting_from_node:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            pygame.draw.line(self.screen, ACCENT_COLOR,
                           (self.connecting_from_node.x, self.connecting_from_node.y),
                           (mouse_x, mouse_y), 2)

    def _draw_workspace_layers(self):
        for layer in self.layers:
            self._draw_layer(layer)
        
        for operation in self.operations:
            self._draw_operation(operation)

    def _draw_layer(self, layer):
        color = SECONDARY_COLOR if layer.is_selected else (60, 80, 100)
        pygame.draw.rect(self.screen, color, layer.get_rect(), 2)

        display_name = layer.get_display_name()
        for i, line in enumerate(display_name.split('\n')):
            text = self.font_label.render(line, True, TEXT_COLOR)
            text_rect = text.get_rect(center=(layer.x, layer.y - 5 + i * 15))
            self.screen.blit(text, text_rect)

        self._draw_layer_size_info(layer)
        self._draw_layer_nodes(layer)

    def _draw_layer_size_info(self, layer):
        actual_input = self._get_layer_input_size(layer)
        info_text = f"{actual_input} → {layer.output_size}"
        text = self.font_small.render(info_text, True, (150, 150, 150))
        text_rect = text.get_rect(center=(layer.x, layer.y + 20))
        self.screen.blit(text, text_rect)

    def _draw_layer_nodes(self, layer):
        pygame.draw.circle(self.screen, ACCENT_COLOR, (int(layer.input_node.x), int(layer.input_node.y)), 5)
        pygame.draw.circle(self.screen, ACCENT_COLOR, (int(layer.output_node.x), int(layer.output_node.y)), 5)

    def _draw_operation(self, operation):
        color = SECONDARY_COLOR if operation.is_selected else (100, 60, 80)
        pygame.draw.rect(self.screen, color, operation.get_rect(), 2)

        display_name = operation.get_display_name()
        text = self.font_label.render(display_name, True, TEXT_COLOR)
        text_rect = text.get_rect(center=(operation.x, operation.y))
        self.screen.blit(text, text_rect)

        for input_node in operation.input_nodes:
            pygame.draw.circle(self.screen, ACCENT_COLOR, (int(input_node.x), int(input_node.y)), 5)
        pygame.draw.circle(self.screen, ACCENT_COLOR, (int(operation.output_node.x), int(operation.output_node.y)), 5)

    def _draw_training_graph(self, width, height):
        graph_x = self.workspace_x + 20
        graph_y = 110
        graph_width = 450
        graph_height = 180
        
        pygame.draw.rect(self.screen, (25, 25, 35), (graph_x, graph_y, graph_width, graph_height))
        pygame.draw.rect(self.screen, ACCENT_COLOR, (graph_x, graph_y, graph_width, graph_height), 1)
        
        title = self.font_label.render("Training Loss", True, TEXT_COLOR)
        self.screen.blit(title, (graph_x + 10, graph_y + 5))
        
        if len(self.loss_history) > 1:
            all_losses = self.loss_history + self.ce_history
            max_loss = max(all_losses)
            min_loss = min(all_losses)
            loss_range = max(max_loss - min_loss, 0.1)
            
            for i in range(len(self.loss_history) - 1):
                x1 = graph_x + 10 + (i / max(len(self.loss_history) - 1, 1)) * (graph_width - 20)
                x2 = graph_x + 10 + ((i + 1) / max(len(self.loss_history), 1)) * (graph_width - 20)
                
                y1_mse = graph_y + graph_height - 30 - ((self.loss_history[i] - min_loss) / loss_range) * (graph_height - 60)
                y2_mse = graph_y + graph_height - 30 - ((self.loss_history[i + 1] - min_loss) / loss_range) * (graph_height - 60)
                pygame.draw.line(self.screen, (100, 150, 255), (x1, y1_mse), (x2, y2_mse), 3)
                
                y1_ce = graph_y + graph_height - 30 - ((self.ce_history[i] - min_loss) / loss_range) * (graph_height - 60)
                y2_ce = graph_y + graph_height - 30 - ((self.ce_history[i + 1] - min_loss) / loss_range) * (graph_height - 60)
                pygame.draw.line(self.screen, (255, 150, 100), (x1, y1_ce), (x2, y2_ce), 3)
        
        if not self.training and len(self.loss_history) > 0:
            accuracy_pct = self.accuracy * 100
            pct_text = self.font_label.render(f"{accuracy_pct:.1f}%", True, SUCCESS_COLOR if accuracy_pct >= 85 else SECONDARY_COLOR)
            pct_rect = pct_text.get_rect(topright=(graph_x + graph_width - 10, graph_y + 5))
            self.screen.blit(pct_text, pct_rect)
        
        legend_y = graph_y + graph_height - 18
        pygame.draw.line(self.screen, (100, 150, 255), (graph_x + 10, legend_y), (graph_x + 30, legend_y), 3)
        self.screen.blit(self.font_small.render("MSE", True, TEXT_COLOR), (graph_x + 35, legend_y - 6))
        
        pygame.draw.line(self.screen, (255, 150, 100), (graph_x + 80, legend_y), (graph_x + 100, legend_y), 3)
        self.screen.blit(self.font_small.render("CE", True, TEXT_COLOR), (graph_x + 105, legend_y - 6))
        
        if self.training:
            progress_text = self.font_small.render(f"Training: {self.training_progress * 100:.0f}%", True, ACCENT_COLOR)
            text_rect = progress_text.get_rect(right=graph_x + graph_width - 10, centery=legend_y)
            self.screen.blit(progress_text, text_rect)
        elif len(self.loss_history) > 0:
            final_text = self.font_small.render(f"MSE: {self.loss_history[-1]:.3f} CE: {self.ce_history[-1]:.3f}", True, (150, 150, 150))
            text_rect = final_text.get_rect(right=graph_x + graph_width - 10, centery=legend_y)
            self.screen.blit(final_text, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self._handle_mouse_down(event)
        elif event.type == pygame.MOUSEBUTTONUP:
            self._handle_mouse_up(event)
        elif event.type == pygame.MOUSEMOTION:
            self._handle_mouse_motion(event)
        elif event.type == pygame.KEYDOWN:
            self._handle_key_down(event)

    def _handle_mouse_down(self, event):
        mouse_pos = pygame.mouse.get_pos()
        
        if self._check_back_button_click(mouse_pos):
            return

        if self._check_complete_button_click(mouse_pos):
            return

        if event.button == 3:
            self._handle_right_click(mouse_pos)
            return

        if self._check_inventory_click(mouse_pos):
            return

        height = self.screen.get_height()
        if self._check_button_click(mouse_pos, height):
            return

        if self._check_node_click(mouse_pos):
            return

        if self._check_layer_click(mouse_pos):
            return

    def _check_back_button_click(self, pos):
        width = self.screen.get_width()
        back_button_width = 80
        back_button_height = 30
        back_button_x = width - back_button_width - 10
        back_button_y = 10
        back_rect = pygame.Rect(back_button_x, back_button_y, back_button_width, back_button_height)
        if back_rect.collidepoint(pos):
            if self.on_level_complete:
                self.on_level_complete()
            return True
        return False

    def _check_complete_button_click(self, pos):
        if not self._is_level_passed():
            return False
        
        width = self.screen.get_width()
        button_width = 120
        button_height = 30
        button_x = width - 100 - button_width - 10
        button_y = 10
        
        complete_button = pygame.Rect(button_x, button_y, button_width, button_height)
        
        if complete_button.collidepoint(pos):
            self._complete_level()
            return True
        
        return False

    def _complete_level(self):
        LEVELS[self.level_id].is_completed = True
        if self.on_level_complete:
            self.on_level_complete()

    def _check_inventory_click(self, pos):
        if pos[0] > self.inventory_width:
            return False

        y_offset = self.progress_height + 80
        for activation in self.level.available_activations:
            item_rect = pygame.Rect(10, y_offset, self.inventory_width - 20, 30)
            if item_rect.collidepoint(pos):
                self._create_layer_from_activation(activation)
                return True
            y_offset += 35
        
        for operation in self.level.available_operations:
            item_rect = pygame.Rect(10, y_offset, self.inventory_width - 20, 30)
            if item_rect.collidepoint(pos):
                self._create_operation_block(operation)
                return True
            y_offset += 35

        return False

    def _create_layer_from_activation(self, activation):
        self.layer_counter += 1
        layer_id = f"layer_{self.layer_counter}"
        x = 500 + self.layer_counter * 30
        y = 300 + (self.layer_counter % 2) * 100

        layer = Layer(layer_id, self.level.available_layers[1], x, y,
                     self.level.dataset.input_features, 20, activation)
        self.layers.append(layer)
        self._update_cost()

    def _create_operation_block(self, operation):
        self.operation_counter += 1
        block_id = f"op_{self.operation_counter}"
        x = 500 + self.operation_counter * 30
        y = 300 + (self.operation_counter % 2) * 100

        operation_block = OperationBlock(block_id, operation, x, y)
        self.operations.append(operation_block)
        self._update_cost()

    def _check_button_click(self, pos, height):
        if pos[0] > self.inventory_width:
            return False
            
        button_y = height - 120
        test_button = pygame.Rect(10, button_y, self.inventory_width - 20, 35)
        run_button = pygame.Rect(10, button_y + 45, self.inventory_width - 20, 35)

        if test_button.collidepoint(pos):
            self._test_model()
            return True
        if run_button.collidepoint(pos):
            self._run_model()
            return True
            
        return False

    def _check_node_click(self, pos):
        if self.input_node.contains_point(pos[0], pos[1]):
            self._start_connection(self.input_node)
            return True

        if self.output_node.contains_point(pos[0], pos[1]):
            self._start_connection(self.output_node)
            return True

        for layer in self.layers:
            if layer.input_node.contains_point(pos[0], pos[1]):
                self._start_connection(layer.input_node)
                return True
            if layer.output_node.contains_point(pos[0], pos[1]):
                self._start_connection(layer.output_node)
                return True

        for operation in self.operations:
            for input_node in operation.input_nodes:
                if input_node.contains_point(pos[0], pos[1]):
                    self._start_connection(input_node)
                    return True
            if operation.output_node.contains_point(pos[0], pos[1]):
                self._start_connection(operation.output_node)
                return True

        return False

    def _start_connection(self, from_node):
        if self.connecting_from_node is None:
            self.connecting_from_node = from_node
        else:
            self._complete_connection(from_node)

    def _complete_connection(self, to_node):
        if self.connecting_from_node and self._is_valid_connection(self.connecting_from_node, to_node):
            from_node = self.connecting_from_node
            if from_node.type == "input" and to_node.type == "output":
                from_node, to_node = to_node, from_node
            connection = Connection(from_node, to_node)
            self.connections.append(connection)
            self._update_layer_sizes()
        self.connecting_from_node = None

    def _is_valid_connection(self, from_node, to_node):
        if from_node == to_node:
            return False
        
        if not ((from_node.type == "output" and to_node.type == "input") or 
                (from_node.type == "input" and to_node.type == "output")):
            return False
        
        actual_from = from_node if from_node.type == "output" else to_node
        actual_to = to_node if to_node.type == "input" else from_node
        
        if actual_from.id == "input" and actual_to.id == "output":
            return False
        
        existing = any((c.from_node == actual_from and c.to_node == actual_to) or
                      (c.from_node == actual_to and c.to_node == actual_from) for c in self.connections)
        if existing:
            return False
        
        for operation in self.operations:
            for input_node in operation.input_nodes:
                if actual_to == input_node:
                    has_connection = any(c.to_node == input_node for c in self.connections)
                    if has_connection:
                        return False
        
        return True

    def _check_layer_click(self, pos):
        for operation in self.operations:
            if operation.contains_point(pos[0], pos[1]):
                self.selected_layer = operation
                self.dragging_layer = operation
                self.drag_offset_x = pos[0] - operation.x
                self.drag_offset_y = pos[1] - operation.y
                operation.is_selected = True
                return True

        for layer in self.layers:
            if layer.contains_point(pos[0], pos[1]):
                self.selected_layer = layer
                self.dragging_layer = layer
                self.drag_offset_x = pos[0] - layer.x
                self.drag_offset_y = pos[1] - layer.y
                layer.is_selected = True
                return True

        self.selected_layer = None
        for layer in self.layers:
            layer.is_selected = False
        for operation in self.operations:
            operation.is_selected = False
        return False

    def _handle_mouse_up(self, event):
        if self.dragging_layer:
            self.dragging_layer = None

    def _handle_mouse_motion(self, event):
        if self.dragging_layer:
            new_x = event.pos[0] - self.drag_offset_x
            new_y = event.pos[1] - self.drag_offset_y
            self.dragging_layer.update_position(new_x, new_y)

    def _handle_right_click(self, pos):
        for operation in self.operations:
            if operation.contains_point(pos[0], pos[1]):
                self._delete_operation(operation)
                return
        
        for layer in self.layers:
            if layer.contains_point(pos[0], pos[1]):
                self._delete_layer(layer)
                return

    def _handle_key_down(self, event):
        if event.key == pygame.K_DELETE and self.selected_layer:
            self._delete_layer(self.selected_layer)
        if event.key == pygame.K_ESCAPE:
            self.connecting_from_node = None

    def _delete_layer(self, layer):
        if layer in self.layers:
            self.layers.remove(layer)
            self.connections = [c for c in self.connections
                              if (c.from_node.id != layer.input_node.id and 
                                  c.from_node.id != layer.output_node.id and
                                  c.to_node.id != layer.input_node.id and 
                                  c.to_node.id != layer.output_node.id)]
            self._update_cost()

    def _delete_operation(self, operation):
        if operation in self.operations:
            self.operations.remove(operation)
            self.connections = [c for c in self.connections
                              if (c.from_node not in operation.input_nodes and 
                                  c.from_node != operation.output_node and
                                  c.to_node not in operation.input_nodes and 
                                  c.to_node != operation.output_node)]
            self._update_cost()

    def _test_model(self):
        self.model_run = False
        self._run_model()
        self.model_tested = True

    def _run_model(self):
        if not self.layers or not self._validate_network():
            self.accuracy = 0.0
            self.loss_history = []
            self.l1_history = []
            self.ce_history = []
            self.model_run = False
            self.model_tested = False
            return

        self.training = True
        self.training_progress = 0.0
        self.epoch = 0
        self.loss_history = []
        self.l1_history = []
        self.ce_history = []
        self.accuracy = 0.0
        self.model_run = False
        self.model_tested = False
        self._update_layer_sizes()
        self._initialize_network()

    def _validate_network(self):
        if not self.connections:
            return False
        
        input_connected = any(c.from_node == self.input_node for c in self.connections)
        output_connected = any(c.to_node == self.output_node for c in self.connections)
        
        return input_connected and output_connected

    def _initialize_network(self):
        self.weights = []
        self.biases = []
        
        sorted_layers = self._get_sorted_layers()
        layer_sizes = [self.level.dataset.input_features]
        for layer in sorted_layers:
            layer_sizes.append(layer.output_size)
        layer_sizes.append(self.level.dataset.output_classes)
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.1
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def _apply_activation(self, x, activation):
        if activation == ActivationLayer.RELU:
            return np.maximum(0, x)
        elif activation == ActivationLayer.TANH:
            return np.tanh(x)
        elif activation == ActivationLayer.SOFTMAX:
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        elif activation == ActivationLayer.SIGMOID:
            return 1 / (1 + np.exp(-x))
        return x

    def _activation_derivative(self, x, activation):
        if activation == ActivationLayer.RELU:
            return (x > 0).astype(float)
        elif activation == ActivationLayer.TANH:
            return 1 - np.tanh(x) ** 2
        elif activation == ActivationLayer.SIGMOID:
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        return np.ones_like(x)

    def _train_step(self):
        learning_rate = 0.05
        batch_size = 32
        
        indices = np.random.choice(len(self.X_train), batch_size, replace=False)
        X_batch = self.X_train[indices]
        y_batch = self.y_train[indices]
        
        sorted_layers = self._get_sorted_layers()
        activations = [X_batch]
        zs = []
        
        for i in range(len(self.weights)):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            zs.append(z)
            if i < len(sorted_layers):
                a = self._apply_activation(z, sorted_layers[i].activation)
            else:
                a = z
            activations.append(a)
        
        predictions = activations[-1]
        mse_loss = np.mean((predictions - y_batch) ** 2)
        l1_loss = np.mean(np.abs(predictions - y_batch))
        
        predictions_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)
        y_clipped = np.clip(y_batch, 1e-7, 1 - 1e-7)
        ce_loss = -np.mean(y_clipped * np.log(predictions_clipped) + (1 - y_clipped) * np.log(1 - predictions_clipped))
        
        delta = 2 * (predictions - y_batch) / batch_size
        for i in range(len(self.weights) - 1, -1, -1):
            self.weights[i] -= learning_rate * (activations[i].T @ delta)
            self.biases[i] -= learning_rate * np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                if i - 1 < len(sorted_layers):
                    delta = (delta @ self.weights[i].T) * self._activation_derivative(zs[i-1], sorted_layers[i-1].activation)
                else:
                    delta = delta @ self.weights[i].T
        
        return mse_loss, l1_loss, ce_loss

    def _evaluate_model(self):
        sorted_layers = self._get_sorted_layers()
        activations = self.X_test
        for i in range(len(self.weights)):
            z = activations @ self.weights[i] + self.biases[i]
            if i < len(sorted_layers):
                activations = self._apply_activation(z, sorted_layers[i].activation)
            else:
                activations = z
        
        predictions = activations
        
        if self.level.dataset.output_classes > 1:
            predicted_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(self.y_test, axis=1)
            self.accuracy = np.mean(predicted_classes == true_classes)
        else:
            mse = np.mean((predictions - self.y_test) ** 2)
            self.accuracy = max(0, 1 - mse)

    def _update_cost(self):
        self.total_cost = sum(10.0 + layer.output_size * 0.1 for layer in self.layers)

    def _get_layer_input_size(self, layer):
        for conn in self.connections:
            if conn.to_node.id == layer.input_node.id:
                if conn.from_node.id == "input":
                    return self.level.dataset.input_features
                for other_layer in self.layers:
                    if other_layer.output_node.id == conn.from_node.id:
                        return other_layer.output_size
        return layer.input_size

    def _get_layer_output_size(self, layer):
        for conn in self.connections:
            if conn.from_node.id == layer.output_node.id:
                if conn.to_node.id == "output":
                    return self.level.dataset.output_classes
        return layer.output_size

    def _update_layer_sizes(self):
        for layer in self.layers:
            layer.input_size = self._get_layer_input_size(layer)
            layer.input_node.input_size = layer.input_size
            layer.output_size = self._get_layer_output_size(layer)
            layer.output_node.output_size = layer.output_size

    def _get_sorted_layers(self):
        sorted_layers = []
        visited = set()
        
        def add_layer_and_dependencies(layer):
            if layer.id in visited:
                return
            for conn in self.connections:
                if conn.to_node.id == layer.input_node.id:
                    for other_layer in self.layers:
                        if other_layer.output_node.id == conn.from_node.id:
                            add_layer_and_dependencies(other_layer)
            visited.add(layer.id)
            sorted_layers.append(layer)
        
        for layer in self.layers:
            add_layer_and_dependencies(layer)
        
        return sorted_layers

    def get_model_config(self):
        return {
            "input_size": self.input_node.input_size,
            "output_size": self.output_node.output_size,
            "layers": [(layer.type, layer.output_size, layer.activation) for layer in self.layers],
            "connections": [(c.from_node.id, c.to_node.id) for c in self.connections]
        }
