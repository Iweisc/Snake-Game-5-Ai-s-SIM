import pygame
import random
import math
import logging
from collections import deque
import os
from datetime import datetime

# Set up logging
def setup_logging():
    # Create logs directory if it doesn't exist
    logs_dir = 'game_logs'
    os.makedirs(logs_dir, exist_ok=True)
    # Generate unique log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(logs_dir, f'snake_game_log_{timestamp}.txt')
    # Configure logging - Set to INFO to reduce logging overhead
    logging.basicConfig(
        level=logging.INFO,  # Changed from DEBUG to INFO
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return log_filename

# Pygame and game setup
pygame.init()
width, height = 1920, 950
win = pygame.display.set_mode((width, height))
pygame.display.set_caption('AI Pathfinding Arena - Slither Edition')

# Colors and game constants
black      = (0, 0, 0)
dark_gray  = (40, 40, 40)
white      = (255, 255, 255)
green      = (0, 255, 0)       # A* snake
blue       = (0, 0, 255)       # Dijkstra snake
red        = (255, 0, 0)       # Random snake
purple     = (128, 0, 128)     # BFS snake
orange     = (255, 165, 0)     # Greedy Best-First snake
yellow     = (255, 255, 0)
pink       = (255, 192, 203)   # Speed food
cyan       = (0, 255, 255)     # Portal food
silver     = (192, 192, 192)   # Obstacles
gold       = (255, 215, 0)     # Bonus food

# Gambling-related colors
casino_gold = (255, 215, 0)
casino_red = (220, 20, 60)
casino_green = (50, 205, 50)
jackpot_purple = (153, 50, 204)

# Configure Slither.io style settings
snake_block = 10
snake_speed = 30  # Increased for smoother movement
segment_spacing = 5  # Spacing between segments for smoother appearance
font_style = pygame.font.SysFont("bahnschrift", 20)
score_font = pygame.font.SysFont("segoeui", 30)
title_font = pygame.font.SysFont("segoeui", 36)
bet_font = pygame.font.SysFont("segoeui", 24)

# Pre-calculate all possible moves to avoid recreating this list repeatedly
ALL_MOVES = [
    (0, -snake_block),     # Up
    (0, snake_block),      # Down
    (-snake_block, 0),     # Left
    (snake_block, 0)       # Right
]

# Player betting variables
player_coins = 1000  # Starting coins
current_bet = 0
bet_snake_id = None
bet_placed = False
bet_result = None
bet_multipliers = {1: 2.5, 2: 3.0, 3: 4.0, 4: 3.5, 5: 3.0}  # Multipliers for each snake

# Game objects
class Food:
    def __init__(self, pos, value=1, color=white, type='regular', size=None):
        self.pos = pos
        self.value = value
        self.color = color
        self.type = type
        self.size = size if size else snake_block // 2
        self.creation_time = pygame.time.get_ticks()

    def draw(self, surface):
        pos = self.pos
        if self.type == 'regular':
            pygame.draw.circle(surface, self.color,
                          (int(pos[0]), int(pos[1])),
                          self.size)
        elif self.type == 'super':
            pygame.draw.circle(surface, self.color,
                          (int(pos[0]), int(pos[1])),
                          self.size)
            # Add pulsating effect
            pulse = abs(math.sin((pygame.time.get_ticks() - self.creation_time) * 0.005)) * 3
            pygame.draw.circle(surface, self.color,
                          (int(pos[0]), int(pos[1])),
                          self.size + pulse, 1)
        elif self.type == 'speed':
            pygame.draw.circle(surface, self.color,
                          (int(pos[0]), int(pos[1])),
                          self.size)
            # Add speed lines
            for i in range(4):
                angle = i * math.pi/2
                end_x = pos[0] + math.cos(angle) * self.size * 2
                end_y = pos[1] + math.sin(angle) * self.size * 2
                pygame.draw.line(surface, self.color,
                             (pos[0], pos[1]),
                             (end_x, end_y), 1)
        elif self.type == 'bonus':
            # Draw star shape
            points = []
            for i in range(5):
                # Outer points
                angle = i * 2 * math.pi / 5 - math.pi/2
                x = pos[0] + math.cos(angle) * self.size
                y = pos[1] + math.sin(angle) * self.size
                points.append((x, y))
                # Inner points
                angle += math.pi / 5
                x = pos[0] + math.cos(angle) * self.size / 2
                y = pos[1] + math.sin(angle) * self.size / 2
                points.append((x, y))
            pygame.draw.polygon(surface, self.color, points)
        elif self.type == 'portal':
            # Draw portal effect
            pulse = abs(math.sin((pygame.time.get_ticks() - self.creation_time) * 0.005)) * 3
            for radius in range(1, 5):
                pygame.draw.circle(surface, self.color,
                              (int(pos[0]), int(pos[1])),
                              self.size - radius + pulse, 1)
        elif self.type == 'remains':
            # Draw with slight transparency for remains
            s = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.color, 200), (self.size, self.size), self.size)
            surface.blit(s, (int(pos[0])-self.size, int(pos[1])-self.size))
        elif self.type == 'jackpot':
            # Draw jackpot food with special effects
            pygame.draw.circle(surface, self.color,
                          (int(pos[0]), int(pos[1])),
                          self.size)
            # Add dollar sign in the middle
            if self.size >= 6:
                dollar_color = (255, 255, 255)
                font = pygame.font.SysFont("arial", self.size)
                dollar = font.render("$", True, dollar_color)
                surface.blit(dollar, (int(pos[0] - dollar.get_width()/2),
                                  int(pos[1] - dollar.get_height()/2)))
            # Rotating outer ring
            angle = (pygame.time.get_ticks() / 5) % 360
            for i in range(8):
                outer_angle = angle + (i * 45)
                rad_angle = math.radians(outer_angle)
                outer_x = pos[0] + math.cos(rad_angle) * (self.size * 1.5)
                outer_y = pos[1] + math.sin(rad_angle) * (self.size * 1.5)
                pygame.draw.circle(surface, casino_gold, (int(outer_x), int(outer_y)), 2)

class RiskZone:
    def __init__(self, pos, radius, multiplier):
        self.pos = pos
        self.radius = radius
        self.multiplier = multiplier  # Reward multiplier when inside
        self.creation_time = pygame.time.get_ticks()
        self.active_time = random.randint(10000, 20000)  # 10-20 seconds lifetime
        self.danger_level = random.uniform(0.5, 2.0)  # Higher = more dangerous

    def is_active(self):
        current_time = pygame.time.get_ticks()
        return current_time - self.creation_time < self.active_time

    def contains_point(self, point):
        return euclidean_distance(self.pos, point) <= self.radius

    def draw(self, surface):
        if not self.is_active():
            return
        # Calculate opacity based on remaining time
        time_percent = 1.0 - ((pygame.time.get_ticks() - self.creation_time) / self.active_time)
        alpha = int(100 * time_percent)
        # Create transparent surface
        zone_surf = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        # Draw filled circle with pulsing effect
        pulse = abs(math.sin(pygame.time.get_ticks() * 0.002)) * 0.2
        color = (casino_red[0], casino_red[1], casino_red[2], alpha)
        pygame.draw.circle(zone_surf, color, (self.radius, self.radius),
                       self.radius * (0.9 + pulse))
        # Draw border
        border_color = (casino_gold[0], casino_gold[1], casino_gold[2],
                     min(255, alpha + 50))
        pygame.draw.circle(zone_surf, border_color, (self.radius, self.radius),
                       self.radius, 3)
        # Draw multiplier text
        font = pygame.font.SysFont("impact", 30)
        text = font.render(f"{self.multiplier}x", True, (255, 255, 255))
        zone_surf.blit(text, (self.radius - text.get_width()//2,
                          self.radius - text.get_height()//2))
        # Draw dollar signs around the perimeter
        for i in range(8):
            angle = (pygame.time.get_ticks() / 10 + i * 45) % 360
            rad_angle = math.radians(angle)
            x = self.radius + math.cos(rad_angle) * (self.radius * 0.7)
            y = self.radius + math.sin(rad_angle) * (self.radius * 0.7)
            dollar = font.render("$", True, (255, 255, 255, int(150 * time_percent)))
            zone_surf.blit(dollar, (x - dollar.get_width()//2, y - dollar.get_height()//2))
        # Blit to main surface
        surface.blit(zone_surf, (int(self.pos[0] - self.radius), int(self.pos[1] - self.radius)))

class Obstacle:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)

    def draw(self, surface):
        pygame.draw.rect(surface, silver, self.rect)
        pygame.draw.rect(surface, dark_gray, self.rect, 1)

class Snake:
    def __init__(self, start_pos, color, algorithm_name, algorithm_id):
        self.segments = [start_pos]  # Head is at index 0
        self.exact_positions = [list(start_pos)]  # For smooth movement
        self.direction = [1, 0]  # Initial direction vector
        self.length = 5  # Start with a bit more length
        self.score = 0
        self.color = color
        self.algorithm = algorithm_name
        self.algorithm_id = algorithm_id
        self.path = []
        self.path_target = None
        self.speed_boost = 0
        self.target_indicator = None
        self.alive = True
        self.speed = 2.0  # Base speed
        self.boost_available = 100  # Boost meter (0-100)
        self.boosting = False
        self.radius = snake_block // 2  # Base segment size
        self.glow_colors = []
        self.invincible_frames = 60  # Start with 60 frames (1 second) of invincibility
        # Gambling related properties
        self.multiplier = 1.0  # Current score multiplier
        self.in_risk_zone = False
        self.jackpot_mode = False
        self.jackpot_timer = 0
        self.recent_winnings = []  # List of recent coin wins to show
        # Generate slightly varying colors for glow effect
        base_r, base_g, base_b = color
        for i in range(10):
            factor = 0.7 + (i / 10) * 0.5  # Values from 0.7 to 1.2
            r = min(255, int(base_r * factor))
            g = min(255, int(base_g * factor))
            b = min(255, int(base_b * factor))
            self.glow_colors.append((r, g, b))

    def get_head(self):
        return self.segments[0]

    def get_head_exact(self):
        return self.exact_positions[0]

    def move(self, target, dt):
        # Update invincibility frames
        if self.invincible_frames > 0:
            self.invincible_frames -= 1
        # Calculate direction vector to target
        head = self.get_head_exact()
        dx = target[0] - head[0]
        dy = target[1] - head[1]
        # Normalize direction
        distance = math.sqrt(dx*dx + dy*dy)
        if distance > 0:
            dx /= distance
            dy /= distance
        # Update direction gradually (smooth turning)
        turn_rate = 0.15
        self.direction[0] = self.direction[0] * (1-turn_rate) + dx * turn_rate
        self.direction[1] = self.direction[1] * (1-turn_rate) + dy * turn_rate
        # Normalize direction again
        dir_len = math.sqrt(self.direction[0]**2 + self.direction[1]**2)
        if dir_len > 0:
            self.direction[0] /= dir_len
            self.direction[1] /= dir_len
        # Calculate current speed
        current_speed = self.speed
        if self.boosting and self.boost_available > 0:
            current_speed *= 1.7  # 70% speed boost
            self.boost_available -= 1
        elif not self.boosting and self.boost_available < 100:
            self.boost_available += 0.2  # Recharge boost
        # Jackpot mode speed boost
        if self.jackpot_mode:
            current_speed *= 1.5
            self.jackpot_timer -= 1
            if self.jackpot_timer <= 0:
                self.jackpot_mode = False
        # Move head
        new_x = head[0] + self.direction[0] * current_speed * dt
        new_y = head[1] + self.direction[1] * current_speed * dt
        # Ensure head stays within bounds
        new_x = max(self.radius, min(width - self.radius, new_x))
        new_y = max(self.radius, min(height - self.radius, new_y))
        # Update head position
        self.exact_positions[0] = [new_x, new_y]
        self.segments[0] = (int(new_x), int(new_y))
        # Update body segments to follow the head
        target_spacing = segment_spacing
        if self.boosting:
            target_spacing *= 1.5  # Increase spacing when boosting
        for i in range(1, len(self.segments)):
            # Calculate direction to segment ahead
            prev = self.exact_positions[i-1]
            curr = self.exact_positions[i]
            # Move segment toward previous segment
            dx = prev[0] - curr[0]
            dy = prev[1] - curr[1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > target_spacing:
                # Only move if we're further than the target spacing
                move_dist = min(dist - target_spacing, current_speed * dt)
                if dist > 0:  # Avoid division by zero
                    curr[0] += (dx / dist) * move_dist
                    curr[1] += (dy / dist) * move_dist
                self.exact_positions[i] = curr
                self.segments[i] = (int(curr[0]), int(curr[1]))
        # Update recent winnings display (fade them out over time)
        for i, win_info in enumerate(self.recent_winnings[:]):
            win_info['time'] -= 1
            if win_info['time'] <= 0:
                self.recent_winnings.remove(win_info)

    def set_boost(self, boosting):
        self.boosting = boosting and self.boost_available > 0

    def grow(self, amount=1):
        # Add new segments at the end of the snake
        for _ in range(amount):
            if len(self.segments) > 0:
                # Get the last segment's position
                last_pos = self.exact_positions[-1].copy() if self.exact_positions else [width//2, height//2]
                # Add a slight offset to prevent exact overlap
                if len(self.exact_positions) > 1:
                    second_last = self.exact_positions[-2]
                    direction = [last_pos[0] - second_last[0], last_pos[1] - second_last[1]]
                    dir_len = math.sqrt(direction[0]**2 + direction[1]**2)
                    if dir_len > 0:
                        direction[0] /= dir_len
                        direction[1] /= dir_len
                        # Place new segment slightly behind the last one
                        last_pos[0] += direction[0] * segment_spacing
                        last_pos[1] += direction[1] * segment_spacing
                else:
                    # For the first body segment, place it behind the head based on direction
                    direction = [-self.direction[0], -self.direction[1]]  # Opposite of movement direction
                    last_pos[0] += direction[0] * segment_spacing * 3  # Triple spacing for safety
                    last_pos[1] += direction[1] * segment_spacing * 3
                self.segments.append((int(last_pos[0]), int(last_pos[1])))
                self.exact_positions.append(last_pos)

    def activate_jackpot_mode(self, duration=180):
        self.jackpot_mode = True
        self.jackpot_timer = duration

    def add_winnings(self, amount):
        self.recent_winnings.append({
            'amount': amount,
            'pos': [self.get_head()[0], self.get_head()[1] - 30],
            'time': 60,  # Display for 60 frames
            'vel': [random.uniform(-1, 1), -2]  # Upward movement
        })

    def check_collision(self, other_snakes, obstacles):
        """Check if snake collides with obstacles or other snakes"""
        if not self.alive:
            return False
        # Skip collision detection during invincibility frames
        if self.invincible_frames > 0:
            return False
        head = self.get_head()
        head_radius = self.radius * (0.8 if self.boosting else 1.0)  # Smaller head when boosting
        # Check collision with boundaries
        x, y = head
        if x - head_radius < 0 or x + head_radius > width or y - head_radius < 0 or y + head_radius > height:
            logging.info(f"Snake {self.algorithm} died from boundary collision: ({x}, {y})")
            return True
        # Check collision with obstacles
        for obstacle in obstacles:
            if obstacle.rect.collidepoint(head):
                logging.info(f"Snake {self.algorithm} died from obstacle collision")
                return True
        # Check collision with other snakes
        for other_snake in other_snakes:
            if not other_snake.alive or other_snake.invincible_frames > 0:
                continue
            other_head = other_snake.get_head()
            other_radius = other_snake.radius * (0.8 if other_snake.boosting else 1.0)
            # Head-to-head collision
            head_dist = math.sqrt((head[0]-other_head[0])**2 + (head[1]-other_head[1])**2)
            if other_snake != self and head_dist < head_radius + other_radius:
                # Determine winner based on size
                if len(self.segments) < len(other_snake.segments):
                    logging.info(f"Snake {self.algorithm} died from head-to-head collision with {other_snake.algorithm}")
                    return True
                elif len(self.segments) > len(other_snake.segments):
                    logging.info(f"Snake {other_snake.algorithm} died from head-to-head collision with {self.algorithm}")
                    other_snake.alive = False
                    return False
                else:
                    # If equal size, randomly determine winner
                    if random.random() < 0.5:
                        logging.info(f"Snake {self.algorithm} died from equal-size head-to-head collision")
                        return True
                    else:
                        logging.info(f"Snake {other_snake.algorithm} died from equal-size head-to-head collision")
                        other_snake.alive = False
                        return False
            # Head-to-body collision (skip the head)
            for i, segment in enumerate(other_snake.segments[1:], 1):
                seg_radius = other_snake.radius * (0.8 if i < 5 and other_snake.boosting else 1.0)
                dist = math.sqrt((head[0]-segment[0])**2 + (head[1]-segment[1])**2)
                if other_snake != self and dist < head_radius + seg_radius:  # Added check to avoid self-collisions
                    logging.info(f"Snake {self.algorithm} died from head-to-body collision with {other_snake.algorithm}")
                    return True
        return False

    def generate_death_orbs(self):
        """Generate food orbs when snake dies"""
        orbs = []
        # Calculate how many orbs to drop
        num_orbs = min(30, max(5, len(self.segments) // 3))
        # Create orbs at segment positions, spaced out
        segment_step = max(1, len(self.segments) // num_orbs)
        for i in range(0, len(self.segments), segment_step):
            segment = self.segments[i]
            value = max(1, min(5, self.score // 20))  # Scale value with score
            size = random.randint(3, 8)  # Varying sizes
            # Add some random offset
            pos = (
                segment[0] + random.randint(-10, 10),
                segment[1] + random.randint(-10, 10)
            )
            orbs.append(Food(pos, value=value, color=self.color, type='remains', size=size))
        return orbs

    def draw(self, surface):
        if not self.alive:
            return
        # Draw the segments from tail to head (so head is on top)
        for i in range(len(self.segments)-1, -1, -1):
            pos = self.segments[i]
            # Scale radius based on position (head is bigger)
            segment_radius = self.radius * (1.2 if i == 0 else 1.0)
            # Reduce size when boosting
            if self.boosting:
                segment_radius *= 0.8
            # Draw glow effect
            glow_surface = pygame.Surface((segment_radius*4, segment_radius*4), pygame.SRCALPHA)
            # If in jackpot mode, make the snake glow with gold
            if self.jackpot_mode:
                glow_colors = [(255, 215, 0), (255, 223, 0), (255, 200, 0)]
                for g in range(3):
                    glow_size = segment_radius * (2.0 - g * 0.2)  # Bigger glow in jackpot mode
                    glow_alpha = 180 - g * 40
                    glow_color = (*glow_colors[g % len(glow_colors)], glow_alpha)
                    pygame.draw.circle(
                        glow_surface,
                        glow_color,
                        (segment_radius*2, segment_radius*2),
                        glow_size
                    )
            else:
                for g in range(3):
                    glow_size = segment_radius * (1.5 - g * 0.2)
                    glow_alpha = 150 - g * 40
                    glow_color = (*self.glow_colors[g % len(self.glow_colors)], glow_alpha)
                    pygame.draw.circle(
                        glow_surface,
                        glow_color,
                        (segment_radius*2, segment_radius*2),
                        glow_size
                    )
            surface.blit(
                glow_surface,
                (pos[0]-segment_radius*2, pos[1]-segment_radius*2)
            )
            # Draw main segment
            # If invincible, make it flash
            segment_color = self.color
            if self.invincible_frames > 0 and self.invincible_frames % 10 < 5:
                segment_color = (255, 255, 255)  # Flash white
            elif self.jackpot_mode:
                # In jackpot mode, make snake golden
                segment_color = casino_gold
            pygame.draw.circle(surface, segment_color, pos, segment_radius)
            # Draw "eyes" on the head
            if i == 0:
                # Calculate eye positions based on direction
                eye_offset = segment_radius * 0.5
                eye_size = segment_radius * 0.3
                # Position eyes perpendicular to movement direction
                perp_x, perp_y = -self.direction[1], self.direction[0]
                # Left eye
                left_eye_x = pos[0] + self.direction[0] * eye_offset + perp_x * eye_offset
                left_eye_y = pos[1] + self.direction[1] * eye_offset + perp_y * eye_offset
                # Right eye
                right_eye_x = pos[0] + self.direction[0] * eye_offset - perp_x * eye_offset
                right_eye_y = pos[1] + self.direction[1] * eye_offset - perp_y * eye_offset
                eye_color = black if not self.jackpot_mode else casino_red
                pygame.draw.circle(surface, eye_color, (int(left_eye_x), int(left_eye_y)), int(eye_size))
                pygame.draw.circle(surface, eye_color, (int(right_eye_x), int(right_eye_y)), int(eye_size))
                # If in jackpot mode, draw dollar signs in the eyes
                if self.jackpot_mode and segment_radius > 5:
                    dollar_font = pygame.font.SysFont("arial", int(eye_size * 1.5))
                    dollar = dollar_font.render("$", True, white)
                    surface.blit(dollar, (int(left_eye_x - dollar.get_width()/2),
                                      int(left_eye_y - dollar.get_height()/2)))
                    surface.blit(dollar, (int(right_eye_x - dollar.get_width()/2),
                                      int(right_eye_y - dollar.get_height()/2)))
        # Draw multiplier if active
        if self.multiplier > 1.0:
            mult_text = bet_font.render(f"{self.multiplier}x", True, casino_gold)
            surface.blit(mult_text, (self.segments[0][0] - mult_text.get_width()//2,
                                 self.segments[0][1] - self.radius - 20))
        # Draw boost meter if not full
        if self.boost_available < 100:
            meter_width = 30
            meter_height = 5
            meter_x = self.segments[0][0] - meter_width // 2
            meter_y = self.segments[0][1] - self.radius - 10
            # Background
            pygame.draw.rect(surface, (50, 50, 50),
                        (meter_x, meter_y, meter_width, meter_height))
            # Fill based on available boost
            fill_width = int(meter_width * (self.boost_available / 100))
            if fill_width > 0:
                fill_color = (0, 255, 0) if self.boost_available > 50 else (255, 165, 0) if self.boost_available > 20 else (255, 0, 0)
                pygame.draw.rect(surface, fill_color,
                            (meter_x, meter_y, fill_width, meter_height))
            # Border
            pygame.draw.rect(surface, (200, 200, 200),
                        (meter_x, meter_y, meter_width, meter_height), 1)
        # Draw recent winnings
        for win_info in self.recent_winnings:
            # Move upward with slight randomness
            win_info['pos'][0] += win_info['vel'][0]
            win_info['pos'][1] += win_info['vel'][1]
            # Calculate opacity (fade out)
            alpha = min(255, win_info['time'] * 4)
            # Render text with alpha
            win_font = pygame.font.SysFont("impact", 20)
            win_text = win_font.render(f"+{win_info['amount']}", True, casino_gold)
            # Create surface with alpha
            text_surf = pygame.Surface(win_text.get_size(), pygame.SRCALPHA)
            text_surf.fill((0, 0, 0, 0))  # Transparent
            text_surf.blit(win_text, (0, 0))
            text_surf.set_alpha(alpha)
            # Draw
            surface.blit(text_surf, (win_info['pos'][0] - win_text.get_width()//2,
                                 win_info['pos'][1] - win_text.get_height()//2))

class BettingSystem:
    def __init__(self):
        self.bets = {}  # Dictionary mapping snake IDs to bet amounts
        self.payouts = {}  # Cached payout multipliers
        self.total_bet = 0
        self.last_winner = None
        self.winning_amount = 0
        self.show_result_time = 0
        self.jackpot = 0
        self.jackpot_growth_rate = 0.1  # 10% of all bets go to jackpot

    def place_bet(self, snake_id, amount):
        """Place a bet on a specific snake"""
        global player_coins
        if amount <= 0 or amount > player_coins:
            return False
        self.bets[snake_id] = amount
        self.total_bet += amount
        player_coins -= amount
        # Add to jackpot
        self.jackpot += amount * self.jackpot_growth_rate
        return True

    def calculate_payouts(self, snakes):
        """Calculate payout multipliers based on snake lengths"""
        total_segments = sum(len(snake.segments) for snake in snakes if snake.alive)
        for snake in snakes:
            if not snake.alive:
                self.payouts[snake.algorithm_id] = 0
                continue
            # Base multiplier on relative size (smaller snakes = higher payout)
            if total_segments == 0:
                relative_size = 0
            else:
                relative_size = len(snake.segments) / total_segments
            # Inverse relationship: smaller snakes have higher payouts
            # Use a curve that gives reasonable multipliers
            base_multi = 1.5 + (1.0 - relative_size) * 5
            # Apply some randomness for excitement
            randomizer = random.uniform(0.9, 1.1)
            # Set final multiplier (min 1.5, max 10)
            self.payouts[snake.algorithm_id] = min(10.0, max(1.5, base_multi * randomizer))
        return self.payouts

    def process_winner(self, winner_id):
        """Process a winning bet"""
        global player_coins
        if winner_id in self.bets:
            bet_amount = self.bets[winner_id]
            multiplier = self.payouts.get(winner_id, bet_multipliers.get(winner_id, 2.0))
            winnings = int(bet_amount * multiplier)
            player_coins += winnings
            self.last_winner = winner_id
            self.winning_amount = winnings
            self.show_result_time = 180  # Show result for 3 seconds (60 fps * 3)
            logging.info(f"Player won {winnings} coins from bet on Snake {winner_id}")
            return winnings
        else:
            self.last_winner = None
            self.winning_amount = 0
            self.show_result_time = 120
            return 0

    def trigger_jackpot(self):
        """Trigger the jackpot for a lucky win"""
        global player_coins
        winnings = int(self.jackpot)
        player_coins += winnings
        # Reset jackpot but leave a seed amount
        seed = self.jackpot * 0.1
        self.jackpot = seed
        return winnings

    def reset(self):
        """Reset all bets"""
        self.bets = {}
        self.total_bet = 0

    def update(self):
        """Update bet display time"""
        if self.show_result_time > 0:
            self.show_result_time -= 1

    def draw(self, surface, snakes):
        """Draw betting UI"""
        # Create semi-transparent panel
        panel_width = 250
        panel_height = 400
        panel_x = width - panel_width - 10
        panel_y = 70
        panel = pygame.Surface((panel_width, panel_height))
        panel.fill((20, 20, 30))
        panel.set_alpha(200)
        surface.blit(panel, (panel_x, panel_y))
        # Draw panel border
        pygame.draw.rect(surface, casino_gold, (panel_x, panel_y, panel_width, panel_height), 2)
        # Draw betting header
        header_font = pygame.font.SysFont("impact", 28)
        header = header_font.render("BETTING PARLOR", True, casino_gold)
        surface.blit(header, (panel_x + panel_width//2 - header.get_width()//2, panel_y + 10))
        # Draw jackpot
        jackpot_text = bet_font.render(f"JACKPOT: {int(self.jackpot)} COINS", True, casino_red)
        surface.blit(jackpot_text, (panel_x + panel_width//2 - jackpot_text.get_width()//2, panel_y + 40))
        # Draw player coins
        coins_text = bet_font.render(f"YOUR COINS: {player_coins}", True, white)
        surface.blit(coins_text, (panel_x + panel_width//2 - coins_text.get_width()//2, panel_y + 70))
        # Draw bet options
        y_offset = panel_y + 110
        for snake in snakes:
            # Skip dead snakes
            if not snake.alive:
                continue
            # Snake color indicator
            pygame.draw.circle(surface, snake.color, (panel_x + 20, y_offset + 12), 10)
            # Snake name and current payout
            multiplier = self.payouts.get(snake.algorithm_id, bet_multipliers.get(snake.algorithm_id, 2.0))
            snake_text = bet_font.render(f"{snake.algorithm}: {multiplier:.1f}x", True, white)
            surface.blit(snake_text, (panel_x + 40, y_offset))
            # Draw bet indicator if we've bet on this snake
            if snake.algorithm_id in self.bets:
                bet_text = bet_font.render(f"BET: {self.bets[snake.algorithm_id]}", True, casino_gold)
                surface.blit(bet_text, (panel_x + panel_width - 80, y_offset))
            y_offset += 35
        # Draw betting instructions
        y_offset = panel_y + 300
        instr_font = pygame.font.SysFont("segoeui", 16)
        instructions = [
            "Press 1-5 to select a snake",
            "Press UP/DOWN to adjust bet",
            "Press ENTER to place bet",
            "Press J for jackpot bet (25% chance)"
        ]
        for i, line in enumerate(instructions):
            instr_text = instr_font.render(line, True, (200, 200, 200))
            surface.blit(instr_text, (panel_x + 10, y_offset + i*20))
        # Show betting result if active
        if self.show_result_time > 0:
            # Create popup with fade effect
            alpha = min(255, self.show_result_time * 3)
            popup = pygame.Surface((400, 100))
            popup.fill((0, 0, 0))
            popup.set_alpha(alpha)
            popup_x = width//2 - 200
            popup_y = height//2 - 50
            surface.blit(popup, (popup_x, popup_y))
            # Draw border
            pygame.draw.rect(surface, casino_gold, (popup_x, popup_y, 400, 100), 3)
            # Draw result
            result_font = pygame.font.SysFont("impact", 36)
            if self.last_winner is not None:
                result_text = result_font.render(f"YOU WON {self.winning_amount} COINS!", True, casino_green)
            else:
                result_text = result_font.render("BETTER LUCK NEXT TIME!", True, casino_red)
            # Calculate alpha for text too
            text_surface = pygame.Surface(result_text.get_size(), pygame.SRCALPHA)
            text_surface.fill((0, 0, 0, 0))  # Transparent
            text_surface.blit(result_text, (0, 0))
            text_surface.set_alpha(alpha)
            surface.blit(text_surface, (popup_x + 200 - result_text.get_width()//2,
                                    popup_y + 50 - result_text.get_height()//2))

def generate_obstacles(count, snake_positions):
    obstacles = []
    for _ in range(count):
        # Make obstacles of varying sizes
        obstacle_width = random.randint(3, 8) * snake_block
        obstacle_height = random.randint(3, 8) * snake_block
        # Try to find a position that doesn't collide with snakes
        for _ in range(50):  # Maximum attempts
            # Fixed calculation to account for obstacle size
            x = random.randint(2, (width // snake_block) - (obstacle_width // snake_block) - 2) * snake_block
            y = random.randint(2, (height // snake_block) - (obstacle_height // snake_block) - 2) * snake_block
            # Check if obstacle would overlap with any snake
            overlap = False
            for pos in snake_positions:
                if (abs(pos[0] - x) < obstacle_width + 100 and  # Increased clearance for snake starting positions
                    abs(pos[1] - y) < obstacle_height + 100):
                    overlap = True
                    break
            if not overlap:
                obstacles.append(Obstacle(x, y, obstacle_width, obstacle_height))
                break
    return obstacles

def is_point_in_obstacles(point, obstacles):
    """Check if a point collides with any obstacle"""
    for obstacle in obstacles:
        if obstacle.rect.collidepoint(point):
            return True
    return False

def manhattan_distance(a, b):
    """Calculate Manhattan distance between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean_distance(a, b):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# Vector operations
def normalize_vector(vec):
    mag = math.sqrt(vec[0]**2 + vec[1]**2)
    if mag > 0:
        return [vec[0]/mag, vec[1]/mag]
    return [0, 0]

def find_path_astar(snake, goal, all_snakes, obstacles, max_depth=25):
    """A* pathfinding algorithm adapted for continuous movement."""
    start = snake.get_head()
    # If goal is far, move directly toward it
    if euclidean_distance(start, goal) > max_depth * snake_block:
        direction = normalize_vector([goal[0] - start[0], goal[1] - start[1]])
        return start[0] + direction[0] * 20, start[1] + direction[1] * 20
    # Discretize space into a grid for pathfinding
    grid_size = snake_block
    # Create a discrete grid-based start and goal
    grid_start = (int(start[0] // grid_size), int(start[1] // grid_size))
    grid_goal = (int(goal[0] // grid_size), int(goal[1] // grid_size))
    # Define grid moves
    grid_moves = [(0, -1), (1, 0), (0, 1), (-1, 0), (1, -1), (1, 1), (-1, 1), (-1, -1)]
    # A* implementation
    open_set = {grid_start}
    closed_set = set()
    g_score = {grid_start: 0}
    f_score = {grid_start: manhattan_distance(grid_start, grid_goal)}
    came_from = {}
    while open_set and min(f_score.get(node, float('inf')) for node in open_set) < float('inf'):
        current = min(open_set, key=lambda node: f_score.get(node, float('inf')))
        if current == grid_goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path = path[::-1]  # Reverse path
            # If path found, return the next point
            if path:
                next_grid = path[0]
                # Convert back to world coordinates (center of grid cell)
                return (next_grid[0] * grid_size + grid_size//2,
                        next_grid[1] * grid_size + grid_size//2)
            break
        open_set.remove(current)
        closed_set.add(current)
        # If we've explored too many nodes, stop for performance
        if len(closed_set) > max_depth:
            break
        for dx, dy in grid_moves:
            neighbor = (current[0] + dx, current[1] + dy)
            # Skip if outside bounds
            if (neighbor[0] < 0 or neighbor[0] >= width // grid_size or
                neighbor[1] < 0 or neighbor[1] >= height // grid_size):
                continue
            # Skip if already evaluated
            if neighbor in closed_set:
                continue
            # Skip if this would go through an obstacle
            world_pos = (neighbor[0] * grid_size + grid_size//2,
                          neighbor[1] * grid_size + grid_size//2)
            if is_point_in_obstacles(world_pos, obstacles):
                continue
            # Skip if too close to other snakes
            too_close = False
            for other_snake in all_snakes:
                if other_snake != snake and other_snake.alive:
                    for segment in other_snake.segments:
                        if euclidean_distance(world_pos, segment) < snake.radius * 2.5:
                            too_close = True
                            break
                    if too_close:
                        break
            if too_close:
                continue
            # Calculate tentative g_score (diagonal moves cost more)
            move_cost = 1.4 if dx != 0 and dy != 0 else 1.0
            tentative_g_score = g_score[current] + move_cost
            if neighbor not in open_set:
                open_set.add(neighbor)
            elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue
            # This path is the best so far
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + manhattan_distance(neighbor, grid_goal)
    # If no path found, move in direction of goal avoiding obstacles
    direction = [goal[0] - start[0], goal[1] - start[1]]
    direction = normalize_vector(direction)
    # Check a few directions
    test_angles = [0, 15, -15, 30, -30, 45, -45, 60, -60]
    for angle in test_angles:
        # Rotate the direction vector
        rad_angle = math.radians(angle)
        rotated_dir = [
            direction[0] * math.cos(rad_angle) - direction[1] * math.sin(rad_angle),
            direction[0] * math.sin(rad_angle) + direction[1] * math.cos(rad_angle)
        ]
        # Test this direction
        test_dist = 20  # How far ahead to look
        test_pos = (start[0] + rotated_dir[0] * test_dist,
                    start[1] + rotated_dir[1] * test_dist)
        # Check if this position is clear
        if (test_pos[0] > 0 and test_pos[0] < width and
            test_pos[1] > 0 and test_pos[1] < height and
            not is_point_in_obstacles(test_pos, obstacles)):
            # Also check if it's not too close to other snakes
            too_close = False
            for other_snake in all_snakes:
                if other_snake != snake and other_snake.alive:
                    for segment in other_snake.segments:
                        if euclidean_distance(test_pos, segment) < snake.radius * 2.5:
                            too_close = True
                            break
                    if too_close:
                        break
            if not too_close:
                return test_pos
    # If no clear direction found, just try the original direction
    return start[0] + direction[0] * 10, start[1] + direction[1] * 10

def find_path_dijkstra(snake, goal, all_snakes, obstacles, max_depth=25):
    """Dijkstra's algorithm adapted for continuous movement."""
    start = snake.get_head()
    # If goal is far, move directly toward it
    if euclidean_distance(start, goal) > max_depth * snake_block:
        direction = normalize_vector([goal[0] - start[0], goal[1] - start[1]])
        return start[0] + direction[0] * 20, start[1] + direction[1] * 20
    # Discretize space into a grid for pathfinding
    grid_size = snake_block
    # Create a discrete grid-based start and goal
    grid_start = (int(start[0] // grid_size), int(start[1] // grid_size))
    grid_goal = (int(goal[0] // grid_size), int(goal[1] // grid_size))
    # Define grid moves
    grid_moves = [(0, -1), (1, 0), (0, 1), (-1, 0), (1, -1), (1, 1), (-1, 1), (-1, -1)]
    # Dijkstra implementation
    open_set = {grid_start}
    closed_set = set()
    g_score = {grid_start: 0}
    came_from = {}
    while open_set:
        # Get node with lowest g_score
        current = min(open_set, key=lambda node: g_score.get(node, float('inf')))
        if current == grid_goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path = path[::-1]  # Reverse path
            # If path found, return the next point
            if path:
                next_grid = path[0]
                # Convert back to world coordinates (center of grid cell)
                return (next_grid[0] * grid_size + grid_size//2,
                        next_grid[1] * grid_size + grid_size//2)
            break
        open_set.remove(current)
        closed_set.add(current)
        # If we've explored too many nodes, stop for performance
        if len(closed_set) > max_depth:
            break
        for dx, dy in grid_moves:
            neighbor = (current[0] + dx, current[1] + dy)
            # Skip if outside bounds
            if (neighbor[0] < 0 or neighbor[0] >= width // grid_size or
                neighbor[1] < 0 or neighbor[1] >= height // grid_size):
                continue
            # Skip if already evaluated
            if neighbor in closed_set:
                continue
            # Skip if this would go through an obstacle
            world_pos = (neighbor[0] * grid_size + grid_size//2,
                          neighbor[1] * grid_size + grid_size//2)
            if is_point_in_obstacles(world_pos, obstacles):
                continue
            # Skip if too close to other snakes
            too_close = False
            for other_snake in all_snakes:
                if other_snake != snake and other_snake.alive:
                    for segment in other_snake.segments:
                        if euclidean_distance(world_pos, segment) < snake.radius * 2.5:
                            too_close = True
                            break
                    if too_close:
                        break
            if too_close:
                continue
            # Calculate tentative g_score (diagonal moves cost more)
            move_cost = 1.4 if dx != 0 and dy != 0 else 1.0
            tentative_g_score = g_score[current] + move_cost
            if neighbor not in open_set:
                open_set.add(neighbor)
            elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue
            # This path is the best so far
            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
    # If no path found, move in direction of goal avoiding obstacles
    direction = [goal[0] - start[0], goal[1] - start[1]]
    direction = normalize_vector(direction)
    # Check a few directions
    test_angles = [0, 15, -15, 30, -30, 45, -45, 60, -60]
    for angle in test_angles:
        # Rotate the direction vector
        rad_angle = math.radians(angle)
        rotated_dir = [
            direction[0] * math.cos(rad_angle) - direction[1] * math.sin(rad_angle),
            direction[0] * math.sin(rad_angle) + direction[1] * math.cos(rad_angle)
        ]
        # Test this direction
        test_dist = 20  # How far ahead to look
        test_pos = (start[0] + rotated_dir[0] * test_dist,
                    start[1] + rotated_dir[1] * test_dist)
        # Check if this position is clear
        if (test_pos[0] > 0 and test_pos[0] < width and
            test_pos[1] > 0 and test_pos[1] < height and
            not is_point_in_obstacles(test_pos, obstacles)):
            # Also check if it's not too close to other snakes
            too_close = False
            for other_snake in all_snakes:
                if other_snake != snake and other_snake.alive:
                    for segment in other_snake.segments:
                        if euclidean_distance(test_pos, segment) < snake.radius * 2.5:
                            too_close = True
                            break
                    if too_close:
                        break
            if not too_close:
                return test_pos
    # If no clear direction found, just try the original direction
    return start[0] + direction[0] * 10, start[1] + direction[1] * 10

def find_path_bfs(snake, goal, all_snakes, obstacles, max_depth=25):
    """BFS adapted for continuous movement."""
    start = snake.get_head()
    # If goal is far, move directly toward it
    if euclidean_distance(start, goal) > max_depth * snake_block:
        direction = normalize_vector([goal[0] - start[0], goal[1] - start[1]])
        return start[0] + direction[0] * 20, start[1] + direction[1] * 20
    # Discretize space into a grid for pathfinding
    grid_size = snake_block
    # Create a discrete grid-based start and goal
    grid_start = (int(start[0] // grid_size), int(start[1] // grid_size))
    grid_goal = (int(goal[0] // grid_size), int(goal[1] // grid_size))
    # Define grid moves (4-way movement for BFS)
    grid_moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    # BFS implementation
    queue = deque([(grid_start, [])])
    visited = {grid_start}
    while queue:
        current, path = queue.popleft()
        if current == grid_goal:
            # If path found, return the next point
            if path:
                next_grid = path[0]
                # Convert back to world coordinates (center of grid cell)
                return (next_grid[0] * grid_size + grid_size//2,
                        next_grid[1] * grid_size + grid_size//2)
            break
        # If we've explored too many nodes, stop for performance
        if len(visited) > max_depth:
            break
        for dx, dy in grid_moves:
            neighbor = (current[0] + dx, current[1] + dy)
            # Skip if outside bounds
            if (neighbor[0] < 0 or neighbor[0] >= width // grid_size or
                neighbor[1] < 0 or neighbor[1] >= height // grid_size):
                continue
            # Skip if already visited
            if neighbor in visited:
                continue
            # Skip if this would go through an obstacle
            world_pos = (neighbor[0] * grid_size + grid_size//2,
                          neighbor[1] * grid_size + grid_size//2)
            if is_point_in_obstacles(world_pos, obstacles):
                continue
            # Skip if too close to other snakes
            too_close = False
            for other_snake in all_snakes:
                if other_snake != snake and other_snake.alive:
                    for segment in other_snake.segments:
                        if euclidean_distance(world_pos, segment) < snake.radius * 2.5:
                            too_close = True
                            break
                    if too_close:
                        break
            if too_close:
                continue
            # Add to visited and queue
            visited.add(neighbor)
            new_path = path + [neighbor] if not path else [neighbor]
            queue.append((neighbor, new_path))
    # If no path found, move in direction of goal avoiding obstacles
    direction = [goal[0] - start[0], goal[1] - start[1]]
    direction = normalize_vector(direction)
    # Check a few directions
    test_angles = [0, 15, -15, 30, -30, 45, -45, 60, -60]
    for angle in test_angles:
        # Rotate the direction vector
        rad_angle = math.radians(angle)
        rotated_dir = [
            direction[0] * math.cos(rad_angle) - direction[1] * math.sin(rad_angle),
            direction[0] * math.sin(rad_angle) + direction[1] * math.cos(rad_angle)
        ]
        # Test this direction
        test_dist = 20  # How far ahead to look
        test_pos = (start[0] + rotated_dir[0] * test_dist,
                    start[1] + rotated_dir[1] * test_dist)
        # Check if this position is clear
        if (test_pos[0] > 0 and test_pos[0] < width and
            test_pos[1] > 0 and test_pos[1] < height and
            not is_point_in_obstacles(test_pos, obstacles)):
            # Also check if it's not too close to other snakes
            too_close = False
            for other_snake in all_snakes:
                if other_snake != snake and other_snake.alive:
                    for segment in other_snake.segments:
                        if euclidean_distance(test_pos, segment) < snake.radius * 2.5:
                            too_close = True
                            break
                    if too_close:
                        break
            if not too_close:
                return test_pos
    # If no clear direction found, just try the original direction
    return start[0] + direction[0] * 10, start[1] + direction[1] * 10

def find_path_greedy(snake, goal, all_snakes, obstacles, max_depth=25):
    """Greedy Best-First Search adapted for continuous movement."""
    start = snake.get_head()
    # If goal is far, move directly toward it
    if euclidean_distance(start, goal) > max_depth * snake_block:
        direction = normalize_vector([goal[0] - start[0], goal[1] - start[1]])
        return start[0] + direction[0] * 20, start[1] + direction[1] * 20
    # Discretize space into a grid for pathfinding
    grid_size = snake_block
    # Create a discrete grid-based start and goal
    grid_start = (int(start[0] // grid_size), int(start[1] // grid_size))
    grid_goal = (int(goal[0] // grid_size), int(goal[1] // grid_size))
    # Define grid moves
    grid_moves = [(0, -1), (1, 0), (0, 1), (-1, 0), (1, -1), (1, 1), (-1, 1), (-1, -1)]
    # Greedy Best-First implementation
    open_set = {grid_start}
    closed_set = set()
    h_score = {grid_start: manhattan_distance(grid_start, grid_goal)}
    came_from = {}
    while open_set:
        # Get node with lowest h_score (closest to goal)
        current = min(open_set, key=lambda node: h_score.get(node, float('inf')))
        if current == grid_goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path = path[::-1]  # Reverse path
            # If path found, return the next point
            if path:
                next_grid = path[0]
                # Convert back to world coordinates (center of grid cell)
                return (next_grid[0] * grid_size + grid_size//2,
                        next_grid[1] * grid_size + grid_size//2)
            break
        open_set.remove(current)
        closed_set.add(current)
        # If we've explored too many nodes, stop for performance
        if len(closed_set) > max_depth:
            break
        for dx, dy in grid_moves:
            neighbor = (current[0] + dx, current[1] + dy)
            # Skip if outside bounds
            if (neighbor[0] < 0 or neighbor[0] >= width // grid_size or
                neighbor[1] < 0 or neighbor[1] >= height // grid_size):
                continue
            # Skip if already evaluated
            if neighbor in closed_set:
                continue
            # Skip if this would go through an obstacle
            world_pos = (neighbor[0] * grid_size + grid_size//2,
                          neighbor[1] * grid_size + grid_size//2)
            if is_point_in_obstacles(world_pos, obstacles):
                continue
            # Skip if too close to other snakes
            too_close = False
            for other_snake in all_snakes:
                if other_snake != snake and other_snake.alive:
                    for segment in other_snake.segments:
                        if euclidean_distance(world_pos, segment) < snake.radius * 2.5:
                            too_close = True
                            break
                    if too_close:
                        break
            if too_close:
                continue
            if neighbor not in open_set:
                h_score[neighbor] = manhattan_distance(neighbor, grid_goal)
                open_set.add(neighbor)
                came_from[neighbor] = current
    # If no path found, move in direction of goal avoiding obstacles
    direction = [goal[0] - start[0], goal[1] - start[1]]
    direction = normalize_vector(direction)
    # Check a few directions
    test_angles = [0, 15, -15, 30, -30, 45, -45, 60, -60]
    for angle in test_angles:
        # Rotate the direction vector
        rad_angle = math.radians(angle)
        rotated_dir = [
            direction[0] * math.cos(rad_angle) - direction[1] * math.sin(rad_angle),
            direction[0] * math.sin(rad_angle) + direction[1] * math.cos(rad_angle)
        ]
        # Test this direction
        test_dist = 20  # How far ahead to look
        test_pos = (start[0] + rotated_dir[0] * test_dist,
                    start[1] + rotated_dir[1] * test_dist)
        # Check if this position is clear
        if (test_pos[0] > 0 and test_pos[0] < width and
            test_pos[1] > 0 and test_pos[1] < height and
            not is_point_in_obstacles(test_pos, obstacles)):
            # Also check if it's not too close to other snakes
            too_close = False
            for other_snake in all_snakes:
                if other_snake != snake and other_snake.alive:
                    for segment in other_snake.segments:
                        if euclidean_distance(test_pos, segment) < snake.radius * 2.5:
                            too_close = True
                            break
                    if too_close:
                        break
            if not too_close:
                return test_pos
    # If no clear direction found, just try the original direction
    return start[0] + direction[0] * 10, start[1] + direction[1] * 10

def get_random_target(snake, all_snakes, obstacles, foods):
    """Get a random target for the random snake, with some smarts to avoid danger."""
    head = snake.get_head()
    # Prioritize nearby food if available
    closest_food = None
    min_dist = float('inf')
    for food in foods:
        dist = euclidean_distance(head, food.pos)
        if dist < min_dist and dist < 150:  # Only consider food within range
            min_dist = dist
            closest_food = food
    if closest_food:
        # 75% chance to go for nearby food
        if random.random() < 0.75:
            return closest_food.pos
    # Generate random directions and test them
    for _ in range(10):
        angle = random.uniform(0, 2 * math.pi)
        direction = [math.cos(angle), math.sin(angle)]
        # Test this direction
        test_dist = random.uniform(30, 100)  # Vary the distance
        test_pos = (head[0] + direction[0] * test_dist,
                    head[1] + direction[1] * test_dist)
        # Check if this position is clear
        if (test_pos[0] > 0 and test_pos[0] < width and
            test_pos[1] > 0 and test_pos[1] < height and
            not is_point_in_obstacles(test_pos, obstacles)):
            # Also check if it's not too close to other snakes
            too_close = False
            for other_snake in all_snakes:
                if other_snake != snake and other_snake.alive:
                    for segment in other_snake.segments:
                        if euclidean_distance(test_pos, segment) < snake.radius * 3:
                            too_close = True
                            break
                    if too_close:
                        break
            if not too_close:
                return test_pos
    # If all else fails, just pick a safe-ish direction
    angle = random.uniform(0, 2 * math.pi)
    return (head[0] + math.cos(angle) * 50, head[1] + math.sin(angle) * 50)

def create_trail_effect(snake):
    """Create a trail effect surface for a snake"""
    if len(snake.segments) <= 1:
        return None
    trail_surf = pygame.Surface((width, height), pygame.SRCALPHA)
    # Get color components
    r, g, b = snake.color
    # Draw trail segments with decreasing opacity
    for i in range(1, min(20, len(snake.segments))):
        # Decrease opacity as we move back in the snake
        alpha = int(100 * (1 - (i / 20)))
        # Use golden trails for jackpot mode
        if snake.jackpot_mode:
            trail_color = (casino_gold[0], casino_gold[1], casino_gold[2], alpha)
        else:
            trail_color = (r, g, b, alpha)
        pos = snake.segments[i]
        # Draw a smaller circle for the trail effect
        radius = max(1, snake.radius * (1 - i/25) * 0.8)
        pygame.draw.circle(trail_surf, trail_color, pos, radius)
    return trail_surf

def spawn_food(snake_segments, obstacles, food_type='regular'):
    """Spawn food at a valid location"""
    attempts = 0
    max_attempts = 100
    # Food characteristics
    if food_type == 'regular':
        value = 1
        color = white
        size = random.randint(3, 5)
    elif food_type == 'super':
        value = 3
        color = yellow
        size = random.randint(6, 8)
    elif food_type == 'speed':
        value = 2
        color = pink
        size = random.randint(4, 6)
    elif food_type == 'bonus':
        value = 5
        color = gold
        size = random.randint(7, 10)
    elif food_type == 'portal':
        value = 2
        color = cyan
        size = random.randint(5, 7)
    elif food_type == 'jackpot':
        value = random.randint(5, 20)  # High value for jackpot food
        color = jackpot_purple
        size = random.randint(8, 12)
    while attempts < max_attempts:
        # Random position
        pos = (
            random.randint(size, width - size),
            random.randint(size, height - size)
        )
        # Check distance from all snake segments
        too_close = False
        for segments in snake_segments:
            for segment in segments:
                if euclidean_distance(pos, segment) < size + 10:  # Minimum spacing
                    too_close = True
                    break
            if too_close:
                break
        if too_close:
            attempts += 1
            continue
        # Check if in obstacle
        if is_point_in_obstacles(pos, obstacles):
            attempts += 1
            continue
        # Valid position found
        return Food(pos, value=value, color=color, type=food_type, size=size)
    # If couldn't find position after max attempts
    return None

def spawn_multiple_food(count, snake_segments, obstacles, preferences=None):
    """Spawn multiple food items with given preferences"""
    if preferences is None:
        preferences = [
            ('regular', 0.5),
            ('super', 0.2),
            ('speed', 0.1),
            ('bonus', 0.08),
            ('portal', 0.07),
            ('jackpot', 0.05)  # Added jackpot food type
        ]
    food_items = []
    for _ in range(count):
        # Select food type based on preferences
        food_type = random.choices([p[0] for p in preferences],
                                [p[1] for p in preferences])[0]
        food = spawn_food(snake_segments, obstacles, food_type)
        if food:
            food_items.append(food)
    return food_items

def create_background(width, height):
    """Create a slither.io style background"""
    background = pygame.Surface((width, height))
    # Fill with dark color
    background.fill((5, 5, 10))
    # Draw grid pattern
    grid_spacing = 50
    grid_color = (30, 30, 40)
    for x in range(0, width, grid_spacing):
        pygame.draw.line(background, grid_color, (x, 0), (x, height))
    for y in range(0, height, grid_spacing):
        pygame.draw.line(background, grid_color, (0, y), (width, y))
    # Add some random dots for texture
    for _ in range(300):
        x = random.randint(0, width)
        y = random.randint(0, height)
        size = random.randint(1, 3)
        brightness = random.randint(20, 50)
        pygame.draw.circle(background, (brightness, brightness, brightness), (x, y), size)
    return background

def generate_start_positions():
    """Generate snake starting positions that are far from each other"""
    positions = []
    min_distance = 200  # Minimum distance between starting positions
    for i in range(5):
        attempts = 0
        while attempts < 100:
            # Start in a safer area (away from walls)
            margin = 100
            x = random.randint(margin, width - margin)
            y = random.randint(margin, height - margin)
            pos = (x, y)
            # Check if far enough from other positions
            too_close = False
            for other_pos in positions:
                if euclidean_distance(pos, other_pos) < min_distance:
                    too_close = True
                    break
            if not too_close:
                positions.append(pos)
                break
            attempts += 1
        if attempts >= 100:
            # Fallback position
            positions.append((
                width // 2 + random.randint(-200, 200),
                height // 2 + random.randint(-200, 200)
            ))
    return positions

def spawn_risk_zone(obstacles, snakes):
    """Create a new risk zone in a valid location"""
    attempts = 0
    max_attempts = 50
    while attempts < max_attempts:
        # Random position, size and multiplier
        radius = random.randint(80, 150)
        pos = (
            random.randint(radius, width - radius),
            random.randint(radius, height - radius)
        )
        multiplier = random.uniform(1.5, 5.0)
        multiplier = round(multiplier * 2) / 2  # Round to nearest 0.5
        # Check if overlaps with obstacles
        valid = True
        for obstacle in obstacles:
            if (abs(obstacle.rect.centerx - pos[0]) < radius + obstacle.rect.width//2 and
                abs(obstacle.rect.centery - pos[1]) < radius + obstacle.rect.height//2):
                valid = False
                break
        # Check if overlaps with snake starting areas
        for snake in snakes:
            if snake.alive and euclidean_distance(snake.get_head(), pos) < radius + 50:
                valid = False
                break
        if valid:
            return RiskZone(pos, radius, multiplier)
        attempts += 1
    return None

def gameLoop():
    # Set up logging and get log filename
    log_filename = setup_logging()
    logging.info("Game started with slither.io mechanics and gambling features")
    game_over = False
    # Create background
    background = create_background(width, height)
    # Snakes setup
    snake_colors = {
        1: green,   # A*
        2: blue,    # Dijkstra
        3: red,     # Random
        4: purple,  # BFS
        5: orange   # Greedy
    }
    # Algorithm names
    algorithm_names = [
        'A*',
        'Dijkstra',
        'Random',
        'BFS',
        'Greedy BFS'
    ]
    # Initialize snakes in different parts of the map
    start_positions = generate_start_positions()
    logging.info(f"Starting positions: {start_positions}")
    # Create snake objects
    snakes = [
        Snake(start_positions[i], snake_colors[i+1], algorithm_names[i], i+1)
        for i in range(5)
    ]
    # Initially grow snakes a bit
    for snake in snakes:
        snake.grow(4)  # Start with length 5
    # Generate obstacles (fewer for more open space)
    obstacles = generate_obstacles(5, start_positions)
    # Generate initial food
    foods = spawn_multiple_food(20, [snake.segments for snake in snakes], obstacles)
    # Create risk zones
    risk_zones = []
    for _ in range(3):
        zone = spawn_risk_zone(obstacles, snakes)
        if zone:
            risk_zones.append(zone)
    # Create betting system
    betting = BettingSystem()
    global player_coins, current_bet, bet_snake_id, bet_placed, bet_result
    # Game loop variables
    clock = pygame.time.Clock()
    frame_count = 0
    last_time = pygame.time.get_ticks()
    # Stats tracking
    kill_count = {snake.algorithm_id: 0 for snake in snakes}
    # Particles for visual effects
    particles = []
    # Create fonts for UI
    leaderboard_font = pygame.font.SysFont("segoeui", 24)
    stats_font = pygame.font.SysFont("segoeui", 20)
    # Startup delay - give snakes time to position properly
    startup_frames = 60
    # Betting selection state
    selected_snake_id = None
    bet_amount = 50  # Default bet amount
    while not game_over:
        frame_count += 1
        current_time = pygame.time.get_ticks()
        dt = (current_time - last_time) / 1000.0  # Delta time in seconds
        last_time = current_time
        # Cap delta time to avoid huge jumps
        dt = min(dt, 0.1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            # Player control inputs
            elif event.type == pygame.KEYDOWN:
                # Space key to enable boost
                if event.key == pygame.K_SPACE:
                    # Random chance for which snake gets the boost command
                    alive_snakes = [s for s in snakes if s.alive]
                    if alive_snakes:
                        random.choice(alive_snakes).set_boost(True)
                # Betting controls: select snake (1-5)
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]:
                    snake_num = int(event.unicode)
                    if 1 <= snake_num <= 5:
                        selected_snake_id = snake_num
                        if any(s.algorithm_id == selected_snake_id and s.alive for s in snakes):
                            logging.info(f"Selected snake {snake_num} for betting")
                # Adjust bet amount
                elif event.key == pygame.K_UP:
                    bet_amount = min(player_coins, bet_amount + 50)
                elif event.key == pygame.K_DOWN:
                    bet_amount = max(50, bet_amount - 50)
                # Place bet
                elif event.key == pygame.K_RETURN:
                    if selected_snake_id is not None and not bet_placed:
                        if betting.place_bet(selected_snake_id, bet_amount):
                            bet_placed = True
                            bet_snake_id = selected_snake_id
                            current_bet = bet_amount
                            logging.info(f"Placed bet of {bet_amount} on snake {selected_snake_id}")
                # Jackpot bet (high risk, big reward)
                elif event.key == pygame.K_j:
                    if not bet_placed and player_coins >= 100:
                        jackpot_bet = min(player_coins, 100)
                        player_coins -= jackpot_bet
                        # 25% chance to win jackpot
                        if random.random() < 0.25:
                            winnings = int(betting.trigger_jackpot())
                            logging.info(f"JACKPOT WIN! Player won {winnings} coins!")
                            # Find a random snake to give jackpot mode to
                            lucky_snake = random.choice([s for s in snakes if s.alive])
                            lucky_snake.activate_jackpot_mode(300)  # 5 seconds of jackpot mode
                            lucky_snake.add_winnings(winnings)
                            # Add visual effect
                            for _ in range(50):
                                particles.append({
                                    'pos': lucky_snake.get_head(),
                                    'vel': (random.uniform(-5, 5), random.uniform(-5, 5)),
                                    'color': casino_gold,
                                    'life': random.randint(30, 60),
                                    'size': random.randint(3, 10)
                                })
                        else:
                            logging.info("JACKPOT LOSS! Better luck next time.")
                            # Add to jackpot
                            betting.jackpot += jackpot_bet * 0.5
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    for snake in snakes:
                        snake.set_boost(False)
        # Update betting system
        betting.update()
        # Calculate payout multipliers occasionally
        if frame_count % 60 == 0:
            betting.calculate_payouts(snakes)
        # Get a list of active snakes
        active_snakes = [snake for snake in snakes if snake.alive]
        # Occasionally activate boost for random snakes (after startup period)
        if frame_count > startup_frames and frame_count % 60 == 0:  # Every ~2 seconds
            for snake in active_snakes:
                if random.random() < 0.3:  # 30% chance
                    snake.set_boost(True)
                else:
                    snake.set_boost(False)
        # Spawn new risk zones occasionally
        if frame_count % 300 == 0:  # Every ~5 seconds
            # Remove expired zones
            risk_zones = [zone for zone in risk_zones if zone.is_active()]
            # Add new zones if fewer than 3
            if len(risk_zones) < 3:
                new_zone = spawn_risk_zone(obstacles, snakes)
                if new_zone:
                    risk_zones.append(new_zone)
                    # Add spawning effect
                    for _ in range(20):
                        particles.append({
                            'pos': new_zone.pos,
                            'vel': (random.uniform(-3, 3), random.uniform(-3, 3)),
                            'color': casino_gold,
                            'life': random.randint(20, 40),
                            'size': random.randint(2, 5)
                        })
        # Process each snake
        for snake in active_snakes:
            # Check if in risk zone and update multiplier
            snake.in_risk_zone = False
            snake.multiplier = 1.0
            for zone in risk_zones:
                if zone.is_active() and zone.contains_point(snake.get_head()):
                    snake.in_risk_zone = True
                    snake.multiplier = zone.multiplier
                    break
            # Determine target based on algorithm
            if snake.algorithm_id == 3:  # Random snake
                target = get_random_target(snake, snakes, obstacles, foods)
            else:
                # Find closest food
                closest_food = None
                min_dist = float('inf')
                for food in foods:
                    dist = euclidean_distance(snake.get_head(), food.pos)
                    # If in jackpot mode, prioritize jackpot food
                    if snake.jackpot_mode and food.type == 'jackpot':
                        dist *= 0.5  # Make jackpot food seem closer
                    # If we're in a risk zone, adjust distance based on food value
                    if snake.in_risk_zone and food.value > 1:
                        dist *= 0.7  # Prioritize valuable food in risk zones
                    if dist < min_dist:
                        min_dist = dist
                        closest_food = food
                if closest_food:
                    # Use appropriate algorithm for each snake
                    if snake.algorithm_id == 1:  # A* (green snake)
                        target = find_path_astar(snake, closest_food.pos, snakes, obstacles)
                    elif snake.algorithm_id == 2:  # Dijkstra (blue snake)
                        target = find_path_dijkstra(snake, closest_food.pos, snakes, obstacles)
                    elif snake.algorithm_id == 4:  # BFS (purple snake)
                        target = find_path_bfs(snake, closest_food.pos, snakes, obstacles)
                    elif snake.algorithm_id == 5:  # Greedy (orange snake)
                        target = find_path_greedy(snake, closest_food.pos, snakes, obstacles)
                else:
                    # If no food, move randomly
                    target = get_random_target(snake, snakes, obstacles, foods)
            # Move snake toward target
            snake.move(target, dt)
            # Check for collisions (only after startup frames)
            if frame_count > startup_frames and snake.check_collision(snakes, obstacles):
                snake.alive = False
                # Generate death orbs
                death_orbs = snake.generate_death_orbs()
                foods.extend(death_orbs)
                # Update kill counts - find who was closest to the dying snake
                closest_killer = None
                min_dist = float('inf')
                for other_snake in snakes:
                    if other_snake != snake and other_snake.alive:
                        dist = euclidean_distance(snake.get_head(), other_snake.get_head())
                        if dist < min_dist:
                            min_dist = dist
                            closest_killer = other_snake
                if closest_killer and min_dist < 100:  # Only count if reasonably close
                    kill_count[closest_killer.algorithm_id] += 1
                    closest_killer.score += 10  # Bonus for kill
                # Add visual effect for death
                for _ in range(30):
                    particles.append({
                        'pos': snake.get_head(),
                        'vel': (random.uniform(-5, 5), random.uniform(-5, 5)),
                        'color': snake.color,
                        'life': random.randint(20, 40),
                        'size': random.randint(3, 8)
                    })
                # Log death
                logging.info(f"Snake {snake.algorithm} died at score {snake.score}")
                # Check if this was the snake the player bet on
                if bet_placed and bet_snake_id == snake.algorithm_id:
                    bet_result = False
                    betting.process_winner(None)  # Process loss
                    bet_placed = False
            # Check food consumption
            for i, food in enumerate(foods[:]):
                head = snake.get_head()
                head_radius = snake.radius * (0.8 if snake.boosting else 1.0)
                if euclidean_distance(head, food.pos) < head_radius + food.size:
                    # Apply multiplier if in risk zone
                    base_value = food.value
                    actual_value = int(base_value * snake.multiplier)
                    # Grow snake
                    growth = actual_value
                    snake.grow(growth)
                    # Add to score
                    snake.score += actual_value
                    # Add visual effect for food consumption
                    for _ in range(10):
                        particles.append({
                            'pos': food.pos,
                            'vel': (random.uniform(-3, 3), random.uniform(-3, 3)),
                            'color': food.color,
                            'life': random.randint(10, 20),
                            'size': random.randint(2, 4)
                        })
                    # Show point gain if multiplier is active
                    if snake.multiplier > 1.0:
                        snake.add_winnings(actual_value)
                    # Special effects based on food type
                    if food.type == 'speed':
                        # Give a speed boost
                        snake.speed_boost = 100  # Lasts longer
                    elif food.type == 'portal':
                        # Teleport to a random safe location
                        for _ in range(50):
                            new_x = random.randint(50, width - 50)
                            new_y = random.randint(50, height - 50)
                            if not is_point_in_obstacles((new_x, new_y), obstacles):
                                snake.segments[0] = (new_x, new_y)
                                snake.exact_positions[0] = [new_x, new_y]
                                # Teleport particles at both locations
                                for _ in range(20):
                                    particles.append({
                                        'pos': food.pos,
                                        'vel': (random.uniform(-3, 3), random.uniform(-3, 3)),
                                        'color': cyan,
                                        'life': random.randint(15, 30),
                                        'size': random.randint(3, 6)
                                    })
                                for _ in range(20):
                                    particles.append({
                                        'pos': (new_x, new_y),
                                        'vel': (random.uniform(-3, 3), random.uniform(-3, 3)),
                                        'color': cyan,
                                        'life': random.randint(15, 30),
                                        'size': random.randint(3, 6)
                                    })
                                break
                    elif food.type == 'jackpot':
                        # Activate jackpot mode
                        snake.activate_jackpot_mode(180)  # 3 seconds of jackpot mode
                        # Special effects
                        for _ in range(30):
                            particles.append({
                                'pos': food.pos,
                                'vel': (random.uniform(-4, 4), random.uniform(-4, 4)),
                                'color': casino_gold,
                                'life': random.randint(30, 60),
                                'size': random.randint(4, 8)
                            })
                    # Remove consumed food
                    foods.remove(food)
                    # Add new food of random type
                    if random.random() < 0.8:  # 80% chance for replacement
                        new_food = spawn_food([s.segments for s in snakes], obstacles)
                        if new_food:
                            foods.append(new_food)
        # Add new food occasionally
        if frame_count % 30 == 0 and len(foods) < 30:
            new_foods = spawn_multiple_food(random.randint(1, 3),
                                     [snake.segments for snake in snakes],
                                     obstacles)
            foods.extend(new_foods)
        # Update particles
        for particle in particles[:]:
            particle['life'] -= 1
            if particle['life'] <= 0:
                particles.remove(particle)
                continue
            particle['pos'] = (
                particle['pos'][0] + particle['vel'][0],
                particle['pos'][1] + particle['vel'][1]
            )
            # Slow down particles gradually
            particle['vel'] = (
                particle['vel'][0] * 0.95,
                particle['vel'][1] * 0.95
            )
        # Drawing
        win.blit(background, (0, 0))  # Draw background
        # Draw risk zones
        for zone in risk_zones:
            zone.draw(win)
        # Draw obstacles
        for obstacle in obstacles:
            obstacle.draw(win)
        # Draw food
        for food in foods:
            food.draw(win)
        # Draw snake trails first (behind snakes)
        for snake in active_snakes:
            trail = create_trail_effect(snake)
            if trail:
                win.blit(trail, (0, 0))
        # Draw snakes
        for snake in active_snakes:
            snake.draw(win)
        # Draw particles
        for particle in particles:
            size = particle['size'] * (particle['life'] / 30)  # Fade out by size
            r, g, b = particle['color']
            # Add alpha for fading
            alpha = min(255, particle['life'] * 10)
            color = (r, g, b, alpha)
            pygame.draw.circle(win, color,
                          (int(particle['pos'][0]), int(particle['pos'][1])),
                          size)
        # Draw scoreboard
        # Background for scores
        score_bg = pygame.Surface((width, 60))
        score_bg.fill((20, 20, 30))
        score_bg.set_alpha(200)
        win.blit(score_bg, (0, 0))
        # Title
        title_text = title_font.render("AI Pathfinding Arena - Slither Edition", True, white)
        win.blit(title_text, (width//2 - title_text.get_width()//2, 10))
        # Sort snakes by score for leaderboard
        sorted_snakes = sorted(
            snakes,
            key=lambda s: s.score + (kill_count[s.algorithm_id] * 10),
            reverse=True
        )
        # Leaderboard display
        x_offset = 20
        for i, snake in enumerate(sorted_snakes):
            # Create rank indicator
            rank_text = leaderboard_font.render(f"#{i+1}", True, (200, 200, 200))
            # Create score text with algorithm name
            status = "ALIVE" if snake.alive else "DEAD"
            score_text = leaderboard_font.render(
                f"{snake.algorithm}: {snake.score} pts | {kill_count[snake.algorithm_id]} kills | {status}",
                True,
                snake.color if snake.alive else (*snake.color, 150)
            )
            # Draw rank and score
            win.blit(rank_text, (x_offset, height - 40))
            win.blit(score_text, (x_offset + 40, height - 40))
            x_offset += score_text.get_width() + 80
        # Draw betting UI
        betting.draw(win, snakes)
        # Check if game should end (all but one snake dead or timeout)
        alive_count = sum(1 for snake in snakes if snake.alive)
        if alive_count <= 1 or frame_count > 6000:  # End after ~3 minutes max
            if alive_count == 1:
                winner = next(snake for snake in snakes if snake.alive)
                logging.info(f"Winner: {winner.algorithm} with score {winner.score}")
                # Process bet result if player bet
                if bet_placed and bet_snake_id == winner.algorithm_id:
                    winnings = betting.process_winner(winner.algorithm_id)
                    bet_result = True
                    logging.info(f"Player won {winnings} coins from bet")
                elif bet_placed:
                    betting.process_winner(None)  # Process loss
                    bet_result = False
                # Display winner message
                winner_bg = pygame.Surface((600, 100))
                winner_bg.fill((0, 0, 0))
                winner_bg.set_alpha(200)
                win.blit(winner_bg, (width//2 - 300, height//2 - 50))
                winner_text = title_font.render(
                    f"WINNER: {winner.algorithm} with {winner.score} points!",
                    True,
                    winner.color
                )
                win.blit(winner_text, (width//2 - winner_text.get_width()//2, height//2 - 20))
                pygame.display.update()
                pygame.time.delay(3000)  # Show for 3 seconds
            game_over = True
        pygame.display.update()
        clock.tick(60)  # Higher frame rate for smoother animation
    # Log location of the log file
    print(f"Game log saved to: {log_filename}")
    return log_filename

# Run the game
log_file = gameLoop()
