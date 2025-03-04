
import pygame
import random
import math
from collections import deque

# Set verbose mode for debug output
VERBOSE = True

# Log file path
LOG_FILE = "game_debug.log"

# Ensure log file is empty at the start of the game
with open(LOG_FILE, "w") as f:
    f.write("=== Game Debug Log ===\n")

def debug_print(snake_id, algo, head, target, path, next_move, fallback=False):
    log_entry = f"Snake {snake_id} ({algo}): Head = {head}\n"
    log_entry += f"   Target Food: {target}\n"
    
    if path:
        log_entry += f"   Computed path: {path}\n"
    else:
        log_entry += "   No valid path computed.\n"

    if fallback:
        log_entry += f"   Fallback move used: {next_move}\n"
    else:
        log_entry += f"   Next move: {next_move}\n"

    # Print to console
    if VERBOSE:
        print(log_entry)

    # Save to log file
    with open(LOG_FILE, "a") as log_file:
        log_file.write(log_entry + "\n")

# Initialize Pygame
pygame.init()

# Set up display
width, height = 500, 500
win = pygame.display.set_mode((width, height))
pygame.display.set_caption('Snake Game with Multiple AIs - Debug Mode')

# Define colors
black      = (0, 0, 0)
dark_gray  = (40, 40, 40)
white      = (255, 255, 255)
green      = (0, 255, 0)       # A* snake
blue       = (0, 0, 255)       # Dijkstra snake
red        = (255, 0, 0)       # Random snake
purple     = (128, 0, 128)     # BFS snake
orange     = (255, 165, 0)     # Greedy Best-First snake
yellow     = (255, 255, 0)

# Snake properties
snake_block = 10
snake_speed = 15

font_style = pygame.font.SysFont("bahnschrift", 20)

def draw_grid():
    for x in range(0, width, snake_block):
        pygame.draw.line(win, dark_gray, (x, 0), (x, height))
    for y in range(0, height, snake_block):
        pygame.draw.line(win, dark_gray, (0, y), (width, y))

def draw_snake(snake, color):
    for segment in snake:
        rect = pygame.Rect(segment[0], segment[1], snake_block, snake_block)
        pygame.draw.rect(win, color, rect)
        pygame.draw.rect(win, black, rect, 1)  # Border for a nicer look

def draw_food(foods, color, is_super=False):
    for food in foods:
        center = (int(food[0] + snake_block / 2), int(food[1] + snake_block / 2))
        radius = snake_block // 2
        pygame.draw.circle(win, color, center, radius)
        if is_super:
            # Add an outline to super food for distinction
            pygame.draw.circle(win, black, center, radius, 2)

def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(start, goal, snakes):
    open_set = {start}
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Return reversed path

        open_set.remove(current)
        for dx, dy in [(0, -snake_block), (0, snake_block), (-snake_block, 0), (snake_block, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < width and 0 <= neighbor[1] < height and
                    all(neighbor not in snake for snake in snakes) and neighbor != current):
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    open_set.add(neighbor)
    return []

def dijkstra(start, goal, snakes):
    unvisited = {start}
    distances = {start: 0}
    came_from = {}
    
    while unvisited:
        current = min(unvisited, key=lambda x: distances.get(x, float('inf')))
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        unvisited.remove(current)
        for dx, dy in [(0, -snake_block), (0, snake_block), (-snake_block, 0), (snake_block, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < width and 0 <= neighbor[1] < height and 
                    all(neighbor not in snake for snake in snakes) and neighbor != current):
                tentative_distance = distances[current] + 1
                if tentative_distance < distances.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    distances[neighbor] = tentative_distance
                    unvisited.add(neighbor)
    return []

def bfs(start, goal, snakes):
    queue = deque([start])
    came_from = {}
    visited = {start}
    while queue:
        current = queue.popleft()
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        for dx, dy in [(0, -snake_block), (0, snake_block), (-snake_block, 0), (snake_block, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < width and 0 <= neighbor[1] < height and
                all(neighbor not in snake for snake in snakes) and neighbor not in visited):
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)
    return []

def greedy_bfs(start, goal, snakes):
    open_set = [start]
    came_from = {}
    visited = {start}
    while open_set:
        current = min(open_set, key=lambda x: heuristic(x, goal))
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        open_set.remove(current)
        for dx, dy in [(0, -snake_block), (0, snake_block), (-snake_block, 0), (snake_block, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < width and 0 <= neighbor[1] < height and
                all(neighbor not in snake for snake in snakes) and neighbor not in visited):
                visited.add(neighbor)
                came_from[neighbor] = current
                open_set.append(neighbor)
    return []

def find_closest_food(snake_head, food_positions):
    closest_food = None
    closest_distance = float('inf')
    for food in food_positions:
        distance = heuristic(snake_head, food)
        if distance < closest_distance:
            closest_distance = distance
            closest_food = food
    return closest_food

def get_random_valid_move(head, snakes):
    valid_moves = []
    for dx, dy in [(0, -snake_block), (0, snake_block), (-snake_block, 0), (snake_block, 0)]:
        new_head = (head[0] + dx, head[1] + dy)
        if (0 <= new_head[0] < width and 0 <= new_head[1] < height and
                all(new_head not in snake for snake in snakes)):
            valid_moves.append(new_head)
    if valid_moves:
        return random.choice(valid_moves)
    return head  # No valid moves available

def debug_print(snake_id, algo, head, target, path, next_move, fallback=False):
    if VERBOSE:
        print(f"Snake {snake_id} ({algo}): Head = {head}")
        print(f"   Target Food: {target}")
        if path:
            print(f"   Computed path: {path}")
        else:
            print("   No valid path computed.")
        if fallback:
            print(f"   Fallback move used: {next_move}")
        else:
            print(f"   Next move: {next_move}")

def gameLoop():
    game_over = False

    # Initialize snakes with random starting positions
    ai1 = [(random.randint(0, width // snake_block - 1) * snake_block,
            random.randint(0, height // snake_block - 1) * snake_block)]
    ai2 = [(random.randint(0, width // snake_block - 1) * snake_block,
            random.randint(0, height // snake_block - 1) * snake_block)]
    ai3 = [(random.randint(0, width // snake_block - 1) * snake_block,
            random.randint(0, height // snake_block - 1) * snake_block)]
    ai4 = [(random.randint(0, width // snake_block - 1) * snake_block,
            random.randint(0, height // snake_block - 1) * snake_block)]
    ai5 = [(random.randint(0, width // snake_block - 1) * snake_block,
            random.randint(0, height // snake_block - 1) * snake_block)]
    
    lengths = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
    scores  = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    food_positions = [(
        round(random.randrange(0, width - snake_block) / 10.0) * 10.0,
        round(random.randrange(0, height - snake_block) / 10.0) * 10.0
    )]
    super_food_positions = [(
        round(random.randrange(0, width - snake_block) / 10.0) * 10.0,
        round(random.randrange(0, height - snake_block) / 10.0) * 10.0
    )]

    clock = pygame.time.Clock()

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        # Combine all snake segments for collision detection
        all_snakes = [ai1, ai2, ai3, ai4, ai5]

        # --- AI Snake 1 - A* ---
        head1 = ai1[0]
        target_food = find_closest_food(head1, food_positions + super_food_positions)
        path1 = a_star(head1, target_food, all_snakes)
        if path1:
            next_move = path1[0]
            debug_print(1, "A*", head1, target_food, path1, next_move)
            fallback_used = False
        else:
            next_move = get_random_valid_move(head1, all_snakes)
            debug_print(1, "A*", head1, target_food, None, next_move, fallback=True)
            fallback_used = True
        if next_move != head1:
            ai1.insert(0, next_move)
            if next_move in food_positions:
                lengths[1] += 1
                scores[1] += 1
                food_positions.remove(next_move)
                food_positions.append((
                    round(random.randrange(0, width - snake_block) / 10.0) * 10.0,
                    round(random.randrange(0, height - snake_block) / 10.0) * 10.0
                ))
            elif next_move in super_food_positions:
                lengths[1] += 2
                scores[1] += 5
                super_food_positions.remove(next_move)
                super_food_positions.append((
                    round(random.randrange(0, width - snake_block) / 10.0) * 10.0,
                    round(random.randrange(0, height - snake_block) / 10.0) * 10.0
                ))
            else:
                if len(ai1) > lengths[1]:
                    ai1.pop()

        # --- AI Snake 2 - Dijkstra ---
        head2 = ai2[0]
        target_food = find_closest_food(head2, food_positions + super_food_positions)
        path2 = dijkstra(head2, target_food, all_snakes)
        if path2:
            next_move = path2[0]
            debug_print(2, "Dijkstra", head2, target_food, path2, next_move)
            fallback_used = False
        else:
            next_move = get_random_valid_move(head2, all_snakes)
            debug_print(2, "Dijkstra", head2, target_food, None, next_move, fallback=True)
            fallback_used = True
        if next_move != head2:
            ai2.insert(0, next_move)
            if next_move in food_positions:
                lengths[2] += 1
                scores[2] += 1
                food_positions.remove(next_move)
                food_positions.append((
                    round(random.randrange(0, width - snake_block) / 10.0) * 10.0,
                    round(random.randrange(0, height - snake_block) / 10.0) * 10.0
                ))
            elif next_move in super_food_positions:
                lengths[2] += 2
                scores[2] += 5
                super_food_positions.remove(next_move)
                super_food_positions.append((
                    round(random.randrange(0, width - snake_block) / 10.0) * 10.0,
                    round(random.randrange(0, height - snake_block) / 10.0) * 10.0
                ))
            else:
                if len(ai2) > lengths[2]:
                    ai2.pop()

        # --- AI Snake 3 - Random Movement ---
        head3 = ai3[0]
        directions = [(0, -snake_block), (0, snake_block), (-snake_block, 0), (snake_block, 0)]
        random_direction = random.choice(directions)
        next_move = (head3[0] + random_direction[0], head3[1] + random_direction[1])
        if VERBOSE:
            print(f"Snake 3 (Random): Head = {head3}, Random direction = {random_direction}, Next move = {next_move}")
        if (0 <= next_move[0] < width and 0 <= next_move[1] < height and
                all(next_move not in snake for snake in [ai1, ai2, ai4, ai5]) and next_move != head3):
            ai3.insert(0, next_move)
            if next_move in food_positions:
                lengths[3] += 1
                scores[3] += 1
                food_positions.remove(next_move)
                food_positions.append((
                    round(random.randrange(0, width - snake_block) / 10.0) * 10.0,
                    round(random.randrange(0, height - snake_block) / 10.0) * 10.0
                ))
            elif next_move in super_food_positions:
                lengths[3] += 2
                scores[3] += 5
                super_food_positions.remove(next_move)
                super_food_positions.append((
                    round(random.randrange(0, width - snake_block) / 10.0) * 10.0,
                    round(random.randrange(0, height - snake_block) / 10.0) * 10.0
                ))
            else:
                if len(ai3) > lengths[3]:
                    ai3.pop()

        # --- AI Snake 4 - BFS ---
        head4 = ai4[0]
        target_food = find_closest_food(head4, food_positions + super_food_positions)
        path4 = bfs(head4, target_food, all_snakes)
        if path4:
            next_move = path4[0]
            debug_print(4, "BFS", head4, target_food, path4, next_move)
            fallback_used = False
        else:
            next_move = get_random_valid_move(head4, all_snakes)
            debug_print(4, "BFS", head4, target_food, None, next_move, fallback=True)
            fallback_used = True
        if next_move != head4:
            ai4.insert(0, next_move)
            if next_move in food_positions:
                lengths[4] += 1
                scores[4] += 1
                food_positions.remove(next_move)
                food_positions.append((
                    round(random.randrange(0, width - snake_block) / 10.0) * 10.0,
                    round(random.randrange(0, height - snake_block) / 10.0) * 10.0
                ))
            elif next_move in super_food_positions:
                lengths[4] += 2
                scores[4] += 5
                super_food_positions.remove(next_move)
                super_food_positions.append((
                    round(random.randrange(0, width - snake_block) / 10.0) * 10.0,
                    round(random.randrange(0, height - snake_block) / 10.0) * 10.0
                ))
            else:
                if len(ai4) > lengths[4]:
                    ai4.pop()

        # --- AI Snake 5 - Greedy Best-First Search ---
        head5 = ai5[0]
        target_food = find_closest_food(head5, food_positions + super_food_positions)
        path5 = greedy_bfs(head5, target_food, all_snakes)
        if path5:
            next_move = path5[0]
            debug_print(5, "Greedy BFS", head5, target_food, path5, next_move)
            fallback_used = False
        else:
            next_move = get_random_valid_move(head5, all_snakes)
            debug_print(5, "Greedy BFS", head5, target_food, None, next_move, fallback=True)
            fallback_used = True
        if next_move != head5:
            ai5.insert(0, next_move)
            if next_move in food_positions:
                lengths[5] += 1
                scores[5] += 1
                food_positions.remove(next_move)
                food_positions.append((
                    round(random.randrange(0, width - snake_block) / 10.0) * 10.0,
                    round(random.randrange(0, height - snake_block) / 10.0) * 10.0
                ))
            elif next_move in super_food_positions:
                lengths[5] += 2
                scores[5] += 5
                super_food_positions.remove(next_move)
                super_food_positions.append((
                    round(random.randrange(0, width - snake_block) / 10.0) * 10.0,
                    round(random.randrange(0, height - snake_block) / 10.0) * 10.0
                ))
            else:
                if len(ai5) > lengths[5]:
                    ai5.pop()

        # --- Drawing ---
        win.fill(black)
        draw_grid()
        draw_food(food_positions, white)
        draw_food(super_food_positions, yellow, is_super=True)
        
        draw_snake(ai1, green)
        draw_snake(ai2, blue)
        draw_snake(ai3, red)
        draw_snake(ai4, purple)
        draw_snake(ai5, orange)
        
        # Score display with a background for readability
        score_text = font_style.render(
            f"A*: {scores[1]}  Dijk: {scores[2]}  Rand: {scores[3]}  BFS: {scores[4]}  Greedy: {scores[5]}",
            True, white)
        text_bg = pygame.Surface((score_text.get_width() + 10, score_text.get_height() + 10))
        text_bg.fill(dark_gray)
        win.blit(text_bg, (5, 5))
        win.blit(score_text, (10, 10))
        
        pygame.display.update()
        clock.tick(snake_speed)

    pygame.quit()
    quit()

gameLoop()

