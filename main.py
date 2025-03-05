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
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    return log_filename

# Pygame and game setup
pygame.init()
width, height = 1000, 1000
win = pygame.display.set_mode((width, height))
pygame.display.set_caption('Snake Game with Multiple AIs')

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

snake_block = 10
snake_speed = 15

font_style = pygame.font.SysFont("bahnschrift", 20)

def get_all_moves():
    return [
        (0, -snake_block),     # Up
        (0, snake_block),      # Down
        (-snake_block, 0),     # Left
        (snake_block, 0)       # Right
    ]

def is_valid_move(head, move, snake_bodies):
    new_head = (head[0] + move[0], head[1] + move[1])
    
    # Check boundary conditions
    if (new_head[0] < 0 or new_head[0] >= width or 
        new_head[1] < 0 or new_head[1] >= height):
        logging.debug(f"Invalid move: Out of bounds - {new_head}")
        return False
    
    # Check collision with snake bodies
    for body in snake_bodies:
        if new_head in body:
            logging.debug(f"Invalid move: Collision with snake body - {new_head}")
            return False
    
    return True

def find_path_to_food(start, goal, snake_bodies, max_depth=100):
    logging.debug(f"Finding path from {start} to {goal}")
    
    # Breadth-first search with depth limit
    queue = deque([(start, [start])])
    visited = set([start])
    
    while queue:
        current, path = queue.popleft()
        
        # Stop if path is too long
        if len(path) > max_depth:
            logging.debug(f"Path search exceeded max depth of {max_depth}")
            continue
        
        if current == goal:
            logging.debug(f"Path found: {path}")
            return path[1:] if len(path) > 1 else []
        
        for move in get_all_moves():
            next_pos = (current[0] + move[0], current[1] + move[1])
            
            if next_pos not in visited and is_valid_move(current, move, snake_bodies):
                queue.append((next_pos, path + [next_pos]))
                visited.add(next_pos)
    
    logging.debug("No path found")
    return []

def get_safe_move(head, snake_bodies):
    possible_moves = get_all_moves()
    safe_moves = [move for move in possible_moves if is_valid_move(head, move, snake_bodies)]
    
    # If no safe moves, return current position to prevent crash
    if not safe_moves:
        logging.warning(f"No safe moves found for head {head}")
        return (0, 0)
    
    chosen_move = random.choice(safe_moves)
    logging.debug(f"Safe move chosen: {chosen_move}")
    return chosen_move

def gameLoop():
    # Set up logging and get log filename
    log_filename = setup_logging()
    logging.info("Game started")
    
    game_over = False
    game_iterations = 0

    # Snakes setup
    snake_colors = {
        1: green,   # A* 
        2: blue,    # Dijkstra
        3: red,     # Random
        4: purple,  # BFS
        5: orange   # Greedy
    }
    
    # Initialize unique snake positions
    def generate_start():
        start = (
            round(random.randint(0, (width // snake_block) - 1) * snake_block),
            round(random.randint(0, (height // snake_block) - 1) * snake_block)
        )
        logging.debug(f"Generated snake start position: {start}")
        return start
    
    snake_data = {
        id: {
            'body': [generate_start()],
            'length': 1,
            'score': 0,
            'color': color,
            'algorithm': ['A*', 'Dijkstra', 'Random', 'BFS', 'Greedy BFS'][id-1]
        } for id, color in snake_colors.items()
    }

    # Food generation
    def spawn_food(existing_foods, snake_bodies):
        attempts = 0
        max_attempts = 100
        
        while attempts < max_attempts:
            food = (
                round(random.randrange(0, width - snake_block) / 10.0) * 10.0,
                round(random.randrange(0, height - snake_block) / 10.0) * 10.0
            )
            # Ensure food doesn't spawn on snakes or other food
            if all(food not in body for body in snake_bodies) and \
               all(food not in foods for foods in existing_foods):
                logging.debug(f"Food spawned at {food}")
                return food
            
            attempts += 1
        
        logging.error(f"Failed to spawn food after {max_attempts} attempts")
        return None

    # Initialize food lists
    food_positions = []
    super_food_positions = []

    # Spawn initial foods
    all_snake_bodies = [snake['body'] for snake in snake_data.values()]
    food_positions = [spawn_food([super_food_positions], all_snake_bodies) for _ in range(3)]
    super_food_positions = [spawn_food([food_positions], all_snake_bodies) for _ in range(2)]

    clock = pygame.time.Clock()

    while not game_over:
        game_iterations += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        # Get all snake bodies for collision checks
        all_snake_bodies = [snake['body'] for snake in snake_data.values()]

        # Process each snake
        for snake_id, snake in snake_data.items():
            head = snake['body'][0]
            
            # Determine next move based on snake type
            if snake_id in [1, 2, 4, 5]:  # Pathfinding snakes
                # Prioritize super food, then normal food
                food_targets = super_food_positions + food_positions
                
                if food_targets:
                    # Find closest food
                    target = min(food_targets, key=lambda f: math.hypot(head[0]-f[0], head[1]-f[1]))
                    
                    # Find path to food
                    path = find_path_to_food(head, target, all_snake_bodies)
                    
                    # If no path, get a safe move
                    if not path:
                        move = get_safe_move(head, all_snake_bodies)
                        next_pos = (head[0] + move[0], head[1] + move[1])
                        logging.warning(f"{snake['algorithm']} snake used fallback move")
                    else:
                        next_pos = path[0]
                else:
                    # If no food, move randomly
                    move = get_safe_move(head, all_snake_bodies)
                    next_pos = (head[0] + move[0], head[1] + move[1])
            
            elif snake_id == 3:  # Random snake
                move = get_safe_move(head, all_snake_bodies)
                next_pos = (head[0] + move[0], head[1] + move[1])

            # Move snake
            snake['body'].insert(0, next_pos)

            # Check food consumption
            if next_pos in food_positions:
                snake['length'] += 1
                snake['score'] += 1
                logging.info(f"{snake['algorithm']} snake ate normal food. Score: {snake['score']}")
                food_positions.remove(next_pos)
                new_food = spawn_food([super_food_positions], all_snake_bodies)
                if new_food:
                    food_positions.append(new_food)
            elif next_pos in super_food_positions:
                snake['length'] += 2
                snake['score'] += 5
                logging.info(f"{snake['algorithm']} snake ate super food. Score: {snake['score']}")
                super_food_positions.remove(next_pos)
                new_food = spawn_food([food_positions], all_snake_bodies)
                if new_food:
                    super_food_positions.append(new_food)
            else:
                # Remove tail if not growing
                if len(snake['body']) > snake['length']:
                    snake['body'].pop()

        # Drawing (same as previous version)
        win.fill(black)
        
        # Draw grid
        for x in range(0, width, snake_block):
            pygame.draw.line(win, dark_gray, (x, 0), (x, height))
        for y in range(0, height, snake_block):
            pygame.draw.line(win, dark_gray, (0, y), (width, y))

        # Draw food
        for food in food_positions:
            pygame.draw.circle(win, white, 
                               (int(food[0] + snake_block/2), int(food[1] + snake_block/2)), 
                               snake_block//2)
        
        for food in super_food_positions:
            pygame.draw.circle(win, yellow, 
                               (int(food[0] + snake_block/2), int(food[1] + snake_block/2)), 
                               snake_block//2, 2)

        # Draw snakes
        for id, snake in snake_data.items():
            for segment in snake['body']:
                rect = pygame.Rect(segment[0], segment[1], snake_block, snake_block)
                pygame.draw.rect(win, snake_colors[id], rect)
                pygame.draw.rect(win, black, rect, 1)

        # Score display
        score_text = font_style.render(
            " | ".join(f"{snake['algorithm']}: {snake['score']}" for id, snake in snake_data.items()),
            True, white)
        win.blit(score_text, (10, 10))
        
        pygame.display.update()
        clock.tick(snake_speed)

        # Optional: Add a game length limit
        if game_iterations > 5000:
            logging.warning("Game reached maximum iteration limit")
            game_over = True

    # Game over logging
    logging.info("Game ended")
    logging.info("Final Scores:")
    for id, snake in snake_data.items():
        logging.info(f"{snake['algorithm']} Snake - Score: {snake['score']}, Length: {snake['length']}")
    
    # Close Pygame
    pygame.quit()
    
    # Log location of the log file
    print(f"Game log saved to: {log_filename}")
    
    return log_filename

# Run the game
log_file = gameLoop()
