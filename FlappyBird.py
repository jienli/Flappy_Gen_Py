import pygame
import sys
import random 
import time

# meta-parameters / Game Variables
Window_Width = 576
Window_Height = 1024

gravity = 0.25
bird_movement = 0

score = 0
high_score = 0

def draw_floor():
    screen.blit(floor_surface, (floor_x, Window_Height - 100))
    screen.blit(floor_surface, (floor_x + Window_Width, Window_Height - 100))

def create_pipe():
    random_pipe_pos = random.choice([400, 500, 600, 700, 800])
    bottom_pipe = pipe_surface.get_rect(midtop = (Window_Width + 100, random_pipe_pos))
    top_pipe = pipe_surface.get_rect(midbottom = (Window_Width + 100, random_pipe_pos - 300))
    return bottom_pipe, top_pipe

def move_pipes(pipes):
    for pipe in pipes:
        pipe.centerx -= 5
    return pipes

def draw_pipes(pipes):
    for pipe in pipes:
        if pipe.bottom > Window_Height:
            screen.blit(pipe_surface, pipe)
        else:
            screen.blit(pipe_flip_surface, pipe)

def rotate_bird(bird_surface):
    new_bird = pygame.transform.rotozoom(bird_surface, bird_movement * -3, 1)
    return new_bird

def check_collision(pipes):
    for pipe in pipes:
        if bird_rect.colliderect(pipe):
            game_over()
    if bird_rect.top <= -100 or bird_rect.bottom >= Window_Height - 100:
        game_over()

def score_display():
    display_score = max(int(score - 1.8), 0)
    score_surface = game_font.render("Score: " + str(display_score), True, (255, 255, 255))
    score_rect = score_surface.get_rect(center = (Window_Width / 2, 100))
    screen.blit(score_surface, score_rect)

    if not game_active:
        display_high_score = max(int(high_score - 1.8), 0)
        high_score_surface = game_font.render("High Score: " + str(display_high_score), True, (255, 255, 255))
        high_score_rect = high_score_surface.get_rect(center = (Window_Width / 2, Window_Height / 2 - 100))
        screen.blit(high_score_surface, high_score_rect)

        notification_surface = game_font.render("press space to restart", True, (255, 255, 255))
        notification_surface_rect = notification_surface.get_rect(center = (Window_Width / 2, Window_Height / 2))
        screen.blit(notification_surface, notification_surface_rect)

def game_over():
    global game_active
    game_active = False
    global high_score
    if score > high_score:
        high_score = score

def new_game():
    global score
    score = 0
    global bird_movement
    bird_movement = 0
    global game_active 
    game_active = True
    pipe_list.clear()
    bird_rect.center = (100, 250)



pygame.init()

screen = pygame.display.set_mode((Window_Width, Window_Height))
clock = pygame.time.Clock()
game_font = pygame.font.Font("04B_19.ttf", 40)


# Non-Interactable Game Variables

game_active = True

bg_surface = pygame.image.load("assets/background-day.png").convert()
bg_surface = pygame.transform.scale2x(bg_surface)

floor_surface = pygame.image.load("assets/base.png").convert()
floor_surface = pygame.transform.scale2x(floor_surface)
floor_x = 0

bird_surface = pygame.image.load("assets/bluebird-midflap.png").convert_alpha()
bird_surface = pygame.transform.scale2x(bird_surface)
bird_rect = bird_surface.get_rect(center = (100, 250))

pipe_surface = pygame.image.load("assets/pipe-green.png")
pipe_surface = pygame.transform.scale2x(pipe_surface)
pipe_flip_surface = pygame.transform.flip(pipe_surface, False, True)
pipe_list = []
pipe_list.extend(create_pipe())
SPAWNPIPE = pygame.USEREVENT
pygame.time.set_timer(SPAWNPIPE, 1200)


while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            # sys.exit()        // this or exit()
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if game_active:
                    bird_movement = -10
                else:
                    new_game()
                
        if event.type == SPAWNPIPE and game_active:
            pipe_list.extend(create_pipe())
            

    # Background
    screen.blit(bg_surface, (0, 0))

    # Bird
    if game_active:
        bird_movement += gravity
        bird_rect.centery += bird_movement
    rotated_bird_surface = rotate_bird(bird_surface)
    screen.blit(rotated_bird_surface, bird_rect)

    # Pipes
    if game_active:
        pipe_list = move_pipes(pipe_list)
    draw_pipes(pipe_list) 

    # Floor
    if game_active:
        floor_x = (floor_x - 6) % - Window_Width
    else:
        floor_x = (floor_x - 1) % - Window_Width
    draw_floor()

    if game_active:
        score += 1/1.2/60
    score_display()

    if game_active:
        check_collision(pipe_list)
        
    
    pygame.display.update()
    clock.tick(60)

pygame.quit() 