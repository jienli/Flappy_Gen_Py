import numpy as np
from typing import Tuple, Optional, Union, Set, Dict, Any
from fractions import Fraction
import random
import pygame
import sys
import os
import json

# from misc import *
from typing import List, Tuple, Union, Dict
from genetic_algorithm.individual import Individual
from neural_network import FeedForwardNetwork, linear, sigmoid, tanh, relu, leaky_relu, ActivationFunction, get_activation_by_name
from settings import settings

BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,180,0)
BLUE = (50,200,255)

class Bird(Individual):
    def __init__(self, 
                chromosome: Optional[Dict[str, List[np.ndarray]]] = None,
                x_pos: Optional[int] = settings['init_bird_x_pos'], 
                y_pos: Optional[int] = settings['init_bird_y_pos'],
                # x_pos: Optional[int] = 100, 
                # y_pos: Optional[int] = 250,
                y_speed: Optional[int] = 0,
                hidden_layer_architecture: Optional[List[int]] = [12, 20],
                hidden_activation: Optional[ActivationFunction] = 'relu',
                output_activation: Optional[ActivationFunction] = 'sigmoid'
                ):

        self.Window_Width = settings['Window_Width']
        self.Window_Height = settings['Window_Height']

        self.x_pos = x_pos
        self.y_pos = y_pos
        self.y_speed = y_speed

        self.bird_surface = pygame.image.load("assets/bluebird-midflap.png").convert_alpha()
        self.bird_surface = pygame.transform.scale2x(self.bird_surface)
        self.bird_rect = self.bird_surface.get_rect(center = (self.x_pos, self.y_pos))

        self._fitness = 0  # Overall fitness
        self.score = 0
        self.x_distance_to_next_pipe_center = 0
        self.y_distance_to_next_pipe_center = 0
        self.is_alive = True

        self.hidden_layer_architecture = hidden_layer_architecture
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation



        # Setting up network architecture
        num_inputs = 5 #@TODO: how many
        self.network_architecture = [num_inputs]                          # Inputs
        self.network_architecture.extend(self.hidden_layer_architecture)  # Hidden layers
        self.network_architecture.append(2)                               # 2 output, jump or not jump
        self.network = FeedForwardNetwork(self.network_architecture,
                                          get_activation_by_name(self.hidden_activation),
                                          get_activation_by_name(self.output_activation)
        )

        # If chromosome is set, take it
        if chromosome:
            # self._chromosome = chromosome
            self.network.params = chromosome
            # self.decode_chromosome()
        else:
            # self._chromosome = {}
            # self.encode_chromosome()
            pass

        

    @property
    def fitness(self):
        return self._fitness
    
    def calculate_fitness(self):
        # Give positive minimum fitness for roulette wheel selection
        # self._fitness = (self._frames) + ((2**self.score) + (self.score**2.1)*500) - (((.25 * self._frames)**1.3) * (self.score**1.2))
        # self._fitness = (self._frames) + ((2**self.score) + (self.score**2.1)*500) - (((.25 * self._frames)) * (self.score))
        print(self.y_distance_to_next_pipe_center)
        self._fitness = (2 ** self.score + self.score * 2.1) * 200 + 1000 - (self.x_distance_to_next_pipe_center + abs(self.y_distance_to_next_pipe_center)) * 1.2
        self._fitness = max(self._fitness, .1)

    @property
    def chromosome(self):
        # return self._chromosome
        pass

    def encode_chromosome(self):
        # # L = len(self.network.params) // 2
        # L = len(self.network.layer_nodes)
        # # Encode weights and bias
        # for layer in range(1, L):
        #     l = str(layer)
        #     self._chromosome['W' + l] = self.network.params['W' + l].flatten()
        #     self._chromosome['b' + l] = self.network.params['b' + l].flatten()
        pass

    def decode_chromosome(self):
        # # L = len(self.network.params) // 2
        # L = len(self.network.layer_nodes)
        # # Decode weights and bias
        # for layer in range(1, L):
        #     l = str(layer)
        #     w_shape = (self.network_architecture[layer], self.network_architecture[layer-1])
        #     b_shape = (self.network_architecture[layer], 1)
        #     self.network.params['W' + l] = self._chromosome['W' + l].reshape(w_shape)
        #     self.network.params['b' + l] = self._chromosome['b' + l].reshape(b_shape)
        pass

    # restart the game for this individual
    def reset(self):
        self._fitness = 0
        self.score = 0
        self.x_distance_to_next_pipe_center = 0
        self.y_distance_to_next_pipe_center = 0
        self.x_pos = settings['init_bird_x_pos']
        self.y_pos = settings['init_bird_y_pos']
        self.y_speed = 0
        self.is_alive = True


    def update(self, inputs):
        self.network.feed_forward(inputs)
        if self.network.out == 0:
            # do nothing
            pass
        elif self.network.out == 1:
            # jump
            self.y_speed = settings['bird_jump_speed']

        return True

    def move(self, pipes):
        self.y_pos += self.y_speed
        self.y_speed += settings['gravity']
        self.score += 1/1.2/60
        self.check_collision(pipes)
        # self.distance_travelled += abs(self.xspeed)
        # if self.x_pos < 0:
        #     self.x_pos = 0
        # elif self.x_pos > self.board_size[0]-100:
        #     self.x_pos = self.board_size[0]-100


    def check_collision(self, pipes):
        for pipe in pipes:
            if self.bird_rect.colliderect(pipe):
                self.is_alive = False 
        if self.bird_rect.top <= -100 or self.bird_rect.bottom >= self.Window_Height - 100:
            self.is_alive = False 


    def rotate_bird(self, bird_surface):
        new_bird = pygame.transform.rotozoom(bird_surface, self.y_speed * -3, 1)
        return new_bird

    # Draw the bird
    def draw(self, screen, winner=False, champion=False):
        if not self.is_alive:
            return
        if champion:
            self.bird_surface = pygame.image.load("assets/redbird-midflap.png").convert_alpha()
            self.bird_surface = pygame.transform.scale2x(self.bird_surface)
        elif winner:
            self.bird_surface = pygame.image.load("assets/yellowbird-midflap.png").convert_alpha()
            self.bird_surface = pygame.transform.scale2x(self.bird_surface)
        


        self.bird_rect.centery = self.y_pos
        rotated_bird_surface = self.rotate_bird(self.bird_surface)
        screen.blit(rotated_bird_surface, self.bird_rect)
        # TODO: include different color for champion and winner
        # if champion:
        #     pygame.draw.rect(screen,BLACK,[self.x_pos,self.y_pos,100,20])
        #     pygame.draw.rect(screen,GREEN,[self.x_pos+2,self.y_pos+2,100-4,20-4])
        # elif winner:
        #     pygame.draw.rect(screen,BLACK,[self.x_pos,self.y_pos,100,20])
        #     pygame.draw.rect(screen,BLUE,[self.x_pos+2,self.y_pos+2,100-4,20-4])
        # else:
        #     pygame.draw.rect(screen,BLACK,[self.x_pos,self.y_pos,100,20])
        #     pygame.draw.rect(screen,WHITE,[self.x_pos+2,self.y_pos+2,100-4,20-4])

        
        
       

# def save_snake(population_folder: str, individual_name: str, snake: Snake, settings: Dict[str, Any]) -> None:
#     # Make population folder if it doesn't exist
#     if not os.path.exists(population_folder):
#         os.makedirs(population_folder)

#     # Save off settings
#     if 'settings.json' not in os.listdir(population_folder):
#         f = os.path.join(population_folder, 'settings.json')
#         with open(f, 'w', encoding='utf-8') as out:
#             json.dump(settings, out, sort_keys=True, indent=4)

#     # Make directory for the individual
#     individual_dir = os.path.join(population_folder, individual_name)
#     os.makedirs(individual_dir)

#     # Save some constructor information for replay
#     # @NOTE: No need to save chromosome since that is saved as .npy
#     # @NOTE: No need to save board_size or hidden_layer_architecture
#     #        since these are taken from settings
#     constructor = {}
#     constructor['start_pos'] = snake.start_pos.to_dict()
#     constructor['apple_seed'] = snake.apple_seed
#     constructor['initial_velocity'] = snake.initial_velocity
#     constructor['starting_direction'] = snake.starting_direction
#     snake_constructor_file = os.path.join(individual_dir, 'constructor_params.json')

#     # Save
#     with open(snake_constructor_file, 'w', encoding='utf-8') as out:
#         json.dump(constructor, out, sort_keys=True, indent=4)

#     L = len(snake.network.layer_nodes)
#     for l in range(1, L):
#         w_name = 'W' + str(l)
#         b_name = 'b' + str(l)

#         weights = snake.network.params[w_name]
#         bias = snake.network.params[b_name]

#         np.save(os.path.join(individual_dir, w_name), weights)
#         np.save(os.path.join(individual_dir, b_name), bias)

# def load_snake(population_folder: str, individual_name: str, settings: Optional[Union[Dict[str, Any], str]] = None) -> Snake:
#     if not settings:
#         f = os.path.join(population_folder, 'settings.json')
#         if not os.path.exists(f):
#             raise Exception("settings needs to be passed as an argument if 'settings.json' does not exist under population folder")
        
#         with open(f, 'r', encoding='utf-8') as fp:
#             settings = json.load(fp)

#     elif isinstance(settings, dict):
#         settings = settings

#     elif isinstance(settings, str):
#         filepath = settings
#         with open(filepath, 'r', encoding='utf-8') as fp:
#             settings = json.load(fp)

#     params = {}
#     for fname in os.listdir(os.path.join(population_folder, individual_name)):
#         extension = fname.rsplit('.npy', 1)
#         if len(extension) == 2:
#             param = extension[0]
#             params[param] = np.load(os.path.join(population_folder, individual_name, fname))
#         else:
#             continue

#     # Load constructor params for the specific snake
#     constructor_params = {}
#     snake_constructor_file = os.path.join(population_folder, individual_name, 'constructor_params.json')
#     with open(snake_constructor_file, 'r', encoding='utf-8') as fp:
#         constructor_params = json.load(fp)

#     snake = Snake(settings['board_size'], chromosome=params, 
#                   start_pos=Point.from_dict(constructor_params['start_pos']),
#                   apple_seed=constructor_params['apple_seed'],
#                   initial_velocity=constructor_params['initial_velocity'],
#                   starting_direction=constructor_params['starting_direction'],
#                   hidden_layer_architecture=settings['hidden_network_architecture'],
#                   hidden_activation=settings['hidden_layer_activation'],
#                   output_activation=settings['output_layer_activation'],
#                   lifespan=settings['lifespan'],
#                   apple_and_self_vision=settings['apple_and_self_vision']
#                   )
#     return snake