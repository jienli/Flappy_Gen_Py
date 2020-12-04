import sys
from typing import List
from bird import *
from ball import *
import numpy as np
import pygame
from neural_network import FeedForwardNetwork, sigmoid, linear, relu
from settings import settings
from genetic_algorithm.population import Population
from genetic_algorithm.selection import elitism_selection, roulette_wheel_selection, tournament_selection
from genetic_algorithm.mutation import gaussian_mutation, random_uniform_mutation
from genetic_algorithm.crossover import simulated_binary_crossover as SBX
from genetic_algorithm.crossover import uniform_binary_crossover, single_point_binary_crossover
from math import sqrt
from decimal import Decimal
import random

class Main():
    def __init__(self):
        
        self._mutation_bins = np.cumsum([settings['probability_gaussian'],
                                        settings['probability_random_uniform']
        ])
        self._crossover_bins = np.cumsum([settings['probability_SBX'],
                                         settings['probability_SPBX']
        ])
        self._SPBX_type = settings['SPBX_type'].lower()
        self._SBX_eta = settings['SBX_eta']
        self._mutation_rate = settings['mutation_rate']

        # Determine size of next gen based off selection type
        self._next_gen_size = None
        if settings['selection_type'].lower() == 'plus':
            self._next_gen_size = settings['num_parents'] + settings['num_offspring']
        elif settings['selection_type'].lower() == 'comma':
            self._next_gen_size = settings['num_offspring']
        else:
            raise Exception('Selection type "{}" is invalid'.format(settings['selection_type']))

        
        self.Window_Width = settings['Window_Width']
        self.Window_Height = settings['Window_Height']
        


        #### Pygame init ####
        pygame.init()
        self.screen = pygame.display.set_mode((self.Window_Width, self.Window_Height))
        pygame.display.set_caption('Flappy Bird')
        self.game_font = pygame.font.Font("04B_19.ttf", 40)


        individuals: List[Individual] = []
        # Create initial generation
        for _ in range(settings['num_parents']):
            individual = Bird(hidden_layer_architecture=settings['hidden_network_architecture'],
                            hidden_activation=settings['hidden_layer_activation'],
                            output_activation=settings['output_layer_activation'])
            individuals.append(individual)

        self.population = Population(individuals)

        self.current_generation = 0

        # Best of single generation
        self.winner = None
        self.winner_index = -1
        # Best paddle of all generations
        self.champion = None
        self.champion_index = -1
        self.champion_fitness = -1 * np.inf



        #### Pygame ####

        

        # The loop will carry on until the user exit the game (e.g. clicks the close button).
        game_active = True
        # The clock will be used to control how fast the screen updates
        clock = pygame.time.Clock()

        # load assets
        self.bg_surface = pygame.image.load("assets/background-day.png").convert()
        self.bg_surface = pygame.transform.scale2x(self.bg_surface)

        self.floor_surface = pygame.image.load("assets/base.png").convert()
        self.floor_surface = pygame.transform.scale2x(self.floor_surface)
        self.floor_x = 0

        self.pipe_surface = pygame.image.load("assets/pipe-green.png")
        self.pipe_surface = pygame.transform.scale2x(self.pipe_surface)
        self.pipe_flip_surface = pygame.transform.flip(self.pipe_surface, False, True)
        self.pipe_list = []
        self.pipe_list.extend(self.create_pipe())
        # self.SPAWNPIPE = pygame.USEREVENT
        # pygame.time.set_timer(self.SPAWNPIPE, 1200)               # don't use this because sometimes the game lags and reuslts in pipes spawning closer than intended
        self.spawn_pipe_counter = 0

        # scores
        self.score = 0
        self.high_score = 0
        





        while game_active:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    game_active = False
                    pygame.quit()
                    # sys.exit()        // this or exit()
                    exit()
                # if event.type == self.SPAWNPIPE and game_active:
                #     self.pipe_list.extend(self.create_pipe())

            # screen.fill(BLACK)
            self.screen.blit(self.bg_surface, (0, 0))


            # Pipes
            self.spawn_pipe_counter += 1
            
            # if self.spawn_pipe_counter % 72 == 0:
            if self.spawn_pipe_counter % settings['pipe_interval_in_frames'] == 0:
                self.pipe_list.extend(self.create_pipe())
            self.pipe_list = self.move_pipes(self.pipe_list)
            self.draw_pipes(self.pipe_list) 

            # Floor
            self.floor_x = (self.floor_x - 6) % - self.Window_Width
            self.draw_floor()

            # TODO: should change this for better visibility
            font = pygame.font.Font('freesansbold.ttf', 18)
            generation_text = font.render("Generation: %d" % self.current_generation, True, WHITE)
            score_text = font.render("Score: %d" % self.score, True, WHITE)
            best_score_text = font.render("Best: %d" % self.high_score, True, WHITE)
            self.screen.blit(generation_text, (self.Window_Width - 150, 30))
            self.screen.blit(score_text, (self.Window_Width - 150, 60))
            self.screen.blit(best_score_text, (self.Window_Width - 150, 90))


            self.still_alive = 0
            # Loop through the paddles in the generation
            for i, bird in enumerate(self.population.individuals):

                # Update paddel if still alive
                if bird.is_alive:
                    self.still_alive += 1
                    #----------------------------------------inputs for neural network--------------------------------------------
                    # next_pipe = next(pipe for pipe in self.pipe_list if pipe.right > settings['init_bird_x_pos'])
                    # next_next_pipe = next(pipe for pipe in self.pipe_list if pipe.right > next_pipe.right)
                    next_pipe, next_next_pipe = self.get_next_pipes()
                    bird.x_distance_to_next_pipe_center = next_pipe.right - settings['init_bird_x_pos']
                    bird.y_distance_to_next_pipe_center = (next_pipe.top - 150) - bird.y_pos
                    if next_next_pipe != None:
                        bird.x_distance_to_next_next_pipe_center = next_next_pipe.right - settings['init_bird_x_pos']
                        bird.y_distance_to_next_next_pipe_center = (next_next_pipe.top - 150) - bird.y_pos
                    else:
                        bird.x_distance_to_next_next_pipe_center = None
                        bird.y_distance_to_next_next_pipe_center = None
                    inputs = np.array([[bird.y_speed], [bird.y_pos], [bird.x_distance_to_next_pipe_center], [bird.y_distance_to_next_pipe_center], [bird.y_distance_to_next_next_pipe_center]])
                    # inputs = np.array([[paddle.x_pos], [ball_distance_left_wall], [ball_distance_right_wall], [balls[i].xspeed], [paddle.xspeed], [balls[i].y], [balls[i].yspeed]])
                    # inputs = np.array([[paddle.x_pos], [balls[i].xspeed], [paddle.xspeed], [balls[i].x]])
                    #----------------------------------------inputs for neural network--------------------------------------------
                    bird.update(inputs)
                    bird.move(self.pipe_list)
                    self.score = max(self.score, bird.score)
                    self.high_score = max(self.score, self.high_score)
                    
    
                # Draw every paddle except the best ones
                if bird.is_alive and bird != self.winner and bird != self.champion:
                    bird.winner = False
                    bird.champion = False
                    bird.draw(self.screen)



            # Draw the winning and champion paddle last
            if self.winner is not None and self.winner.is_alive:
                self.winner.draw(self.screen, winner=True)
            if self.champion is not None and self.champion.is_alive:
                self.champion.draw(self.screen, champion=True)
            
            # Generate new generation when all have died out
            if self.still_alive == 0:
                self.next_generation()

            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.update()
        
            # --- Limit to 60 frames per second
            clock.tick(60)

        #Once we have exited the main program loop we can stop the game engine:
        pygame.quit()




    def draw_floor(self):
        self.screen.blit(self.floor_surface, (self.floor_x, self.Window_Height - 100))
        self.screen.blit(self.floor_surface, (self.floor_x + self.Window_Width, self.Window_Height - 100))

    def create_pipe(self):
        random_pipe_pos = random.choice([400, 500, 600, 700, 800])
        bottom_pipe = self.pipe_surface.get_rect(midtop = (self.Window_Width + 100, random_pipe_pos))
        top_pipe = self.pipe_surface.get_rect(midbottom = (self.Window_Width + 100, random_pipe_pos - 300))
        return bottom_pipe, top_pipe

    def move_pipes(self, pipes):
        for pipe in pipes:
            pipe.centerx -= 5
        return pipes

    def draw_pipes(self, pipes):
        for pipe in pipes:
            if pipe.bottom > self.Window_Height:
                self.screen.blit(self.pipe_surface, pipe)
            else:
                self.screen.blit(self.pipe_flip_surface, pipe)

    def get_next_pipes(self):
        next_pipe = None
        next_next_pipe = None
        for pipe in self.pipe_list:
            if pipe.right > settings['init_bird_x_pos'] and next_pipe == None:
                next_pipe = pipe
            elif next_pipe != None and next_next_pipe == None:
                next_next_pipe = pipe
        return next_pipe, next_next_pipe





    #### GA Related ####

    def next_generation(self):
        self.current_generation += 1
        # reset for new game
        self.score = 0
        self.pipe_list.clear()
        self.pipe_list.extend(self.create_pipe())
        # reset timer so don't have the second pipe comming too soon
        # pygame.time.set_timer(self.SPAWNPIPE, 0)
        # pygame.time.set_timer(self.SPAWNPIPE, 1200)         ## TODO: test if this is neccessary
        self.spawn_pipe_counter = 0

        # Calculate fitness of individuals
        for individual in self.population.individuals:
            individual.calculate_fitness()

        # Find winner from each generation and champion
        self.winner = self.population.fittest_individual
        self.winner_index = self.population.individuals.index(self.winner)
        if self.winner.fitness > self.champion_fitness:
            self.champion_fitness = self.winner.fitness
            self.champion = self.winner
            self.champion_index = self.winner_index
        self.winner.reset()
        self.champion.reset()

        # Print results from each generation
        print('======================= Gneration {} ======================='.format(self.current_generation))
        print('----Max fitness:', self.population.fittest_individual.fitness)
        # print('----Best Score:', self.population.fittest_individual.score)
        print('----Average fitness:', self.population.average_fitness)
        
        self.population.individuals = elitism_selection(self.population, settings['num_parents'])
        
        random.shuffle(self.population.individuals)
        next_pop: List[Bird] = []

        # parents + offspring selection type ('plus')
        if settings['selection_type'].lower() == 'plus':
            next_pop.append(self.winner)
            next_pop.append(self.champion)

        while len(next_pop) < self._next_gen_size:
            p1, p2 = roulette_wheel_selection(self.population, 2)

            L = len(p1.network.layer_nodes)
            c1_params = {}
            c2_params = {}

            # Each W_l and b_l are treated as their own chromosome.
            # Because of this I need to perform crossover/mutation on each chromosome between parents
            for l in range(1, L):
                p1_W_l = p1.network.params['W' + str(l)]
                p2_W_l = p2.network.params['W' + str(l)]  
                p1_b_l = p1.network.params['b' + str(l)]
                p2_b_l = p2.network.params['b' + str(l)]

                # Crossover
                # @NOTE: I am choosing to perform the same type of crossover on the weights and the bias.
                c1_W_l, c2_W_l, c1_b_l, c2_b_l = self._crossover(p1_W_l, p2_W_l, p1_b_l, p2_b_l)

                # Mutation
                # @NOTE: I am choosing to perform the same type of mutation on the weights and the bias.
                self._mutation(c1_W_l, c2_W_l, c1_b_l, c2_b_l)

                # Assign children from crossover/mutation
                c1_params['W' + str(l)] = c1_W_l
                c2_params['W' + str(l)] = c2_W_l
                c1_params['b' + str(l)] = c1_b_l
                c2_params['b' + str(l)] = c2_b_l

                # Clip to [-1, 1]
                np.clip(c1_params['W' + str(l)], -1, 1, out=c1_params['W' + str(l)])
                np.clip(c2_params['W' + str(l)], -1, 1, out=c2_params['W' + str(l)])
                np.clip(c1_params['b' + str(l)], -1, 1, out=c1_params['b' + str(l)])
                np.clip(c2_params['b' + str(l)], -1, 1, out=c2_params['b' + str(l)])

            # Create children from chromosomes generated above
            c1 = Bird(chromosome=c1_params, 
                    hidden_layer_architecture=p1.hidden_layer_architecture,
                    hidden_activation=p1.hidden_activation, 
                    output_activation=p1.output_activation)
            c2 = Bird(chromosome=c2_params, 
                    hidden_layer_architecture=p2.hidden_layer_architecture,
                    hidden_activation=p2.hidden_activation,
                    output_activation=p2.output_activation)

            # Add children to the next generation
            next_pop.extend([c1, c2])
        
        # Set the next generation
        random.shuffle(next_pop)
        self.population.individuals = next_pop

    def _crossover(self, parent1_weights: np.ndarray, parent2_weights: np.ndarray,
                   parent1_bias: np.ndarray, parent2_bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rand_crossover = random.random()
        crossover_bucket = np.digitize(rand_crossover, self._crossover_bins)
        child1_weights, child2_weights = None, None
        child1_bias, child2_bias = None, None

        # SBX
        if crossover_bucket == 0:
            child1_weights, child2_weights = SBX(parent1_weights, parent2_weights, self._SBX_eta)
            child1_bias, child2_bias =  SBX(parent1_bias, parent2_bias, self._SBX_eta)

        # Single point binary crossover (SPBX)
        elif crossover_bucket == 1:
            child1_weights, child2_weights = single_point_binary_crossover(parent1_weights, parent2_weights, major=self._SPBX_type)
            child1_bias, child2_bias =  single_point_binary_crossover(parent1_bias, parent2_bias, major=self._SPBX_type)
        
        else:
            raise Exception('Unable to determine valid crossover based off probabilities')

        return child1_weights, child2_weights, child1_bias, child2_bias

    def _mutation(self, child1_weights: np.ndarray, child2_weights: np.ndarray,
                  child1_bias: np.ndarray, child2_bias: np.ndarray) -> None:
        scale = .2
        rand_mutation = random.random()
        mutation_bucket = np.digitize(rand_mutation, self._mutation_bins)

        mutation_rate = self._mutation_rate
        if settings['mutation_rate_type'].lower() == 'decaying':
            mutation_rate = mutation_rate / sqrt(self.current_generation + 1)

        # Gaussian
        if mutation_bucket == 0:
            # Mutate weights
            gaussian_mutation(child1_weights, mutation_rate, scale=scale)
            gaussian_mutation(child2_weights, mutation_rate, scale=scale)

            # Mutate bias
            gaussian_mutation(child1_bias, mutation_rate, scale=scale)
            gaussian_mutation(child2_bias, mutation_rate, scale=scale)

        # Uniform random
        elif mutation_bucket == 1:
            # Mutate weights
            random_uniform_mutation(child1_weights, mutation_rate, -1, 1)
            random_uniform_mutation(child2_weights, mutation_rate, -1, 1)

            # Mutate bias
            random_uniform_mutation(child1_bias, mutation_rate, -1, 1)
            random_uniform_mutation(child2_bias, mutation_rate, -1, 1)

        else:
            raise Exception('Unable to determine valid mutation based off probabilities.')


if __name__ == "__main__":
    main = Main()