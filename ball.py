from settings import settings
import pygame

BLACK = (0,0,0)
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,180,0)
BLUE = (50,200,255)

class Ball:
    def __init__(self, x = 50, y = 50, xspeed = 15, yspeed = 15):
	    self.x = x
	    self.y = y
	    self.xlast = x-xspeed
	    self.ylast = y-yspeed
	    self.xspeed = xspeed
	    self.yspeed = yspeed
	    self.alive = True
	    self.distance_travelled = 0
	
    #Update position based on speed 
    def update(self, paddle):

        self.distance_travelled += abs(self.xspeed)
        
        #Accounts for bouncing off walls and paddle
        if self.x<15:
            self.x=15
            self.xspeed *= -1
        elif self.x>settings['board_size'][0]-15:
            self.x=settings['board_size'][0]-15
            self.xspeed *= -1
        elif self.y<35:
            self.y=35
            self.yspeed *= -1
        elif self.x>paddle.x_pos and self.x < paddle.x_pos + 100 and self.ylast < settings['board_size'][1]-35 and self.y >= settings['board_size'][1]-35:
            self.yspeed *= -1
            paddle.hit += 1
            paddle.distance_to_ball = 0
        elif self.y > settings['board_size'][1]:
            self.yspeed *= -1
            paddle.ball_travelled = self.distance_travelled
            paddle.is_alive = False
            paddle.distance_to_ball = abs(self.x - paddle.x_pos)
            
    def update_pos(self):
        self.xlast = self.x
        self.ylast = self.y
        
        self.x += self.xspeed
        self.y += self.yspeed
			
	#Draw ball to screen	   
    def draw(self, screen):
	    pygame.draw.circle(screen, WHITE,[self.x,self.y], 15)