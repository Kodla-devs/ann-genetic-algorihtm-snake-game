from map import *
from pygame.locals import *
from snake import *


class Game:
    def __init__(self):
        self.game_score = 0
        self.game_time = 0

    def start(self, display=False, neural_net=None, playable=False, speed=20):
        if not display:
            return self.run_invisible(neural_net=neural_net)
        else:
            return self.run_visible(neural_net=neural_net, playable=playable, speed=speed)

    def run_invisible(self, neural_net=None):
        snake = Snake(neural_net=neural_net)    
        map = Map(snake)                          

        cont = True                                
        while cont:
            self.game_time += 1
            map.scan()                              
            snake.AI()                              
            snake.update()                         
            map.update()                           
            if not snake.alive:
                cont = False
                self.game_time = 0
        self.game_score = snake.fitness()           
        return self.game_score

    def run_visible(self, playable=False, neural_net=None, speed=20):
        pygame.init()                                                               
        game_window = pygame.display.set_mode((int(WINDOW_SIZE*2), WINDOW_SIZE))    
        pygame.display.set_caption(WINDOW_TITLE)

        snake = Snake(neural_net=neural_net)
        map = Map(snake)

        cont = [True]
        while cont[0]:                             
            pygame.time.Clock().tick(speed)            
            self.inputs_management(snake, cont)     
            if not playable:
                map.scan()                         
                snake.AI()                           
            self.render(game_window, map)           
            snake.update()
            map.update()
            if not snake.alive:
                cont[0] = False

        self.game_score = snake.fitness()           
        return self.game_score

    def inputs_management(self, snake, cont):
        for event in pygame.event.get():
            if event.type == QUIT:
                cont[0] = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:     
                    cont[0] = False
                if event.key == K_RIGHT:      
                    snake.turn_right()
                elif event.key == K_LEFT:      
                    snake.turn_left()
                elif event.key == K_UP:      
                    pass

    def render(self, window, map):
        map.render(window)
        pygame.display.flip()
