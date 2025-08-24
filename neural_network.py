from numba import jit
import numpy as np
import pygame
from constants import *
from pygame import gfxdraw


class NeuralNetwork:
    def __init__(self, shape=None):
        self.shape = shape
        self.biases = []
        self.weights = []
        self.score = 0       
        if shape:
            for y in shape[1:]:                            
                self.biases.append(np.random.randn(y, 1))
            for x, y in zip(shape[:-1], shape[1:]):        
                self.weights.append(np.random.randn(y, x))

    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def save(self, name=None):
        if not name:
            np.save('saved_weights_'+str(self.score), self.weights)
            np.save('saved_biases_'+str(self.score), self.biases)
        else:
            np.save(name + '_weights', self.weights)
            np.save(name + '_biases', self.biases)

    def load(self, filename_weights, filename_biases):
        # self.weights = np.load(filename_weights)
        # self.biases = np.load(filename_biases)
        self.weights = np.load(filename_weights, allow_pickle=True)
        self.biases = np.load(filename_biases, allow_pickle=True)


    def render(self, window, vision):
        network = [np.array(vision)]           
        for i in range(len(self.biases)):
            activation = sigmoid(np.dot(self.weights[i], network[i]) + self.biases[i]) 
            network.append(activation)                                                  

        screen_division = WINDOW_SIZE / (len(network) * 2)     
        step = 1
        for i in range(len(network)):                                           
            for j in range(len(network[i])):                                   
                y = int(WINDOW_SIZE/2 + (j*24) - (len(network[i])-1)/2 * 24)    
                x = int(WINDOW_SIZE + screen_division * step)
                intensity = int(network[i][j][0] * 255)                         

                if i < len(network)-1:
                    for k in range(len(network[i+1])):                                         
                        y2 = int(WINDOW_SIZE/2 + (k * 24) - (len(network[i+1]) - 1) / 2 * 24)  
                        x2 = int(WINDOW_SIZE + screen_division * (step+2))
                        pygame.gfxdraw.line(window, x, y, x2, y2,                              
                                            (intensity/2+30, intensity/2+30, intensity/2+30, intensity/2+30))

                pygame.gfxdraw.filled_circle(window, x, y, 9, (intensity, intensity, intensity))   
                pygame.gfxdraw.aacircle(window, x, y, 9, (205, 205, 205))
            step += 2

@jit(nopython=True)
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
