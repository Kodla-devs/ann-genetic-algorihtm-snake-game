from game import*
from genetic_algorithm import *

net = NeuralNetwork()
game = Game()

# test
net.load(filename_weights='gen_100_weights.npy', filename_biases='gen_100_biases.npy')
game.start(display=True, neural_net=net)

# play 
# game = Game()
# game.start(playable=True, display=True, speed=10)

# train
# gen = GeneticAlgorithm(population_size=1000, crossover_method='neuron', mutation_method='weight')
# gen.start()
