import copy
import multiprocessing
from random import randint
from game import*
from neural_network import *
from joblib import Parallel, delayed

class GeneticAlgorithm:
    def __init__(self, networks=None, networks_shape=None, population_size=1000, generation_number = 100,
                 crossover_rate=0.3, crossover_method='neuron', mutation_rate=0.7, mutation_method='weight'):
        self.networks_shape = networks_shape
        if self.networks_shape is None:             
            self.networks_shape = [21,16,3]        
        self.networks = networks

        if networks is None:                                  
            self.networks = []
            for i in range(population_size):                  
                self.networks.append(NeuralNetwork(self.networks_shape))

        self.population_size = population_size
        self.generation_number = generation_number
        self.crossover_rate = crossover_rate
        self.crossover_method = crossover_method
        self.mutation_rate = mutation_rate
        self.mutation_method = mutation_method

    def start(self):
        networks = self.networks
        population_size = self.population_size
        crossover_number = int(self.crossover_rate*self.population_size)  
        mutation_number = int(self.mutation_rate*self.population_size)     

        num_cores = multiprocessing.cpu_count()        
        gen = 0                                         
        for i in range(self.generation_number):
            gen += 1

            parents = self.parent_selection(networks, crossover_number, population_size)       
            children = self.children_production(crossover_number, parents)                     
            mutations = self.mutation_production(networks, mutation_number, population_size)   

            networks = networks + children + mutations                     
            self.evaluation(networks, num_cores)                           
            networks.sort(key=lambda Network: Network.score, reverse=True)  
            networks[0].save(name="gen_"+str(gen))                         

            for i in range(int(0.2*len(networks))):             
                rand = randint(10, len(networks)-1)
                networks[rand] = self.mutation(networks[rand])

            networks = networks[:population_size]      
            self.print_generation(networks, gen)

    def parent_selection(self, networks, crossover_number, population_size):
        parents = []
        for i in range(crossover_number):
            parent = self.tournament(networks[randint(0, population_size - 1)],      
                                     networks[randint(0, population_size - 1)],
                                     networks[randint(0, population_size - 1)])
            parents.append(parent)                                                   
        return parents

    def children_production(self, crossover_number, parents):
        children = []
        for i in range(crossover_number):
            child = self.crossover(parents[randint(0, crossover_number - 1)],     
                                   parents[randint(0, crossover_number - 1)])
            children.append(child)                                                 
        return children

    def mutation_production(self, networks, mutation_number, population_size):
        mutations = []
        for i in range(mutation_number):
            mut = self.mutation(networks[randint(0, population_size - 1)])     
            mutations.append(mut)                                              
        return mutations

    def evaluation(self, networks, num_cores, ):
        game = Game()

        results1, results2, results3, results4 = [], [], [], []

        # for i in range(len(networks)):
        #     results1.append(game.start(display=True, neural_net=networks[i]))
        #     results2.append(game.start(display=True, neural_net=networks[i]))
        #     results3.append(game.start(display=True, neural_net=networks[i]))
        #     results4.append(game.start(display=True, neural_net=networks[i]))
        results1 = Parallel(n_jobs=num_cores)(delayed(game.start)(neural_net=networks[i]) for i in range(len(networks)))
        results2 = Parallel(n_jobs=num_cores)(delayed(game.start)(neural_net=networks[i]) for i in range(len(networks)))
        results3 = Parallel(n_jobs=num_cores)(delayed(game.start)(neural_net=networks[i]) for i in range(len(networks)))
        results4 = Parallel(n_jobs=num_cores)(delayed(game.start)(neural_net=networks[i]) for i in range(len(networks)))
        for i in range(len(results1)):
            networks[i].score = int(np.mean([results1[i], results2[i], results3[i], results4[i]]))

    def tournament(self, net1, net2, net3):
        game = Game()
        game.start(neural_net=net1)               
        score1 = game.game_score
        game.start(neural_net=net2)
        score2 = game.game_score
        game.start(neural_net=net3)
        score3 = game.game_score
        maxscore = max(score1, score2, score3)    
        if maxscore == score1:
            return net1
        elif maxscore == score2:
            return net2
        else:
            return net3

    def crossover(self, net1, net2):
        res1 = copy.deepcopy(net1)                 
        res2 = copy.deepcopy(net2)
        weights_or_biases = random.randint(0, 1)   
        if weights_or_biases == 0:                 
            if self.crossover_method == 'weight':
                layer = random.randint(0, len(res1.weights) - 1)                           
                neuron = random.randint(0, len(res1.weights[layer]) - 1)                   
                weight = random.randint(0, len(res1.weights[layer][neuron]) - 1)           
                temp = res1.weights[layer][neuron][weight]                                 
                res1.weights[layer][neuron][weight] = res2.weights[layer][neuron][weight]
                res2.weights[layer][neuron][weight] = temp
            elif self.crossover_method == 'neuron':
                layer = random.randint(0, len(res1.weights) - 1)                           
                neuron = random.randint(0, len(res1.weights[layer]) - 1)                    
                temp = copy.deepcopy(res1)                                                 
                res1.weights[layer][neuron] = res2.weights[layer][neuron]
                res2.weights[layer][neuron] = temp.weights[layer][neuron]
            elif self.crossover_method == 'layer':
                layer = random.randint(0, len(res1.weights) - 1)                            
                temp = copy.deepcopy(res1)                                                 
                res1.weights[layer] = res2.weights[layer]
                res2.weights[layer] = temp.weights[layer]
        else:                                                      
            layer = random.randint(0, len(res1.biases) - 1)        
            bias = random.randint(0, len(res1.biases[layer]) - 1)  
            temp = copy.deepcopy(res1)                             
            res1.biases[layer][bias] = res2.biases[layer][bias]
            res2.biases[layer][bias] = temp.biases[layer][bias]

        game = Game()
        game.start(neural_net=res1)     
        score1 = game.game_score
        game.start(neural_net=res2)     
        score2 = game.game_score
        if score1 > score2:            
            return res1
        else:
            return res2

    def mutation(self, net):
        res = copy.deepcopy(net)                   
        weights_or_biases = random.randint(0, 1)   
        if weights_or_biases == 0:                 
            if self.mutation_method == 'weight':
                layer = random.randint(0, len(res.weights) - 1)                  
                neuron = random.randint(0, len(res.weights[layer]) - 1)         
                weight = random.randint(0, len(res.weights[layer][neuron]) - 1) 
                res.weights[layer][neuron][weight] = np.random.randn()          
            elif self.mutation_method == 'neuron':
                layer = random.randint(0, len(res.weights) - 1)                 
                neuron = random.randint(0, len(res.weights[layer]) - 1)
                new_neuron = np.random.randn(len(res.weights[layer][neuron]))
                res.weights[layer][neuron] = new_neuron
        else:                                                      
            layer = random.randint(0, len(res.biases) - 1)         
            bias = random.randint(0, len(res.biases[layer]) - 1)   
            res.weights[layer][bias] = np.random.randn()          
        return res

    def print_generation(self, networks, gen):
        top_mean = int(np.mean([networks[i].score for i in range(6)]))
        bottom_mean = int(np.mean([networks[-i].score for i in range(1, 6)]))
        print("\nBest Fitness gen", gen, " : ", networks[0].score)
        print("Pop size = ", len(networks))
        print("Average top 6 = ", top_mean)
        print("Average last 6 = ", bottom_mean)
