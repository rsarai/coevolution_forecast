from evolution_strategies import ES
from parameters import num_of_individuos
from rbf import RBF


class Coevolution():
    population = []
#    rbfn_population = []
#    lags_population = []

#    def initialize_rbfn_population(self):
#        for i in range(num_of_individuos):
#            rbfn = RBF()
#            self.rbfn_population.append(rbfn)
#
#    def initialize_lags_population(self):
#        for i in range(num_of_individuos):
#            lags = ES(Sphere()).search()
#            self.lags_population.append(rbfn)

    def initialize_population(self):
        for i in range(num_of_individuos):
            solution = ES(RBF())
            self.population(solution)

    def train(self):
        model = ES().search()
        # Train rbf weights with backpropagation


Coevolution().train()
