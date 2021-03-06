import numpy as np


class evolution:
    def __init__(self, mut_prop, mut_per_bit, cross_prop, elite, tournament_size):
        self.mut_prob = mut_prop
        self.mut_per_bit = mut_per_bit
        self.cross_prob = cross_prop
        self.tournament_size = tournament_size
        self.elite_size = elite
        self.elite = None
        self.best = None

    def __call__(self, last_gen, fitness, beginning=0):
        new_gen = last_gen.copy()
        o_indices = np.array(fitness).argsort()[-self.elite_size:]
        new_gen[range(self.elite_size), :] = last_gen[o_indices, :]
        self.best = new_gen[self.elite_size - 1]
        self.elite = new_gen[:self.elite_size]

        for i in range(self.elite_size, len(last_gen), 2):
            random_tournament = np.random.randint(0, len(last_gen), 5)

            tournament_fitness = fitness[random_tournament]
            tournament_indices = np.array(tournament_fitness).argsort()[-2:]

            o_indices = random_tournament[tournament_indices]
            o_1, o_2 = last_gen[o_indices, :]

            # Crossover
            o_1, o_2 = self._crossover(o_1, o_2, beginning)

            # Mutation
            o_1 = self._mutate(o_1, beginning)
            o_2 = self._mutate(o_2, beginning)

            new_gen[i, :] = o_1
            new_gen[i + 1, :] = o_2

        return new_gen

    def _crossover(self, first, second, beginning):
        if np.random.rand() < self.cross_prob:
            index = np.random.randint(beginning, len(first))
            first[index:], second[index:] = second[index:], first[index:]

        return first, second

    def _mutate(self, element, beginning):
        if np.random.rand() < self.mut_prob:
            for i in range(beginning, len(element)):
                if np.random.rand() < self.mut_per_bit:
                    element[i] = (np.random.rand() * 2) - 1
        return element
