"""
FastMap Implementation

Embed objects into K-D space


Contributors:

Niroop Ramdas Sagar  
USC ID: 4897621292    
ramdassa@usc.edu  

Sushma Mahadevaswamy
USC ID: 3939734806
mahadeva@usc.edu

"""


import random
import matplotlib.pyplot as plt
import numpy as np
import math


K = 2
col = 0

class fastmap():
    def __init__(self):
        self.dist_matrix = np.zeros((11,11))
        self.K = K
        self.X = np.zeros((11, self.K + 1))

    def read_input(self):
        with open("fastmap-data.txt",'r') as in_file:
            for line in in_file:
                t_dist = list(map(int,line.split()))
                self.dist_matrix[t_dist[0]][t_dist[1]] = t_dist[2]
                self.dist_matrix[t_dist[1]][t_dist[0]] = t_dist[2]

    def get_farthest_objects(self, col_val):

        max_iters      = 10
        final_objA     = 0
        final_objB     = 0
        final_val      = -math.inf
        objB           = random.randint(1,10)
        objA           = -1
        self.obj_count = self.dist_matrix.shape[0]

        for _ in range(max_iters):

            cur_val = -math.inf

            for i in range(1, self.obj_count):
                cur_dist = self.get_object_distance(objB, i, col_val)
                if cur_dist > cur_val:
                    cur_val = cur_dist
                    objA = i

            if cur_val > final_val:
                final_val  = cur_val
                final_objA = objA
                final_objB = objB
            elif cur_val == final_val :
                cur_min    = min(objA, objB)
                final_min  = min(final_objA, final_objB)

                if cur_min < final_min:
                    final_objA = objA
                    final_objB    = objB
            objB = objA

        if final_objA < final_objB :
            return (final_objA, final_objB)
        else:
            return (final_objB, final_objA)        
                    

    def get_object_distance(self, a, b, col):

        if col == 1:
            return self.dist_matrix[a][b]
        else:
            return math.pow((math.pow(self.get_object_distance(a, b, col - 1), 2) - math.pow((self.X[a][col-1]- self.X[b][col-1]), 2)), 0.5)

    def run(self, k):

        global col

        if k < 1:
            return
        else:
            col += 1
        a, b = self.get_farthest_objects(col)

        print(f"Farthest Objects in Iteration :  {col}")
        print(f"Object A value                :  {a}")
        print(f"Object B value                :  {b}")

        if self.get_object_distance(a, b, col) == 0:
            self.X[:, col] = 0

        for i in range(1, 11):
            if i == a:
                self.X[i, col] = 0
            elif i == b:
                self.X[i, col] = self.get_object_distance(a,b,col)
            else:

                numerator      = ((math.pow(self.get_object_distance(a, i, col), 2) + math.pow(self.get_object_distance(a, b, col), 2) - math.pow(self.get_object_distance(b, i, col), 2)))
                denominator    = 2 * self.get_object_distance(a,b,col)
                self.X[i, col] = numerator / denominator

        self.run(k-1)

    def plot(self):

        self.wordsToMap = []
        with open('fastmap-wordlist.txt') as in_file:
            for line in in_file:
                self.wordsToMap.extend(line.split())

        self.X = [self.X[i][1:] for i in range(1, 11)]

        _, ax = plt.subplots()
        for i in range(10):
            ax.scatter(self.X[i][0], self.X[i][1])
            ax.annotate(self.wordsToMap[i], (self.X[i][0], self.X[i][1]))

        print("\n\n===================Embeddings==================")
        for word, val in zip(self.wordsToMap, self.X):
            print(f'{word:20s} -> {val}')
        print("===================Embeddings==================")

        plt.show()

if __name__ == '__main__':
    fastmap_obj = fastmap()
    fastmap_obj.read_input()
    fastmap_obj.run(K)
    fastmap_obj.plot()
