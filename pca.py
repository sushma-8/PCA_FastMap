"""
PCA Implementation

Dimensionality reduction from K-D space to (K-1)-D space


Contributors:

Niroop Ramdas Sagar  
USC ID: 4897621292    
ramdassa@usc.edu  

Sushma Mahadevaswamy
USC ID: 3939734806
mahadeva@usc.edu

"""


import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

style.use('ggplot')


class algorithms:
    def __init__(self, data):
        self.data = data
        # Number of training data
        self.points_count = data.shape[0]
        # Number of features in the data
        self.axis = data.shape[1]

    def pca_algorithm(self):
        self.data = np.asmatrix(self.data)
        mean = np.mean(self.data, axis=0)
        self.data -= mean
        covar_mat = np.cov(self.data, rowvar=False)

        eigen_val, eigen_vec = np.linalg.eig(covar_mat)
        eigenval_index = np.argsort(eigen_val)[::-1]
        new_eigen_vec = np.mat(eigen_vec[:, eigenval_index[0:2]])
        new_data = np.matmul(self.data, new_eigen_vec)
        print(new_eigen_vec)
        plt.scatter([new_data[:, 0]], [new_data[:, 1]],s=2)
        plt.show()


if __name__ == "__main__":
    data = np.loadtxt(open('pca.txt', 'r'), delimiter='\t', dtype='float')
    obj = algorithms(data)
    obj.pca_algorithm()
