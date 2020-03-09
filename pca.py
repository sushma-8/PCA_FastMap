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
from sklearn.decomposition import PCA
import os
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

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
        # print(mean.shape)
        covar_mat = np.cov(self.data, rowvar=False)

        eigen_val, eigen_vec = np.linalg.eig(covar_mat)
        eigenval_index = np.argsort(eigen_val)[::-1]
        new_eigen_vec = np.mat(eigen_vec[:, eigenval_index[0:2]])
        new_data = np.matmul(self.data, new_eigen_vec)
        print(new_data.shape)

        x = new_data[:,0]
        y = new_data[:,1]
        #points = new_data[:,2:4]
        # color is the length of each vector in `points`
        #color = np.sqrt((points**2).sum(axis = 1))/np.sqrt(2.0)
        #rgb = plt.get_cmap('jet')(color)
        print(x.shape)
        #x, y = new_data[:,0]
        plt.scatter([new_data[:,0]],[new_data[:,1]])
        plt.show()


if __name__ == "__main__":
    data = np.loadtxt(open('pca.txt', 'r'), delimiter='\t', dtype='float')

    """
    #Plot 3D space
    t_data = data.transpose()
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams['legend.fontsize'] = 10
    ax.plot(t_data[0,:], t_data[1,:], t_data[2,:], 'o', markersize=1, color='blue', alpha=0.5, label='data')
    plt.show()
    """

    # print(data)

    # Library implementation
    pca = PCA(n_components=2)
    pca.fit(data)
    print(pca.components_)

    obj = algorithms(data)
    obj.pca_algorithm()
