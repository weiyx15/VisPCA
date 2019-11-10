import numpy as np
import heapq
import json
import matplotlib.pyplot as plt

class PCA:
    @staticmethod
    def _check_real_symmetric(A: np.array) -> bool:
        """check whether a matrix is real symmetric
        """
        return np.allclose(A, A.T, atol=1e-9)

    @staticmethod
    def normalize(A: np.array) -> np.array:
        """
        normalize n * d array s.t. each element is between [0, 1]
        :param A: n * d numpy array
        :return: n * d numpy array
        """
        for i in range(A.shape[1]):
            A[:, i] = (A[:, i] - np.min(A[:, i])) / (np.max(A[:, i] - np.min(A[:, i])))
        return A

    @staticmethod
    def pca(X: np.array, k: int) -> np.array:
        """
        do pca to decomposition X to k-dimension
        :param X: n * d shape numpy array, n: number of samples, d: feature dimension
        :param k: decomposed dimension
        :return: decomposed numpy array, n * k shape
        """
        n, d = X.shape
        X = X - np.mean(X, 0)       # mean value of each dimension
        C = np.dot(np.transpose(X), X)      # covariance matrix
        if not PCA._check_real_symmetric(C):
            raise ArithmeticError('Covariance matrix is not real symmetric')
        eig_val, eig_vec = np.linalg.eig(C)     # eigenvalue, eigenvector
        eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(d)]     # eigen-value-vector tuples
        topk_pairs = heapq.nlargest(k, eig_pairs)           # retrieve top-k eigenvalue pairs
        P = np.array([pair[1] for pair in topk_pairs])      # permutation matrix
        return np.dot(np.real(P), np.transpose(X)).T


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)
    images = np.load(config['image_file'])                                      # (1000, 28, 28)
    labels = np.load(config['label_file']).astype(np.int)                       # (1000, )
    n_samples = images.shape[0]                                                 # 1000
    X = images.reshape((n_samples, -1))                                         # (1000, 28 * 28)
    Y = PCA.pca(X, 2)                                                           # (1000, 2)
    Y = PCA.normalize(Y)
    labelY = [Y[i].tolist() + [int(labels[i]), i] for i in range(n_samples)]    # (1000, 4), (x, y, label, index)
    with open('data.json', 'w') as f:
        json.dump({'data': labelY}, f)
    # plt.figure()
    # axes = [plt.scatter(Y[0, np.where(labels == cat)[0]], Y[1, np.where(labels == cat)[0]]) for cat in range(10)]
    # plt.legend(axes, [str(cat) for cat in range(10)])
    # plt.title('PCA decomposition')
    # plt.show()
