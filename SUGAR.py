import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
import numpy.matlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
SUGAR (Synthesis Using Geometrically Aligned Random-walks) Algorithm for Manifold Sampling
Author: Hossein Rahmani    based on:  https://arxiv.org/abs/1802.04927
'''

'tested by bunny dataset.'


def guassian_kernel(data1, data2, sigma=1, distance_type='euclidean'):
    """
    based on algorithm in section 3.1
    :param data1: Y matrix or if data1==data2(our data matrix in the begining)
    :param data2: our N*D data wich is N vectors in R^D space.
    :param sigma: σ is a bandwidth parameter that controls neighborhood sizes. section 3.3
    :param distance_type: suitable distance_type in scipy function cdist
    :return:
    """
    pair_dist = distance.cdist(data1, data2, distance_type) ** 2
    Kernel = np.exp(-pair_dist / (2 * (sigma ** 2)))
    return Kernel


'tested by bunny dataset.'


def get_kernel_properties(Kernel):
    """
    :param Kernel: Kernel
    :return: degree, sparsity measure and Markov matrix P(or diffusion operator)
    """
    d_hat = np.sum(Kernel, axis=1)
    s_hat = 1 / d_hat
    P = np.diag(1 / d_hat) @ Kernel  # markov matrix
    return d_hat, s_hat, P


'tested by bunny dataset.'


def local_covariance(data, k=5):
    """
    :param data: our sample points (N points in |R^D space.)
    :param k: constant number for k_nearest neighbor algorithms.
    :return: with D*D*N each axis=2 (manzoram ine ke ina ro royeh ham chidim) is D*D local covariance matrix computed with k-n-n.
    """
    N = data.shape[0]
    D = data.shape[1]
    local_cov = np.zeros((D, D, N))
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)
    for i in range(0, N):
        local_cov[:, :, i] = np.cov(data[indices[i]], rowvar=False)

    return local_cov


def numpts(data, d_hat, bandwith, local_cov, equalize=True):
    """
    we suppose bandwith is scalar (we can have a vector also.)
    :param local_cov: local_cov is 3d array with D*D*N each row is D*D local covariance matrix computed with k-n-n.
    :param bandwith: same as guassian sigma
    :param data: our sample points(N points) from R D
    :param d_hat: degree of kernel
    :return:  amount of points generated around each x_i
    """
    maxd = np.max(d_hat)
    N = data.shape[0]
    l = np.zeros(N, dtype=np.int64)
    if equalize:
        for i in range(0, N):
            upper_bound = np.sqrt(
                np.linalg.det(np.eye(local_cov[:, :, i].shape[0], local_cov[:, :, i].shape[1]) + local_cov[:, :, i] / (
                        2 * bandwith ** 2))) * (
                                  maxd - d_hat[i])
            l[i] = np.maximum(int(upper_bound), 1)

    if np.sum(l) == 0:
        print('Point generation estimate < 0 , either provide/increase M or decrease noise_cov')
        l = np.ones(N)
    if np.sum(l) > 10 ^ 6:
        print('Point generation > 1e6, either provide/decrease M or increase noise_cov')
    print('M: ', np.sum(l), "N: ", N)
    return l


def generate_points(data, l, local_cov):
    N = data.shape[0]
    D = data.shape[1]
    M = np.sum(l)
    Y = np.zeros((M, D), dtype=data.dtype)
    index = 0
    for i in range(0, N):
        mio = data[i]
        sig = local_cov[:, :, i]
        for num_sample in range(l[i]):
            Y[index] = np.random.multivariate_normal(mio, sig)
            index = index + 1
    print('M and then index should be same', M, index)
    return Y


#
# def generate_points(data, l, sigma):
#     N = data.shape[0]
#     D= data.shape[1]
#     mio = np.matlib.repmat(data[0], l[0], 1)
#     sigma_0 = sigma[:, :, 0]
#     sig = np.repeat(sigma_0[:, :, np.newaxis], 3, axis=2)
#     for i in range(1, N):
#         mio = np.vstack((mio, np.matlib.repmat(data[i], l[i], 1)))
#         sigma_i = sigma[:, :, i]
#         sig = np.dstack((sig, np.repeat(sigma_i[:, :, np.newaxis], l[i], axis=2)))
#
#     return np.random.multivariate_normal(mio, sig) #in error mide. be khater data fekr konam.


def MGC_Kernel(X, Y, s_hat, sigma=1, distance_type='euclidean'):
    """
    new kernel based on
    kernel_XY = guassian_kernel(X, Y, 1, 'euclidean') ==Kernel_YX.T  interesting fact
    :param X: our data
    :param Y: generated data
    :param s_hat: sparcity mesure
    :param sigma: bandwith
    :param distance_type: default= euclidean
    :return: new Kernel_hat
    """
    kernel_YX = guassian_kernel(Y, X, sigma, distance_type)
    sparcity_times_kernel = kernel_YX * s_hat
    Kernel_hat = sparcity_times_kernel @ kernel_YX.T
    return Kernel_hat


def magic(Y, P_hat, t=1, rescale=True):
    """
    applying P^ operator in step 6 in algorithm sugar MAGIC (Markov Affinity-based Graph Imputation of Cells)
    :param Y: synthesised sample
    :param P: Markov matrix
    :param t: time instant default t=1
    :param rescale: True. it means need rescale. based on algorithm step 7.Rescale 95th percentile of imputed data to match original data.
    :return:
    """
    diffused_points = Y
    for i in range(1, t + 1):
        diffused_points = P_hat @ diffused_points

    if rescale:
        scale_vec = np.percentile(Y, 95, axis=0) / np.percentile(diffused_points, 95, axis=0)
        scale_vec = np.nan_to_num(scale_vec)
        print(scale_vec)
        diffused_points = diffused_points * scale_vec

    return diffused_points


def choose_bandwidth(data, C):
    """
    σ MaxMin based on  Kernel Bandwidth Selection algorithm(section 3.3)
    :param C:  C ∈ [2, 3] randomly or we can choose it.
    :param data: our data
    :return: choose a sigma
    """
    pair_distance = distance.cdist(data, data)
    # pair_distance[pair_distance == 0] = np.inf
    Min_dist = np.min(pair_distance + np.eye(pair_distance.shape[0], pair_distance.shape[1]) * (10 ^ 15), axis=1)
    # which one?
    return C * np.max(Min_dist)


"  sphere  (x,y,z)=(p cos(theta)*sin(phi), p sin(phi)*sin(theta), p cos(theta))"


def xyz_sphere(pho, u, v):
    """
    convert (pho,theta,phi) representation to (x,y,z)
    :param pho:
    :param theta:
    :param phi:
    :return: (x,y,z)
    """
    x = pho * np.outer(np.cos(u), np.sin(v))
    y = pho * np.outer(np.sin(u), np.sin(v))
    z = pho * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

def prepare_circle(number_of_points=100):
    data = np.zeros((number_of_points, 2), dtype=np.float64)
    # theta = np.linspace(-np.pi,np.pi,number_of_points)
    theta = np.random.normal(0,np.pi /2,number_of_points)
    data[:,0]=np.cos(theta)
    data[:,1]=np.sin(theta)
    # X1 = np.arange(0, 1, 0.1)
    # X2 = np.arange(1, 2, 0.1)
    # for i in range(0, 20):
    #     if i < 10:
    #         data[i, 0] = X1[i]
    #         data[i, 1] = -np.sqrt(1 - (1 - X1[i]) ** 2) + 1
    #     else:
    #         data[i, 0] = X2[i - 10]
    #         data[i, 1] = np.sqrt(1 - (1 - X2[i - 10]) ** 2) + 1
    plt.scatter(data[:, 0], data[:, 1])
    return data
# preparing data


def sugar(data,t=1):
    bandwidth = choose_bandwidth(data, 2.5)
    data_Kernel = guassian_kernel(data, data, sigma=bandwidth)
    d_hat, s_hat, P = get_kernel_properties(data_Kernel)
    local_cov = local_covariance(data)
    l = numpts(data, d_hat, bandwidth, local_cov)
    Y = generate_points(data, l, local_cov)
    Kernel_hat = MGC_Kernel(data, Y, s_hat, sigma=bandwidth)
    d_hat_y, s_hat_y, P_hat_y = get_kernel_properties(Kernel_hat)
    diffused_points = magic(Y, P_hat_y, t=t)
    return diffused_points
data = np.random.normal(0,np.pi /2,100).reshape((100,1))
diffused_points = sugar(data,t=1)
plt.scatter(np.cos(diffused_points), np.sin(diffused_points),color='g')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('auto')
# ax.scatter(bunny[:, 0], bunny[:, 1], bunny[:, 2])
# plt.show()

u = diffused_points[:, 0]
v = diffused_points[:, 1]
phi = 1
x = phi * np.outer(np.cos(u), np.sin(v))
y = phi * np.outer(np.sin(u), np.sin(v))
z = phi * np.outer(np.ones(np.size(u)), np.cos(v))
ax.scatter(x, y, z)
plt.show()
# ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.5)



# 1.bunny data
with open('bunny.txt', 'r') as f:
    # reader = csv.reader(f)
    bunny = np.loadtxt(f, delimiter=",")

u1 = np.linspace(0, 2 * np.pi, 20)
v1 = np.linspace(0, np.pi, 20)
data = np.zeros((u1.size, 2), dtype=u1.dtype)
data[:, 0] = u1
data[:, 1] = v1
plt.scatter(data[:, 0], data[:, 1])

mu, sigma = 1, 1  # mean and standard deviation
x = np.linspace(-1, 3, 200)
y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
data = np.zeros((x.size, 2), dtype=x.dtype)
data[:, 0] = x
data[:, 1] = y
plt.scatter(data[:, 0], data[:, 1],color='g')
