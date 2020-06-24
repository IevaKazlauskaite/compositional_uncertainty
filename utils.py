import numpy as np
import tensorflow as tf


dtype = tf.float64


def vec_to_tri(tri, N):
    """ map from vector to lower triangular matrix (adapted from gpflow) """
    indices = list(zip(*np.tril_indices(N)))
    indices = tf.constant([list(i) for i in indices], dtype=tf.int32)
    tri_part = tf.scatter_nd(indices=indices, shape=[N, N], updates=tri)
    return tri_part


def init_triangular(N, diag=None):
    """ Initialize lower triangular parametrization for  covariance matrices (adapted from gpflow) """
    I = int(N*(N+1)/2)
    indices = list(zip(*np.tril_indices(N)))
    diag_indices = np.array([idx for idx, (x, y) in enumerate(indices) if x == y])
    I = np.zeros(I)
    if diag is None:
        I[diag_indices] = 1
    else:
        I[diag_indices] = diag
    return I


def log_det_from_chol(L):
    return 2.0 * tf.reduce_sum(tf.log(tf.diag_part(L)))


def real_variable(initial_value, prior_function=None, name=None, collections=None):
    t_a = tf.Variable(initial_value, dtype=dtype, name=name, collections=collections)
    return t_a


def positive_variable(initial_value, prior_function=None, name=None, collections=None):
    t_a = tf.Variable(tf.log(tf.cast(initial_value, dtype=dtype)), dtype=dtype, name=name, collections=collections)
    return tf.exp(t_a)


class Kernel:
    def __init__(self, covar_matrix_func, covar_diag_func, descriptor, kernels=None):
        self._descriptor = descriptor
        self._covar_matrix_func = covar_matrix_func
        self._covar_diag_func = covar_diag_func
        self._kernels = kernels

    @property
    def descriptor(self):
        return self._descriptor

    def covar_matrix(self, t_X, t_Z):
        return self._covar_matrix_func(t_X=t_X, t_Z=t_Z)

    def covar_diag(self, t_X):
        return self._covar_diag_func(t_X=t_X)


def create_squared_exp_kernel(t_alpha, t_gamma):
    def matrix_func(t_X, t_Z):
        xx = 0.5 * t_gamma * tf.reduce_sum(t_X * t_X, axis=1, keepdims=True)
        zz = 0.5 * t_gamma * tf.reduce_sum(t_Z * t_Z, axis=1, keepdims=True)
        sq_dist_xz = xx + tf.transpose(zz) - t_gamma * tf.matmul(t_X, t_Z, transpose_b=True)
        return t_alpha * tf.exp(- sq_dist_xz)

    def diag_func(t_X):
        t_N = tf.shape(t_X)[0]
        return t_alpha * tf.ones([t_N], dtype=dtype)

    return Kernel(descriptor='Squared Exponential',
                  covar_matrix_func=matrix_func,
                  covar_diag_func=diag_func,
                  kernels=None)


def create_matern32_kernel(t_alpha, t_gamma):
    def matrix_func(t_X, t_Z):
        xx = 0.5 * t_gamma * tf.reduce_sum(t_X * t_X, axis=1, keepdims=True)
        zz = 0.5 * t_gamma * tf.reduce_sum(t_Z * t_Z, axis=1, keepdims=True)
        sq_dist_xz = xx + tf.transpose(zz) - t_gamma * tf.matmul(t_X, t_Z, transpose_b=True)
        sqrt_3 = tf.constant(np.sqrt(3.0), dtype=dtype)
        sqrt_3_dist_xz = sqrt_3 * tf.sqrt(sq_dist_xz + 1.0e-12)
        return t_alpha * (1.0 + sqrt_3_dist_xz) * tf.exp(- sqrt_3_dist_xz)

    def diag_func(t_X):
        t_N = tf.shape(t_X)[0]
        return t_alpha * tf.ones([t_N], dtype=dtype)

    return Kernel(descriptor='Matern 3/2',
                  covar_matrix_func=matrix_func,
                  covar_diag_func=diag_func,
                  kernels=None)


def create_matern32_kernel_ard(t_alpha, t_gammas):
    def matrix_func(t_X, t_Z):
        xx = 0.5 * tf.reduce_sum((t_gammas * t_X) * t_X, axis=1, keepdims=True)
        zz = 0.5 * tf.reduce_sum((t_gammas * t_Z) * t_Z, axis=1, keepdims=True)
        sq_dist_xz = xx + tf.transpose(zz) - tf.matmul(t_gammas * t_X, t_Z, transpose_b=True)
        sqrt_3 = tf.constant(np.sqrt(3.0), dtype=dtype)
        sqrt_3_dist_xz = sqrt_3 * tf.sqrt(sq_dist_xz + 1.0e-12)
        return t_alpha * (1.0 + sqrt_3_dist_xz) * tf.exp(- sqrt_3_dist_xz)

    def diag_func(t_X):
        t_N = tf.shape(t_X)[0]
        return t_alpha * tf.ones([t_N], dtype=dtype)

    return Kernel(descriptor='Matern 3/2',
                  covar_matrix_func=matrix_func,
                  covar_diag_func=diag_func,
                  kernels=None)


def create_periodic_kernel(t_alpha, t_gamma, t_period): # inverse period
    def matrix_func(t_X, t_Z):
        xx = tf.reduce_sum(t_X * t_X, axis=1, keepdims=True)
        zz = tf.reduce_sum(t_Z * t_Z, axis=1, keepdims=True)
        dist_xz = xx + tf.transpose(zz) - 2.0 * tf.matmul(t_X, t_Z, transpose_b=True)
        return t_alpha * tf.exp(- 2.0 * t_gamma * (tf.sin(t_period * tf.sqrt(dist_xz + 1.0e-12)) ** 2))

    def diag_func(t_X):
        t_N = tf.shape(t_X)[0]
        return t_alpha * tf.ones([t_N], dtype=dtype)

    return Kernel(descriptor='Periodic',
                  covar_matrix_func=matrix_func,
                  covar_diag_func=diag_func,
                  kernels=None)

