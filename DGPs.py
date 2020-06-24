import numpy as np
from collections import defaultdict

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from utils import create_periodic_kernel, \
                  create_squared_exp_kernel, \
                  real_variable, \
                  positive_variable, \
                  vec_to_tri, \
                  init_triangular


dtype = tf.float64
jitter = 1e-6


def posterior_gp_integrated_inducing(t_inputs, mean_fn, t_Z, t_U_mean, t_U_Sigma, t_kernel, S, M):
    t_K_zz = t_kernel.covar_matrix(t_Z, t_Z)
    t_L_zz = tf.cholesky(t_K_zz + jitter * tf.eye(M, dtype=dtype))
    
    t_K_xz = tf.stack([t_kernel.covar_matrix(t_inputs[s], t_Z) for s in range(S)])
    t_K_xx = tf.stack([t_kernel.covar_matrix(t_inputs[s], t_inputs[s]) for s in range(S)])
    
    # Posterior mean
    t_L_inv_y = tf.stack([tf.linalg.triangular_solve(t_L_zz, t_U_mean - mean_fn(t_Z))
                          for s in range(S)])
    t_L_inv_K_zx = tf.stack([tf.linalg.triangular_solve(t_L_zz, tf.transpose(t_K_xz[s])) for s in range(S)])
    t_mean = mean_fn(t_inputs) + tf.einsum('smi,smd->sid', t_L_inv_K_zx, t_L_inv_y)
    
    # Posterior covariance
    t_L_S = tf.cholesky(t_U_Sigma)
    t_L_zz_inv_L_S = tf.linalg.triangular_solve(t_L_zz, t_L_S)
    t_cov = t_K_xx - \
            tf.einsum('smi,smj->sij', t_L_inv_K_zx, t_L_inv_K_zx) + \
            tf.einsum('spi,pk,qk,sqj->sij', t_L_inv_K_zx, t_L_zz_inv_L_S, t_L_zz_inv_L_S, t_L_inv_K_zx)
    t_L_cov = tf.cholesky(t_cov + jitter * tf.eye(tf.shape(t_inputs)[1], dtype=dtype))
    return t_mean, t_L_cov


def posterior_gp_sampled_inducing(t_inputs, mean_fn, t_Z, t_U, t_kernel, S, M):
    t_K_zz = t_kernel.covar_matrix(t_Z, t_Z)
    t_L_zz = tf.cholesky(t_K_zz + jitter * tf.eye(M, dtype=dtype))
    
    t_K_xz = tf.stack([t_kernel.covar_matrix(t_inputs[s], t_Z) for s in range(S)])
    t_K_xx = tf.stack([t_kernel.covar_matrix(t_inputs[s], t_inputs[s]) for s in range(S)])
    
    # Posterior mean
    t_L_inv_y = tf.stack([tf.linalg.triangular_solve(t_L_zz, t_U[s] - mean_fn(t_Z))
                          for s in range(S)])
    t_L_inv_K_zx = tf.stack([tf.linalg.triangular_solve(t_L_zz, tf.transpose(t_K_xz[s])) for s in range(S)])
    t_mean = mean_fn(t_inputs) + tf.einsum('smi,smd->sid', t_L_inv_K_zx, t_L_inv_y)
    
    # Posterior covariance
    t_cov = t_K_xx - \
            tf.einsum('smi,smj->sij', t_L_inv_K_zx, t_L_inv_K_zx)
    t_L_cov = tf.cholesky(t_cov + jitter * tf.eye(tf.shape(t_inputs)[1], dtype=dtype))
    return t_mean, t_L_cov


def posterior_gp_inducing_points_locations(t_inputs, mean_fn, t_Z, t_U, t_kernel, S, M):
    t_K_zz = tf.stack([t_kernel.covar_matrix(t_Z[s], t_Z[s]) for s in range(S)])
    t_L_zz = tf.cholesky(t_K_zz + jitter * tf.eye(M, dtype=dtype))
    
    t_K_xz = tf.stack([t_kernel.covar_matrix(t_inputs[s], t_Z[s]) for s in range(S)])
    t_K_xx = tf.stack([t_kernel.covar_matrix(t_inputs[s], t_inputs[s]) for s in range(S)])
    
    # Posterior mean
    t_L_inv_y = tf.stack([tf.linalg.triangular_solve(t_L_zz[s], t_U[s] - mean_fn(t_Z[s])) for s in range(S)])
    t_L_inv_K_zx = tf.stack([tf.linalg.triangular_solve(t_L_zz[s], tf.transpose(t_K_xz[s])) for s in range(S)])
    t_mean = mean_fn(t_inputs) + tf.einsum('smi,smd->sid', t_L_inv_K_zx, t_L_inv_y)
    
    # Posterior covariance
    t_cov = t_K_xx - tf.einsum('smi,smj->sij', t_L_inv_K_zx, t_L_inv_K_zx)
    t_L_cov = tf.cholesky(t_cov + jitter * tf.eye(tf.shape(t_inputs)[1], dtype=dtype))
    return t_mean, t_L_cov


def make_DGP_MF(n_layers, S, M, noise_std, last_periodic=True):
    tf.reset_default_graph()

    t_pi = tf.constant(np.pi, dtype=dtype)

    t_X = tf.placeholder(shape=(None, 1), dtype=dtype)
    t_X_S = tf.tile(tf.reshape(t_X, (1, -1, 1)), [S, 1, 1])
    t_Y = tf.placeholder(shape=(None, 1), dtype=dtype)

    t_beta = tf.constant(1 / noise_std**2, dtype=tf.float64) # 

    mean_function = lambda x: x
    layers = defaultdict(dict)

    layers[0]['t_X_samples'] = t_X_S

    for layer in range(1, n_layers + 1):
        if layer == n_layers:
            mean_function = lambda x: 0 * x

        # Kernel
        t_alpha = positive_variable(1)
        t_gamma = positive_variable(1)
        t_period = positive_variable(1)
        t_kernel = create_squared_exp_kernel(t_alpha, t_gamma)

        if layer == n_layers:
            if last_periodic:
                t_kernel = create_periodic_kernel(t_alpha, t_gamma, t_period)
            else:
                t_kernel = create_squared_exp_kernel(t_alpha, t_gamma)

        # Inducing points
        Z_init = np.linspace(-1, 1, M).reshape(M, 1)
        t_Z = real_variable(Z_init)

        m_init = mean_function(Z_init)
        t_m = real_variable(m_init) # 1e-1 * np.random.randn(*m0_init.shape))

        t_S = tf.Variable(1e-2 * init_triangular(M), dtype=dtype)
        t_S_sqrt = vec_to_tri(t_S, M)
        t_S = tf.matmul(t_S_sqrt, t_S_sqrt, transpose_b=True)

        # Distributions
        t_pSigma = t_kernel.covar_matrix(t_Z, t_Z) + jitter * tf.eye(M, dtype=dtype)
        t_pU = tfp.distributions.MultivariateNormalFullCovariance(
            loc=mean_function(tf.squeeze(t_Z)),
            covariance_matrix=t_pSigma)

        t_qU = tfp.distributions.MultivariateNormalTriL(loc=tf.squeeze(t_m), scale_tril=t_S_sqrt)

        layers[layer]['t_pU'] = t_pU
        layers[layer]['t_qU'] = t_qU

        t_pred_mean, t_L_cov = posterior_gp_integrated_inducing(
            layers[layer - 1]['t_X_samples'],
            mean_function,
            t_Z,
            t_m,
            t_S,
            t_kernel,
            S, M)
        t_output = tfp.distributions.MultivariateNormalTriL(loc=t_pred_mean[:,:,0], scale_tril=t_L_cov)
        t_sampled_output = tf.expand_dims(t_output.sample(), -1)
        layers[layer]['t_X_samples'] = t_sampled_output
        
        t_pred_mean, t_L_cov = posterior_gp_integrated_inducing(
            t_X_S,
            mean_function,
            t_Z,
            t_m,
            t_S,
            t_kernel,
            S, M)
        t_output = tfp.distributions.MultivariateNormalTriL(loc=t_pred_mean[:,:,0], scale_tril=t_L_cov)
        t_sampled_output = tf.expand_dims(t_output.sample(), -1)
        layers[layer]['t_X_samples_input'] = t_sampled_output

    t_p_likelihood = tfp.distributions.MultivariateNormalDiag(
        loc=layers[n_layers]['t_X_samples'][:,:,0],
        scale_identity_multiplier=1 / tf.sqrt(t_beta))
    t_first_term = tf.reduce_mean(t_p_likelihood.log_prob(t_Y[:,0]))
    t_third_term = sum([tfp.distributions.kl_divergence(layers[layer]['t_qU'], layers[layer]['t_pU'])
                    for layer in range(1, n_layers + 1)])
    t_lower_bound = t_first_term - t_third_term
    t_neg_lower_bound = -t_lower_bound
    return layers, t_neg_lower_bound, t_X, t_Y


def make_DGP_jointly_Gaussian(n_layers, S, M, noise_std, last_periodic=True):
    tf.reset_default_graph()

    t_pi = tf.constant(np.pi, dtype=dtype)

    t_X = tf.placeholder(shape=(None, 1), dtype=dtype)
    t_X_S = tf.tile(tf.reshape(t_X, (1, -1, 1)), [S, 1, 1])
    t_Y = tf.placeholder(shape=(None, 1), dtype=dtype)

    t_beta = tf.constant(1 / noise_std**2, dtype=tf.float64) # 

    mean_function = lambda x: x
    layers = defaultdict(dict)

    layers[0]['t_X_samples'] = t_X_S
    
    Z_init = np.linspace(-1, 1, M).reshape(M, 1)
    m_init = mean_function(Z_init)
    t_m = real_variable(np.concatenate([m_init] * (n_layers - 1) + [np.zeros_like(m_init)]))
    t_S = tf.Variable(1e-2 * init_triangular(n_layers * M), dtype=dtype)
    t_S_sqrt = vec_to_tri(t_S, n_layers * M)
    t_S = tf.matmul(t_S_sqrt, t_S_sqrt, transpose_b=True)
    t_qU = tfp.distributions.MultivariateNormalTriL(loc=tf.squeeze(t_m), scale_tril=t_S_sqrt)
    
    t_qU_samples = t_qU.sample(S)
    
    for layer in range(1, n_layers + 1):
        layers[layer]['t_qU_samples'] = tf.expand_dims(t_qU_samples[:, (layer - 1) * M : layer * M], -1)

    for layer in range(1, n_layers + 1):
        if layer == n_layers:
            mean_function = lambda x: 0 * x
            
        layers[layer]['mean_fn'] = mean_function

        # Kernel
        t_alpha = positive_variable(1)
        t_gamma = positive_variable(1)
        t_period = positive_variable(1)
        t_kernel = create_squared_exp_kernel(t_alpha, t_gamma)

        if layer == n_layers:
            if last_periodic:
                t_kernel = create_periodic_kernel(t_alpha, t_gamma, t_period)
            else:
                t_kernel = create_squared_exp_kernel(t_alpha, t_gamma)

        # Inducing points
        Z_init = np.linspace(-1, 1, M).reshape(M, 1)
        t_Z = real_variable(Z_init)
        layers[layer]['t_Z'] = t_Z
        
        layers[layer]['t_pSigma'] = t_kernel.covar_matrix(t_Z, t_Z)

        t_pred_mean, t_L_cov = posterior_gp_sampled_inducing(
            layers[layer - 1]['t_X_samples'],
            mean_function,
            t_Z,
            layers[layer]['t_qU_samples'],
            t_kernel,
            S, M)
        t_output = tfp.distributions.MultivariateNormalTriL(loc=t_pred_mean[:,:,0], scale_tril=t_L_cov)
        t_sampled_output = tf.expand_dims(t_output.sample(), -1)
        layers[layer]['t_X_samples'] = t_sampled_output
        
        t_pred_mean, t_L_cov = posterior_gp_sampled_inducing(
            t_X_S,
            mean_function,
            t_Z,
            layers[layer]['t_qU_samples'],
            t_kernel,
            S, M)
        t_output = tfp.distributions.MultivariateNormalTriL(loc=t_pred_mean[:,:,0], scale_tril=t_L_cov)
        t_sampled_output = tf.expand_dims(t_output.sample(), -1)
        layers[layer]['t_X_samples_input'] = t_sampled_output

    t_p_likelihood = tfp.distributions.MultivariateNormalDiag(
        loc=layers[n_layers]['t_X_samples'][:,:,0],
        scale_identity_multiplier=1 / tf.sqrt(t_beta))
    t_first_term = tf.reduce_mean(t_p_likelihood.log_prob(t_Y[:,0]))
    
    rows = []
    for i in range(n_layers):
        row = [tf.zeros((M, M), dtype=tf.float64) for _ in range(i)] + \
              [layers[i + 1]['t_pSigma']] + \
              [tf.zeros((M, M), dtype=tf.float64) for _ in range(n_layers - i - 1)]
        row = tf.concat(row, axis=1)
        rows.append(row)
    t_pU = tf.concat(rows, axis=0) + jitter * tf.eye(n_layers * M, dtype=tf.float64)
    t_pU = tfp.distributions.MultivariateNormalFullCovariance(
        loc=tf.squeeze(
                tf.concat(
                    [layers[layer]['mean_fn'](layers[layer]['t_Z'])
                     for layer in range(1, n_layers + 1)], axis=0)),
        covariance_matrix=t_pU)
    
    t_third_term = tfp.distributions.kl_divergence(t_qU, t_pU)
    t_lower_bound = t_first_term - t_third_term
    t_neg_lower_bound = -t_lower_bound
    return layers, t_neg_lower_bound, t_X, t_Y


def make_DGP_inducing_locations_points(n_layers, S, M, noise_std, last_periodic=True):
    tf.reset_default_graph()

    t_pi = tf.constant(np.pi, dtype=dtype)

    t_X = tf.placeholder(shape=(None, 1), dtype=dtype)
    t_X_S = tf.tile(tf.reshape(t_X, (1, -1, 1)), [S, 1, 1])
    t_X_test = tf.placeholder(shape=(None, 1), dtype=dtype)
    t_X_test_S = tf.tile(tf.reshape(t_X_test, (1, -1, 1)), [S, 1, 1])
    t_Y = tf.placeholder(shape=(None, 1), dtype=dtype)
    t_Y_S = tf.tile(tf.reshape(t_Y, (1, -1, 1)), [S, 1, 1])
    t_Y_test = tf.placeholder(shape=(None, 1), dtype=dtype)
    t_Y_test_S = tf.tile(tf.reshape(t_Y, (1, -1, 1)), [S, 1, 1])

    t_beta = tf.constant(1 / noise_std**2, dtype=dtype) # positive_variable(1) #
    t_beta_jitter = tf.placeholder(dtype=dtype, shape=()) # tf.constant(1 / 1e-6, dtype=dtype)

    mean_function = lambda x: x
    layers = defaultdict(dict)

    t_Z = real_variable(np.linspace(-1, 1, M))
    t_Z_S = tf.tile(tf.reshape(t_Z, (1, M, 1)), [S, 1, 1])
    layers[0]['t_q_samples'] = t_Z_S
    layers[0]['t_X_samples'] = t_X_S

    for layer in range(1, n_layers + 1):
        if layer == n_layers:
            mean_function = lambda x: 0 * x

        t_alpha = positive_variable(1)
        t_gamma = positive_variable(1)
        t_period = positive_variable(1)
        t_kernel = create_squared_exp_kernel(t_alpha, t_gamma)
        if layer == n_layers:
            if last_periodic:
                t_kernel = create_periodic_kernel(t_alpha, t_gamma, t_period)
            else:
                t_kernel = create_squared_exp_kernel(t_alpha, t_gamma)
        layers[layer]['kernel'] = t_kernel
        layers[layer]['alpha'] = t_alpha
        layers[layer]['gamma'] = t_gamma

        m_init = mean_function(np.linspace(-1, 1, M)) + 1e-2 * np.random.randn(M)
        t_m = real_variable(m_init)
        t_S = tf.Variable(1e-3 * init_triangular(M), dtype=dtype)
        t_S_sqrt = vec_to_tri(t_S, M)
        t_S = tf.matmul(t_S_sqrt, t_S_sqrt, transpose_b=True)
        t_q = tfd.MultivariateNormalTriL(loc=t_m, scale_tril=t_S_sqrt)
        t_q_samples = tf.expand_dims(t_q.sample(S), -1)
        layers[layer]['t_q'] = t_q
        layers[layer]['t_q_samples'] = t_q_samples

        t_X_mu, t_X_L_cov = \
            posterior_gp_inducing_points_locations(
                t_inputs=layers[layer - 1]['t_X_samples'],
                mean_fn=mean_function,
                t_Z=layers[layer - 1]['t_q_samples'],
                t_U=layers[layer]['t_q_samples'],
                t_kernel=t_kernel,
                S=S, M=M)

        t_X_dist = tfd.MultivariateNormalTriL(loc=tf.squeeze(t_X_mu), scale_tril=t_X_L_cov)
        t_X_samples = tf.expand_dims(t_X_dist.sample(), -1)
        layers[layer]['t_X_samples'] = t_X_samples
        
        t_X_mu, t_X_L_cov = \
            posterior_gp_inducing_points_locations(
                t_inputs=t_X_S,
                mean_fn=mean_function,
                t_Z=layers[layer - 1]['t_q_samples'],
                t_U=layers[layer]['t_q_samples'],
                t_kernel=t_kernel,
                S=S, M=M)

        t_X_dist = tfd.MultivariateNormalTriL(loc=tf.squeeze(t_X_mu), scale_tril=t_X_L_cov)
        t_X_samples = tf.expand_dims(t_X_dist.sample(), -1)
        layers[layer]['t_X_samples_input'] = t_X_samples

        t_p_Sigma = tf.stack(
            [t_kernel.covar_matrix(layers[layer - 1]['t_q_samples'][s], layers[layer - 1]['t_q_samples'][s])
             for s in range(S)])
        t_p_Sigma = t_p_Sigma + (1 / t_beta_jitter + jitter) * tf.eye(M, dtype=tf.float64)
        t_p = tfd.MultivariateNormalFullCovariance(
            loc=mean_function(layers[layer - 1]['t_q_samples'][:,:,0]),
            covariance_matrix=t_p_Sigma)
        layers[layer]['t_p'] = t_p

    t_p_likelihood = tfp.distributions.MultivariateNormalDiag(
        loc=layers[n_layers]['t_X_samples'][:,:,0],
        scale_identity_multiplier=1 / tf.sqrt(t_beta))

    t_first_term = tf.reduce_mean(t_p_likelihood.log_prob(t_Y[:,0]))

    KL_terms = []

    for layer in range(1, n_layers + 1):
        t_q = layers[layer]['t_q']
        t_p = layers[layer]['t_p']
        KLs_layer = tfd.kl_divergence(t_q, t_p)
        KL_terms.append(tf.reduce_mean(KLs_layer))

    t_third_term = sum(KL_terms)

    t_lower_bound = t_first_term - t_third_term
    t_neg_lower_bound = -t_lower_bound
    return layers, t_neg_lower_bound, t_X, t_Y, t_beta_jitter


def run_optimisation(t_neg_lower_bound, sess, optimiser, t_lr, feed_dict, n_steps=10000, lr_init=1e-2, lr_reduce=2, print_loss=False):
    losses = []
    
    lr = lr_init * 10
    for _ in range(lr_reduce + 1):
        feed_dict[t_lr] = lr / 10
        for step in range(n_steps + 1):
            _, loss = sess.run([optimiser, t_neg_lower_bound], feed_dict)
            losses.append(loss)
            if print_loss and (step % print_loss == 0):
                print('{}: {:.2f}'.format(step, loss))
        
    return losses


def sample(layers, sess, feed_dict, S):
    tf_samples = \
    (layers[1]['t_X_samples_input'],
     layers[2]['t_X_samples_input'],
     layers[2]['t_X_samples'])

    samples = np.concatenate([sess.run(tf_samples, feed_dict) for _ in range(100 // S)], axis=1)
    return samples

