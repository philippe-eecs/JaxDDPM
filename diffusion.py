import numpy as np
import jax
import jax.numpy as jnp


def compute_vlb_full(model, rng, x_start, T, alphas, alpha_hat, betas, alpha_hat_prev, alpha_hat_next):
    vlbs = []
    for t in list(range(T)):
        data_dim = x_start.shape[-1]
        time = np.expand_dims(np.array([t]).repeat(x_start.shape[0]), axis = 1)


        noise_sample = np.random.multivariate_normal(np.zeros(data_dim),
                                                    np.eye(data_dim),
                                                    size=x_start.shape[0])
        alpha_hats = alpha_hat[time]
        alpha_1 = np.sqrt(alpha_hats)
        alpha_2 = np.sqrt(1 - alpha_hats)
        x_t = alpha_1 * x_start + alpha_2 * noise_sample
        vlb = compute_vlb(model, rng, x_start, x_t, time, alphas, alpha_hat, betas, alpha_hat_prev, alpha_hat_next)
        
        vlbs.append(vlb)
    
    vlbs = jnp.stack(vlbs, axis = 1)
    time = np.expand_dims(np.array([T - 1]).repeat(x_start.shape[0]), axis = 1)
    vlb_prior = compute_prior_kl(model, rng, x_start, time, alphas, alpha_hat, betas, alpha_hat_prev, alpha_hat_next)
    vlb_total = vlb_prior + vlbs.sum(axis=1)

    return vlb_total


@jax.jit
def compute_prior_kl(model, rng, x_start, t, alphas, alpha_hats, betas, alpha_hat_prev, alpha_hat_next):

    mean = jnp.sqrt(alpha_hats)[t] * x_start
    variance = (1 - alpha_hats)[t]
    log_var = jnp.log(1 - alpha_hats)[t]

    kl_term = compute_kl(mean, log_var, 0, 0).mean(axis=-1) / np.log(2.0)
    
    return kl_term

@jax.jit
def compute_vlb(model, rng, x_start, x_t, t, alphas, alpha_hats, betas, alpha_hat_prev, alpha_hat_next):
    posterior_mean, _, posterior_log_variance = compute_q_posterior(model, rng, x_start, x_t, t, alphas, alpha_hats, betas, alpha_hat_prev, alpha_hat_next)

    model_mean = model.apply_fn({'params': model.params}, x_t, t)
    posterior_variances = betas * (1 - alpha_hat_prev) / (1 - alpha_hats)
    posterior_log_variance_clipped = jnp.log(jnp.append(posterior_variances[1], posterior_variances[1:]))
    model_log_variance = posterior_log_variance_clipped[t]
    start_x_pred = predict_xstart_from_eps(x_t, t, model_mean, alpha_hats)
    compare_mean, _, _ = compute_q_posterior(model, rng, start_x_pred, x_t, t, alphas, alpha_hats, betas, alpha_hat_prev, alpha_hat_next)


    kl_term = compute_kl(posterior_mean, posterior_log_variance, compare_mean, model_log_variance).mean(axis=-1) / np.log(2.0)

    return kl_term

@jax.jit
def compute_q_posterior(model, rng, x_start, x_t, t, alphas, alpha_hats, betas, alpha_hat_prev, alpha_hat_next):
    posterior_variances = betas * (1 - alpha_hat_prev) / (1 - alpha_hats)
    posterior_log_variance_clipped = jnp.log(jnp.append(posterior_variances[1], posterior_variances[1:]))

    posterior_mean = alpha_hat_prev[t] * x_start + alpha_hat_next[t] * x_t
    posterior_variance = posterior_variances[t]
    posterior_log_variance = posterior_log_variance_clipped[t]

    return posterior_mean, posterior_variance, posterior_log_variance

@jax.jit
def predict_xstart_from_eps(x_t, t, eps, alpha_hats):
    start = jnp.sqrt(1/alpha_hats[t]) * x_t - jnp.sqrt(1/alpha_hats[t] - 1) * eps
    return jnp.clip(start, -1, 1)

@jax.jit
def compute_kl(mean1, log_var1, mean2, log_var2):
    return 0.5 * (-1 + log_var2 - log_var1 + jnp.exp(log_var1 - log_var2) + ((mean1 - mean2)** 2) * jnp.exp(-log_var2))