import jax
import numpy as np
import optax
from absl import app, flags
from flax.training.train_state import TrainState
from tqdm import tqdm
from functools import partial
from img_utils import scatter_into_array
import wandb
from datasets import get_dataset, skd_func
from mlp import MLP, DiffusionMLP, FourierFeatures
from mlp_resnet_arch import MLPResNetV2
import flax.linen as nn
import jax.numpy as jnp

FLAGS = flags.FLAGS

flags.DEFINE_string('project_name', 'diffusion_pi', 'wandb project name.')
flags.DEFINE_string('arch', 'mlp', 'Architecture.')
flags.DEFINE_string('schedule', 'cosine', 'Beta Schedule.')
flags.DEFINE_integer('T', 50, 'Number of timesteps.')
flags.DEFINE_enum('dataset_name', 'swissroll', skd_func.keys(),
                  'Dataset for p(x)')
flags.DEFINE_integer('dataset_size', int(1e6), 'Dataset size.')
flags.DEFINE_integer('batch_size', 4096, 'Minibatch size.')
flags.DEFINE_integer('sample_batch_size', 50000, 'Minibatch size.')
flags.DEFINE_integer('eval_interval', 10000, 'Minibatch size.')
flags.DEFINE_integer('max_iterations', 100001, 'Training iterations.')
flags.DEFINE_integer('repeat_last_step', 0, 'lmfao.')
flags.DEFINE_float('beta_start', 1e-4, 'Beta start.')
flags.DEFINE_float('beta_end', 0.02, 'Beta end.')
flags.DEFINE_boolean('collect_sample_video', False, 'Minibatch size.')
flags.DEFINE_boolean('truncnorm', False, 'Minibatch size.')
flags.DEFINE_boolean('use_one_hot', False, 'Minibatch size.')
flags.DEFINE_boolean('use_fourier', False, 'Minibatch size.')

def main(_):
    wandb.init(project=FLAGS.project_name) #, entity="diffusion_pi")
    wandb.config.update(FLAGS)

    rng = jax.random.PRNGKey(100)
    rng, mlp_key = jax.random.split(rng, 2)

    samples = get_dataset(FLAGS.dataset_name, FLAGS.dataset_size)
    val_samples = get_dataset(FLAGS.dataset_name, FLAGS.dataset_size // 10)

    data_dim = samples.shape[-1]

    current_x = np.random.multivariate_normal(
                np.zeros(data_dim),
                np.eye(data_dim),
                size=samples.shape[0])

    hist = scatter_into_array(samples, "Data Distribution")
    images = wandb.Image(hist, caption="Data distribution")
    wandb.log({"Data": images})

    
    time_dim = 64

    def mish(x):
        return x * jnp.tanh(nn.softplus(x))

    preprocess_time = FourierFeatures(time_dim, learnable=True)
    #preprocess_inputs = FourierFeatures(time_dim, learnable=True)
    #preprocess_ff = Base2FourierFeatures()
    time_model = MLP([time_dim * 2, time_dim], activate_final=False)
    
    if FLAGS.arch == 'LN_resnet':
        base_model = MLPResNetV2(2, data_dim, act = mish)
        model = DiffusionMLP(time_model, base_model, preprocess_time, use_one_hot=FLAGS.use_one_hot, T = FLAGS.T, use_ff_features=FLAGS.use_fourier)
    elif FLAGS.arch == 'MLP':
        base_model = MLP([256, 256, data_dim], activations = mish, activate_final=False)
        model = DiffusionMLP(time_model, base_model, preprocess_time, use_one_hot=FLAGS.use_one_hot, T = FLAGS.T)
    else:
        raise NotImplementedError
        
    
    observations = np.zeros((1, data_dim))
    if FLAGS.arch == 'ensemble':
        observations = np.zeros((FLAGS.ensemble_size, 1, data_dim))
        time = np.zeros((1, 1))
    else:
        observations = np.zeros((1, data_dim))
        time = np.zeros((1, 1))
    model_params = model.init(mlp_key, observations, time)['params']
    model = TrainState.create(apply_fn=model.apply,
                              params=model_params,
                              tx=optax.adam(learning_rate=3e-4))

    @jax.jit
    def update_step_ddpm(model, time, x_t, noise_sample):

        def model_loss(model_params):
            outputs = model.apply_fn({'params': model_params}, x_t, time, True)
            loss = ((noise_sample - outputs)**2).mean()
            return loss, {'loss': loss}

        grads, info = jax.grad(model_loss, has_aux=True)(model.params)
        return model.apply_gradients(grads=grads), info
    
    @jax.jit
    def sample_step_ddpm(model, time, current_x):

        outputs = model.apply_fn({'params': model.params}, current_x, time)

        return outputs
    

    def cosine_beta_schedule(timesteps, s = 0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        t = jnp.linspace(0, timesteps, steps) / timesteps
        alphas_cumprod = jnp.cos((t + s) / (1 + s) * jnp.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return jnp.clip(betas, 0, 0.999)
    
    def vp_beta_schedule(timesteps):
        t = jnp.arange(1, timesteps + 1)
        T = timesteps
        b_max = 10.
        b_min = 0.1
        alpha = jnp.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
        betas = 1 - alpha
        return betas

    if FLAGS.schedule == 'cosine':
        betas = cosine_beta_schedule(FLAGS.T)
    elif FLAGS.schedule == 'vp':
        betas = vp_beta_schedule(FLAGS.T)
    else:
        raise NotImplementedError
        
    alphas = 1 - betas
    alpha_hat = np.cumprod(alphas, axis=0)

    for i in tqdm(range(FLAGS.max_iterations)):
        sample = samples[np.random.randint(0, samples.shape[0],
                                           FLAGS.batch_size)]
        t = np.random.randint(0, FLAGS.T, FLAGS.batch_size)
        noise_sample = np.random.multivariate_normal(np.zeros(data_dim),
                                                     np.eye(data_dim),
                                                     size=FLAGS.batch_size)
        
        if FLAGS.arch == 'ensemble':
            sample = np.expand_dims(sample, axis=0).repeat(FLAGS.ensemble_size, axis=0)
            noise_sample = np.expand_dims(noise_sample, axis=0).repeat(FLAGS.ensemble_size, axis=0)

        alpha_hats = alpha_hat[t]
        t = np.expand_dims(t, axis=1)
        alpha_1 = np.expand_dims(np.sqrt(alpha_hats), axis=1)
        alpha_2 = np.expand_dims(np.sqrt(1 - alpha_hats), axis=1)
        x_t = alpha_1 * sample + alpha_2 * noise_sample
        model, info = update_step_ddpm(model, t, x_t, noise_sample)
        wandb.log({"training/loss": info['loss'].item()}, step=i)

        if i % FLAGS.eval_interval == 0:
            current_x = np.random.multivariate_normal(
                np.zeros(data_dim),
                np.eye(data_dim),
                size=FLAGS.sample_batch_size)
            
            if FLAGS.arch == 'ensemble':
                current_x = np.expand_dims(current_x, axis=0).repeat(FLAGS.ensemble_size, axis=0)

            images = []
            for t in reversed(range(FLAGS.T)):
                stack_t = np.expand_dims(np.array(t).repeat(
                    FLAGS.sample_batch_size),
                                         axis=1)

                noise_estimate = sample_step_ddpm(model, stack_t, current_x)
                noise_estimate = np.asarray(noise_estimate)

                alpha_1 = 1 / np.sqrt(alphas[t])
                alpha_2 = ((1 - alphas[t]) / (np.sqrt(1 - alpha_hat[t])))

                current_x = alpha_1 * (current_x - alpha_2 * noise_estimate)

                if t > 0:
                    import scipy.stats as stats
                    z = np.random.multivariate_normal(
                        np.zeros(data_dim),
                        np.eye(data_dim),
                        size=FLAGS.sample_batch_size)
                    if FLAGS.truncnorm:
                        z = stats.truncnorm.rvs(-2, 2, size=z.shape)
                    if FLAGS.arch == 'ensemble':
                        z = np.expand_dims(z, axis=0).repeat(FLAGS.ensemble_size, axis=0)
                    
                    current_x += np.sqrt(betas[t]) * z
                else:
                    for j in range(FLAGS.repeat_last_step):
                        noise_estimate = sample_step_ddpm(model, stack_t, current_x)
                        noise_estimate = np.asarray(noise_estimate)

                        alpha_1 = 1 / np.sqrt(alphas[t])
                        alpha_2 = ((1 - alphas[t]) / (np.sqrt(1 - alpha_hat[t])))

                        current_x = alpha_1 * (current_x - alpha_2 * noise_estimate)

                current_x = jnp.clip(current_x, -1.0, 1.0)

                if FLAGS.collect_sample_video:
                    hist = scatter_into_array(current_x)
                    images.append(hist)
            

            if FLAGS.collect_sample_video:
                images = np.stack(images)
                wandb.log(
                    {
                        "video":
                        wandb.Video(images[:, np.newaxis], fps=4, format="gif")
                    },
                    step=i)
            else:
                hist = scatter_into_array(current_x, FLAGS.arch + " DDPM Sampling Distribution")
                image = wandb.Image(hist, caption="Sampled Data distribution.")
                wandb.log({"image": image})



if __name__ == '__main__':
    app.run(main)
