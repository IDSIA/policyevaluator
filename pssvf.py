import torch
import numpy as np
import gym
import core
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import time
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env_name",
    default="Swimmer-v3",
    choices=[
        "Swimmer-v3",
        "Hopper-v3",
        "Ant-v3",
        "Walker2d-v3",
        "InvertedDoublePendulum-v2",
        "HalfCheetah-v3"
    ],
    type=str,
    required=False,
)
parser.add_argument("--verbose", default=0, type=int, required=False)
parser.add_argument("--show_plots", default=0, type=int, required=False)
parser.add_argument("--use_gpu", default=1, type=int, required=False)
parser.add_argument("--seed", default=1234, type=int, required=False)
args = parser.parse_args()

verbose = args.verbose
show_plots = args.show_plots

# Default hyperparameters
config = dict(
    env_name='Hopper-v3',#'Swimmer-v3', # MountainCarContinuous-v0
    neurons_policy=(256,256),
    neurons_vf=(256,256),
    policy_iters=5,
    vf_iters=5,
    batch_size=16,
    learning_rate_policy=2e-6,
    learning_rate_vf=5e-3,
    noise_policy=0.05, # std of distribution generating the noise for the perturbed policy
    size_buffer=10000,
    max_episodes=1000000000,
    max_timesteps=3000000,
    seed=1,
    survival_bonus=False,
    deterministic_actor=True,
    ts_evaluation=10000,
    n_probing_states=200,
    print_stats=True,
    render_prob_states=False,
    save_model=100000000,
    act_clipping=False,
    activation_policy='tanh',
    activation_vf='relu',
    start_steps=0,
    observation_normalization=True,
    algo='pssvf',
    act_noise=0.0,
    use_virtual_class=True,
    update_every_ts=False,
    update_every=1000,
    render=False,
    weighted_sampling=True,
    scale=1.1,
)

# Use GPU or CPU
if args.use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

# Create env
env = gym.make(config["env_name"])
env_test = gym.make(config["env_name"])

torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

if config['env_name'] == 'InvertedDoublePendulum-v2':
    config.update({'max_timesteps': 300000}, allow_val_change=True)
    config.update({'ts_evaluation': 1000}, allow_val_change=True)

if config['env_name'] in ['MountainCarContinuous-v0',  'InvertedPendulum-v2', 'Reacher-v2']:
        config.update({'ts_evaluation': 1000,
                             'max_timesteps': 100000,
                             }, allow_val_change=True)

activation_policy = nn.ReLU if config['activation_policy'] == 'relu' else nn.Tanh
activation_vf = nn.ReLU if config['activation_vf'] == 'relu' else nn.Tanh

# Create replay buffer, policy, vf
buffer = core.Buffer(config['size_buffer'], scale=config['scale'])
statistics = core.Statistics(env.observation_space.shape)
ac = core.MLPActorCritic(config['algo'], env.observation_space, env.action_space, config['n_probing_states'],
                         hidden_sizes_actor=tuple(config['neurons_policy']), activation_policy=activation_policy,
                         activation_vf=activation_vf,
                         hidden_sizes_critic=tuple(config['neurons_vf']), device=device,
                         critic=True, deterministic_actor=config['deterministic_actor'],
                         act_clipping=config['act_clipping'], act_noise=config['act_noise'])

virtual_mlp = core.VirtualMLPPolicy(layer_sizes=[env.observation_space.shape[0]]+
                                                 list(tuple(config['neurons_policy']))+
                                                 [env.action_space.shape[0]],
                                    act_lim=env.action_space.high[0],
                                    nonlinearity=config['activation_policy'])


print("Number of policy params:", len(nn.utils.parameters_to_vector(list(ac.pi.parameters()))))
print("Number of value function params:", len(nn.utils.parameters_to_vector(list(ac.v.parameters()))))
print("Obs dim", env.observation_space.shape[0])
print("Act dim", env.action_space.shape[0])

# Setup optimizer
q_params = ac.v.parameters()
optimize_policy = optim.Adam(ac.pi.parameters(), lr=config['learning_rate_policy'])
optimize_vf = optim.Adam(q_params, lr=config['learning_rate_vf'])


def compute_policy_loss():
    params = nn.utils.parameters_to_vector(list(ac.pi.parameters())).to(device, non_blocking=True).unsqueeze(0)
    return -ac.v.forward(params, use_virtual_module=True, virtual_module=virtual_mlp)


def compute_vf_loss(progs, rew):
    q = ac.v(progs, use_virtual_module=True, virtual_module=virtual_mlp)
    loss_q = ((q - rew) ** 2).mean()
    return loss_q


def perturbe_policy(policy):

    dist = Normal(torch.zeros(len(torch.nn.utils.parameters_to_vector(policy.parameters()))), scale=1)
    delta = dist.sample().to(device=device, non_blocking=True).detach()

    # Perturbe policy parameters
    params = torch.nn.utils.parameters_to_vector(policy.parameters()).detach()
    perturbed_params = params + config['noise_policy'] * delta

    # Copy perturbed parameters into a new policy
    perturbed_policy = core.MLPActorCritic(config['algo'], env.observation_space, env.action_space,
                                           config['n_probing_states'],
                                           hidden_sizes_actor=tuple(config['neurons_policy']),  activation_policy=activation_policy,
                                           activation_vf=activation_vf,
                                           hidden_sizes_critic=tuple(config['neurons_vf']), device=device,
                                           critic=False, deterministic_actor=config['deterministic_actor'],
                                           act_clipping=config['act_clipping'], act_noise=config['act_noise'])

    torch.nn.utils.vector_to_parameters(perturbed_params, perturbed_policy.parameters())

    return perturbed_policy


def update():
    start_time = time.perf_counter()

    for _ in range(config['vf_iters']):
        # Sample batch
        hist = buffer.sample_replay(config['batch_size'], weighted_sampling=config['weighted_sampling'])
        prog, rew, _ = zip(*hist)

        if config['use_virtual_class']:
            prog = torch.stack(prog).to(device)

        rew = torch.from_numpy(np.asarray(rew)).float().to(device=device, non_blocking=True).detach()

        optimize_vf.zero_grad()
        loss_vf = compute_vf_loss(prog, rew)
        loss_vf.backward()
        optimize_vf.step()


    statistics.up_v_time += time.perf_counter() - start_time
    start_time = time.perf_counter()
#    # Freeze PSSVF
    for p in q_params:
        p.requires_grad = False

    # Update policy
    for _ in range(config['policy_iters']):
        optimize_policy.zero_grad()
        loss_policy = compute_policy_loss()
        loss_policy.backward()
        optimize_policy.step()

#    # Unfreeze PSSVF
    for p in q_params:
        p.requires_grad = True
    statistics.up_policy_time += time.perf_counter() - start_time

    log_dict = {'loss_pvf': loss_vf.item(),
               'loss_policy': loss_policy.item()}
    if verbose:
        print(log_dict)
    return


def evaluate(policy):
    rew_evals = []
    with torch.no_grad():
        for _ in range(10):

            # Simulate a trajectory and compute the total reward
            done = False
            obs = env_test.reset()
            rew_eval = 0
            while not done:
                obs = torch.as_tensor(obs, dtype=torch.float32)
                if config['observation_normalization'] and statistics.episode > 0:
                    obs = statistics.normalize(obs)

                with torch.no_grad():
                    action = policy.act(obs.to(device, non_blocking=True).detach())
                if config['render']:
                    env_test.render()
                    #print(action)
                obs_new, r, done, _ = env_test.step(action)

                # Remove survival bonus
                rew_eval += r

                obs = obs_new

            rew_evals.append(rew_eval)

        statistics.rew_eval = np.mean(rew_evals)
        statistics.push_rew(np.mean(rew_evals))
    # Log results


    log_dict = {'rew_eval': statistics.rew_eval,
               'average_reward': np.mean(statistics.rewards),
               'average_last_rewards': np.mean(statistics.last_rewards),
               }
    print(log_dict)

    if config['print_stats']:
        print("Ts", statistics.total_ts, "Ep", statistics.episode, "rew_eval", statistics.rew_eval)
        print("time_sim", statistics.sim_time, "time_up_pi", statistics.up_policy_time, "time_up_v", statistics.up_v_time,
              "total_time", statistics.total_time)
    return


def simulate(policy):

    # Simulate a trajectory and compute the total reward
    done = False
    obs = env.reset()
    rew = 0
    rew_bonus = 0

    while not done:
        obs = torch.as_tensor(obs, dtype=torch.float32)
        if config['observation_normalization']:
            statistics.push_obs(obs)
            if statistics.episode > 0:
                obs = statistics.normalize(obs)

        with torch.no_grad():
            action = policy.act(obs.to(device, non_blocking=True).detach())
        obs_new, r, done, _ = env.step(action)
        # Remove survival bonus

        if not config['survival_bonus']:
            if config['env_name'] == 'Hopper-v3' or config['env_name'] == 'Ant-v3' or config['env_name'] == 'Walker2d-v3':
                rew += r - 1
            elif config['env_name'] == 'Humanoid-v3':
                rew += r - 5
            else:
                rew += r
        else:
            rew += r

        rew_bonus += r

        statistics.total_ts += 1

        if statistics.total_ts % config['save_model'] == 0:
            if config['neurons_policy'] == []:

                torch.save(ac.v, 'models/model_lin' + str(int(100*config['n_probing_states'])) + str(int(100*config['seed'])) + config['env_name'] + "_" + str(statistics.total_ts))
                torch.save(ac.pi, 'policies/pi_lin' + str(int(100*config['n_probing_states'])) + str(int(100*config['seed'])) + config['env_name'] + "_" + str(statistics.total_ts))
                with open('statistics/stat_lin' + str(int(100*config['n_probing_states'])) + str(int(100*config['seed'])) + config['env_name'] + "_" + str(statistics.total_ts), 'wb') as fp:
                    pickle.dump(statistics, fp)
            else:
                torch.save(ac.v, 'models/model' + str(int(100*config['n_probing_states'])) + str(int(100*config['seed'])) + config['env_name'] + "_" + str(statistics.total_ts))
                torch.save(ac.pi, 'policies/pi' + str(int(100*config['n_probing_states'])) + str(int(100*config['seed'])) + config['env_name'] + "_" + str(statistics.total_ts))

                with open('statistics/stat' + str(int(100*config['n_probing_states'])) + str(int(100*config['seed'])) + config['env_name'] + "_" + str(statistics.total_ts), 'wb') as fp:
                    pickle.dump(statistics, fp)





        # Evaluate current policy
        if statistics.total_ts % config['ts_evaluation'] == 0:
            evaluate(ac)

        # Update
        if statistics.total_ts > config['start_steps'] and config['update_every_ts'] and statistics.episode > 0:
            if statistics.total_ts % config['update_every'] == 0:
                update()


        if statistics.total_ts == 1000000:
            log_dict = {'rew_eval_1M': statistics.rew_eval,
               'average_reward_1M': np.mean(statistics.rewards),
               'average_last_rewards_1M': np.mean(statistics.last_rewards)}
            print(log_dict)

        obs = obs_new

    return rew, rew_bonus


def train():
    start_time = time.perf_counter()
    # Collect data with perturbed policy
    perturbed_policy = perturbe_policy(ac.pi)
    # Simulate a trajectory and compute the total reward
    rew, rew_bonus = simulate(perturbed_policy)
    # Store data in replay buffer
    perturbed_params = nn.utils.parameters_to_vector(list(perturbed_policy.parameters())).to(device, non_blocking=True).detach()

    buffer.history.append((perturbed_params, rew, rew_bonus))

    statistics.episode += 1
    statistics.sim_time += time.perf_counter() - start_time
    # Update
    if statistics.total_ts > config['start_steps'] and not config['update_every_ts']:
        update()

    # Log results
    log_dict = {'rew': rew,
               'rew_bonus': rew_bonus,
               'steps': statistics.total_ts,
               'episode': statistics.episode,
               'grads_norm_policy': core.grad_norm(ac.pi.parameters()),
               'norm_policy': core.norm(ac.pi.parameters()),
               'norm_pvf': core.norm(ac.v.parameters()),
               'grads_norm_pvf': core.grad_norm(ac.v.parameters()),
               'norm_prob_states': core.norm(ac.v.probing_states.parameters()),
               'grads_norm_prob_states': core.grad_norm(ac.v.probing_states.parameters()),
               }
    if verbose:
        print(log_dict)

    return


# Initial evaluation
evaluate(ac)

# Loop over episodes
while statistics.total_ts < config['max_timesteps'] and statistics.episode < config['max_episodes']:
    start_time = time.perf_counter()
    train()
    statistics.total_time += time.perf_counter() - start_time





