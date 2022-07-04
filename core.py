import torch
import torch.nn as nn
from gym.spaces import Box, Discrete
import numpy as np
import random


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]

    return nn.Sequential(*layers)


class VirtualModule:
    def __init__(self):
        self._parameter_shapes = self.get_parameter_shapes()

        self._num_parameters = 0
        for shape in self.parameter_shapes.values():
            numel = np.prod(shape)
            self._num_parameters += numel

    def get_parameter_shapes(self):
        # return an OrderedDict with the parameter names and their shape
        return NotImplementedError

    def parameter_initialization(self, num_instances):
        factor = 1 / ((self.num_parameters / 100) ** 0.5)
        initializations = []
        for i in range(num_instances):
            p = []
            for key, shape in self.parameter_shapes.items():
                p.append(torch.randn(shape).view(-1) * factor)
            p = torch.cat(p, dim=0)
            initializations.append(p)
        return initializations

    def split_parameters(self, p):
        if len(p.shape) == 1:
            batch_size = []
        else:
            batch_size = [p.shape[0]]
        pointer = 0
        parameters = []
        for shape in self.parameter_shapes.values():
            numel = np.prod(shape)
            x = p[..., pointer:pointer + numel].view(*(batch_size + list(shape)))
            parameters.append(x)
            pointer += numel
        return parameters

    @property
    def parameter_shapes(self):
        return self._parameter_shapes

    @property
    def num_parameters(self):
        return self._num_parameters


class VirtualModuleWrapper(torch.nn.Module):
    # Allows treating a virtual module as a normal pytorch module (train with standard optimizers etc.)
    def __init__(self, virtual_module):
        super().__init__()
        self.virtual_module = virtual_module
        self.virtual_parameters = torch.nn.Parameter(self.virtual_module.parameter_initialization(1)[0])

    def forward(self, x):
        output = self.virtual_module.forward(x, self.virtual_parameters)
        return output


def linear_multi_parameter(input, weight, bias=None):
    """
    n: input batch dimension
    m: parameter batch dimension (not obligatory)
    i: input feature dimension
    o: output feature dimension
    :param input: n x (m x) i
    :param weight: (m x) o x i
    :param bias:  (m x) o
    :return: n x (m x) o
    """

    if len(weight.shape) == 2:
        # no parameter batch dimension
        x = torch.einsum('ni,oi->no', input, weight)
    elif len(input.shape) == 3:
        # parameter batch dimension for input and weights
        x = torch.einsum('nmi,moi->nmo', input, weight)
    else:
        # no parameter dimension batch for input
        x = torch.einsum('ni,moi->nmo', input, weight)
    if bias is not None:
        x = x + bias.unsqueeze(0)
    return x


class VirtualMLP(VirtualModule):
    def __init__(self, layer_sizes, nonlinearity='tanh', output_activation='linear'):
        self.layer_sizes = layer_sizes

        if nonlinearity == 'tanh':
            self.nonlinearity = torch.tanh
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = torch.sigmoid
        else:
            self.nonlinearity = torch.relu

        if output_activation == 'linear':
            self.output_activation = None
        elif output_activation == 'sigmoid':
            self.output_activation = torch.sigmoid
        elif output_activation == 'tanh':
            self.output_activation = torch.tanh
        elif output_activation == 'softmax':
            self.output_activation = lambda x: torch.softmax(x, dim=-1)

        super(VirtualMLP, self).__init__()

    def get_parameter_shapes(self):
        parameter_shapes = OrderedDict()
        for i in range(1, len(self.layer_sizes)):
            parameter_shapes['w' + str(i)] = (self.layer_sizes[i], self.layer_sizes[i-1])
            parameter_shapes['wb' + str(i)] = ( self.layer_sizes[i],)

        return parameter_shapes

    def forward(self, input, parameters, callback_func=None):
        # input_sequence: input_batch x (parameter_batch x) input_size
        # parameters: (parameter_batch x) num_params
        # return: input_batch x (parameter_batch x) output_size

        p = self.split_parameters(parameters)
        #print("split parameters", p)
        num_layers = len(self.layer_sizes) - 1
        x = input
        for l in range(0, num_layers):
            w = p[l*2]
            a = linear_multi_parameter(x, w, bias=p[l*2 + 1])
            if l < num_layers - 1:
                x = self.nonlinearity(a)
                if callback_func is not None:
                    callback_func(x, l)
            else:
                x = a if self.output_activation is None else self.output_activation(a)
        return x

    def parameter_initialization(self, num_instances, bias_var=0.):
        initializations = []
        for i in range(num_instances):
            p = []
            for i in range(1, len(self.layer_sizes)):
                w = torch.empty(self.parameter_shapes['w' + str(i)])
                torch.nn.init.xavier_normal_(w)
                p.append(w.view(-1))
            if self.bias:
                for i in range(1, len(self.layer_sizes)):
                    b = torch.empty(self.parameter_shapes['wb' + str(i)])
                    if bias_var == 0:
                        torch.nn.init.zeros_(b)
                    else:
                        torch.nn.init.normal_(b, std=bias_var**0.5)
                    p.append(b.view(-1))
            p = torch.cat(p, dim=0)
            initializations.append(p)
        return initializations


class VirtualMLPPolicy(VirtualMLP):
    def __init__(self, layer_sizes, bias=True, act_lim=1, nonlinearity='tanh'):
        super().__init__(layer_sizes=layer_sizes, nonlinearity=nonlinearity, output_activation='tanh')
        self.act_lim = act_lim

    def forward(self, input, parameters, callback_func=None):
        x = super().forward(input, parameters, callback_func)
        x = x * self.act_lim
        return x


class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit, act_clipping):
        super().__init__()
        self.act_clipping = act_clipping
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi_net = mlp(pi_sizes, activation)
        self.act_limit = act_limit
    def forward(self, obs):
        return (self.pi_net(obs))

    def get_probing_action(self, obs):
        if self.act_clipping:
            return (self.pi_net(obs))
        else:
            return (torch.tanh(self.pi_net(obs))*self.act_limit)


class PSSVF(nn.Module):
    def __init__(self, obs_dim, num_probing_states, parameter_space_dim, hidden_sizes, activation):
        super().__init__()

        self.probing_states = nn.ParameterList([nn.Parameter(torch.rand([obs_dim]))
                                       for _ in range(num_probing_states)])
        self.v_net = mlp([parameter_space_dim] + list(hidden_sizes) + [1], activation)



    def forward(self, parameters, use_virtual_module=False, virtual_module=None):
        prob_sates = torch.stack([torch.nn.utils.parameters_to_vector(state) for state in self.probing_states])
        if use_virtual_module:
            actions = virtual_module.forward(prob_sates, parameters).transpose(0,1).reshape(parameters.shape[0], -1)
        else:
            actions = [torch.stack([prog.pi.get_probing_action(prob_sates)]).squeeze() for prog in parameters]
            actions = torch.stack(actions, dim=0).reshape([len(parameters), -1])
        return torch.squeeze(self.v_net(actions), -1)



from collections import OrderedDict


class MLPActorCritic(nn.Module):

    def __init__(self, algo, observation_space, action_space, n_probing_states,
                 hidden_sizes_actor, activation_policy, activation_vf, hidden_sizes_critic, device, critic,
                 deterministic_actor, act_clipping, act_noise):
        super().__init__()

        self.device = device
        self.act_noise = act_noise
        self.act_clipping = act_clipping
        self.algo = algo
        self.deterministic_actor = deterministic_actor
        obs_dim = observation_space.shape[0]
        if isinstance(action_space, Box):
            self.act_dim = action_space.shape[0]
            self.act_limit = action_space.high[0]
        elif isinstance(action_space, Discrete):
            self.act_dim = action_space.n

        self.pi = MLPActor(obs_dim, self.act_dim, hidden_sizes_actor,
                           activation_policy, self.act_limit, self.act_clipping).to(device=device)

        if critic:
                # mean and sd of gaussian
            self.parameters_dim = n_probing_states * self.act_dim

            self.v = PSSVF(obs_dim, n_probing_states, self.parameters_dim, hidden_sizes_critic, activation_vf).to(device=device)


    def act(self, obs):
        with torch.no_grad():
            a = self.pi(obs)
            a += self.act_noise * torch.as_tensor(np.random.randn(self.act_dim)).to(self.device)
            a = (torch.tanh(a)*self.act_limit).to(device='cpu').numpy()
        return a


class Statistics(object):

    def __init__(self, obs_dim):
        super().__init__()

        self.total_ts = 0
        self.episode = 0
        self.len_episode = 0
        self.rew_shaped_eval = 0
        self.rew_eval = 0
        self.rewards = []
        self.last_rewards = []
        self.position = 0
        self.n = 0
        self.mean = torch.zeros(obs_dim)
        self.mean_diff = torch.zeros(obs_dim)
        self.std = torch.zeros(obs_dim)
        self.sim_time = 0
        self.up_policy_time = 0
        self.up_v_time = 0
        self.total_time = 0


    def push_obs(self, obs):
        self.n += 1.
        last_mean = self.mean
        self.mean += (obs - self.mean) / self.n
        self.mean_diff += (obs - last_mean) * (obs - self.mean)
        var = self.mean_diff / (self.n - 1) if self.n > 1 else np.square(self.mean)
        self.std = np.sqrt(var)
        return

    def push_rew(self, rew):
        if len(self.last_rewards) < 20:
            self.last_rewards.append(rew)
        else:
            self.last_rewards[self.position] = rew
            self.position = (self.position + 1) % 20
        self.rewards.append(rew)

    def normalize(self, obs):
        return (obs - self.mean) / (self.std + 1e-8)

    def denormalize(self, obs):
        return obs * (self.std + 1e-8) + self.mean

class Buffer(object):
    def __init__(self, size_buffer, scale=1.0):
        self.history = []
        self.size_buffer = size_buffer
        self.weights = []
        self.scale = scale

    def sample_replay(self, batch_size, weighted_sampling=False):

        if weighted_sampling:
            self.weights = list(np.reciprocal(np.arange(1, len(self.history)+1, dtype=float)))
            self.weights.reverse()
            self.weights = np.array(self.weights) ** self.scale
            self.weights = list(self.weights)
            sampled_hist = random.choices(self.history, weights=self.weights, k=min(int(batch_size), len(self.history)))
        else:
            sampled_hist = random.choices(self.history, k=min(int(batch_size), len(self.history)))
        if len(self.history) > self.size_buffer:
            self.history.pop(0)
        return sampled_hist



def grad_norm(parameters):
    # Compute the norm of the gradient
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(2)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def norm(parameters):
    # Compute the norm of the weights of a model
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(2)
    total_norm = 0
    for p in parameters:
        param_norm = p.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm