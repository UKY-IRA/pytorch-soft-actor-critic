import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    '''a clipped-double DQN implementation of a mapped exploration problem (x, u) -> Q(x,u)'''
    def __init__(self, num_inputs, num_actions, hidden_dim, map_input=(1,50,90), device=torch.device("cpu")):
        super(QNetwork, self).__init__()
        self.num_inputs = num_inputs
        self.device = device
        # DQN1
        # map layers
        self.conv1 = nn.Conv2d(map_input[0], 64, 3) # in channels, out channels, kernel_dim
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.pool = nn.AvgPool2d(3,3) # window_size, stride
        conv_to_fc_size = self._get_conv_output(map_input)
        self.fc1 = nn.Linear(conv_to_fc_size, 2*hidden_dim)
        self.fc2 = nn.Linear(2*hidden_dim, hidden_dim)

        # plane layers
        self.fc3 = nn.Linear(num_inputs + num_actions, hidden_dim) # TODO: pass correct shape here
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)

        # combination layers
        self.fc6 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.fc8 = nn.Linear(int(hidden_dim/2), 1)

        # DQN2
        # map layers
        self.conv3 = nn.Conv2d(map_input[0], 64, 3) # in channels, out channels, kernel_dim
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.fc9 = nn.Linear(conv_to_fc_size, 2*hidden_dim)
        self.fc10 = nn.Linear(2*hidden_dim, hidden_dim)

        # plane layers
        self.fc11 = nn.Linear(num_inputs + num_actions, hidden_dim) # TODO: pass correct shape here
        self.fc12 = nn.Linear(hidden_dim, hidden_dim)
        self.fc13 = nn.Linear(hidden_dim, hidden_dim)

        # combination layers
        self.fc14 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc15 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.fc16 = nn.Linear(int(hidden_dim/2), 1)

        # dropout (for training)
        self.dropout = nn.Dropout(0.2)

        self.apply(weights_init_)

    def _get_conv_output(self, shape):
        batch_size = 1
        xin = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(xin)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        return x

    def forward(self, state, action):
        plane_state = torch.zeros(state.shape[0], 4).to(self.device)
        plane_state[:,0:3] = state[:,0,0]
        plane_state[:,3] = state[:,0,1,0]
        map_state = state[:,1:]
        map_state = map_state.permute(0,3,1,2) # reshape to (batch, chan, xdim, ydim)
        xu = torch.cat([plane_state, action], 1)

        # QDN 1
        x1 = self._forward_features(map_state)
        x1 = self.dropout(x1)
        x1 =  torch.flatten(x1, 1)
        x1 = F.leaky_relu(self.fc1(x1))
        x1 = F.leaky_relu(self.fc2(x1))

        x2 = F.leaky_relu(self.fc3(xu))
        x2 = F.leaky_relu(self.fc4(x2))
        x2 = F.leaky_relu(self.fc5(x2))

        x3 = torch.cat([x1,x2], 1)
        x3 = F.leaky_relu(self.fc6(x3))
        x3 = F.leaky_relu(self.fc7(x3))
        x3 = self.fc8(x3)

        # QDN 2
        x4 = self._forward_features(map_state)
        x4 = self.dropout(x4)
        x4 =  torch.flatten(x4, 1)
        x4 = F.leaky_relu(self.fc9(x4))
        x4 = F.leaky_relu(self.fc10(x4))

        x5 = F.leaky_relu(self.fc11(xu))
        x5 = F.leaky_relu(self.fc12(x5))
        x5 = F.leaky_relu(self.fc13(x5))

        x6 = torch.cat([x4,x5], 1)
        x6 = F.leaky_relu(self.fc14(x6))
        x6 = F.leaky_relu(self.fc15(x6))
        x6 = self.fc16(x6)

        return x3, x6

class GaussianPolicy(nn.Module):
    '''policy network to relate state to action stochastically x -> mu_u, sigma_u'''
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None, map_input=(1,50,90), device=torch.device("cpu")):
        super(GaussianPolicy, self).__init__()
        # map layers
        self.device = device
        self.conv1 = nn.Conv2d(map_input[0], 64, 3) # in channels, out channels, kernel_dim
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.pool = nn.AvgPool2d(3,3) # window_size, stride
        conv_to_fc_size = self._get_conv_output(map_input)
        self.fc1 = nn.Linear(conv_to_fc_size, 2*hidden_dim)
        self.fc2 = nn.Linear(2*hidden_dim, hidden_dim)

        # plane layers
        self.fc3 = nn.Linear(num_inputs, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)

        # combination layers
        self.fc6 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.mean_fc = nn.Linear(int(hidden_dim/2), 1)
        self.log_std_fc = nn.Linear(int(hidden_dim/2), 1)

        self.dropout = nn.Dropout(0.2)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def _get_conv_output(self, shape):
        batch_size = 1
        xin = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(xin)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        return x

    def forward(self, state):
        plane_state = torch.zeros(state.shape[0], 4).to(self.device)
        plane_state[:,0:3] = state[:,0,0]
        plane_state[:,3] = state[:,0,1,0]
        map_state = state[:,1:]
        map_state = map_state.permute(0,3,1,2) # reshape to (batch, chan, xdim, ydim)

        x1 = self._forward_features(map_state)
        x1 = self.dropout(x1)
        x1 =  torch.flatten(x1, 1)
        x1 = F.leaky_relu(self.fc1(x1))
        x1 = F.leaky_relu(self.fc2(x1))

        x2 = F.leaky_relu(self.fc3(plane_state))
        x2 = F.leaky_relu(self.fc4(x2))
        x2 = F.leaky_relu(self.fc5(x2))

        x3 = torch.cat([x1,x2], 1)
        x3 = F.leaky_relu(self.fc6(x3))
        x3 = F.leaky_relu(self.fc7(x3))
        mean = self.mean_fc(x3)
        log_std = self.log_std_fc(x3)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample() # reparameterization trick (mean + std*N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
