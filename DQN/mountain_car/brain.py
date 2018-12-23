import torch
from collections import namedtuple
import torch.optim as optim
import torch.nn.functional as F
import random
from tensorboardX import SummaryWriter


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DNN(torch.nn.Module):

    def __init__(self, n_features, n_actions):
        super(DNN, self).__init__()
        self.layer1 = torch.nn.Linear(n_features, 10)
        self.layer2 = torch.nn.Linear(10, n_actions)
        torch.nn.init.normal_(self.layer1.weight, 0, 0.03)
        torch.nn.init.normal_(self.layer2.weight, 0, 0.03)
        torch.nn.init.constant_(self.layer1.bias, 0.1)
        torch.nn.init.constant_(self.layer2.bias, 0.1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.ReLU()(x)
        x = self.layer2(x)
        return x


class DQNBrain:

    def __init__(
            self,
            n_features,
            n_actions,
            gamma,
            learning_rate,
            eps_end,
            eps_start,
            eps_decay,
            memory_size,
            batch_size,
            model_dir="./model",
            log_dir="./log",
            device="cpu",
            target_update_freq=300,
    ):
        self.n_features = n_features
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.eps_end = eps_end
        self.lr = learning_rate
        self.eps_threshold = self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.device = device
        self.target_update_freq = target_update_freq

        self.n_episode = 0

        self.global_step_num = 0

        self.memory = ReplayMemory(memory_size)

        self.target_net = DNN(self.n_features, self.n_actions)
        self.policy_net = DNN(self.n_features, self.n_actions)

        self.replace_target_params()
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.lr)

        self.summary_writer = SummaryWriter()

    def choose_action(self, state):
        state = torch.Tensor(state).view(1, self.n_features)
        sample = random.random()

        self.global_step_num += 1

        if self.global_step_num % self.target_update_freq == 0:
            self.replace_target_params()

        self.summary_writer.add_scalar('eps_threshold', self.eps_threshold, self.global_step_num)

        if sample > self.eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1).item()
        else:
            # return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long).item()
            return random.randrange(self.n_actions)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).long()
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        q_table = self.policy_net(state_batch)
        state_action_values = q_table.gather(1, action_batch.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        self.summary_writer.add_scalar('q_value', state_action_values.sum().item(), self.global_step_num)

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.summary_writer.add_scalar('loss', loss.item(), self.global_step_num)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save_transition(self, s, a, r, s_):
        s = torch.Tensor(s, device=self.device).unsqueeze(0)
        s_ = torch.Tensor(s_, device=self.device).unsqueeze(0)
        r = torch.Tensor([r], device=self.device)
        a = torch.Tensor([a], device=self.device)
        self.memory.push(s, a, s_, r)

    def replace_target_params(self):
        print("replace target params at {} step".format(self.global_step_num))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # def new_episode(self, i_episode, step):
    #     self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
    #                                    math.exp(-1. * i_episode / self.eps_decay)
    #     self.n_episode = i_episode
    #
    #     # Summary the q_value
    #     transitions = self.memory.sample(self.batch_size)
    #     batch = Transition(*zip(*transitions))
    #     state_batch = torch.cat(batch.state)
    #     action_batch = torch.cat(batch.action).long()
    #
    #     q_table = self.policy_net(state_batch)
    #     state_action_values = q_table.gather(1, action_batch.unsqueeze(1))
    #     self.summary_writer.add_scalar("q_value", state_action_values.sum().item(), i_episode)
    #
    #     self.summary_writer.add_scalar("step", step, i_episode)
