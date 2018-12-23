import numpy as np
import torch
import torch.nn.functional as F


class Memory:
    def __init__(self, discount):
        self.eps_obv = []
        self.eps_a = []
        self.eps_r = []
        self.discount = discount

    def reset(self):
        self.eps_obv = []
        self.eps_a = []
        self.eps_r = []

    def save(self, o, a, r):
        self.eps_a.append(a)
        self.eps_obv.append(o)
        self.eps_r.append(r)

    def _get_discount_reward(self):
        rewards = np.zeros_like(self.eps_r)
        pre_r = 0
        for i in reversed(range(0, len(rewards))):
            pre_r = self.eps_r[i] + pre_r * self.discount
            rewards[i] = pre_r

        # normalization
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards)
        return rewards

    def get_sar(self):
        norm_rewards = self._get_discount_reward()
        return self.eps_obv, self.eps_a, self.eps_r, norm_rewards


class Network(torch.nn.Module):
    def __init__(self, n_features, n_actions):
        super(Network, self).__init__()
        self.layer1 = torch.nn.Linear(n_features, 10)
        self.layer2 = torch.nn.Linear(10, n_actions)
        torch.nn.init.normal_(self.layer1.weight, 0, 0.3)
        torch.nn.init.normal_(self.layer2.weight, 0, 0.3)
        torch.nn.init.constant_(self.layer1.bias, 0.1)
        torch.nn.init.constant_(self.layer2.bias, 0.1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.Tanh()(x)
        x = self.layer2(x)
        return x


class PolicyGradient:
    def __init__(self, n_features, n_actions, discount, lr):
        self.discount = discount
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr

        self.memory = Memory(self.discount)
        self.actor = Network(self.n_features, self.n_actions)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

    def choose_action(self, s):
        self.actor.eval()
        s = torch.Tensor(s)
        pro_a = self.actor(s).softmax(-1)
        return np.random.choice(range(self.n_actions), p=pro_a.detach().numpy())

    def learn(self):
        self.actor.train()
        eps_s, eps_a, eps_r, eps_return = self.memory.get_sar()
        self.memory.reset()
        pro_a = self.actor(torch.Tensor(eps_s)).softmax(-1)
        log = -torch.log(pro_a)
        hot = torch.zeros(len(eps_s), self.n_actions).scatter(1, torch.LongTensor(eps_a).view(len(eps_a), 1), 1)
        neg_log_prob = torch.sum(log * hot, -1)

        loss = (neg_log_prob * torch.Tensor(eps_return)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return eps_return

    def save_transition(self, o, a, r):
        self.memory.save(o, a, r)
