import torch
import torch.nn as nn
from A3C.utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from A3C.shared_adam import SharedAdam
import gym
import os
from tensorboardX import SummaryWriter

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
MAX_EP = 4000

MODEL_FILE = 'A3C/cart_pole/models/model.pt'
MODEL_SAVE_FREQ = 100

env_name = 'CartPole-v0'
# env_name = 'MountainCar-v0'
env = gym.make(env_name)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 100)
        self.pi2 = nn.Linear(100, a_dim)
        self.v1 = nn.Linear(s_dim, 100)
        self.v2 = nn.Linear(100, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])

    def forward(self, x):
        pi1 = F.relu(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = F.relu(self.v1(x))
        values = self.v2(v1)
        return logits, values


class Agent:
    def __init__(self, s_dim, a_dim):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.net = Net(s_dim, a_dim)
        # load
        try:
            self.net.load_state_dict(torch.load(MODEL_FILE))
        except Exception as e:
            print(e)
        self.distribution = torch.distributions.Categorical

    def choose_action(self, s):
        self.net.eval()
        logits, _ = self.net(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.net.train()
        logits, values = self.net(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss

    def get_net(self):
        return self.net


class Worker(mp.Process):
    def __init__(self, g_agent, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.g_agent, self.opt = g_agent, opt
        self.l_agent = Agent(N_S, N_A)           # local agent
        self.env = gym.make(env_name).unwrapped

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                if self.name == 'w0':
                    self.env.render()
                a = self.l_agent.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a)
                if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.l_agent, self.g_agent, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)


if __name__ == "__main__":
    writer = SummaryWriter()

    g_agent = Agent(N_S, N_A)        # global network
    gnet = g_agent.get_net()
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=0.0001)      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(g_agent, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
            print(global_ep.value, r)
            writer.add_scalar('value', r, global_step=global_ep.value)

            if global_ep.value % MODEL_SAVE_FREQ == 0:
                torch.save(gnet.state_dict(), MODEL_FILE)

        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
