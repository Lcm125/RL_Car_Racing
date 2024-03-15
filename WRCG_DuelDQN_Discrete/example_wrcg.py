import numpy as np
import torch
from model import CNNActionValue
import torch.nn.functional as F
import cv2
from env import Env


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.s = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.a = np.zeros((max_size, *action_dim), dtype=np.int64)
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.s_prime = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def update(self, s, a, r, s_prime, terminated):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_prime[self.ptr] = s_prime
        self.terminated[self.ptr] = terminated

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.s[ind]),
            torch.FloatTensor(self.a[ind]),
            torch.FloatTensor(self.r[ind]),
            torch.FloatTensor(self.s_prime[ind]),
            torch.FloatTensor(self.terminated[ind]),
        )

    def save(self):
        print('saving buffer')
        np.save('saved_paras/s.npy', self.s)
        np.save('saved_paras/a.npy', self.a)
        np.save('saved_paras/r.npy', self.r)
        np.save('saved_paras/s_prime.npy', self.s_prime)
        np.save('saved_paras/terminated.npy', self.terminated)
        np.save('saved_paras/para.npy', np.array([self.ptr, self.size, self.max_size]))

    def load(self):
        print('loading buffer')
        self.s = np.load('saved_paras/s.npy')
        self.a = np.load('saved_paras/a.npy')
        self.r = np.load('saved_paras/r.npy')
        self.s_prime = np.load('saved_paras/s_prime.npy')
        self.terminated = np.load('saved_paras/terminated.npy')
        self.ptr, self.size, self.max_size = np.load('saved_paras/para.npy')


class Agent:
    def __init__(
            self,
            state_dim,
            action_dim,
            lr=0.00025,
            epsilon=0.5,
            epsilon_min=0.0,
            gamma=0.99,
            batch_size=128,
            warmup_steps=5000,
            buffer_size=int(5e4),
            target_update_interval=10000,
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval

        self.network = CNNActionValue(state_dim[0], action_dim)
        self.target_network = CNNActionValue(state_dim[0], action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr)

        self.buffer = ReplayBuffer(state_dim, (1,), buffer_size)
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
        self.network.to(self.device)
        self.target_network.to(self.device)

        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 1e5

    @torch.no_grad()
    def act(self, x, training=True):
        self.network.train(training)
        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
            a = np.random.randint(0, self.action_dim)
        else:
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)
            q = self.network(x)
            a = torch.argmax(q).item()
        return a

    def learn(self):
        s, a, r, s_prime, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))

        next_q = self.target_network(s_prime).detach()
        current_q = self.network(s_prime).detach()
        td_target = r + (1. - terminated) * self.gamma * next_q.gather(1, torch.max(current_q, 1)[1].unsqueeze(1))
        # td_target = r + (1. - terminated) * self.gamma * next_q.max(dim=1, keepdim=True).values
        loss = F.mse_loss(self.network(s).gather(1, a.long()), td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print('{:<10} {:<10.4f}'.format(self.total_steps, loss.item()))
        result = {
            'total_steps': self.total_steps,
            'value_loss': loss.item()
        }
        return result

    def process(self, transition):
        result = {}
        self.total_steps += 1
        self.buffer.update(*transition)

        if self.total_steps > self.warmup_steps:
            result = self.learn()
            # print(result)

        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        self.epsilon -= self.epsilon_decay
        return result

    def load(self, steps):
        self.network.load_state_dict(torch.load('dqn_{}.pt'.format(steps)))
        self.target_network.load_state_dict(torch.load('dqn_{}.pt'.format(steps)))
        self.buffer.load()
        self.epsilon = self.epsilon - self.epsilon_decay * steps
        self.total_steps = steps
        print('epsilon:', self.epsilon)


def evaluate(wrcg_env, agent):
    scores = 0
    for i in range(5):
        ret = 0
        s = wrcg_env.reset_game()
        while True:
            a = agent.act(s, training=False)
            s_prime, r, done, err = wrcg_env.step(a)
            if done or err:
                break
            s = s_prime
            ret += r
        scores += ret
    return np.round(scores / 5, 4)


def continous_evaluate(wrcg_env, agent):
    ret = 0
    reset_num = 0
    s = wrcg_env.reset_game()
    while True:
        a = agent.act(s, training=False)
        s_prime, r, done, err = wrcg_env.step(a)
        if done or err:
            s = wrcg_env.reset_car()
            reset_num += 1
            continue
        s = s_prime
        ret += r
    return ret, reset_num

def train(wrcg_env, agent):
    # s = wrcg_env.reset_game()
    # wrcg_env.pause_game()
    s = wrcg_env.reset_car()
    max_steps = int(2e6)
    eval_interval = 10000

    while True:
        a = agent.act(s)
        s_prime, r, done, err = wrcg_env.step(a)
        if err == 3:
            s = wrcg_env.reset_game()
            continue
        elif err != 0:
            s = wrcg_env.reset_car()
            continue
        result = agent.process(
            (s, a, r, s_prime, done))  # You can track q-losses over training from `result` variable.

        s = s_prime
        if done:
            s = wrcg_env.reset_car()

        if agent.total_steps % eval_interval == 0:
            torch.save(agent.network.state_dict(), 'dqn_{}.pt'.format(agent.total_steps))
            agent.buffer.save()

            ret = evaluate(wrcg_env, agent)
            print(agent.total_steps, ret)
            with open('log.txt', 'a+') as f:
                f.write('{} {}\n'.format(agent.total_steps, ret))

        if agent.total_steps > max_steps:
            break



if __name__ == '__main__':
    wrcg_env = Env(0x00130E64)
    state_dim = (4, 112, 112)
    action_dim = 4
    agent = Agent(state_dim, action_dim)

    agent.load(80000)

    # train(wrcg_env, agent)
    # print(evaluate(wrcg_env, agent))
    print(continous_evaluate(wrcg_env, agent))
