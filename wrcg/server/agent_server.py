import time

from model import CNNActionValue
import torch.nn.functional as F
import torch
from buffer import ReplayBuffer
import numpy as np


class Agent:
    def __init__(
            self,
            state_dim,
            action_dim,
            lr=0.00025,
            epsilon=0.8,
            epsilon_min=0.0,
            gamma=0.99,
            batch_size=256,
            warmup_steps=0,
            buffer_size=int(2e5),
            update_interval=100,
            target_update_interval=10000,
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval
        self.update_interval = update_interval

        self.network = CNNActionValue(state_dim[0], action_dim)
        self.target_network = CNNActionValue(state_dim[0], action_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr)

        self.buffer = ReplayBuffer(state_dim, (1,), buffer_size)
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        self.network.to(self.device)
        self.target_network.to(self.device)

        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 1e5


    def learn(self):
        s, a, r, s_prime, terminated = map(lambda x: x.to(self.device), self.buffer.sample(self.batch_size))

        s /= 255.
        s_prime /= 255.

        next_q = self.target_network(s_prime).detach()
        current_q = self.network(s_prime).detach()
        td_target = r + (1. - terminated) * self.gamma * next_q.gather(1, torch.max(current_q, 1)[1].unsqueeze(1))
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

    def process(self, transition, conn):
        result = {}
        self.total_steps += 1
        self.buffer.update(*transition)

        if self.total_steps > self.warmup_steps and self.total_steps % self.update_interval == 0:
            result = self.learn()

            print(result)
            # t = time.time()
            self.push_network(conn)
            # print(time.time() - t)

        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        return result

    def load(self, steps, training=True):
        self.network.load_state_dict(torch.load('checkpoints/dqn_{}.pt'.format(steps)))
        self.target_network.load_state_dict(torch.load('checkpoints/dqn_{}.pt'.format(steps)))
        if training:
            self.buffer.load()
        self.epsilon = self.epsilon - self.epsilon_decay * steps
        self.total_steps = steps
        print('epsilon:', self.epsilon)

    def push_network(self, conn):
        d = self.network.state_dict()
        mixed = np.array([x.cpu().numpy() for x in list(d.values())], dtype=object)
        conn.sendall(mixed)


