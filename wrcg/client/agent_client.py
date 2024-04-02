from model import CNNActionValue
import numpy as np
import torch

class Agent:
    def __init__(
            self,
            state_dim,
            action_dim,
            epsilon=0.8,
            epsilon_min=0.0,
            warmup_steps=0,
            update_interval=100,
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.warmup_steps = warmup_steps
        self.update_interval = update_interval

        self.network = CNNActionValue(state_dim[0], action_dim)
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        self.network.to(self.device)

        self.total_steps = 0
        self.epsilon_decay = (epsilon - epsilon_min) / 1e5

    @torch.no_grad()
    def act(self, x, training=True):
        self.network.train(training)
        if training and ((np.random.rand() < self.epsilon) or (self.total_steps < self.warmup_steps)):
            a = np.random.randint(0, self.action_dim)
        else:
            x = torch.from_numpy(x).float().unsqueeze(0).to(self.device) / 255.
            q = self.network(x)
            a = torch.argmax(q).item()
        return a

    def process(self, transition, socket):
        self.total_steps += 1
        s, a, r, s_prime, done = transition
        mixed = np.array([s, a, r, s_prime, done], dtype=object)

        try:
            socket.sendall(mixed)
        except Exception:
            pass

        self.epsilon -= self.epsilon_decay

        if self.total_steps > self.warmup_steps and self.total_steps % self.update_interval == 0:
            # t = time.time()
            self.update_network(socket)
            # print(time.time() - t)
            return True
        return False

    def update_network(self, socket):
        new_values = socket.recv()
        model_dict = self.network.state_dict()
        for (k, v), new_v in zip(model_dict.items(), new_values):
            model_dict[k] = torch.Tensor(new_v)
        self.network.load_state_dict(model_dict)





