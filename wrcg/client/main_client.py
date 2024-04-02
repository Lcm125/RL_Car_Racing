from env import Env
from agent_client import Agent
from numpysocket import NumpySocket
import numpy as np
import time


def evaluate(wrcg_env, agent):
    ret = 0
    s = wrcg_env.reset_game()
    for i in range(1000):
        a = agent.act(s, training=False)
        s_prime, r, done, err = wrcg_env.step(a)
        if done or err == 1 or err == 2:
            s = wrcg_env.reset_car()
            continue
        if err == 3:
            s = wrcg_env.reset_game()
            continue
        s = s_prime
        ret += r
    return np.round(ret, 4)


def train(wrcg_env, agent, ip, port):
    # s = wrcg_env.reset_game()
    # wrcg_env.pause_game()
    with NumpySocket() as socket:
        socket.connect((ip, port))
        s = wrcg_env.reset_car()
        max_steps = int(1e6)
        # t = time.time()
        while True:
            a = agent.act(s)
            s_prime, r, done, err = wrcg_env.step(a)
            if err == 3:
                s = wrcg_env.reset_game()
                continue
            elif err != 0:
                s = wrcg_env.reset_car()
                continue

            reset = agent.process(
                (s, a, r, s_prime, done), socket)  # You can track q-losses over training from `result` variable.
            # t2 = time.time()
            # print(t2 - t)
            # t = t2
            s = s_prime
            if done or reset:
                s = wrcg_env.reset_car()

            if agent.total_steps > max_steps:
                break


if __name__ == '__main__':
    wrcg_env = Env(0x000C0984)
    state_dim = (4, 112, 112)
    action_dim = 4
    agent = Agent(state_dim, action_dim)
    train(wrcg_env, agent, "10.23.149.92", 9999)
    # print(evaluate(wrcg_env, agent))
    # print(continous_evaluate(wrcg_env, agent))