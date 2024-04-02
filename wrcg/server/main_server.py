import time
from numpysocket import NumpySocket
from agent_server import Agent
from tqdm import tqdm


if __name__ == '__main__':
    state_dim = (4, 112, 112)
    action_dim = 4
    agent = Agent(state_dim, action_dim)

    with NumpySocket() as socket:
        socket.bind(("", 9999))
        socket.listen()
        conn, addr = socket.accept()
        print(f"connected: {addr}")
        t = time.time()
        with tqdm(total=100, ncols=100, leave=False) as pbar:
            pbar.set_description("training")
            while conn:
                data = conn.recv()
                if len(data) == 0:
                    break
                s, a, r, s_prime, done = data
                agent.process((s, a, r, s_prime, done), conn)
                t2 = time.time()
                # print(t2 - t)
                t = t2
                pbar.update(100 / agent.total_steps)
        print(f"disconnected: {addr}")
