import numpy as np
from settings import BATCH_SIZE
from collections import namedtuple, deque



class ReplayBuffer(object):
    def __init__(self, memory_size=50000, burn_in=1000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.buffer = namedtuple('Buffer',
            field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size,
                                   replace=False)
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def save_transition(self, state, action, reward, next_state, done):
        self.replay_memory.append(
            self.buffer(state, action, reward, next_state, done))

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in
