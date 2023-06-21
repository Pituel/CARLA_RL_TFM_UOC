import sys
import time
import random
import numpy as np
import torch
from simulation.connection import ClientConnection
from simulation.environment import CarlaEnvironment
from networks.off_policy.ddqn.agent import DQNAgent
from encoder_init import EncodeState
from settings import *



def runner():

    town = "Town02"

    #Seeding to reproduce the results 
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # setup the conecction with Server.
    try:
        client, world = ClientConnection(town).setup()
    except:
        ConnectionRefusedError

    # Initialize environment, encoder and agent
    env = CarlaEnvironment(client, world)
    encode = EncodeState(LATENT_DIM)
    agent = DQNAgent(env, encode)

    try:
        time.sleep(1)

        if TRAIN:

            agent.train()

            print("Terminating the run.")
            sys.exit()
        else:
            agent.test()

            print("Terminating the run.")

            sys.exit()

    finally:
        sys.exit()


if __name__ == "__main__":
    try:    
        runner()

    except KeyboardInterrupt:
        sys.exit()
    finally:
        print('\nExit')
