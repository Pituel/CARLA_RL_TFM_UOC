

MODE = 1 # 1: CNN 2: VAE
TRAIN = False # True:Train False:Test

'''
Setting variables
'''

HOST = "localhost"
PORT = 2000
TIMEOUT = 30.0

CAR_NAME = 't2'
VISUAL_DISPLAY = True


RGB_CAMERA = 'sensor.camera.rgb'
SSC_CAMERA = 'sensor.camera.semantic_segmentation'

'''
Training hyperparametres and variables
'''

SEED = 0
BATCH_SIZE = 64
IM_WIDTH = 160
IM_HEIGHT = 80
GAMMA = 0.99
MEMORY_SIZE = 150000
EPISODES =  3500
BURN_IN = 1000


#VAE Bottleneck
LATENT_DIM = 95

DQN_LEARNING_RATE = 0.00005
EPSILON = 1.00
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

REPLACE_NETWORK = 500
UPDATE_FREQ = 5
DQN_CHECKPOINT_DIR_VAE = 'preTrained_models/VAE'
DQN_CHECKPOINT_DIR_CNN = 'preTrained_models/CNN'
MODEL = 'carla_dueling_dqn.pth'



