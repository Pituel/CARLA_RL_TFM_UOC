import sys
import torch
from autoencoder.encoder import VariationalEncoder
from settings import MODE

class EncodeState():
    def __init__(self, latent_dim):
        # Initialize the encoder
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.conv_encoder = VariationalEncoder(self.latent_dim).to(self.device)
            self.conv_encoder.load()
            self.conv_encoder.eval()

            for params in self.conv_encoder.parameters():
                params.requires_grad = False
        except Exception as e:
            print(e)
            print('Encoder could not be initialized.')
            sys.exit()
    
    def process(self, observation):
        """
        Process the observation and prepare it for the dueling dqn network.
        The network need observations as array.
        But process transformation ar doing over tensors because it is faster.


        Args:
            observation: current observation.

        Returns:
            observations.
        """

        if MODE == 1:

            # Normalize observation and convert to tensor.
            image_obs = torch.tensor(observation[0]/255, dtype=torch.float).to(self.device)
            # Permute the image to prepare for pytorch.
            image_obs = image_obs.permute(2,1,0)
            # Convert tensor into numpy array
            image_obs = image_obs.cpu().detach().numpy()
            
            driving_obs = torch.tensor(observation[1], dtype=torch.float).to(self.device)
            driving_obs = driving_obs.cpu().detach().numpy()

            # image_obs = observation[0]/255
            # image_obs = np.transpose(observation[0]/255, (2,1,0))

            # driving_obs = observation[1]

            return  image_obs, driving_obs

        else:
            # Normalize observation and convert to tensor.
            image_obs = torch.tensor(observation[0]/255, dtype=torch.float).to(self.device)
            # Prepare for pytorch.
            image_obs = image_obs.unsqueeze(0)
            image_obs = image_obs.permute(0,3,2,1)
            # Preprocess the image with the encoder
            image_obs = self.conv_encoder(image_obs)
            # Concat image and driving observation.
            driving_obs = torch.tensor(observation[1], dtype=torch.float).to(self.device)
            observation = torch.cat((image_obs.view(-1), driving_obs), -1)

            # Convert tensor into numpy array
            observation = observation.cpu().detach().numpy()

            return observation
        
        