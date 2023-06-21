import torch
import numpy as np
from networks.off_policy.ddqn.dueling_dqn import DuelingDQnetwork
from networks.off_policy.ddqn.dueling_dqn_vae import DuelingDQnetworkVAE
from networks.off_policy.replay_buffer import ReplayBuffer
from settings import *
from copy import deepcopy
import time
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker



class DQNAgent(object):

    def __init__(self, env, encode):
        # Initialize all the variables
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_end = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY
        self.mem_size = MEMORY_SIZE
        self.batch_size = BATCH_SIZE
        self.encode = encode
        self.replay_buffer = ReplayBuffer(MEMORY_SIZE, BURN_IN)
        # Better performance for this environment
        n_actions = 5
        if MODE == 1:
            # Early stopping for training.
            self.limit = 1350
            self.q_network = DuelingDQnetwork(n_actions, MODEL)
        else:
            self.limit = 800
            self.q_network = DuelingDQnetworkVAE(n_actions, MODEL)
        self.action_space = np.arange(n_actions)
        self.target_network = deepcopy(self.q_network)
        self.env = env
        self.cumulative_score = 0
        self.scores = []
        self.mean_scores = []
        self.training_loss = []
        self.epsilon_ev = []
        self.update_loss = []
        self.observation = self.env.reset()
        self.observation = self.encode.process(self.observation)
        self.step_episode_count = 0
        self.max_score = 0
        self.step_count = 0
        self.current_ep_reward = 0

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def save_model(self):
        """
        Save the model
        """
        self.q_network.save_checkpoint()

    def load_model(self):
        """
        Load the model
        """
        self.q_network.load_checkpoint()
    
    def take_step(self, mode='train'):
        """
        Take an action.
        Take a step with this action and get new observation, reward and done info.
        Save transition in buffer.
        Reset the environment if episode is done.

        Args:
            mode: train or explore.

        Returns:
            done: info about the episode is done or not.
        """
        if mode == 'explore':
            action = np.random.choice(self.action_space)
        else:
            action = self.q_network.get_action(self.observation, self.epsilon)
            self.step_count += 1

        
        new_observation, reward, done, info = self.env.step(action)
        new_observation = self.encode.process(new_observation)
   
        self.current_ep_reward += reward

        # Save experience in buffer
        self.replay_buffer.save_transition(self.observation, action, reward, new_observation, done)
        self.observation = new_observation
        
        if done:
            self.observation = self.env.reset()
            self.observation = self.encode.process(self.observation)

        return done
    
    def calculate_loss(self, batch):
        """
        Calculate loss throught Bellman equation.

        Args:
            batch: size of the batch.

        Returns:
            loss
        """
        states, actions, rewards, next_states, dones= [i for i in batch]


        rewards_vals = torch.FloatTensor(rewards).to(device=self.device).reshape(-1,1)
        actions_vals = torch.LongTensor(np.array(actions)).reshape(-1,1).to(
            device=self.device)
        dones_t = torch.BoolTensor(dones).to(device=self.device)

        # Get Q-values from main network
        qvals = torch.gather(self.q_network.get_qvals(states), 1, actions_vals)

        #DQN update
        next_actions = torch.max(self.q_network.get_qvals(next_states), dim=-1)[1]
        if self.device == 'cuda':
            next_actions_vals = next_actions.reshape(-1,1).to(device=self.device)
        else:
            next_actions = torch.LongTensor(next_actions).reshape(-1,1).to(device=self.device)
            
        # Get Q-values from target network
        target_qvals = self.target_network.get_qvals(next_states)
        qvals_next = torch.gather(target_qvals, 1, next_actions_vals).detach()

        qvals_next[dones_t] = 0 # 0 en estados terminales

        # Bellman equation
        expected_qvals = self.gamma * qvals_next + rewards_vals

        # Calculate the loss
        loss = torch.nn.MSELoss()(qvals, expected_qvals.reshape(-1,1))
        
        return loss



    def update(self):
        """
        Update the network
        """
        # Set the gradients to zero
        self.q_network.optimizer.zero_grad()
        # Select a batch from buffer
        batch = self.replay_buffer.sample_batch(batch_size=self.batch_size)
        # Calculate loss
        loss = self.calculate_loss(batch) 
        # Compute the gradients. 
        loss.backward()
        # Apply gradients to main network.
        self.q_network.optimizer.step()
        # Save loss
        if self.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())


    def train(self):
        """
        Train the network
        """
        start = time.time()

        # Start the Codecarbon tracker.
        tracker = EmissionsTracker()
        tracker.start()
        # Fill the buffer with random experience.
        print("Filling replay buffer...")
        while self.replay_buffer.burn_in_capacity() < 1:
            self.take_step(mode='explore')

        episode = 0
        
        training = True
        print("Training...")
        while training:
            self.current_ep_reward = 0
            print('Episode: ', episode, ', Epsilon:  {:.3f}'.format(self.epsilon), ', ', end="")

            epdone = False
            while epdone == False:
                epdone = self.take_step(mode='train')
                # Update the main network
                if self.step_count % UPDATE_FREQ == 0:
                    self.update()

                # Sincornize main and target networks
                if self.step_count % REPLACE_NETWORK == 0:
                    self.target_network.load_state_dict(
                        self.q_network.state_dict())
                    
                self.step_episode_count += 1

                if epdone:
                    # Update of variables if episode is done.
                    episode += 1

                    self.scores.append(self.current_ep_reward)
                    self.cumulative_score = np.mean(self.scores[-50:])
                    self.mean_scores.append(self.cumulative_score)
                    mean_loss = np.mean(self.update_loss)
                    self.training_loss.append(mean_loss)
                    self.epsilon_ev.append(self.epsilon)
                    
                    # Print the info.
                    print('Reward:  {:.2f}'.format(self.current_ep_reward), ', Average Reward:  {:.2f}'.format(self.cumulative_score), ', Training Loss:  {:.2f}'.format(mean_loss), ', Steps: {}'.format(self.step_episode_count))

                    # Restart variables
                    self.update_loss = []
                    self.step_episode_count = 0


                    if episode >= 10 and episode % 100 == 0:
                        
                        # Save the plots each 100 episdodes to see the results if the training fail.
                        plt.close('all')
                        self.plot_rewards(self.scores, self.mean_scores)
                        self.plot_loss(self.training_loss)
                        self.plot_epsilon(self.epsilon_ev)

                    if episode >= EPISODES or self.cumulative_score >= self.limit:
                        # End time counter and print the results.
                        end = time.time()
                        print("Training time: {} minutes".format(round((end-start)/60,2)))

                        # Start the Codecarbon tracker and print the results.
                        emissions: float = tracker.stop()
                        print(emissions)

                        # Flag to end the training.
                        training = False

                        # Save the model
                        self.save_model() 
                        
                        # Save the plots
                        plt.close('all')
                        self.plot_rewards(self.scores, self.mean_scores)
                        self.plot_loss(self.training_loss)
                        self.plot_epsilon(self.epsilon_ev)

                        # End message.
                        if episode < EPISODES:
                            print('\nTarget score reached.')
                        else:
                            print('\nEpisode limit reached.')
                        break
                    
                    # Update epsilon.
                    self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

    def test(self):
        """
        Load the network and test it.
        """
        self.load_model()
        total_reward = 0
        while True:
            action = self.q_network.get_action(self.observation, epsilon=0.0)
            new_observation, reward, done, info = self.env.step(action)
            new_observation = self.encode.process(new_observation)

            self.observation = new_observation

            total_reward += reward

            if done:
                print('Reward: {:.2f}'.format(total_reward))
                break
        

    def plot_rewards(self, tr_rewards, mean_tr_rewards):
        """
        Plot the reward and mean reward and save it.
        """
        plt.figure(figsize=(8,4))
        plt.plot(tr_rewards, label='Rewards')
        plt.plot(mean_tr_rewards, label='Mean Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.legend(loc="upper left")
        plt.savefig('plots/Rewards.png')

    def plot_loss(self, tr_loss):
        """
        Plot the loss and save it.
        """
        plt.figure(figsize=(8,4))
        plt.plot(tr_loss, label='Loss')
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.legend(loc="upper left")
        plt.savefig('plots/Loss.png')

    def plot_epsilon(self, eps_evolution):
        """
        Plot the evolution of epsilon and save it.
        """
        plt.figure(figsize=(8,4))
        plt.plot(eps_evolution, label='Epsilon')
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        plt.legend(loc="upper right")
        plt.savefig('plots/eps.png')

                   