from collections import namedtuple, deque
import random
import torchvision.transforms as T
from PIL import Image
import numpy as np
import torch
from itertools import count
from gym import wrappers
import torch.optim as optim
import torch.nn as nn
import os
import matplotlib
from torchsummary import summary
from matplotlib import pyplot as plt


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class SnakeUtil():
    def __init__(self, name : str, env, policy_net, target_net, device, exploration_rate, exploration_decay, exploration_min, memory_size, target_update, batch_size, gamma):
        self.env = env
        self.policy_net = policy_net
        self.target_net = target_net
        self.resize = T.Compose([T.ToPILImage(), T.Resize(
            16, interpolation=Image.BOX), T.ToTensor()])
        self.device = device
        self.exploration_rate_init = exploration_rate
        self.exploration_decay_init = exploration_decay
        self.exploration_min_init = exploration_min
        self.exploration_rate = 0.0
        self.exploration_decay = 0.0
        self.exploration_min = 0.0
        self.name = name
        self.memory = ReplayMemory(memory_size)
        self.TARGET_UPDATE = target_update
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.steps_done = 0
        self.get_net_loss_and_optimizer()
        self.printerDataStrings = []

    def get_net_loss_and_optimizer(self):
        """ Optimizer """
        # ref: https://ai.stackexchange.com/questions/9298/neural-network-optimizers-in-reinforcement-learning-non-well-behaved-environment
        # ref: https://stackoverflow.com/questions/59833217/best-reinforcement-learner-optimizer
        #optimizer = optim.RMSprop(policy_net.parameters(), lr=learning_rate, weight_decay=decay_rate)
        #optimizer = optim.RMSprop(policy_net.parameters())
        self.optimizer = optim.Adam(self.policy_net.parameters())
        #optimizer = optim.SGD(policy_net.parameters())

        """ Loss function """
        #criterion = nn.CrossEntropyLoss()
        #criterion = nn.NLLLoss()
        self.criterion = nn.SmoothL1Loss()
        #criterion = nn.MSELoss()

    # Get screen size so that we can initialize layers correctly based on shape
    # returned from AI gym. Typical dimensions at this point are close to 3x40x90
    # which is the result of a clamped and down-scaled render buffer in get_screen()
    # screen_height, screen_width, _ = screen = env.render(mode='rgb_array').shape 
    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))

        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW) #TODO: CHANGE
        return self.resize(screen).unsqueeze(0)

    def record_episodes(self, num_episodes=10):

        env = gym.wrappers.RecordVideo(
            self.env, 'video/' + self.name, episode_trigger=lambda x: x % 1 == 0)

        episode_reward = 0
        rewards = []

        for i_episode in range(num_episodes):

            # Initialize the environment and state
            env.reset()
            last_screen = self.get_screen()
            current_screen = self.get_screen()
            state = current_screen + last_screen

            for t in count():
                # Select and perform an action
                action = self.target_net(state).max(1)[1].view(1, 1)
                _, reward, done, _ = env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                episode_reward += reward

                # Observe new state
                last_screen = current_screen
                current_screen = self.get_screen()
                if not done:
                    next_state = current_screen + last_screen
                else:
                    next_state = None

                # Move to the next state
                state = next_state

                if done:
                    rewards.append(episode_reward.item())
                    break
        
    def reset(self):
        self.env.reset()


    def select_action(self,state):
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)
        self.steps_done += 1
        if np.random.rand() > self.exploration_rate:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.env.action_space.n)]], device=self.device, dtype=torch.long)

    """ Model Optimization """
    def optimize_model(self):

        if len(self.memory) < self.BATCH_SIZE:
            return

        self.exploration_rate = self.exploration_rate_init
        self.exploration_decay = self.exploration_decay_init
        self.exploration_min = self.exploration_min_init    

        self.losses = []
                

        transitions = self.memory.sample(self.BATCH_SIZE)
        
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
                                            
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        
        #print(expected_state_action_values.unsqueeze(1))
        #print(state_action_values)
        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # self.losses.append(loss.to('cpu').detach().numpy())
        self.mean_loss.append(np.mean(loss.to('cpu').detach().numpy()))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def training_printer(self, i_epi, num_epi):      
        if i_epi % 100 == 0 and i_epi != 0:
            display.clear_output()
            episodeStr = "Playing episode: " + str(i_epi) + "/" + str(num_epi)
            lossStr = "\tMean loss => " + str(round(self.mean_loss[-1],4))
            rewardStr = "\tMean reward => " + str(round(self.moving_average_reward[-1], 4))
            self.printerDataStrings.append(episodeStr + " " + lossStr + " " + rewardStr)
            if self.printerDataStrings.__len__() > 10:
                self.printerDataStrings.pop(0)
            for i in range(self.printerDataStrings.__len__()):
                print(self.printerDataStrings[i])
            self._plot(i_epi)


    def _plot(self, episode):
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        print("episode", episode)
        print(len(self.mean_loss))
        print(len(self.rewards))
        fig.suptitle('Analysis of the model')
        ax1.plot(episode, self.mean_loss[0])
        ax1.set_title('Model 1')
        ax2.plot(episode, 1)
        ax2.set_title('Model 2')
        ax3.plot(episode,1)
        plt.show()



    """ Training """
    def train(self, num_episodes=10000):
        
        self.steps_done = 0

        save_path = 'models/'+str(self.name)+'.pkl'

        self.losses = []
        self.mean_loss = []
        self.rewards = []
        self.moving_average_reward = []

        for i_episode in range(num_episodes):
            self.training_printer(i_episode, num_episodes)
            episode_reward = 0

            # Initialize the environment and state
            self.env.reset()
            last_screen = self.get_screen()
            current_screen = self.get_screen()
            state = current_screen + last_screen


            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                _, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device).long()
                episode_reward += reward

                # Observe new state
                last_screen = current_screen
                current_screen = self.get_screen()
                if not done:
                    next_state = current_screen + last_screen
                else:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                
                if done:
                    self.rewards.append(episode_reward.item())
                    # Take the mean of the last 100 rewards
                    self.moving_average_reward.append(np.mean(self.rewards[-100:]))
                    break

                # Update the target network, copying all weights and biases in DQN
                if t % self.TARGET_UPDATE == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    torch.save(self.policy_net.state_dict(), save_path)
