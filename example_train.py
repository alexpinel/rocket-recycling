import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import numpy as np
import torch
from rocket import Rocket
from policy import ActorCritic
import matplotlib.pyplot as plt
import glob
import traceback

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(net, ckpt_dir):
    try:
        if os.path.exists(ckpt_dir):
            checkpoint = torch.load(ckpt_dir)
            net.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded weights from {ckpt_dir}")
            return checkpoint['episode_id'], checkpoint['REWARDS']
        else:
            print(f"No checkpoint found at {ckpt_dir}")
            return 0, []
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return 0, []

class TrainingManager:
    def __init__(self, rocket_env, agent, initial_difficulty=0):
        self.rocket_env = rocket_env
        self.agent = agent
        self.successful_landings = 0
        self.difficulty = initial_difficulty

    def train(self, episodes):
        REWARDS = []
        try:
            for episode in range(episodes):
                state = self.rocket_env.reset(difficulty=self.difficulty)
                episode_reward = 0
                done = False
                
                states, actions, rewards, log_probs = [], [], [], []
                
                while not done:
                    action, log_prob = self.agent.get_action(state)
                    next_state, reward, done, _ = self.rocket_env.step(action)
                    
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    log_probs.append(log_prob)

                    state = next_state
                    episode_reward += reward

                # Update the policy at the end of each episode
                loss = self.agent.update(states, actions, rewards, log_probs)

                REWARDS.append(episode_reward)
                print(f'episode id: {episode}, episode reward: {episode_reward:.3f}, loss: {loss:.3f}')

                if self.rocket_env.already_landing:
                    self.successful_landings += 1

                if self.successful_landings >= 10:
                    self.difficulty += 1
                    self.successful_landings = 0
                    print(f"Difficulty increased to {self.difficulty}")

                if episode % 100 == 0:
                    self.plot_rewards(REWARDS, episode)

        except Exception as e:
            print(f"An error occurred during training: {str(e)}")
            traceback.print_exc()

        return REWARDS

    def plot_rewards(self, REWARDS, episode):
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(REWARDS)
            
            if len(REWARDS) >= 50:
                plt.plot(np.convolve(REWARDS, np.ones(50)/50, mode='valid'))
            
            if len(REWARDS) >= 2:
                x = np.arange(len(REWARDS))
                z = np.polyfit(x, REWARDS, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), "r--", linewidth=2)
                plt.legend(['episode reward', 'moving avg', 'trend'], loc='lower right')
            else:
                plt.legend(['episode reward'], loc='lower right')
            
            plt.xlabel('episode')
            plt.ylabel('reward')
            plt.title(f'Training Progress (Difficulty: {self.difficulty})')
            plt.savefig(os.path.join(ckpt_folder, f'rewards_{episode:08d}.jpg'))
            plt.close()

            torch.save({
                'episode_id': episode,
                'REWARDS': REWARDS,
                'model_state_dict': self.agent.state_dict(),
                'difficulty': self.difficulty
            }, os.path.join(ckpt_folder, f'ckpt_{episode:08d}.pt'))
        except Exception as e:
            print(f"An error occurred while plotting rewards: {str(e)}")

if __name__ == '__main__':
    try:
        task = 'landing'
        max_m_episode = 15000
        max_steps = 800

        env = Rocket(task=task, max_steps=max_steps)
        ckpt_folder = os.path.join('./', task + '_ckpt')
        if not os.path.exists(ckpt_folder):
            os.mkdir(ckpt_folder)

        net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
        
        last_episode_id = 0
        REWARDS = []
        
        if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
            ckpt_path = glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1]
            last_episode_id, REWARDS = load_model(net, ckpt_path)

        training_manager = TrainingManager(env, net, initial_difficulty=0)
        REWARDS = training_manager.train(episodes=max_m_episode)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        traceback.print_exc()