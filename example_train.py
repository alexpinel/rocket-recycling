import numpy as np
import torch
from rocket import Rocket
from policy import ActorCritic
import matplotlib.pyplot as plt
import utils
import os
import glob

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(net, ckpt_dir):
    if os.path.exists(ckpt_dir):
        checkpoint = torch.load(ckpt_dir)
        state_dict = checkpoint['model_G_state_dict']
        compatible_state_dict = {k: v for k, v in state_dict.items() if k in net.state_dict() and v.shape == net.state_dict()[k].shape}
        net.load_state_dict(compatible_state_dict, strict=False)
        print(f"Loaded compatible weights from {ckpt_dir}")
        return checkpoint['episode_id'], checkpoint['REWARDS']
    else:
        print(f"No checkpoint found at {ckpt_dir}")
        return 0, []

class TrainingManager:
    def __init__(self, rocket_env, agent, initial_difficulty=0):
        self.rocket_env = rocket_env
        self.agent = agent
        self.successful_landings = 0
        self.difficulty = initial_difficulty

    def train(self, episodes):
        REWARDS = []
        for episode in range(episodes):
            state = self.rocket_env.reset(difficulty=self.difficulty)
            rewards, log_probs, values, masks = [], [], [], []
            for step_id in range(self.rocket_env.max_steps):
                action, log_prob, value = self.agent.get_action(state)
                next_state, reward, done, _ = self.rocket_env.step(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)
                masks.append(1-done)
                state = next_state

                if episode % 100 == 1:
                    env.render()

                if done or step_id == self.rocket_env.max_steps-1:
                    _, _, Qval = self.agent.get_action(state)
                    self.agent.update_ac(self.agent, rewards, log_probs, values, masks, Qval, gamma=0.999)
                    break

            REWARDS.append(np.sum(rewards))
            print('episode id: %d, episode reward: %.3f' % (episode, np.sum(rewards)))

            if self.rocket_env.already_landing:
                self.successful_landings += 1

            if self.successful_landings >= 10:  # Example threshold
                self.difficulty += 1
                self.successful_landings = 0
                print(f"Difficulty increased to {self.difficulty}")

            if episode % 100 == 1:
                plt.figure()
                plt.plot(REWARDS), plt.plot(utils.moving_avg(REWARDS, N=50))
                plt.legend(['episode reward', 'moving avg'], loc=2)
                plt.xlabel('m episode')
                plt.ylabel('reward')
                plt.savefig(os.path.join(ckpt_folder, 'rewards_' + str(episode).zfill(8) + '.jpg'))
                plt.close()

                torch.save({'episode_id': episode,
                            'REWARDS': REWARDS,
                            'model_G_state_dict': self.agent.state_dict()},
                           os.path.join(ckpt_folder, 'ckpt_' + str(episode).zfill(8) + '.pt'))

        return REWARDS

if __name__ == '__main__':
    task = 'landing'  # 'hover' or 'landing'

    max_m_episode = 800000
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

    training_manager = TrainingManager(env, net)
    REWARDS = training_manager.train(episodes=max_m_episode)
