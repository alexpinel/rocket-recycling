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
        # Get the state dict from the checkpoint
        state_dict = checkpoint['model_G_state_dict']
        
        # Filter out incompatible keys
        compatible_state_dict = {k: v for k, v in state_dict.items() if k in net.state_dict() and v.shape == net.state_dict()[k].shape}
        
        # Load the compatible state dict
        net.load_state_dict(compatible_state_dict, strict=False)
        print(f"Loaded compatible weights from {ckpt_dir}")
        return checkpoint['episode_id'], checkpoint['REWARDS']
    else:
        print(f"No checkpoint found at {ckpt_dir}")
        return 0, []

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
        # load the last ckpt
        ckpt_path = glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1]
        last_episode_id, REWARDS = load_model(net, ckpt_path)

    for episode_id in range(last_episode_id, max_m_episode):
        # training loop
        state = env.reset()
        rewards, log_probs, values, masks = [], [], [], []
        for step_id in range(max_steps):
            action, log_prob, value = net.get_action(state)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            masks.append(1-done)
            state = next_state
            #if episode_id % 100 == 1:
            #    env.render()

            if done or step_id == max_steps-1:
                _, _, Qval = net.get_action(state)
                net.update_ac(net, rewards, log_probs, values, masks, Qval, gamma=0.999)
                break

        REWARDS.append(np.sum(rewards))
        print('episode id: %d, episode reward: %.3f'
              % (episode_id, np.sum(rewards)))

        if episode_id % 100 == 1:
            plt.figure()
            plt.plot(REWARDS), plt.plot(utils.moving_avg(REWARDS, N=50))
            plt.legend(['episode reward', 'moving avg'], loc=2)
            plt.xlabel('m episode')
            plt.ylabel('reward')
            plt.savefig(os.path.join(ckpt_folder, 'rewards_' + str(episode_id).zfill(8) + '.jpg'))
            plt.close()

            torch.save({'episode_id': episode_id,
                        'REWARDS': REWARDS,
                        'model_G_state_dict': net.state_dict()},
                       os.path.join(ckpt_folder, 'ckpt_' + str(episode_id).zfill(8) + '.pt'))