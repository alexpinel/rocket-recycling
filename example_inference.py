import torch
from rocket import Rocket
from policy import ActorCritic
import os
import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(net, ckpt_dir):
    if os.path.exists(ckpt_dir):
        checkpoint = torch.load(ckpt_dir)
        state_dict_keys = ['model_G_state_dict', 'model_state_dict', 'state_dict']
        state_dict = None
        for key in state_dict_keys:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        
        if state_dict is None:
            print(f"Could not find state dict in checkpoint. Available keys: {checkpoint.keys()}")
            return

        compatible_state_dict = {k: v for k, v in state_dict.items() if k in net.state_dict() and v.shape == net.state_dict()[k].shape}
        
        net.load_state_dict(compatible_state_dict, strict=False)
        print(f"Loaded compatible weights from {ckpt_dir}")
    else:
        print(f"No checkpoint found at {ckpt_dir}")

if __name__ == '__main__':
    task = 'landing'  # 'hover' or 'landing'
    max_steps = 800
    ckpt_dir = glob.glob(os.path.join(task+'_ckpt', '*.pt'))[-1]  # last ckpt

    env = Rocket(task=task, max_steps=max_steps)
    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    
    load_model(net, '/Users/alexpinel/Desktop/instinct2d/landing_ckpt/ckpt_00007100.pt')

    state = env.reset()
    total_reward = 0
    for step_id in range(max_steps):
        action, log_prob = net.get_action(state)  # Now expecting only two return values
        state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render(window_name='test')
        if done:
            break

    print(f"Simulation ended after {step_id + 1} steps with total reward: {total_reward}")