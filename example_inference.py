import torch
from rocket import Rocket
from policy import ActorCritic
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
    else:
        print(f"No checkpoint found at {ckpt_dir}")

if __name__ == '__main__':
    task = 'landing'  # 'hover' or 'landing'
    max_steps = 800
    ckpt_dir = glob.glob(os.path.join(task+'_ckpt', '*.pt'))[-1]  # last ckpt

    env = Rocket(task=task, max_steps=max_steps)
    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    
    load_model(net, '/Users/alexpinel/Desktop/instinct2d/landing_ckpt/ckpt_00011401.pt')

    state = env.reset()
    for step_id in range(max_steps):
        action, log_prob, value = net.get_action(state)
        state, reward, done, _ = env.step(action)
        env.render(window_name='test')
        if done:
            break

    print(f"Simulation ended after {step_id + 1} steps with reward: {reward}")