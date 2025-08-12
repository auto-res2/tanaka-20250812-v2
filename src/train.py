import os
import time
import random
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # for headless mode
import matplotlib.pyplot as plt
import seaborn as sns
import gym

# Set up directory for saving images
IMAGE_DIR = os.path.join('.research', 'iteration1', 'images')
os.makedirs(IMAGE_DIR, exist_ok=True)

# ---------------- Utility Functions ----------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ---------------- Model and Network Definitions ----------------

class TinyUNet(nn.Module):
    """A light UNet-like MLP for quick tests."""
    def __init__(self, obs_dim, act_dim, hidden=128):
        super(TinyUNet, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(obs_dim + 1 + act_dim, hidden),  # +t dimension
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim)
        )

    def forward(self, x_t, t, s):
        # t is expected to be a tensor of shape (batch,) so we unsqueeze to match
        t = t.unsqueeze(-1).float()
        inp = torch.cat([x_t, t, s], dim=-1)
        return self.f(inp)

class MLP(nn.Module):
    def __init__(self, inp, out, hidden=256):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out)
        )

    def forward(self, x):
        return self.net(x)

# ---------------- Distiller Definition ----------------

class DistillerConfig:
    def __init__(self, config_dict):
        self.obs_dim = config_dict.get('obs_dim', 4)
        self.act_dim = config_dict.get('act_dim', 2)
        self.lr = config_dict.get('lr', 3e-4)
        self.dropout_p = config_dict.get('dropout_p', 0.1)
        self.timewarp_k = config_dict.get('timewarp_k', 3)
        self.device = config_dict.get('device', 'cpu')
        self.batch_size = config_dict.get('batch_size', 256)
        self.train_steps = config_dict.get('train_steps', 1024)

class ConsistencyDistiller(nn.Module):
    def __init__(self, cfg):
        super(ConsistencyDistiller, self).__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.f = TinyUNet(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.q_net = MLP(cfg.obs_dim + cfg.act_dim, 1).to(self.device)
        self.cbf_net = MLP(cfg.obs_dim + cfg.act_dim, 1).to(self.device)
        self.opt = torch.optim.Adam(self.parameters(), lr=cfg.lr)
        self.step_ctr = 0

    def forward(self, x_t, t, s):
        return self.f(x_t, t, s)

    def train_loop(self, steps=None):
        if steps is None:
            steps = self.cfg.train_steps
        losses = []
        for _ in range(steps):
            batch_size = self.cfg.batch_size
            s = torch.randn(batch_size, self.cfg.obs_dim, device=self.device)
            a0 = torch.tanh(torch.randn(batch_size, self.cfg.act_dim, device=self.device))
            t = torch.randint(0, self.cfg.timewarp_k, (batch_size,), device=self.device)
            # Add noise proportional to t
            noise = torch.randn_like(a0) * (1 + t.float().unsqueeze(-1))
            x_t = a0 + noise
            pred_a0 = self.forward(x_t, t, s)
            l_cons = F.mse_loss(pred_a0, a0)
            q_val = self.q_net(torch.cat([s, pred_a0], dim=-1)).mean()
            cbf_val = F.relu(self.cbf_net(torch.cat([s, pred_a0], dim=-1))).mean()
            loss = l_cons - 0.1 * q_val + 0.2 * cbf_val
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses.append(loss.item())
            self.step_ctr += 1
        return losses

# ---------------- Plotting Utility ----------------

def plot_metric(data, title, ylabel, filename):
    sns.set_theme(style='darkgrid')
    plt.figure(figsize=(4,3))
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Steps')
    plt.ylabel(ylabel)
    plt.tight_layout()
    save_path = os.path.join(IMAGE_DIR, f'{filename}.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    print(f'[train.py] Saved plot to {save_path}')
    plt.close()

# ---------------- Main Training Function ----------------

def run_train():
    # Load configuration from config/config.yaml
    config_path = os.path.join('config', 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = {}
    cfg = DistillerConfig(config_dict)
    set_seed(config_dict.get('seed', 42))

    # Optionally infer dims from Gym environment
    env_name = config_dict.get('env_name', 'CartPole-v1')
    try:
        env = gym.make(env_name)
        cfg.obs_dim = env.observation_space.shape[0]
        if hasattr(env.action_space, 'shape') and env.action_space.shape:
            cfg.act_dim = env.action_space.shape[0]
        else:
            cfg.act_dim = 1
        env.close()
    except Exception as e:
        print(f'[train.py] Warning: Unable to init gym env: {e}')

    model = ConsistencyDistiller(cfg)
    print('[train.py] Starting training...')
    losses = model.train_loop()
    print('[train.py] Training completed.')

    # Save training loss plot
    plot_metric(losses, 'Training Loss (toy)', 'Loss', 'training_loss')

    # Save model
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'consistency_distiller.pth')
    torch.save(model.state_dict(), model_path)
    print(f'[train.py] Model saved to {model_path}')

if __name__ == '__main__':
    run_train()
