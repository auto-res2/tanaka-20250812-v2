import os
import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import yaml

# ---------------- Utility Functions ----------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

# ---------------- Model and Network Definitions (should match train.py) ----------------

class TinyUNet(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super(TinyUNet, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(obs_dim + 1 + act_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim)
        )
    def forward(self, x_t, t, s):
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

class DistillerConfig:
    def __init__(self, config_dict):
        self.obs_dim = config_dict.get('obs_dim', 4)
        self.act_dim = config_dict.get('act_dim', 2)
        self.lr = config_dict.get('lr', 3e-4)
        self.dropout_p = config_dict.get('dropout_p', 0.1)
        self.timewarp_k = config_dict.get('timewarp_k', 3)
        self.device = config_dict.get('device', 'cpu')

class ConsistencyDistiller(nn.Module):
    def __init__(self, cfg):
        super(ConsistencyDistiller, self).__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.f = TinyUNet(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.q_net = MLP(cfg.obs_dim + cfg.act_dim, 1).to(self.device)
        self.cbf_net = MLP(cfg.obs_dim + cfg.act_dim, 1).to(self.device)

    def forward(self, x_t, t, s):
        return self.f(x_t, t, s)

    @torch.no_grad()
    def act(self, obs, project=True):
        # obs is a numpy array
        if isinstance(obs, np.ndarray):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.cfg.device).unsqueeze(0)
        else:
            obs_t = obs.to(self.cfg.device).unsqueeze(0)
        x_1 = torch.zeros((1, self.cfg.act_dim), device=self.cfg.device)
        t = torch.zeros(1, dtype=torch.long, device=self.cfg.device)
        a_star = self.forward(x_1, t, obs_t).squeeze(0)
        # Dummy safety projection: if first element positive then zero it out
        if project and a_star[0] > 0:
            a_star = a_star * 0.0
        return a_star.cpu().numpy()

# Dummy constraint function

def dummy_cbf(s, a):
    # Returns violation if s[0] + a[0] exceeds 0.8
    x_pos = s[0] if isinstance(s, np.ndarray) else s.item()
    return max(0, x_pos + a[0] - 0.8)

# Evaluation helper

def eval_policy(policy_fn, env_name, episodes=10):
    env = gym.make(env_name)
    returns = []
    latencies = []
    violations = []
    for ep in range(episodes):
        obs = env.reset()[0] if isinstance(env.reset(), (list, tuple)) else env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            t_start = time.time()
            act = policy_fn(obs)
            latency = (time.time() - t_start) * 1e6  # in microseconds
            latencies.append(latency)
            obs, rew, done, trunc, info = env.step(act)
            ep_ret += rew
            # compute dummy violation
            viol = dummy_cbf(np.array(obs), act)
            violations.append(viol)
            if done or trunc:
                break
        returns.append(ep_ret)
    env.close()
    results = {
        'return_mean': np.mean(returns),
        'latency_mean_us': np.mean(latencies),
        'violation_rate_%': 100.0 * np.mean([1 if v > 0 else 0 for v in violations])
    }
    print('[evaluate.py] Evaluation results:', json.dumps(results, indent=2))
    return results


def run_evaluate():
    # Load config
    config_path = os.path.join('config', 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = {}
    cfg = DistillerConfig(config_dict)
    set_seed(config_dict.get('seed', 42))

    # Create model and load weights
    model = ConsistencyDistiller(cfg)
    model_path = os.path.join('models', 'consistency_distiller.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        print(f'[evaluate.py] Loaded model from {model_path}')
    else:
        print('[evaluate.py] Model file not found. Evaluating with untrained model.')

    env_name = config_dict.get('env_name', 'CartPole-v1')
    # Evaluate the policy
    results = eval_policy(model.act, env_name, episodes=5)
    return results

if __name__ == '__main__':
    run_evaluate()
