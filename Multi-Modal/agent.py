import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import defaultdict
from model import Actor, Critic
from crop_utils import INPUT_DIM

ACTOR_LR     = 1e-4
CRITIC_LR    = 1e-3
GAMMA        = 0.99
ENTROPY_BETA = 0.01
GRAD_CLIP    = 1.0


class ActorCriticAgent:

    def __init__(self, input_dim=INPUT_DIM, actor_lr=ACTOR_LR,
                 critic_lr=CRITIC_LR, gamma=GAMMA):
        self.actor  = Actor(input_dim)
        self.critic = Critic(input_dim)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self._log_probs_water    = []
        self._log_probs_nitrogen = []
        self._entropies_water    = []
        self._entropies_nitrogen = []
        self._values             = []
        self._rewards            = []
        self.crop_stats = defaultdict(lambda: {"episodes": 0, "total_reward": 0.0})

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        value   = self.critic(state_t)
        water_probs, nitrogen_probs = self.actor(state_t)

        wd = Categorical(water_probs)
        nd = Categorical(nitrogen_probs)
        wa = wd.sample()
        na = nd.sample()

        self._log_probs_water.append(wd.log_prob(wa))
        self._log_probs_nitrogen.append(nd.log_prob(na))
        self._entropies_water.append(wd.entropy())
        self._entropies_nitrogen.append(nd.entropy())
        self._values.append(value.squeeze())
        return wa.item(), na.item()

    def store_reward(self, reward):
        self._rewards.append(float(reward))

    def update(self, crop_name=None):
        T = len(self._rewards)
        if T == 0:
            return 0.0, 0.0, 0.0, 0.0

        returns = torch.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = self._rewards[t] + self.gamma * G
            returns[t] = G

        values     = torch.stack(self._values)
        advantages = returns - values.detach()

        log_probs_w = torch.stack(self._log_probs_water)
        log_probs_n = torch.stack(self._log_probs_nitrogen)
        entropies_w = torch.stack(self._entropies_water)
        entropies_n = torch.stack(self._entropies_nitrogen)

        # Policy loss with entropy bonus to encourage exploration
        actor_loss = (
            -(log_probs_w + log_probs_n) * advantages
            - ENTROPY_BETA * (entropies_w + entropies_n)
        ).mean()

        critic_loss = F.mse_loss(values, returns)

        # Mean entropy across both action heads for this episode — tracked
        # separately from the loss so we can see if the policy is collapsing
        # to near-deterministic (low entropy) even while actor_loss ~ 0.
        mean_entropy = (entropies_w + entropies_n).mean().item()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
        self.critic_optimizer.step()

        if crop_name:
            self.crop_stats[crop_name]["episodes"]     += 1
            self.crop_stats[crop_name]["total_reward"] += sum(self._rewards)

        total_return = returns[0].item()
        self._log_probs_water    = []
        self._log_probs_nitrogen = []
        self._entropies_water    = []
        self._entropies_nitrogen = []
        self._values             = []
        self._rewards            = []
        return actor_loss.item(), critic_loss.item(), total_return, mean_entropy

    def save(self, path_prefix):
        torch.save(self.actor.state_dict(),  f"{path_prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pth")

    def load(self, path_prefix):
        self.actor.load_state_dict( torch.load(f"{path_prefix}_actor.pth",  map_location='cpu', weights_only=True))
        self.critic.load_state_dict(torch.load(f"{path_prefix}_critic.pth", map_location='cpu', weights_only=True))