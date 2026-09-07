import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import defaultdict
from model import Actor, Critic
from crop_utils import INPUT_DIM, RunningNormalizer

ACTOR_LR     = 1e-4
CRITIC_LR    = 3e-4
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
        # BUG FIX (nitrogen entropy stuck near max): nitrogen is only
        # consequential on ~14% of timesteps (NITROGEN_SLOTS windows in
        # main.py) -- the rest of the season, main.py forces nitrogen_kg
        # to 0.0 regardless of what the agent picked. Previously, every
        # timestep's nitrogen log_prob/entropy was still used in the loss
        # with that timestep's advantage, even though the sampled
        # nitrogen action had zero effect on the environment on ~86% of
        # steps. That's a noisy, largely irrelevant gradient signal
        # diluting real learning -- exactly why the water head (relevant
        # every day) learned fine while nitrogen stayed near max entropy.
        # This mask tracks, per timestep, whether the nitrogen slot was
        # actually open, so update() can exclude closed-slot timesteps
        # from the nitrogen head's loss and entropy logging.
        self._nitrogen_active    = []
        self.crop_stats = defaultdict(lambda: {"episodes": 0, "total_reward": 0.0})

        self.advantage_normalizer = RunningNormalizer(size=1, epsilon=1e-4, clip=10.0)

    def select_action(self, state, nitrogen_active=True):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        value   = self.critic(state_t)
        water_probs, nitrogen_probs = self.actor(state_t)

        # DEBUG GUARD: entropy of a K-way categorical is bounded by ln(K).
        # If either head's probs are corrupted (NaN, don't sum to 1, etc.)
        # that bound breaks silently and only shows up several steps later
        # as a nonsensical logged entropy (e.g. ent_n=215 when max=ln(5)).
        # Catch it here, at the source, with the actual bad tensor printed.
        for name, probs in [("water", water_probs), ("nitrogen", nitrogen_probs)]:
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                raise RuntimeError(
                    f"Corrupted {name}_probs from actor: {probs.detach().numpy()} "
                    f"(state stats: min={state_t.min().item():.3f}, "
                    f"max={state_t.max().item():.3f}). "
                    f"Likely a bad/incompatible checkpoint -- delete checkpoints/ "
                    f"and retrain from scratch."
                )
            psum = probs.sum().item()
            if abs(psum - 1.0) > 1e-3:
                raise RuntimeError(
                    f"{name}_probs don't sum to 1 (sum={psum:.6f}): "
                    f"{probs.detach().numpy()}. Likely a bad/incompatible "
                    f"checkpoint -- delete checkpoints/ and retrain from scratch."
                )

        wd = Categorical(water_probs)
        nd = Categorical(nitrogen_probs)
        wa = wd.sample()
        na = nd.sample()

        # DEBUG GUARD 2: entropy of a K-way categorical is bounded by ln(K),
        # regardless of how "bad" the distribution is, as long as it's a
        # valid distribution. Guard 1 (above) checks the probs are valid but
        # ent_n has still been observed at ~200+ (max=ln(5)=1.609), so check
        # the entropy value itself at the source and dump everything needed
        # to diagnose it, rather than continuing to guess from downstream logs.
        w_entropy = wd.entropy()
        n_entropy = nd.entropy()
        for hname, ent, probs, K in [
            ("water", w_entropy, water_probs, water_probs.shape[-1]),
            ("nitrogen", n_entropy, nitrogen_probs, nitrogen_probs.shape[-1]),
        ]:
            max_ent = math.log(K)
            if (ent > max_ent + 1e-3).any():
                raise RuntimeError(
                    f"Impossible {hname} entropy {ent.item():.4f} > ln({K})={max_ent:.4f}. "
                    f"probs={probs.detach().numpy()} "
                    f"probs.sum={probs.sum().item():.8f} "
                    f"probs.dtype={probs.dtype} "
                    f"any_negative={bool((probs < 0).any())} "
                    f"state_stats: min={state_t.min().item():.3f} max={state_t.max().item():.3f}"
                )

        self._log_probs_water.append(wd.log_prob(wa))
        self._log_probs_nitrogen.append(nd.log_prob(na))
        self._entropies_water.append(wd.entropy())
        self._entropies_nitrogen.append(nd.entropy())
        self._values.append(value.squeeze())
        self._nitrogen_active.append(1.0 if nitrogen_active else 0.0)
        return wa.item(), na.item()

    def store_reward(self, reward):
        self._rewards.append(float(reward))

    def update(self, crop_name=None):
        T = len(self._rewards)
        if T == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        returns = torch.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = self._rewards[t] + self.gamma * G
            returns[t] = G

        values     = torch.stack(self._values)
        raw_advantages = returns - values.detach()

        for adv_value in raw_advantages.detach().numpy():
            self.advantage_normalizer.update(np.array([adv_value]))
        advantages = torch.tensor(
            [self.advantage_normalizer.normalize(np.array([a]))[0] for a in raw_advantages.detach().numpy()],
            dtype=torch.float32,
        )

        log_probs_w = torch.stack(self._log_probs_water).squeeze(-1)
        log_probs_n = torch.stack(self._log_probs_nitrogen).squeeze(-1)
        entropies_w = torch.stack(self._entropies_water).squeeze(-1)
        entropies_n = torch.stack(self._entropies_nitrogen).squeeze(-1)

        # Guard against the exact broadcasting bug this replaced: if any of
        # these come out with an extra dim (e.g. batch size > 1 introduced
        # later), elementwise ops against nitrogen_mask (shape [T]) would
        # silently broadcast to [T,T] instead of erroring -- so check shape
        # explicitly rather than relying on downstream math to catch it.
        for name, t in [("log_probs_w", log_probs_w), ("log_probs_n", log_probs_n),
                         ("entropies_w", entropies_w), ("entropies_n", entropies_n)]:
            if t.shape != (T,):
                raise RuntimeError(
                    f"{name} has shape {tuple(t.shape)}, expected ({T},). "
                    f"This will silently broadcast incorrectly against "
                    f"nitrogen_mask -- fix before continuing."
                )

        # BUG FIX (nitrogen entropy stuck near max): mask out nitrogen's
        # contribution on timesteps where the slot was closed, so the
        # nitrogen head only receives gradient signal -- and only gets
        # "credit" in the logged entropy average -- on timesteps where
        # its choice actually affected the environment.
        nitrogen_mask = torch.tensor(self._nitrogen_active, dtype=torch.float32)
        log_probs_n_masked = log_probs_n * nitrogen_mask
        entropies_n_masked = entropies_n * nitrogen_mask

        # Mean entropy per head, logged BEFORE the lists are cleared below.
        # Nitrogen entropy is now averaged ONLY over active-slot timesteps
        # (not all timesteps), so it reflects whether the agent is actually
        # learning to differentiate nitrogen amounts when it matters,
        # rather than being diluted by the ~86% of steps where nitrogen
        # was forced to 0 regardless of the sampled action.
        mean_entropy_water = entropies_w.mean().item()
        n_active_count = nitrogen_mask.sum().item()
        mean_entropy_nitrogen = (
            entropies_n_masked.sum().item() / n_active_count
            if n_active_count > 0 else entropies_n.mean().item()
        )

        # Policy loss with entropy bonus to encourage exploration
        actor_loss = (
            -(log_probs_w + log_probs_n_masked) * advantages
            - ENTROPY_BETA * (entropies_w + entropies_n_masked)
        ).mean()

        critic_loss = F.mse_loss(values, returns)

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
        self._nitrogen_active    = []
        return (actor_loss.item(), critic_loss.item(), total_return,
                mean_entropy_water, mean_entropy_nitrogen)

    def save(self, path_prefix):
        torch.save(self.actor.state_dict(),  f"{path_prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pth")
        self.advantage_normalizer.save(f"{path_prefix}_advnorm")

    def load(self, path_prefix):
        self.actor.load_state_dict( torch.load(f"{path_prefix}_actor.pth",  map_location='cpu', weights_only=True))
        self.critic.load_state_dict(torch.load(f"{path_prefix}_critic.pth", map_location='cpu', weights_only=True))
        try:
            self.advantage_normalizer.load(f"{path_prefix}_advnorm")
        except FileNotFoundError:
            pass