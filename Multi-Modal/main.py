import random, os, csv
import numpy as np
import gym
import gym_dssat_pdi

from agent import ActorCriticAgent
from crop_utils import CROPS, augment_state, action_index_to_amounts, INPUT_DIM
from dssat_configuration import get_dssat_config
from reward_shaping import SeasonalRewardShaper

NUM_EPISODES   = 20000
LOG_INTERVAL   = 100
SAVE_INTERVAL  = 1000
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

RANDOM_BASELINE_EPISODES = 10  # episodes per crop used to estimate the random baseline

# TODO: these cotton values are PLACEHOLDER estimates based on typical
# cotton growth-stage timing (squaring ~35-45 DAP, early flowering ~55-65 DAP,
# peak bloom ~75-85 DAP). Cotton seasons run longer than maize (~150-180 DAP
# vs ~120 DAP), and actual N/irrigation needs depend on your cultivar, soil,
# and climate. Verify against agronomy literature or local extension data
# before trusting results built on these numbers.
NITROGEN_SLOTS = {
    "maize":  [20, 40, 55],
    "cotton": [35, 60, 80],
}
SLOT_WINDOW    = 3

IRRIGATION_FLOOR = {
    "maize": [
        (1,   20,  10),
        (21,  55,  15),
        (56,  90,  20),
        (91,  999,  5),
    ],
    "cotton": [
        (1,   30,   5),   # establishment — light water
        (31,  70,  15),   # squaring/early flowering — ramping up
        (71, 120,  20),   # peak bloom/boll development — highest demand
        (121, 999,  5),   # boll maturation/defoliation — cut back
    ],
}

def get_irrigation_floor(crop_name, dap):
    for dap_start, dap_end, min_mm in IRRIGATION_FLOOR.get(crop_name, []):
        if dap_start <= dap <= dap_end:
            return min_mm
    return 0.0

def vs_baseline(value, baseline, eps=1e-6):
    denom = abs(baseline) if abs(baseline) > eps else eps
    return ((value - baseline) / denom) * 100

def nitrogen_is_allowed(crop_name, dap):
    for slot_dap in NITROGEN_SLOTS.get(crop_name, []):
        if abs(dap - slot_dap) <= SLOT_WINDOW:
            return True
    return False


def run_random_episode(env, crop_name, shaper):
    """Runs one episode with RANDOM actions, scored through the SAME
    SeasonalRewardShaper used in training. This is what makes the resulting
    baseline comparable to agent performance — the old hardcoded
    RANDOM_BASELINE = -29000.0 was computed on raw DSSAT profit (a totally
    different scale) and made vs_baseline_pct meaningless (~99.8% no matter
    what the agent did)."""
    shaper.reset(crop_name, episode=0)
    raw = env.reset()
    raw = raw[0] if isinstance(raw, tuple) else raw

    done = False
    episode_reward = 0.0
    while not done:
        obs_dict = raw if isinstance(raw, dict) else {}
        dap = int(obs_dict.get("dap", 0))
        slot_open = nitrogen_is_allowed(crop_name, dap)

        water_mm = random.choice([0, 10, 20, 30, 40])
        nitrogen_kg = random.choice([0, 5, 10, 20, 40]) if slot_open else 0.0
        floor_mm = get_irrigation_floor(crop_name, dap)
        water_mm = max(water_mm, floor_mm)

        step_out = env.step({"amir": float(water_mm), "anfer": float(nitrogen_kg)})
        if len(step_out) == 5:
            raw_next, dssat_reward, done, truncated, info = step_out
            done = done or truncated
        else:
            raw_next, dssat_reward, done, info = step_out

        obs_dict_next = raw_next if isinstance(raw_next, dict) else {}
        reward = shaper.step(obs_dict_next, dssat_reward, done,
                              last_water_action=water_mm, last_n_action=nitrogen_kg)
        episode_reward += reward
        raw = raw_next
    return episode_reward


def compute_random_baseline(envs, active_crops, n_episodes=RANDOM_BASELINE_EPISODES):
    print(f"Estimating random-policy baseline ({n_episodes} episodes/crop, "
          f"same reward scale as training)...")
    baseline_shaper = SeasonalRewardShaper(active_crops[0], verbose=False)
    all_returns = []
    for crop_name in active_crops:
        crop_returns = [run_random_episode(envs[crop_name], crop_name, baseline_shaper)
                        for _ in range(n_episodes)]
        print(f"  {crop_name:8s} random avg: {np.mean(crop_returns):.2f}")
        all_returns.extend(crop_returns)
    baseline = float(np.mean(all_returns))
    print(f"Random baseline (all crops): {baseline:.2f}\n")
    return baseline

print("Initialising environments...")
envs = {}
for crop in CROPS:
    try:
        envs[crop] = gym.make("GymDssatPdi-v0", **get_dssat_config(crop))
        print(f"  OK  {crop}")
    except Exception as e:
        print(f"  FAIL  {crop}: {e}")

if not envs:
    raise RuntimeError("No environments loaded.")

active_crops = list(envs.keys())
print(f"\nActive crops: {active_crops}  |  INPUT_DIM: {INPUT_DIM}\n")

RANDOM_BASELINE = compute_random_baseline(envs, active_crops)

# ── Resume from best checkpoint if available ───────────────────────────────
agent  = ActorCriticAgent(input_dim=INPUT_DIM)
best_actor_path = f"{CHECKPOINT_DIR}/best_actor.pth"
best_critic_path = f"{CHECKPOINT_DIR}/best_critic.pth"
if os.path.exists(best_actor_path) and os.path.exists(best_critic_path):
    agent.load(f"{CHECKPOINT_DIR}/best")
    print(f"Resumed from best checkpoint.")
else:
    print("Starting from scratch.")

shaper = SeasonalRewardShaper(active_crops[0], verbose=True)

episode_returns = []
best_return     = -float("inf")
per_crop        = {c: {"returns": [], "al": [], "cl": [], "entropy": [],
                       "total_water": [], "total_nitrogen": []} for c in active_crops}

# ── CSV logging setup ──────────────────────────────────────────────────────
CSV_PATH = "./training_log.csv"
csv_file  = open(CSV_PATH, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "episode", "crop", "season_return", "avg100_return",
    "actor_loss", "critic_loss", "entropy",
    "total_water_mm", "total_nitrogen_kgha",
    "vs_baseline_pct"
])
print(f"CSV logging → {CSV_PATH}\n")

for episode in range(1, NUM_EPISODES + 1):
    crop_name = random.choice(active_crops)
    env       = envs[crop_name]

    shaper.reset(crop_name, episode=episode)

    raw  = env.reset()
    raw  = raw[0] if isinstance(raw, tuple) else raw

    init_obs  = raw if isinstance(raw, dict) else {}
    init_dap  = int(init_obs.get("dap", 0))
    init_slot = nitrogen_is_allowed(crop_name, init_dap)
    state     = augment_state(raw, crop_name, is_nitrogen_slot=init_slot)

    done            = False
    episode_reward  = 0.0
    season_water_mm = 0.0
    season_n_kgha   = 0.0

    while not done:
        obs_dict_pre = raw if isinstance(raw, dict) else {}
        dap          = int(obs_dict_pre.get("dap", 0))

        if dap == 0:
            step_out = env.step({"amir": 0.0, "anfer": 0.0})
            if len(step_out) == 5:
                raw_next, dssat_reward, done, truncated, info = step_out
                done = done or truncated
            else:
                raw_next, dssat_reward, done, info = step_out

            obs_dict = raw_next if isinstance(raw_next, dict) else {}
            reward   = shaper.step(obs_dict, dssat_reward, done,
                                   last_water_action=0.0, last_n_action=0.0)
            episode_reward += reward
            next_dap  = int(obs_dict.get("dap", 0))
            next_slot = nitrogen_is_allowed(crop_name, next_dap)
            raw       = raw_next
            state     = augment_state(raw_next, crop_name, is_nitrogen_slot=next_slot)
            continue

        slot_open = nitrogen_is_allowed(crop_name, dap)

        water_idx, nitrogen_idx = agent.select_action(state)
        water_mm, nitrogen_kg   = action_index_to_amounts(water_idx, nitrogen_idx)

        floor_mm = get_irrigation_floor(crop_name, dap)
        water_mm = max(water_mm, floor_mm)

        if not slot_open:
            nitrogen_kg = 0.0

        season_water_mm += water_mm
        season_n_kgha   += nitrogen_kg

        step_out = env.step({"amir": float(water_mm), "anfer": float(nitrogen_kg)})
        if len(step_out) == 5:
            raw_next, dssat_reward, done, truncated, info = step_out
            done = done or truncated
        else:
            raw_next, dssat_reward, done, info = step_out

        obs_dict = raw_next if isinstance(raw_next, dict) else {}
        reward   = shaper.step(obs_dict, dssat_reward, done,
                               last_water_action=water_mm,
                               last_n_action=nitrogen_kg)

        agent.store_reward(reward)
        episode_reward += reward

        next_dap  = int(obs_dict.get("dap", 0))
        next_slot = nitrogen_is_allowed(crop_name, next_dap)
        raw       = raw_next
        state     = augment_state(raw_next, crop_name, is_nitrogen_slot=next_slot)

    al, cl, _, entropy = agent.update(crop_name=crop_name)

    episode_returns.append(episode_reward)
    per_crop[crop_name]["returns"].append(episode_reward)
    per_crop[crop_name]["al"].append(al)
    per_crop[crop_name]["cl"].append(cl)
    per_crop[crop_name]["entropy"].append(entropy)
    per_crop[crop_name]["total_water"].append(season_water_mm)
    per_crop[crop_name]["total_nitrogen"].append(season_n_kgha)

    window     = episode_returns[-100:]
    avg_return = sum(window) / len(window)
    vs_baseline_pct = vs_baseline(avg_return, RANDOM_BASELINE)

    if avg_return > best_return:
        best_return = avg_return
        agent.save(f"{CHECKPOINT_DIR}/best")

    # ── Write one CSV row per episode ──────────────────────────────────────
    csv_writer.writerow([
        episode, crop_name,
        round(episode_reward, 4),
        round(avg_return, 4),
        round(al, 6),
        round(cl, 6),
        round(entropy, 6),
        round(season_water_mm, 1),
        round(season_n_kgha, 1),
        round(vs_baseline_pct, 2)
    ])
    if episode % 100 == 0:
        csv_file.flush()

    if episode % LOG_INTERVAL == 0 or episode == 1:
        improvement = vs_baseline_pct
        print(f"\nEp {episode:6d} | avg100: {avg_return:8.2f} | "
              f"best: {best_return:8.2f} | "
              f"vs baseline: {improvement:+.1f}%")

        for c in active_crops:
            r = per_crop[c]["returns"]
            if r:
                w         = r[-LOG_INTERVAL:]
                avg_water = np.mean(per_crop[c]["total_water"][-LOG_INTERVAL:])
                avg_n     = np.mean(per_crop[c]["total_nitrogen"][-LOG_INTERVAL:])
                crop_impr = vs_baseline(np.mean(w), RANDOM_BASELINE)
                print(f"  {c:8s} | n={len(r):5d} | "
                      f"avg={np.mean(w):8.2f} | "
                      f"vs_baseline={crop_impr:+.1f}% | "
                      f"water={avg_water:.0f}mm | "
                      f"N={avg_n:.0f}kg/ha | "
                      f"al={np.mean(per_crop[c]['al'][-LOG_INTERVAL:]):7.4f} | "
                      f"cl={np.mean(per_crop[c]['cl'][-LOG_INTERVAL:]):.4f} | "
                      f"entropy={np.mean(per_crop[c]['entropy'][-LOG_INTERVAL:]):.4f}")
        print()

    if episode % SAVE_INTERVAL == 0:
        agent.save(f"{CHECKPOINT_DIR}/ep{episode:06d}")

csv_file.close()
agent.save(f"{CHECKPOINT_DIR}/final")
for e in envs.values():
    e.close()

final_improvement = vs_baseline(best_return, RANDOM_BASELINE)
print(f"\nDone.")
print(f"Best avg100 return : {best_return:.2f}")
print(f"Random baseline    : {RANDOM_BASELINE:.2f}")
print(f"Best improvement   : {final_improvement:+.1f}% over random policy")
print(f"Training log saved : {CSV_PATH}")