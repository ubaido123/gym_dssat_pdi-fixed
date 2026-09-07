import random, os, csv
import numpy as np
import gym
import gym_dssat_pdi

from agent import ActorCriticAgent
from crop_utils import (CROPS, augment_state, action_index_to_amounts, INPUT_DIM,
                         flatten_obs, STATE_NORMALIZER)
from dssat_configuration import get_dssat_config, YIELD_PRICE, WATER_COST, N_COST
from reward_shaping import SeasonalRewardShaper

NUM_EPISODES   = 5000
LOG_INTERVAL   = 100
SAVE_INTERVAL  = 1000
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

RANDOM_BASELINE = -29000.0

NITROGEN_SLOTS = {"maize": [20, 40, 55]}
SLOT_WINDOW    = 3

IRRIGATION_FLOOR = {
    "maize": [
        (1,   20,  5),
        (21,  55,  8),
        (56,  90,  10),
        (91,  999,  0),
    ]
}
# NOTE: lowered from (10,15,20,5) after diagnosing that the previous floor
# summed to ~1575-1625mm for a typical ~120-130 DAP season -- almost
# exactly AC's observed water-use plateau (~1640mm). That meant the floor,
# not the learned policy, was setting AC's water use. This halves the
# floor to give the agent real room to find a lower-water strategy. Keep
# an eye on grain_yield_kgha after this change -- if yield starts
# crashing (the earlier under-irrigation exploit this floor was added to
# prevent), raise these back up incrementally rather than reverting to
# the old values outright.

def get_irrigation_floor(crop_name, dap):
    for dap_start, dap_end, min_mm in IRRIGATION_FLOOR.get(crop_name, []):
        if dap_start <= dap <= dap_end:
            return min_mm
    return 0.0

def nitrogen_is_allowed(crop_name, dap):
    for slot_dap in NITROGEN_SLOTS.get(crop_name, []):
        if abs(dap - slot_dap) <= SLOT_WINDOW:
            return True
    return False

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

# ── Resume from best checkpoint if available ───────────────────────────────
agent  = ActorCriticAgent(input_dim=INPUT_DIM)
best_actor_path = f"{CHECKPOINT_DIR}/best_actor.pth"
best_critic_path = f"{CHECKPOINT_DIR}/best_critic.pth"
best_normalizer_path = f"{CHECKPOINT_DIR}/best_normalizer.npz"
if os.path.exists(best_actor_path) and os.path.exists(best_critic_path):
    agent.load(f"{CHECKPOINT_DIR}/best")
    if os.path.exists(best_normalizer_path):
        STATE_NORMALIZER.load(best_normalizer_path)
        print(f"Resumed from best checkpoint (normalizer stats loaded, "
              f"count={STATE_NORMALIZER.count:.0f}).")
    else:
        print("Resumed from best checkpoint, but no saved normalizer stats found — "
              "starting normalizer fresh. This will cause a distribution shift; "
              "prefer retraining from scratch if this happens.")
else:
    print("Starting from scratch.")

shaper = SeasonalRewardShaper(active_crops[0], verbose=True)

episode_returns = []
best_return     = -float("inf")
per_crop        = {c: {"returns": [], "al": [], "cl": [],
                       "entropy_w": [], "entropy_n": [],
                       "total_water": [], "total_nitrogen": [],
                       "profit": []} for c in active_crops}

# ── CSV logging setup ──────────────────────────────────────────────────────
CSV_PATH = "./training_log.csv"
csv_file  = open(CSV_PATH, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "episode", "crop", "season_return", "avg100_return",
    "actor_loss", "critic_loss",
    "entropy_water", "entropy_nitrogen",
    "total_water_mm", "total_nitrogen_kgha",
    "vs_baseline_pct",
    "grain_yield_kgha", "profit_usd_ha", "avg100_profit_usd_ha"
])
print(f"CSV logging → {CSV_PATH}\n")

# ── Per-DAP detail logging (for LAI trajectory / application-pattern figures) ──
# Full per-episode logging isn't needed for these figures — just a handful of
# representative episodes across training. Logged episodes: 1, then every
# DAILY_LOG_INTERVAL after that.
DAILY_LOG_INTERVAL = 500
DAILY_LOG_PATH = "./daily_log.csv"
daily_csv_file = open(DAILY_LOG_PATH, "w", newline="")
daily_csv_writer = csv.writer(daily_csv_file)
daily_csv_writer.writerow([
    "episode", "crop", "dap", "xlai", "topwt", "water_mm", "nitrogen_kg"
])
print(f"Daily (per-DAP) logging → {DAILY_LOG_PATH} "
      f"(episode 1, then every {DAILY_LOG_INTERVAL} episodes)\n")

def should_log_daily(ep):
    return ep == 1 or ep % DAILY_LOG_INTERVAL == 0

profit_history = []

for episode in range(1, NUM_EPISODES + 1):
    crop_name = random.choice(active_crops)
    env       = envs[crop_name]

    shaper.reset(crop_name, episode=episode)

    raw  = env.reset()
    raw  = raw[0] if isinstance(raw, tuple) else raw

    init_obs  = raw if isinstance(raw, dict) else {}
    init_dap  = int(init_obs.get("dap", 0))
    init_slot = nitrogen_is_allowed(crop_name, init_dap)
    STATE_NORMALIZER.update(flatten_obs(raw))
    state     = augment_state(raw, crop_name, is_nitrogen_slot=init_slot)

    done            = False
    episode_reward  = 0.0
    season_water_mm = 0.0
    season_n_kgha   = 0.0
    final_yield_kgha = 0.0
    last_obs_dict    = init_obs

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
            if should_log_daily(episode):
                daily_csv_writer.writerow([
                    episode, crop_name, dap,
                    round(float(obs_dict.get("xlai", 0.0)), 4),
                    round(float(obs_dict.get("topwt", 0.0)), 2),
                    0.0, 0.0
                ])
            next_dap  = int(obs_dict.get("dap", 0))
            next_slot = nitrogen_is_allowed(crop_name, next_dap)
            raw       = raw_next
            STATE_NORMALIZER.update(flatten_obs(raw_next))
            state     = augment_state(raw_next, crop_name, is_nitrogen_slot=next_slot)
            if obs_dict:
                last_obs_dict = obs_dict
            continue

        slot_open = nitrogen_is_allowed(crop_name, dap)

        water_idx, nitrogen_idx = agent.select_action(state, nitrogen_active=slot_open)
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

        if should_log_daily(episode):
            daily_csv_writer.writerow([
                episode, crop_name, dap,
                round(float(obs_dict.get("xlai", 0.0)), 4),
                round(float(obs_dict.get("topwt", 0.0)), 2),
                round(water_mm, 1), round(nitrogen_kg, 1)
            ])

        next_dap  = int(obs_dict.get("dap", 0))
        next_slot = nitrogen_is_allowed(crop_name, next_dap)
        raw       = raw_next
        STATE_NORMALIZER.update(flatten_obs(raw_next))
        state     = augment_state(raw_next, crop_name, is_nitrogen_slot=next_slot)
        if obs_dict:
            last_obs_dict = obs_dict

    # ── Terminal grain yield & common-scale profit ──────────────────────────
    # Same $/ha formula as the REINFORCE baseline (reward.py), extended with
    # an explicit nitrogen cost term so the two systems compare on one axis.
    final_yield_kgha = float(last_obs_dict.get("grnwt", 0.0))
    episode_profit = (
        YIELD_PRICE * final_yield_kgha
        - WATER_COST * season_water_mm
        - N_COST     * season_n_kgha
    )
    profit_history.append(episode_profit)
    avg100_profit = sum(profit_history[-100:]) / len(profit_history[-100:])

    print(f"    [EP {episode:5d} SUMMARY] "
          f"grain_yield={final_yield_kgha:7.1f} kg/ha | "
          f"profit=${episode_profit:8.2f}/ha")

    al, cl, _, ent_w, ent_n = agent.update(crop_name=crop_name)

    episode_returns.append(episode_reward)
    per_crop[crop_name]["returns"].append(episode_reward)
    per_crop[crop_name]["al"].append(al)
    per_crop[crop_name]["cl"].append(cl)
    per_crop[crop_name]["entropy_w"].append(ent_w)
    per_crop[crop_name]["entropy_n"].append(ent_n)
    per_crop[crop_name]["total_water"].append(season_water_mm)
    per_crop[crop_name]["total_nitrogen"].append(season_n_kgha)
    per_crop[crop_name]["profit"].append(episode_profit)

    window     = episode_returns[-100:]
    avg_return = sum(window) / len(window)
    vs_baseline_pct = ((avg_return - RANDOM_BASELINE) / abs(RANDOM_BASELINE)) * 100

    if avg_return > best_return:
        best_return = avg_return
        agent.save(f"{CHECKPOINT_DIR}/best")
        STATE_NORMALIZER.save(f"{CHECKPOINT_DIR}/best_normalizer")

    # ── Write one CSV row per episode ──────────────────────────────────────
    csv_writer.writerow([
        episode, crop_name,
        round(episode_reward, 4),
        round(avg_return, 4),
        round(al, 6),
        round(cl, 6),
        round(ent_w, 4),
        round(ent_n, 4),
        round(season_water_mm, 1),
        round(season_n_kgha, 1),
        round(vs_baseline_pct, 2),
        round(final_yield_kgha, 1),
        round(episode_profit, 2),
        round(avg100_profit, 2)
    ])
    if episode % 100 == 0:
        csv_file.flush()
        daily_csv_file.flush()

    if episode % LOG_INTERVAL == 0 or episode == 1:
        improvement = vs_baseline_pct
        print(f"\nEp {episode:6d} | avg100: {avg_return:8.2f} | "
              f"best: {best_return:8.2f} | "
              f"vs baseline: {improvement:+.1f}% | "
              f"profit(avg100): ${avg100_profit:8.2f}/ha")

        for c in active_crops:
            r = per_crop[c]["returns"]
            if r:
                w         = r[-LOG_INTERVAL:]
                avg_water = np.mean(per_crop[c]["total_water"][-LOG_INTERVAL:])
                avg_n     = np.mean(per_crop[c]["total_nitrogen"][-LOG_INTERVAL:])
                avg_prof  = np.mean(per_crop[c]["profit"][-LOG_INTERVAL:])
                avg_ent_w = np.mean(per_crop[c]["entropy_w"][-LOG_INTERVAL:])
                avg_ent_n = np.mean(per_crop[c]["entropy_n"][-LOG_INTERVAL:])
                crop_impr = ((np.mean(w) - RANDOM_BASELINE) / abs(RANDOM_BASELINE)) * 100
                print(f"  {c:8s} | n={len(r):5d} | "
                      f"avg={np.mean(w):8.2f} | "
                      f"vs_baseline={crop_impr:+.1f}% | "
                      f"water={avg_water:.0f}mm | "
                      f"N={avg_n:.0f}kg/ha | "
                      f"profit=${avg_prof:7.2f}/ha | "
                      f"al={np.mean(per_crop[c]['al'][-LOG_INTERVAL:]):7.4f} | "
                      f"cl={np.mean(per_crop[c]['cl'][-LOG_INTERVAL:]):.4f} | "
                      f"ent_w={avg_ent_w:.3f} | ent_n={avg_ent_n:.3f} "
                      f"(max=ln(5)={np.log(5):.3f})")
        print()

    if episode % SAVE_INTERVAL == 0:
        agent.save(f"{CHECKPOINT_DIR}/ep{episode:06d}")
        STATE_NORMALIZER.save(f"{CHECKPOINT_DIR}/ep{episode:06d}_normalizer")

csv_file.close()
daily_csv_file.close()
agent.save(f"{CHECKPOINT_DIR}/final")
STATE_NORMALIZER.save(f"{CHECKPOINT_DIR}/final_normalizer")
for e in envs.values():
    e.close()

final_improvement = ((best_return - RANDOM_BASELINE) / abs(RANDOM_BASELINE)) * 100
best_profit = max(profit_history) if profit_history else 0.0
print(f"\nDone.")
print(f"Best avg100 return : {best_return:.2f}")
print(f"Random baseline    : {RANDOM_BASELINE:.2f}")
print(f"Best improvement   : {final_improvement:+.1f}% over random policy")
print(f"Best single-episode profit : ${best_profit:.2f}/ha")
print(f"Training log saved : {CSV_PATH}")