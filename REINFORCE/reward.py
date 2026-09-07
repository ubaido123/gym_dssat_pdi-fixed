import dssat_configuration

def compute_reward(raw_obs, next_obs, irrigation_depth, done):
    # FIX #1: When the episode is done, next_obs IS the terminal state S+
    # where yield is realised. The paper defines R+ = p * Yield at S+.
    # The original code incorrectly read yield from raw_obs (the prior state).
    # We only fall back to raw_obs if next_obs is None (mid-episode None returns).
    if next_obs is None:
        next_obs = raw_obs

    # Daily penalty: economic cost of water applied (paper: R_{t+1} = -c * A_t)
    reward = -dssat_configuration.WATER_COST * irrigation_depth

    # Terminal bonus: grain revenue at harvest (paper: R+ = p * Yield)
    # Read from next_obs (the terminal observation S+), not raw_obs.
    if done:
        yield_kg = next_obs.get('grnwt', 0.0)
        reward += dssat_configuration.YIELD_PRICE * yield_kg

    return reward