# Hyperparameters
GAMMA = 0.99            # Discount factor for future rewards
LR = 1e-7               # Matches the paper's explicit step-size alpha 
EPISODES = 5000       # Match paper: 20,000 crop seasons

# Economic reward parameters (directly from paper)
WATER_COST  = 0.6       # Cost per mm of irrigation ($/mm)
YIELD_PRICE = 0.25      # Revenue per kg/ha of grain ($/kg)

# Action Space: 0, 10, 20, 30, 40 mm (matches paper exactly) [cite: 1325]
ACTION_MAP = {0: 0.0, 1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0}

# State vector: stage + lai + 5 top ESW layers + cumulative rainfall + cumulative irrigation = 9 
INPUT_DIM   = 9
NUM_ACTIONS = len(ACTION_MAP)  # 5