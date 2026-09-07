from crop_utils import CROPS

# ── Shared economic constants for profit comparison (AC vs. REINFORCE) ─────
# Must stay numerically identical to REINFORCE/dssat_configuration.py so the
# two systems are compared on one common $/ha profit scale.
YIELD_PRICE = 0.25   # $ per kg/ha grain yield
WATER_COST  = 0.6    # $ per mm irrigation applied
N_COST      = 1.00   # $ per kg/ha nitrogen applied
                      # ASSUMPTION (placeholder, pending literature verification,
                      # same caveat status as the cotton agronomic parameters):
                      # approximate applied urea/N cost for a West African
                      # smallholder context. Adjust to your actual source figure
                      # or currency before final reporting.


def get_dssat_config(crop_name):
    if crop_name not in CROPS:
        raise ValueError(f"Unknown crop '{crop_name}'. Supported: {CROPS}")
    return {
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
        "cultivar":           "maize",
        "mode":               "irrigation",
        "random_weather":     True,
    }