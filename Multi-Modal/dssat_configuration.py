from crop_utils import CROPS

# TODO: verify these against your gym_dssat_pdi install.
# "cultivar" must match a cultivar code gym_dssat_pdi/DSSAT recognizes for
# that crop's experiment file. "maize" below is a placeholder — check what
# your DSSAT genotype files (.CUL) actually call the cotton cultivar, and
# whether cotton needs mode="all" (like your AC-NITROGEN maize setup) or
# a different experiment file path.
CROP_CONFIGS = {
    "maize": {
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
        "cultivar":           "maize",
        "mode":               "irrigation",
        "random_weather":     True,
    },
    "cotton": {
        "run_dssat_location": "/opt/dssat_pdi/run_dssat",
        "cultivar":           "cotton",  # TODO: replace with real cultivar code
        "mode":               "irrigation",
        "random_weather":     True,
    },
}


def get_dssat_config(crop_name):
    if crop_name not in CROPS:
        raise ValueError(f"Unknown crop '{crop_name}'. Supported: {CROPS}")
    if crop_name not in CROP_CONFIGS:
        raise ValueError(f"No DSSAT config defined for crop '{crop_name}'.")
    return CROP_CONFIGS[crop_name]