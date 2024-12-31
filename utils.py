import itertools
import yaml

def generate_sweep_config(config, slurm_index, sweep_params=None):
    """
    Generates a configuration dictionary based on the sweep parameters.
    
    Args:
        config (dict): Original configuration dictionary.
        sweep_params (list of str): List of parameter names to sweep over.
        slurm_index (int): Index to pick the specific combination (e.g., from SLURM job array).

    Returns:
        dict: Updated configuration dictionary.
    """
    def ensure_list(value):
        """Ensures the value is a list."""
        return value if isinstance(value, list) else [value]
    
    # If no sweep parameters are provided, sweep over all params
    sweep_params = sweep_params or list(config.keys())

    # Create combinations of all parameter values specified in `sweep_params`
    param_combinations = list(itertools.product(*[ensure_list(config.get(param, [])) for param in sweep_params]))

    # Extract the parameter values for the given SLURM index
    selected_combination = param_combinations[slurm_index]

    # Update the config with selected parameter values
    updated_config = config.copy()
    for param, value in zip(sweep_params, selected_combination):
        updated_config[param] = value

    # Handle any specific logic (example: setting OPT to 'NONE' if LR3 == 0)
    if updated_config.get("LAST_LAYER_LRS", 1) == 0:
        updated_config["OPT"] = "NONE"

    return updated_config

# Config loader and helper functions
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)