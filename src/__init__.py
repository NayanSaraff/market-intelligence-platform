"""
src package initializer
Sets a reproducible global random seed for the project.
"""
import os
import random
import numpy as np

# Global seed for reproducibility
GLOBAL_RANDOM_SEED = int(os.getenv('STOCKGRO_SEED', 42))
os.environ['PYTHONHASHSEED'] = str(GLOBAL_RANDOM_SEED)
random.seed(GLOBAL_RANDOM_SEED)
np.random.seed(GLOBAL_RANDOM_SEED)

# Export seed constant
__all__ = ["GLOBAL_RANDOM_SEED"]
