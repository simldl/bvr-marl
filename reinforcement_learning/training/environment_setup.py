"""Environment setup and initialization for Windows/Linux compatibility."""

import os
import platform
import warnings


def setup_environment():
    """Configure environment variables for distributed training and Windows compatibility."""
    # ---------- Distributed training setup ----------
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"

    # Windows-specific configuration
    if platform.system() == "Windows":
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Suppress warnings
    os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
    warnings.filterwarnings("ignore", category=DeprecationWarning)


def setup_project_path():
    """Add project root to Python path."""
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For reproducibility (may reduce performance slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("=" * 80)
    print(f"RANDOM SEED SET TO: {seed}")
    print("=" * 80)
