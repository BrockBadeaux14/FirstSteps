"""Test setup: same GPU memory behavior as training (set before JAX starts)."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
