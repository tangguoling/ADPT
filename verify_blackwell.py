"""Fail-fast verification for an RTX 50-series TensorFlow installation."""

import tensorflow as tf

from core.gpu_runtime import validate_blackwell_runtime


print("TensorFlow:", tf.__version__)
print("Build:", tf.sysconfig.get_build_info())
print("GPUs:", tf.config.list_physical_devices("GPU"))
validate_blackwell_runtime(run_kernel=True)
print("Blackwell sm_120 elementwise + cuDNN Conv2D tests: OK")
