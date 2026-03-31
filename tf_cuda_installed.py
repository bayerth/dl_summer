import os
import platform
import sys
import timeit


def import_tensorflow():
    try:
        import tensorflow as tf  # pylint: disable=import-outside-toplevel
        return tf, False
    except Exception as exc:  # pragma: no cover - import-time environment issue
        message = str(exc)
        is_metal_plugin_error = (
                "libmetal_plugin.dylib" in message and "_pywrap_tensorflow_internal.so" in message
        )
        if not is_metal_plugin_error:
            raise

        original_exists = os.path.exists

        def patched_exists(path):
            if path.endswith("tensorflow-plugins"):
                return False
            return original_exists(path)

        # Retry without auto-loading broken pluggable device libraries so the
        # script can still benchmark CPU execution and print a precise diagnosis.
        os.path.exists = patched_exists
        try:
            import tensorflow as tf  # pylint: disable=import-outside-toplevel,reimported
            return tf, True
        finally:
            os.path.exists = original_exists


tf, skipped_broken_metal_plugin = import_tensorflow()

print("Tensorflow v", tf.__version__)
print("Python", platform.python_version())
gpus = tf.config.list_physical_devices("GPU")
print("Num GPUs Available:", len(gpus))
print(gpus)
if len(gpus) == 0:
    print("No GPU detected. Exiting.")
    sys.exit(0)

if skipped_broken_metal_plugin:
    print(
            "\nDetected an incompatible tensorflow-metal installation. "
            "TensorFlow was imported without loading tensorflow-plugins, so GPU "
            "execution is disabled for this run.\n"
            "Long-term fix: install matching versions of tensorflow and "
            "tensorflow-metal, or remove tensorflow-metal if you only need CPU."
            )

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as exc:
        print(exc)

height = 200
width = 200
batch_size = 128


# filters = [[[1, 1], [1, 1],[0,1],[1,1]]]


def cpu():
    with tf.device("/cpu:0"):
        random_image_cpu = tf.random.normal((batch_size, height, width, 3))
        net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
        return tf.math.reduce_sum(net_cpu)


def gpu():
    with tf.device("/device:GPU:0"):
        random_image_gpu = tf.random.normal((batch_size, height, width, 3))
        net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
        return tf.math.reduce_sum(net_gpu)


# We run each op once to warm up; see: https://stackoverflow.com/a/45067900
print("--- warmup --------------------------")
print("Start CPU")
cpu()
gpu_available = bool(gpus) and not skipped_broken_metal_plugin
if gpu_available:
    print("Start GPU")
    gpu()
print("--- warmup finished -----------------")
number_of_runs = 30

print(
        f"Time (s) to convolve 32x7x7x3 filter over random images "
        f"{height}x{width}x3 (batch x height x width x channel). "
        f"Sum of {number_of_runs} runs."
        )
cpu_time = timeit.timeit("cpu()", number=number_of_runs, setup="from __main__ import cpu")
print("CPU (s): {:,.4f}".format(cpu_time))

if gpu_available:
    gpu_time = timeit.timeit("gpu()", number=number_of_runs, setup="from __main__ import gpu")
    print("GPU (s): {:,.4f}".format(gpu_time))
    print(f"GPU speedup over CPU: {cpu_time / gpu_time:.1f}x")
else:
    print("GPU benchmark skipped because no working TensorFlow GPU device is available.")

# with tensorflow-metal
# Start GPU
# --- warmup finished -----------------
# Time (s) to convolve 32x7x7x3 filter over random images 200x200x3 (batch x height x width x channel). Sum of 30 runs.
# CPU (s): 7.6102
# GPU (s): 0.6176
# GPU speedup over CPU: 12.3x
