# gpu_smoketest_tf.py
import os
# Logs plus calmes + stabilité cuDNN + mémoire incrémentale
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_CUDNN_AUTOTUNE_DEFAULT"] = "0"
os.environ["TF_CUDNN_WORKSPACE_LIMIT_IN_MB"] = "512"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
# Optionnel: pinner un GPU précis
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import time
import tensorflow as tf

tf.random.set_seed(1)

print("TF:", tf.__version__)
print("Build:", tf.sysconfig.get_build_info())
gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus)
if not gpus:
    raise SystemExit("❌ Aucun GPU vu par TensorFlow.")

for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

# Test 1 — Matmul sur GPU (rapide)
with tf.device('/GPU:0'):
    a = tf.random.normal([2000, 2000])
    t0 = time.time()
    c = tf.matmul(a, a)
    _ = c.numpy()
    print(f"✅ Matmul GPU ok en {time.time()-t0:.3f}s, c.shape={c.shape}")

# Test 2 — Conv2D cuDNN (forward)
x = tf.random.normal([8, 64, 64, 3])
conv = tf.keras.layers.Conv2D(64, 3, padding='same')
t0 = time.time()
y = conv(x, training=False)
_ = y.numpy()
print(f"✅ Conv2D forward ok en {time.time()-t0:.3f}s, y.shape={y.shape}")

# Test 3 — Mini entraînement (backward via cuDNN)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['acc'],
              jit_compile=False)

xs = tf.random.normal([32, 64, 64, 3])
ys = tf.random.uniform([32], maxval=10, dtype=tf.int32)

print("🚀 Entraînement 2 itérations…")
history = model.fit(xs, ys, batch_size=8, epochs=2, verbose=2)
print("✅ Entraînement minimal ok.")
