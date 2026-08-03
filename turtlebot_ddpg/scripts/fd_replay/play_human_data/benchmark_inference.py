#!/usr/bin/env python
# Benchmark: pure input-to-output computation latency of the actor network.
# Measures only the neural network forward pass (state vector -> action),
# isolated from ROS message I/O and Gazebo simulation stepping.

import time
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.compat.v1.keras.backend as K

STATE_DIM = 28
N_WARMUP = 50
N_TRIALS = 1000


def build_actor_model():
    state_input = Input(shape=(STATE_DIM,))
    h1 = Dense(500, activation='relu')(state_input)
    h2 = Dense(500, activation='relu')(h1)
    h3 = Dense(500, activation='relu')(h2)
    delta_theta = Dense(1, activation='tanh')(h3)
    speed = Dense(1, activation='sigmoid')(h3)
    output = Concatenate()([delta_theta, speed])
    model = Model(inputs=state_input, outputs=output)
    return model


def main():
    sess = tf.Session()
    K.set_session(sess)

    model = build_actor_model()
    model.load_weights('actormodel-450-500.h5')

    rng = np.random.default_rng(0)
    states = rng.random((N_TRIALS, 1, STATE_DIM)).astype(np.float32)

    # warmup (first calls include graph/session overhead, not representative)
    for i in range(N_WARMUP):
        _ = model.predict(states[0], verbose=0)

    latencies_ms = []
    for i in range(N_TRIALS):
        t0 = time.perf_counter()
        _ = model.predict(states[i], verbose=0)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms = np.array(latencies_ms)
    print("=" * 60)
    print("Actor network forward-pass latency (state -> action)")
    print("N trials: %d (after %d warmup calls)" % (N_TRIALS, N_WARMUP))
    print("mean   : %.4f ms" % latencies_ms.mean())
    print("std    : %.4f ms" % latencies_ms.std())
    print("min    : %.4f ms" % latencies_ms.min())
    print("median : %.4f ms" % np.median(latencies_ms))
    print("p95    : %.4f ms" % np.percentile(latencies_ms, 95))
    print("max    : %.4f ms" % latencies_ms.max())
    print("=> effective throughput: %.1f Hz" % (1000.0 / latencies_ms.mean()))
    print("=" * 60)


if __name__ == "__main__":
    main()
