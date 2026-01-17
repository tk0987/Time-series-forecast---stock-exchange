import numpy as np
import tensorflow as tf
import pandas as pd
import os
import gc
from datetime import datetime

tf.config.run_functions_eagerly(True)
# =======================
# Normalization utilities etc
# =======================
def split_inputs(batch_states):
    # batch_states: (B, window_size, 35)
    x1, x2, x3, x4, x5 = tf.split(
        batch_states, [5, 5, 5, 5, 15], axis=-1
    )
    return x1, x2, x3, x4, x5
def normed(dataset):
    mean = np.mean(dataset, axis=(0, 1), keepdims=True)
    std = np.std(dataset, axis=(0, 1), keepdims=True) + 1e-6
    return (dataset - mean) / std


def norm_per_interval(data, n_intervals, n_features):
    """
    data: (batch, time, n_intervals * n_features)
    """
    data = data.copy()
    for i in range(n_intervals):
        start = i * n_features
        end = start + n_features
        mean = data[:, :, start:end].mean(axis=(0, 1), keepdims=True)
        std = data[:, :, start:end].std(axis=(0, 1), keepdims=True) + 1e-6
        data[:, :, start:end] = (data[:, :, start:end] - mean) / std
    return data


def norm_flat(x, eps=1e-8):
    """
    Flat-feature normalization (used for 5m auxiliary block)
    x: (symbols, timesteps, features)
    """
    mean = np.mean(x, axis=(0, 1), keepdims=True)
    std = np.std(x, axis=(0, 1), keepdims=True)
    return (x - mean) / (std + eps)

# =======================
# Embedding helpers (kept as-is)
# =======================

def build_interval_embedding(time_len, embed_dim):
    interval_ids = np.arange(time_len)
    interval_ids = tf.constant(interval_ids[np.newaxis, :], dtype=tf.int32)

    emb_layer = tf.keras.layers.Embedding(
        input_dim=len(INTERVALS),
        output_dim=embed_dim
    )
    return emb_layer(interval_ids)


def positional_encoding(seq_len, embed_dim):
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim)
    )
    pe = np.zeros((seq_len, embed_dim))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return tf.constant(pe[np.newaxis, ...], dtype=tf.float32)



def add_interval_embeddings(inputs, embed_dim):
    """
    inputs: (batch, time, interval * features)
    returns: (batch, time, interval * embed_dim)
    """
    interval_blocks = []
    n_features = len(FEATURES)

    for i, interval in enumerate(INTERVALS):
        start = i * n_features
        end = start + n_features

        xi = inputs[:, :, start:end]
        xi = tf.keras.layers.Dense(embed_dim, use_bias=False)(xi)

        interval_id = tf.constant([i], dtype=tf.int32)
        emb = tf.keras.layers.Embedding(
            input_dim=len(INTERVALS),
            output_dim=embed_dim
        )(interval_id)

        emb = tf.reshape(emb, (1, 1, embed_dim))
        xi = xi + emb
        interval_blocks.append(xi)

    return tf.keras.layers.Concatenate(axis=-1)(interval_blocks)

# =======================
# Repro / GPU
# =======================

tf.random.set_seed(1836)

print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

# =======================
# Global constants
# =======================

INTERVALS = ["15m", "30m", "1h", "4h"]
BASE_INTERVAL = "15m"
AUX_INTERVAL = "5m"

FEATURES = ["open", "high", "low", "close", "volume"]

data_dir = "all_data/binance"
window_size=96
model_path="new_arch.keras"

initial_cash=tf.constant(100,dtype=tf.float32)

# =======================
# Data loading with 5m auxiliary
# =======================

with open("symbols_used.txt", "r") as f:
    ordered_symbols = [s.strip().lower() for s in f if s.strip()]

files = {
    f.split("_")[0].lower(): f
    for f in os.listdir(data_dir)
    if f.endswith(".txt")
}

data_arrays = []

# How many 5m candles per interval
interval_to_5m = {
    "15m": 3,
    "30m": 6,
    "1h": 12,
    "4h": 48
}

for symbol in ordered_symbols:
    if symbol not in files:
        print(f"Missing file for {symbol}")
        continue

    df = pd.read_csv(
        os.path.join(data_dir, files[symbol]),
        sep="\t",
        header=None,
        names=["timestamp", "interval", *FEATURES]
    ).sort_values("timestamp")

    # ---- main intervals
    interval_arrays = {}
    for iv in INTERVALS:
        arr = df[df["interval"] == iv][FEATURES].to_numpy(np.float32)
        if arr.shape[0] == 0:
            break
        interval_arrays[iv] = arr

    if len(interval_arrays) != len(INTERVALS):
        print(f"Skipping {symbol}: missing intervals")
        continue

    # ---- auxiliary 5m
    aux_array = df[df["interval"] == AUX_INTERVAL][FEATURES].to_numpy(np.float32)
    if aux_array.shape[0] == 0:
        print(f"Skipping {symbol}: missing 5m")
        continue

    # ---- determine minimal aligned length
    # align all main intervals by number of 15m candles
    len_15m = interval_arrays["15m"].shape[0]
    len_30m = interval_arrays["30m"].shape[0] * 2
    len_1h  = interval_arrays["1h"].shape[0] * 4
    len_4h  = interval_arrays["4h"].shape[0] * 16

    min_len = min(len_15m, len_30m, len_1h, len_4h)
    if min_len < 10:
        print(f"Skipping {symbol}: too short after alignment")
        continue

    # ---- build main features tensor (all aligned to 15m)
    merged = []
    for iv in INTERVALS:
        arr = interval_arrays[iv]
        reps = interval_to_5m[iv] // 3  # convert to 15m steps
        if iv == "15m":
            merged.append(arr[:min_len])
        else:
            # upsample slower interval by repeating
            factor = interval_to_5m[iv] // 3
            upsampled = np.repeat(arr[: (min_len + factor - 1) // factor], factor, axis=0)[:min_len]
            merged.append(upsampled)

    merged = np.concatenate(merged, axis=1)  # shape: (min_len, n_intervals * n_features)

    # ---- 5m → 15m auxiliary
    aux = []
    for i in range(min_len):
        start_idx = i * 3
        end_idx = start_idx + 3
        if end_idx <= aux_array.shape[0]:
            window = aux_array[start_idx:end_idx]
        else:
            # if we run out of 5m data, repeat last known
            window = aux_array[-3:]
        aux.append(window.flatten())

    aux = np.stack(aux, axis=0)  # shape: (min_len, 15)
    merged = np.concatenate([merged, aux], axis=1)

    data_arrays.append(merged)

# =======================
# Final tensor + normalization
# =======================
if not data_arrays:
    raise RuntimeError("No valid symbols loaded")

data = np.stack(data_arrays).astype(np.float32)

main_feat_count = len(INTERVALS) * len(FEATURES)

data[..., :main_feat_count] = norm_per_interval(
    data[..., :main_feat_count],
    n_intervals=len(INTERVALS),
    n_features=len(FEATURES)
)

data[..., main_feat_count:] = norm_flat(
    data[..., main_feat_count:]
)

print("Final data shape:", data.shape)
print("NaNs:", np.isnan(data).sum())

gc.collect()
assert data.ndim == 3
assert data.shape[-1] == main_feat_count + 15


# -----------------------
# Model building blocks
# -----------------------
def transformer_encoder_block(
    x,
    embed_dim,
    num_heads,
    ff_dim,
    dropout=0.1
):
    # Layer normalization (Pre-Norm)
    x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    # Self-attention
    attn_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim,
        dropout=dropout
    )(x_norm, x_norm)

    attn_output = tf.keras.layers.Dropout(dropout)(attn_output)
    x = tf.keras.layers.Add()([x, attn_output])

    # Feed-forward network
    x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x_norm)
    ffn_output = tf.keras.layers.Dropout(dropout)(ffn_output)
    ffn_output = tf.keras.layers.Dense(embed_dim)(ffn_output)

    ffn_output = tf.keras.layers.Dropout(dropout)(ffn_output)
    return tf.keras.layers.Add()([x, ffn_output])

# from tensorflow.keras import layers

def transformer_encoder_double_attention(
    x,
    embed_dim,
    num_heads,
    ff_dim,
    dropout=0.1
):
    # ---- Attention 1 ----
    x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    attn1 = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim,
        dropout=dropout
    )(x_norm, x_norm)

    attn1 = tf.keras.layers.Dropout(dropout)(attn1)
    x = tf.keras.layers.Add()([x, attn1])

    # ---- Attention 2 ----
    x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    attn2 = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim,
        dropout=dropout
    )(x_norm, x_norm)

    attn2 = tf.keras.layers.Dropout(dropout)(attn2)
    x = tf.keras.layers.Add()([x, attn2])

    # ---- Feed Forward ----
    x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    ffn = tf.keras.layers.Dense(ff_dim, activation="relu")(x_norm)
    ffn = tf.keras.layers.Dropout(dropout)(ffn)
    ffn = tf.keras.layers.Dense(embed_dim)(ffn)
    ffn = tf.keras.layers.Dropout(dropout)(ffn)

    return tf.keras.layers.Add()([x, ffn])

def transformer_encoder_double_attention_resh(
    x,
    embed_dim,
    num_heads,
    ff_dim,
    dropout=0.1
):
    # ---- Attention 1 ----
    x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.Permute((2, 1))(x)
    x_norm = tf.keras.layers.Permute((2, 1))(x_norm)

    attn1 = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim,
        dropout=dropout
    )(x_norm, x_norm)

    attn1 = tf.keras.layers.Dropout(dropout)(attn1)
    x = tf.keras.layers.Add()([x, attn1])

    # ---- Attention 2 ----
    x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    attn2 = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim,
        dropout=dropout
    )(x_norm, x_norm)

    attn2 = tf.keras.layers.Dropout(dropout)(attn2)
    x = tf.keras.layers.Add()([x, attn2])

    # ---- Feed Forward ----
    x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    ffn = tf.keras.layers.Dense(ff_dim, activation="relu")(x_norm)
    ffn = tf.keras.layers.Dropout(dropout)(ffn)
    ffn = tf.keras.layers.Dense(embed_dim)(ffn)
    ffn = tf.keras.layers.Dropout(dropout)(ffn)
    
    last = tf.keras.layers.Add()([x, ffn])
    
    return tf.keras.layers.Permute((2, 1))(last)

def build_actor_critic_transformer(
    embed_dim=512,
    num_heads=8,
    ff_dim=2048,
    num_layers=3,
    dropout=0.04,
    num_actions=2
):
    inputs15m = tf.keras.layers.Input(shape=(window_size, 5))
    inputs30m = tf.keras.layers.Input(shape=(window_size, 5))
    inputs1h  = tf.keras.layers.Input(shape=(window_size, 5))
    inputs4h  = tf.keras.layers.Input(shape=(window_size, 5))
    inputs5m  = tf.keras.layers.Input(shape=(window_size, 15))

    tensors = [inputs15m, inputs30m, inputs1h, inputs4h, inputs5m]

    processed = []
    for t in tensors:
        processed.append(tf.keras.layers.Dense(16, activation="relu")(t))

    x = tf.keras.layers.Concatenate(axis=-1)(processed)
    x = tf.keras.layers.Dense(embed_dim)(x)
    x += positional_encoding(window_size, embed_dim)
    x = tf.keras.layers.Dropout(dropout)(x)

    x2 = x
    for _ in range(num_layers):
        x = transformer_encoder_double_attention(
            x, embed_dim, num_heads, ff_dim, dropout
        )
        x2 = transformer_encoder_double_attention_resh(
            x2, window_size, num_heads, ff_dim, dropout
        )

    x = tf.keras.layers.Add()([x, x2])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # -------- ACTOR --------
    policy = tf.keras.layers.Dense(
        num_actions, activation="softmax", name="policy"
    )(x)

    # -------- CRITIC --------
    value = tf.keras.layers.Dense(
        1, activation=None, name="value"
    )(x)

    return tf.keras.Model(
        inputs=[inputs15m, inputs30m, inputs1h, inputs4h, inputs5m],
        outputs=[policy, value]
    )

def simulate_trade_step(policy, curr_price, next_price,
                        position, cash, fee_rate=0.001):
    """
    policy: model output probabilities (1,2)
    curr_price: current close price
    next_price: next close price
    position: number of units currently held (0 or 1 for simplicity)
    cash: current cash balance
    fee_rate: 0.001 = 0.1% Binance fee

    Returns:
        new_position, new_cash, trade_action, pnl
    """

    # Model decision: 1 = go long, 0 = stay flat
    predicted_label = int(tf.argmax(policy, axis=1).numpy()[0]) # 0=down, 1=up

    trade_action = "HOLD"
    pnl = 0.0

    # ---- CASE 1: Model wants to be LONG ----
    if predicted_label == 1:
        if position == 0:
            # BUY
            trade_action = "BUY"
            cost = curr_price * (1 + fee_rate)
            if cash >= cost:
                cash -= cost
                position = 1

        else:
            # Already long → HOLD
            trade_action = "HOLD"

    # ---- CASE 2: Model wants to be FLAT ----
    else:
        if position == 1:
            # SELL
            trade_action = "SELL"
            revenue = curr_price * (1 - fee_rate)
            cash += revenue
            position = 0

            # Profit from the trade
            pnl = (curr_price - next_price)  # optional: mark-to-market

        else:
            # Already flat → HOLD
            trade_action = "HOLD"

    return position, cash, trade_action, pnl


'''if os.path.exists(model_path):
    net = tf.keras.models.load_model(model_path,
    compile=False)
    print("\n\n\nLoaded model architecture + weights.\n\n\n")
else:
    net = build_time_series_transformer(
        embed_dim = 512,
        num_heads = 8,
        ff_dim = 2048,
        num_layers=3,
        num_outputs=2,
        dropout=0.04,
        task="classification"
    )
    print("Built new model.")
net.summary()'''

# =========================
# A2C / PPO CONFIG
# =========================
USE_PPO = True          # <- set False = A2C, True = PPO
gamma = 0.99
rollout_len = 16
entropy_coef = 0.01
value_coef = 0.5

# PPO only
clip_eps = 0.2
ppo_epochs = 4

# =========================
# OPTIMIZER
# =========================
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=3.5e-4,
    weight_decay=0.01,
    beta_1=0.9,
    beta_2=0.98
)

# =========================
# ROLLOUT BUFFER
# =========================
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.logps = []

    def clear(self):
        self.__init__()

# =========================
# A2C UPDATE
# =========================
@tf.function
def a2c_update(net, optimizer, buffer, last_value):
    # ---- compute returns ----
    returns = []
    R = last_value
    for r in reversed(buffer.rewards):
        R = r + gamma * R
        returns.insert(0, R)

    returns = tf.convert_to_tensor(returns, tf.float32)
    values = tf.convert_to_tensor(buffer.values, tf.float32)
    advantages = returns - values

    with tf.GradientTape() as tape:
        actor_loss = -tf.reduce_mean(
            tf.stack(buffer.logps) * tf.stop_gradient(advantages)
        )
        critic_loss = tf.reduce_mean(tf.square(returns - values))
        loss = actor_loss + value_coef * critic_loss

    grads = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return loss

# =========================
# PPO UPDATE
# =========================
@tf.function
def ppo_update(net, optimizer, states, actions, returns, advantages, old_logps):
    for _ in range(ppo_epochs):
        with tf.GradientTape() as tape:

            # ---- FIX: split states into 5 inputs ----
            x1, x2, x3, x4, x5 = split_inputs(states)
            policy, values = net([x1, x2, x3, x4, x5], training=True)

            action_onehot = tf.one_hot(actions, 2)
            logps = tf.reduce_sum(
                action_onehot * tf.math.log(policy + 1e-8), axis=1
            )

            ratios = tf.exp(logps - old_logps)
            clipped = tf.clip_by_value(
                ratios, 1.0 - clip_eps, 1.0 + clip_eps
            )

            actor_loss = -tf.reduce_mean(
                tf.minimum(ratios * advantages, clipped * advantages)
            )

            critic_loss = tf.reduce_mean(
                tf.square(returns - tf.squeeze(values))
            )

            entropy = -tf.reduce_mean(
                tf.reduce_sum(policy * tf.math.log(policy + 1e-8), axis=1)
            )

            loss = (
                actor_loss
                + value_coef * critic_loss
                - entropy_coef * entropy
            )

        grads = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))

        


# =========================
# TRAINING LOOP
# =========================
net = build_actor_critic_transformer()
net.summary()

for epoch in range(200):
    print(f"\nEpoch {epoch+1}")
    epoch_loss = 0.0

    for symbol in range(data.shape[0]):
        buffer = RolloutBuffer()
        position, cash = 0, 1000.0

        for i in range(data.shape[1] - window_size - 1):

            window = tf.convert_to_tensor(
                data[symbol, i:i+window_size], tf.float32
            )
            curr_close = data[symbol, i+window_size-1, 3]
            next_close = data[symbol, i+window_size, 3]

            inp = tf.expand_dims(window, axis=0)
            x1, x2, x3, x4, x5 = tf.split(inp, [5,5,5,5,15], axis=-1)

            policy, value = net([x1,x2,x3,x4,x5], training=True)

            # ---- sample action ----
            action = tf.random.categorical(tf.math.log(policy), 1)[0, 0]
            logp = tf.math.log(policy[0, action] + 1e-8)

            # ---- reward ----
            reward = 1e4 * (next_close - curr_close) / curr_close
            

            buffer.states.append(inp)
            buffer.actions.append(action)
            buffer.rewards.append(reward)
            buffer.values.append(value[0,0])
            buffer.logps.append(logp)

            position, cash, _, _ = simulate_trade_step(
                policy, curr_close, next_close, position, cash
            )
            if i%25==0:
                print(f"reward: {reward:.4f}")
                print(position,cash)
            # ---- update ----
            if len(buffer.rewards) == rollout_len:
                _, last_val = net([x1,x2,x3,x4,x5], training=False)

                if USE_PPO:
                    returns = []
                    R = last_val[0,0]
                    for r in reversed(buffer.rewards):
                        R = r + gamma * R
                        returns.insert(0, R)

                    returns = tf.convert_to_tensor(returns, tf.float32)
                    values = tf.convert_to_tensor(buffer.values, tf.float32)
                    advantages = returns - values

                    ppo_update(
                        net, optimizer,
                        tf.concat(buffer.states, axis=0),
                        tf.convert_to_tensor(buffer.actions),
                        returns,
                        advantages,
                        tf.convert_to_tensor(buffer.logps)
                    )

                else:
                    loss = a2c_update(
                        net, optimizer, buffer, last_val[0,0]
                    )
                    epoch_loss += float(loss)

                buffer.clear()

        print(f"Epoch loss: {epoch_loss:.4f}")

        net.save(model_path)
