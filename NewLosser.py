import numpy as np
import tensorflow as tf
import pandas as pd
import os
import gc
from datetime import datetime

tf.config.run_functions_eagerly(False)
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


# def simulate_trade_step(policy, curr_price, next_price,
#                         position, cash, holding, fee_rate=0.001):
#     """
#     policy: model output probabilities (1,2)
#             [p_no, p_yes]
#     curr_price: current close price
#     next_price: next close price
#     position: 0 or 1 (kept for compatibility)
#     cash: current cash balance
#     holding: amount of crypto units held
#     fee_rate: 0.001 = 0.1% fee

#     Returns:
#         new_position, new_cash, trade_action, pnl, holding
#     """

#     # --- Extract buy probability ---
#     policy = tf.squeeze(policy).numpy()
#     p_buy = float(policy[1])   # ∈ [0, 1]

#     trading_fund = 0.25 * cash
#     # free_cash = cash - trading_fund

#     trade_action = "HOLD"
#     pnl = 0.0

#     # Desired exposure in USD
#     target_value = p_buy * trading_fund

#     # Current exposure in USD
#     current_value = holding * curr_price

#     value_diff = target_value - current_value

#     # ---- BUY (increase exposure) ----
#     if value_diff > 0:
#         trade_action = "BUY"

#         buy_value = min(value_diff, trading_fund, cash)
#         fee = buy_value * fee_rate

#         units_bought = (buy_value - fee) / curr_price

#         cash -= buy_value
#         holding += units_bought

#     # ---- SELL (decrease exposure) ----
#     elif value_diff < 0:
#         trade_action = "SELL"

#         sell_value = min(-value_diff, current_value)
#         units_sold = sell_value / curr_price

#         revenue = sell_value * (1 - fee_rate)

#         cash += revenue
#         holding -= units_sold

#     # Update position flag (compatibility)
#     position = 1 if holding > 0 else 0

#     # Mark-to-market PnL
#     pnl = holding * (next_price - curr_price)

#     return position, cash, trade_action, pnl, holding
def simulate_trade_step(
    policy,
    curr_price,
    next_price,
    cash,
    holding,
    fee_rate=0.001,
    max_trade_frac=0.25
):
    """
    Fully differentiable trade step.
    All inputs/outputs are tf.Tensor.
    """

    policy = tf.squeeze(policy)
    p_buy = policy[1]                       # tensor ∈ [0,1]

    curr_price = tf.convert_to_tensor(curr_price, tf.float32)
    next_price = tf.convert_to_tensor(next_price, tf.float32)
    cash = tf.convert_to_tensor(cash, tf.float32)
    holding = tf.convert_to_tensor(holding, tf.float32)

    # ---- desired allocation ----
    trading_fund = max_trade_frac * cash
    target_value = p_buy * trading_fund

    # ---- current exposure ----
    current_value = holding * curr_price

    # ---- rebalance delta (USD) ----
    value_diff = target_value - current_value

    # ---- smooth buy/sell split ----
    buy_value = tf.nn.relu(value_diff)
    sell_value = tf.nn.relu(-value_diff)

    # ---- cap by available resources ----
    buy_value = tf.minimum(buy_value, cash)
    sell_value = tf.minimum(sell_value, current_value)

    # ---- apply fees ----
    buy_units = (buy_value * (1.0 - fee_rate)) / curr_price
    sell_units = sell_value / curr_price

    # ---- update state ----
    new_holding = holding + buy_units - sell_units
    new_cash = cash - buy_value + sell_value * (1.0 - fee_rate)

    # ---- mark-to-market pnl ----
    pnl = new_holding * (next_price - curr_price)

    return new_cash, new_holding, pnl

# =========================
# A2C / PPO CONFIG
# =========================
# USE_PPO = False         # <- set False = A2C, True = PPO
gamma = 0.99
rollout_len = 16
entropy_coef = 0.01
value_coef = 0.5

# PPO only
clip_eps = 0.2
# ppo_epochs = 4

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
        self.rewards = []
        self.values = []
        self.equity = []

    def clear(self):
        self.__init__()

# =========================
# A2C UPDATE
# =========================

@tf.function(reduce_retracing=True)
def a2c_update(net, optimizer, states, returns, advantages, equity):
    with tf.GradientTape() as tape:
        x1, x2, x3, x4, x5 = split_inputs(states)
        policy, values = net([x1, x2, x3, x4, x5], training=True)

        # allocation = buy probability
        alloc = policy[:, 1]

        actor_loss = -tf.reduce_mean(
            alloc * tf.stop_gradient(advantages)
        )

        critic_loss = tf.reduce_mean(
            tf.square(returns - tf.squeeze(values))
        )
        equity_loss=tf.reduce_mean(equity)
        a2c_loss = actor_loss + value_coef * critic_loss
        total_loss = a2c_loss - equity_loss

    grads = tape.gradient(total_loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    return total_loss


        
if os.path.exists(model_path):
    net = tf.keras.models.load_model(model_path,
    compile=False)
    print("\n\n\nLoaded model architecture + weights.\n\n\n")
else:
    net = build_actor_critic_transformer()
net.summary()
# USE_PPO=False

# =========================
# TRAINING LOOP
# =========================

for epoch in range(2000):
    print(f"\nEpoch {epoch+1}")
    epoch_loss = 0.0

    for symbol in range(data.shape[0]):
        buffer = RolloutBuffer()

        cash = tf.constant(1000.0,dtype=tf.float32)
        holding = tf.constant(0.0, dtype=tf.float32)

        # equity_curve = []

        for i in range(data.shape[1] - window_size - 1):

            # ---- build input window ----
            window = tf.convert_to_tensor(
                data[symbol, i:i + window_size], tf.float32
            )

            curr_close = data[symbol, i + window_size - 1, 3]
            next_close = data[symbol, i + window_size, 3]

            inp = tf.expand_dims(window, axis=0)
            x1, x2, x3, x4, x5 = tf.split(inp, [5, 5, 5, 5, 15], axis=-1)

            # ---- forward pass ----
            policy, value = net([x1, x2, x3, x4, x5], training=True)

            # ---- equity BEFORE trade ----
            prev_equity = cash + holding * curr_close

            # ---- environment step (deterministic policy) ----
            cash, holding, pnl = simulate_trade_step(
                policy=policy,
                curr_price=curr_close,
                next_price=next_close,
                cash=cash,
                holding=holding
            )

            # ---- equity AFTER price move ----
            equity = cash + holding * next_close
            buffer.equity.append(equity)
            # equity_curve.append(equity)

            # ---- reward = normalized equity delta ----
            reward = equity - prev_equity
            reward = 1e5*reward / (prev_equity + 1e-8)

            # ---- store rollout ----
            buffer.states.append(inp.numpy())
            buffer.rewards.append(float(reward))
            buffer.values.append(float(value[0,0]))

            # equity_tensor = tf.convert_to_tensor(equity_curve, dtype=tf.float32)

            # ---- logging ----
            if i % 25 == 0:
                print(
                    f"[{i:04d}] "
                    f"price={curr_close:.6f}->{next_close:.6f} | "
                    f"reward={reward:+.4f} | "
                    f"equity={equity-1001.0:.2f}"
                )

            # ---- update ----
            if len(buffer.rewards) == rollout_len:
                _, last_val = net([x1, x2, x3, x4, x5], training=False)
                last_val = last_val[0, 0]

                # ---- compute returns & advantages ----
                returns = []
                R = last_val
                for r in reversed(buffer.rewards):
                    R = r + gamma * R
                    returns.insert(0, R)

                returns = tf.convert_to_tensor(returns, tf.float32)
                values = tf.convert_to_tensor(buffer.values, tf.float32)
                advantages = returns - values

                # ---- update network (A2C-style) ----
                loss = a2c_update(
                    net,
                    optimizer,
                    tf.concat(buffer.states, axis=0),
                    returns,
                    advantages,
                    buffer.equity
                )


                epoch_loss += float(loss)
                buffer.clear()

        print(
            f"Symbol done | "
            f"Final equity: {equity:.2f} | "
        )

    print(f"Epoch loss: {epoch_loss:.4f}")
    net.save(model_path)
 
