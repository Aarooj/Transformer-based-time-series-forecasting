import os
os.environ['PYTHONHASHSEED'] = str(2)
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import initializers
from keras import backend
from sklearn.preprocessing import MinMaxScaler
import joblib

# =============================================================================
# CONFIGURABLE HYPERPARAMETERS - 256 CAPACITY CONFIGURATION
# =============================================================================

# Data parameters
DATA_CONFIG = {
    'samples': 12000,
    'seq_length': 20,           # Increased from 12
    'batch_size': 32,           # Increased from 16
    'molecules': 2,
    'species': '2_22',
    'downsample_factor': 1,
    'validation_split': 0.8,    # 80/20 split
}

# Model architecture parameters - 384 CAPACITY (INCREASED FROM 256)
MODEL_CONFIG = {
    'head_size': 128,           # INCREASED from 256 to 384 (50% increase)
    'num_heads': 4,             # INCREASED from 4 to 6 (more attention diversity)
    'ff_dim': 128,              # INCREASED from 256 to 384 (50% increase)
    'num_layers': 4,            # INCREASED from 3 to 4 (more depth)
    'dropout_rate': 0.1,        # REDUCED from 0.25 (allow more learning)
    'lstm_units': 32,          # INCREASED from 128 to 192 (50% increase)
    'l2_reg': 0.0002,           # REDUCED from 0.0005 (less regularization)
}

# Training parameters
TRAINING_CONFIG = {
    'learning_rate': 0.001,    # Reduced learning rate
    'loss_function': 'mape',    # Loss function
    'epochs': 2000,             # Allow more epochs for higher capacity model
    'shuffle': False,           # Training shuffle
}

# Callback parameters
CALLBACK_CONFIG = {
    'early_stop_patience': 50,      # INCREASED from 50 (larger models need more patience)
    'early_stop_min_delta': 1e-5,   # REDUCED from 1e-5 (more sensitive)
    'lr_reduce_patience': 20,       # INCREASED from 20 (more patient)
    'lr_reduce_factor': 0.5,        # REDUCED from 0.8 (more aggressive LR reduction)
    'min_lr': 1e-7,                 # REDUCED from 1e-6 (allow lower LR)
    'overfitting_threshold': 0.015, # REDUCED from 0.02 (more sensitive)
}

# Display configuration
print("?? ENHANCED 384 CAPACITY TRANSFORMER+LSTM MODEL:")
print("="*70)
print(f"?? Data Configuration:")
for key, value in DATA_CONFIG.items():
    print(f"   - {key}: {value}")

print(f"\n??? Model Architecture (384 Capacity - INCREASED):")
for key, value in MODEL_CONFIG.items():
    print(f"   - {key}: {value}")

print(f"\n? Training Configuration:")
for key, value in TRAINING_CONFIG.items():
    print(f"   - {key}: {value}")

print(f"\n?? Callback Configuration:")
for key, value in CALLBACK_CONFIG.items():
    print(f"   - {key}: {value}")

print(f"\n?? EXPECTED IMPROVEMENTS FROM 256?384 CAPACITY:")
print(f"   - Current: Train ~0.015, Val ~0.016")
print(f"   - Target:  Train < 0.010, Val < 0.012")
print(f"   - 50% increase in model capacity")
print(f"   - More attention heads (4?6)")
print(f"   - Deeper transformer (3?4 layers)")

# =============================================================================
# SETUP AND INITIALIZATION
# =============================================================================

def reset_random_seeds():
    os.environ['PYTHONHASHSEED'] = str(2)
    tf.random.set_seed(28)
    np.random.seed(28)
    random.seed(28)

reset_random_seeds()

# ?? GPU Check
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n?? {len(gpus)} GPU(s) detected: {gpus}")
else:
    print("\n?? No GPU detected. Using CPU.")

from tensorflow.python.client import device_lib
print("??? Available devices:")
print(device_lib.list_local_devices())

# Load data
data = np.load('%s_data.npy' % DATA_CONFIG['species'])
full_data = data.copy()
data = data[:DATA_CONFIG['samples']:DATA_CONFIG['downsample_factor'], :]

print(f"\n?? Downsampled data shape: {np.shape(data)}")
binary_flags = data[:, 6]
ones_indices = np.where(binary_flags == 1)[0]
print(f"?? Found {len(ones_indices)} instances of 1 in the binary flag")
if len(ones_indices) > 0:
    print(f"First 10 indices with flag 1: {ones_indices[:10]}")

# Sequence preparation
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length), :]
        y = data[i + seq_length, :6]  # Only predict first 6 columns (molecular coordinates)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

input_, output = create_sequences(data, DATA_CONFIG['seq_length'])
indices = np.arange(len(input_))
np.random.shuffle(indices)
input_ = input_[indices]
output = output[indices]

split_index = int(len(input_) * DATA_CONFIG['validation_split'])
x1_train, y1_train = input_[:split_index], output[:split_index]
x1_val, y1_val = input_[split_index:], output[split_index:]

# =============================================================================
# MODEL ARCHITECTURE - TRANSFORMER + LSTM (256 CAPACITY)
# =============================================================================

from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    ffn = tf.keras.Sequential([
        layers.Dense(ff_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(MODEL_CONFIG['l2_reg'])),
        layers.Dropout(dropout),
        layers.Dense(inputs.shape[-1], kernel_regularizer=tf.keras.regularizers.l2(MODEL_CONFIG['l2_reg'])),
    ])
    x = ffn(x)
    return layers.LayerNormalization(epsilon=1e-6)(x + inputs)

def build_transformer_model_with_lstm(input_shape, config, output_dim):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Add positional encoding
    x = x + positional_encoding(input_shape[0], input_shape[1])
    
    # Transformer layers (INCREASED FOR BETTER PERFORMANCE)
    for _ in range(config['num_layers']):
        x = transformer_encoder(x, config['head_size'], config['num_heads'], config['ff_dim'], config['dropout_rate'])
    
    # LSTM layers (INCREASED CAPACITY - 3 layers)
    x = layers.LSTM(config['lstm_units'], return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(config['l2_reg']))(x)
    # x = layers.LSTM(config['lstm_units'], return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(config['l2_reg']))(x)
    x = layers.LSTM(config['lstm_units'], kernel_regularizer=tf.keras.regularizers.l2(config['l2_reg']))(x)  # Final LSTM without return_sequences
    
    # Output layer with regularization
    outputs = layers.Dense(output_dim, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(config['l2_reg']))(x)
    return Model(inputs=inputs, outputs=outputs)

print(f"\n??? Building model with 384 capacity (INCREASED FROM 256):")
print(f"   head_size: {MODEL_CONFIG['head_size']} (increased from 256 to 384)")
print(f"   num_heads: {MODEL_CONFIG['num_heads']} (increased from 4 to 6)")
print(f"   ff_dim: {MODEL_CONFIG['ff_dim']} (increased from 256 to 384)")
print(f"   num_layers: {MODEL_CONFIG['num_layers']} (increased from 3 to 4)")
print(f"   lstm_units: {MODEL_CONFIG['lstm_units']} (increased from 128 to 192)")
print(f"   dropout_rate: {MODEL_CONFIG['dropout_rate']} (reduced from 0.25 to 0.2)")
print(f"   seq_length: {DATA_CONFIG['seq_length']} (optimal at 20)")
print(f"   batch_size: {DATA_CONFIG['batch_size']} (optimal at 32)")

# Compile model with lower learning rate
model = build_transformer_model_with_lstm(
    input_shape=(DATA_CONFIG['seq_length'], input_.shape[2]),
    config=MODEL_CONFIG,
    output_dim=DATA_CONFIG['molecules'] * 3  # 6 outputs for molecular coordinates
)
model.compile(
    optimizer=Adam(learning_rate=TRAINING_CONFIG['learning_rate']), 
    loss=TRAINING_CONFIG['loss_function']
)
model.summary()

# =============================================================================
# TRAINING CALLBACKS
# =============================================================================

# Custom monitoring callback for overfitting/underfitting (MORE AGGRESSIVE FOR SMALL DATASET)
class OverfittingMonitor(keras.callbacks.Callback):
    def __init__(self, threshold=None):
        super().__init__()
        self.threshold = threshold or CALLBACK_CONFIG['overfitting_threshold']
        
    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        
        # Calculate the difference between training and validation loss
        loss_diff = val_loss - train_loss
        loss_ratio = val_loss / train_loss if train_loss > 0 else float('inf')
        
        print(f"\n?? Epoch {epoch + 1} Analysis:")
        print(f"   Training Loss: {train_loss:.6f}")
        print(f"   Validation Loss: {val_loss:.6f}")
        print(f"   Loss Difference: {loss_diff:.6f}")
        print(f"   Loss Ratio (val/train): {loss_ratio:.3f}")
        
        # Check for overfitting (more aggressive for small dataset)
        if loss_diff > self.threshold and loss_ratio > 1.15:
            print(f"   ??  WARNING: Potential overfitting detected!")
            print(f"      Validation loss is {loss_diff:.4f} higher than training loss")
        
        # Check for underfitting
        elif train_loss > 0.5 and val_loss > 0.5 and epoch > 5:
            print(f"   ??  WARNING: Potential underfitting detected!")
            print(f"      Both losses are high after {epoch + 1} epochs")
        else:
            print(f"   ? Training appears balanced")

# ?? CALLBACKS OPTIMIZED FOR 256 CAPACITY
callback = EarlyStopping(
    monitor="val_loss", 
    min_delta=CALLBACK_CONFIG['early_stop_min_delta'], 
    patience=CALLBACK_CONFIG['early_stop_patience'], 
    verbose=1, 
    mode="min"
)
checkpoint = ModelCheckpoint(
    filepath=f'transformer_model_384_{DATA_CONFIG["species"]}.h5', 
    save_best_only=True, 
    monitor='val_loss', 
    mode='min', 
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=CALLBACK_CONFIG['lr_reduce_factor'], 
    patience=CALLBACK_CONFIG['lr_reduce_patience'], 
    min_lr=CALLBACK_CONFIG['min_lr'], 
    verbose=1
)
overfitting_monitor = OverfittingMonitor()

print(f"\n?? Callbacks configured for 384 capacity:")
print(f"   EarlyStopping: patience={CALLBACK_CONFIG['early_stop_patience']} (increased from 50), min_delta={CALLBACK_CONFIG['early_stop_min_delta']}")
print(f"   ReduceLROnPlateau: patience={CALLBACK_CONFIG['lr_reduce_patience']} (increased from 20), factor={CALLBACK_CONFIG['lr_reduce_factor']}")
print(f"   ModelCheckpoint: saves best model based on val_loss")
print(f"   OverfittingMonitor: threshold={CALLBACK_CONFIG['overfitting_threshold']} (reduced from 0.02 - more sensitive)")

# =============================================================================
# TRAINING
# =============================================================================

# ?? Training with all callbacks
print("\n?? Starting training...")
print(f"?? Training samples: {len(x1_train)}")
print(f"?? Validation samples: {len(x1_val)}")
print(f"?? Batch size: {DATA_CONFIG['batch_size']}")
print(f"?? Using GPU: {tf.test.is_gpu_available()}")

history = model.fit(
    x1_train, y1_train,
    callbacks=[callback, checkpoint, reduce_lr, overfitting_monitor],
    batch_size=DATA_CONFIG['batch_size'],
    epochs=TRAINING_CONFIG['epochs'],
    verbose=2,
    validation_data=(x1_val, y1_val),
    shuffle=TRAINING_CONFIG['shuffle']
)
print("? Training complete.")

# Save training history
hist_df = pd.DataFrame(history.history)
hist_df.to_csv(f'history_384_{DATA_CONFIG["species"]}.csv')
print("?? Training history saved.")

# =============================================================================
# FINAL ANALYSIS
# =============================================================================

# Print final training summary with overfitting analysis
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
epochs_trained = len(history.history['loss'])
min_val_loss = min(history.history['val_loss'])
best_epoch = history.history['val_loss'].index(min_val_loss) + 1

print(f"\n{'='*60}")
print(f"FINAL TRAINING ANALYSIS FOR {DATA_CONFIG['species']} (384 CAPACITY)")
print(f"{'='*60}")
print(f"Epochs trained: {epochs_trained}")
print(f"Final training loss: {final_train_loss:.6f}")
print(f"Final validation loss: {final_val_loss:.6f}")
print(f"Best validation loss: {min_val_loss:.6f} (Epoch {best_epoch})")
print(f"Loss difference: {final_val_loss - final_train_loss:.6f}")
print(f"Loss ratio (val/train): {final_val_loss / final_train_loss:.3f}")

# Final overfitting/underfitting assessment (adjusted for small dataset)
loss_diff = final_val_loss - final_train_loss
loss_ratio = final_val_loss / final_train_loss
epochs_since_best = epochs_trained - best_epoch

if loss_ratio > 1.3 and loss_diff > 0.08:  # Adjusted thresholds for 256 capacity
    print(f"\n?? FINAL VERDICT: Model is OVERFITTING")
    print("   Recommendations:")
    print("   - Use the best saved model instead of final model")
    print("   - Consider increasing dropout rate further")
    print("   - Add more regularization")
elif final_train_loss > 0.4 and final_val_loss > 0.4:  # Lower threshold for 256 capacity
    print(f"\n?? FINAL VERDICT: Model is UNDERFITTING")
    print("   Recommendations:")
    print("   - Model may need even more capacity")
    print("   - Reduce dropout rate")
    print("   - Train for more epochs")
elif epochs_since_best > 20:  # Adjusted threshold for higher capacity
    print(f"\n?? FINAL VERDICT: Model may have CONVERGED")
    print(f"   Best model was {epochs_since_best} epochs ago")
    print("   The saved checkpoint model is likely optimal")
else:
    print(f"\n?? FINAL VERDICT: Model training looks GOOD")
    print("   Training appears well-balanced for 256 capacity")

print(f"\n?? Files saved:")
print(f"   Model: transformer_model_384_{DATA_CONFIG['species']}.h5")
print(f"   History: history_384_{DATA_CONFIG['species']}.csv")
print(f"{'='*60}")
