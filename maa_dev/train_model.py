import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import json

print("╔════════════════════════════════════════╗")
print("║  TinyML Behavioral Model Training      ║")
print("║  Enhanced & Corrected Version          ║")
print("╚════════════════════════════════════════╝\n")

# ============================================
# 1. LOAD DATA
# ============================================
print("[1/9] Loading training data...")
df = pd.read_csv('behavioral_training_data_natural.csv')
print(f"✓ Loaded {len(df):,} samples")

print(f"\nDataset shape: {df.shape}")
print(f"Labels: {df['label'].unique()}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())

# ============================================
# 2. DATA AUGMENTATION (CORRECTED)
# ============================================
print("\n[2/9] Applying data augmentation...")

def augment_data(df, num_augmentations=3):
    """
    Create augmented copies with slight variations
    num_augmentations: number of ADDITIONAL copies (not including original)
    Final multiplier = num_augmentations + 1
    """
    augmented_dfs = [df]  # Start with original
    
    for i in range(num_augmentations):
        df_aug = df.copy()
        
        # Add small random noise to sensor readings
        df_aug['temperature'] += np.random.normal(0, 0.3, len(df))
        df_aug['aqi'] += np.random.normal(0, 4, len(df))
        df_aug['pressure'] += np.random.normal(0, 0.3, len(df))
        
        # Slightly modify rate of change
        df_aug['temp_change_rate'] += np.random.normal(0, 0.05, len(df))
        df_aug['aqi_change_rate'] += np.random.normal(0, 1, len(df))
        
        # CORRECTED: Time shifts in minutes (±5 minutes)
        time_shift_minutes = np.random.randint(-5, 6, len(df))
        
        # Apply minute shift
        new_minutes = df_aug['minute'] + time_shift_minutes
        
        # Handle minute overflow/underflow
        hour_carry = (new_minutes // 60)  # Hours to add
        df_aug['minute'] = new_minutes % 60  # Wrap minutes
        df_aug['hour'] = (df_aug['hour'] + hour_carry) % 24  # Wrap hours
        
        # Recalculate cyclic features
        if 'hour_sin' in df_aug.columns:
            hour_decimal = df_aug['hour'] + df_aug['minute'] / 60.0
            df_aug['hour_sin'] = np.sin(2 * np.pi * hour_decimal / 24)
            df_aug['hour_cos'] = np.cos(2 * np.pi * hour_decimal / 24)
        
        # Recalculate time-based binary features
        df_aug['is_night'] = df_aug['hour'].apply(lambda x: 1 if (x >= 22 or x <= 6) else 0)
        df_aug['is_morning'] = df_aug['hour'].apply(lambda x: 1 if (6 <= x <= 9) else 0)
        df_aug['is_cooking_hours'] = df_aug['hour'].apply(lambda x: 1 if (19 <= x <= 21) else 0)
        
        augmented_dfs.append(df_aug)
        print(f"  Augmentation {i+1}/{num_augmentations} complete")
    
    # Combine all augmented datasets
    df_combined = pd.concat(augmented_dfs, ignore_index=True)
    df_combined = shuffle(df_combined, random_state=42)
    
    actual_multiplier = len(df_combined) / len(df)
    print(f"\n✓ Multiplier: {actual_multiplier:.1f}x (original + {num_augmentations} augmented copies)")
    
    return df_combined, num_augmentations + 1

# Apply augmentation
df_augmented, actual_multiplier = augment_data(df, num_augmentations=3)
print(f"✓ Final dataset: {len(df_augmented):,} samples")

# ============================================
# 3. FEATURE SELECTION & PREPARATION
# ============================================
print("\n[3/9] Preparing features...")

# Select features for training
feature_columns = [
    'day_of_week',
    'temperature',
    'pressure',
    'aqi',
    'temp_change_rate',
    'aqi_change_rate',
    'temp_avg_15min',
    'aqi_avg_15min',
    'is_night',
    'is_morning',
    'is_cooking_hours',
    'is_weekend',
    'hour_sin',  # Cyclic encoding instead of raw hour
    'hour_cos'
]

# Ensure all features exist
available_features = [f for f in feature_columns if f in df_augmented.columns]
print(f"Using {len(available_features)} features:")
for i, f in enumerate(available_features, 1):
    print(f"  {i}. {f}")

X = df_augmented[available_features].values
y = df_augmented['label'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print(f"\nFeatures shape: {X.shape}")
print(f"Labels: {label_encoder.classes_}")
print(f"Number of classes: {num_classes}")

# Check class balance after augmentation
print("\nClass distribution after augmentation:")
unique, counts = np.unique(y_encoded, return_counts=True)
for label_idx, count in zip(unique, counts):
    label_name = label_encoder.classes_[label_idx]
    percentage = count / len(y_encoded) * 100
    print(f"  {label_name:15s}: {count:6,} ({percentage:5.1f}%)")

# ============================================
# 4. TRAIN/VALIDATION/TEST SPLIT
# ============================================
print("\n[4/9] Splitting data...")

# Split: 70% train, 15% validation, 15% test (stratified)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation set: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

# ============================================
# 5. FEATURE SCALING
# ============================================
print("\n[5/9] Scaling features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✓ Features normalized (zero mean, unit variance)")

# Save scaler parameters for ESP32 deployment
scaler_params = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist(),
    'feature_names': available_features,
    'num_features': len(available_features)
}

with open('scaler_params.json', 'w') as f:
    json.dump(scaler_params, f, indent=2)

print("✓ Scaler parameters saved to scaler_params.json")

# ============================================
# 6. BUILD OPTIMIZED TINYML MODEL
# ============================================
print("\n[6/9] Building optimized TinyML model...")

def create_tiny_model(input_shape, num_classes):
    """
    Ultra-small model for ESP32
    Target: <20KB after quantization
    """
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        
        # Single hidden layer often sufficient for tabular data
        keras.layers.Dense(
            24, 
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.001),  # L2 instead of heavy dropout
            name='dense_hidden'
        ),
        
        keras.layers.Dropout(0.2),  # Light dropout
        
        # Output layer
        keras.layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model

model = create_tiny_model(
    input_shape=(len(available_features),),
    num_classes=num_classes
)

# Compile with Adam optimizer
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n" + "="*50)
model.summary()
print("="*50)

total_params = model.count_params()
print(f"\nTotal parameters: {total_params:,}")
print(f"Estimated float32 size: ~{total_params * 4 / 1024:.1f} KB")
print(f"Expected int8 size: ~{total_params / 1024:.1f} KB")

# ============================================
# 7. TRAIN MODEL
# ============================================
print("\n[7/9] Training model...")

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=0.00001,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=150,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Training complete!")

# ============================================
# 8. EVALUATE MODEL
# ============================================
print("\n[8/9] Evaluating model...")

# Keras model evaluation
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nKeras Model Performance:")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy*100:.2f}%")

# Per-class accuracy
y_pred = model.predict(X_test_scaled, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\n--- Per-Class Accuracy (Keras) ---")
for i, label in enumerate(label_encoder.classes_):
    mask = y_test == i
    if mask.sum() > 0:
        class_acc = (y_pred_classes[mask] == y_test[mask]).mean()
        print(f"{label:15s}: {class_acc*100:.1f}% ({mask.sum():,} samples)")

# ============================================
# 9. CONVERT TO TENSORFLOW LITE (CORRECTED)
# ============================================
print("\n[9/9] Converting to TensorFlow Lite...")
print("="*50)

# Save full Keras model
model.save('behavior_model_full.h5')
print("✓ Saved Keras model: behavior_model_full.h5")

# CORRECTED: Better representative dataset
def representative_dataset():
    """
    Use random samples across training set for better quantization
    """
    num_samples = min(500, len(X_train_scaled))  # Use 500 samples
    indices = np.random.choice(len(X_train_scaled), num_samples, replace=False)
    
    for i in indices:
        yield [X_train_scaled[i:i+1].astype(np.float32)]

print("\nQuantizing to int8...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

# Full integer quantization
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert
tflite_model_quant = converter.convert()

# Save quantized model
tflite_filename = 'behavior_model_int8.tflite'
with open(tflite_filename, 'wb') as f:
    f.write(tflite_model_quant)

tflite_size = len(tflite_model_quant) / 1024
print(f"✓ TFLite model saved: {tflite_filename}")
print(f"✓ TFLite model size: {tflite_size:.2f} KB")

# ============================================
# 10. TEST TFLITE MODEL LOCALLY
# ============================================
print("\n" + "="*50)
print("TESTING TFLITE MODEL LOCALLY")
print("="*50)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=tflite_filename)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput details:")
print(f"  Shape: {input_details[0]['shape']}")
print(f"  Type: {input_details[0]['dtype']}")
print(f"  Quantization: scale={input_details[0]['quantization'][0]:.6f}, zero_point={input_details[0]['quantization'][1]}")

print("\nOutput details:")
print(f"  Shape: {output_details[0]['shape']}")
print(f"  Type: {output_details[0]['dtype']}")
print(f"  Quantization: scale={output_details[0]['quantization'][0]:.6f}, zero_point={output_details[0]['quantization'][1]}")

# Test on subset
print("\nTesting TFLite accuracy on test set...")

# Get quantization parameters
input_scale = input_details[0]['quantization'][0]
input_zero_point = input_details[0]['quantization'][1]
output_scale = output_details[0]['quantization'][0]
output_zero_point = output_details[0]['quantization'][1]

def quantize_input(x, scale, zero_point):
    """Quantize float to int8"""
    return np.clip(np.round(x / scale) + zero_point, -128, 127).astype(np.int8)

def dequantize_output(x_quant, scale, zero_point):
    """Dequantize int8 to float"""
    return (x_quant.astype(np.float32) - zero_point) * scale

# Test on sample
num_test_samples = min(1000, len(X_test_scaled))
tflite_predictions = []

for i in range(num_test_samples):
    # Quantize input
    input_data = quantize_input(
        X_test_scaled[i:i+1],
        input_scale,
        input_zero_point
    )
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Dequantize output
    output_probs = dequantize_output(output_data[0], output_scale, output_zero_point)
    
    # Apply softmax (if needed)
    output_probs = np.exp(output_probs) / np.sum(np.exp(output_probs))
    
    tflite_predictions.append(np.argmax(output_probs))

tflite_predictions = np.array(tflite_predictions)
tflite_accuracy = (tflite_predictions == y_test[:num_test_samples]).mean()

print(f"✓ TFLite accuracy: {tflite_accuracy*100:.2f}% (on {num_test_samples} samples)")
print(f"  Keras accuracy: {test_accuracy*100:.2f}%")
print(f"  Accuracy drop: {(test_accuracy - tflite_accuracy)*100:.2f}%")

if abs(test_accuracy - tflite_accuracy) < 0.05:
    print("✓ Quantization quality: GOOD (< 5% drop)")
elif abs(test_accuracy - tflite_accuracy) < 0.10:
    print("⚠ Quantization quality: ACCEPTABLE (5-10% drop)")
else:
    print("⚠ Quantization quality: POOR (> 10% drop) - consider adjusting")

# ============================================
# 11. CONVERT TO C HEADER
# ============================================
print("\n" + "="*50)
print("GENERATING C HEADER FOR ESP32")
print("="*50)

def convert_to_c_array(data, var_name):
    c_str = f'// TinyML Behavioral Model\n'
    c_str += f'// Model size: {len(data)} bytes ({len(data)/1024:.2f} KB)\n'
    c_str += f'// Quantization: int8\n\n'
    c_str += f'#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n'
    c_str += f'const unsigned char {var_name}[] = {{\n'
    
    for i, byte in enumerate(data):
        if i % 12 == 0:
            c_str += '  '
        c_str += f'0x{byte:02x}'
        if i < len(data) - 1:
            c_str += ', '
        if (i + 1) % 12 == 0:
            c_str += '\n'
    
    c_str += '\n};\n\n'
    c_str += f'const unsigned int {var_name}_len = {len(data)};\n\n'
    
    return c_str

header_content = convert_to_c_array(tflite_model_quant, 'model_data')

# Add label mappings
header_content += '// Label mappings\n'
header_content += 'const char* behavior_labels[] = {\n'
for label in label_encoder.classes_:
    header_content += f'  "{label}",\n'
header_content += '};\n\n'
header_content += f'const int num_labels = {num_classes};\n\n'

# Add quantization parameters
header_content += '// Quantization parameters\n'
header_content += f'const float input_scale = {input_scale:.8f}f;\n'
header_content += f'const int input_zero_point = {input_zero_point};\n'
header_content += f'const float output_scale = {output_scale:.8f}f;\n'
header_content += f'const int output_zero_point = {output_zero_point};\n\n'
header_content += '#endif // MODEL_DATA_H\n'

with open('model_data.h', 'w') as f:
    f.write(header_content)

print("✓ C header saved: model_data.h")

# ============================================
# 12. PLOT TRAINING HISTORY
# ============================================
print("\nGenerating training plots...")

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("✓ Saved training_history.png")

# ============================================
# 13. SAVE METADATA (CORRECTED)
# ============================================
metadata = {
    'model_info': {
        'type': 'TinyML Behavioral Classifier',
        'framework': 'TensorFlow Lite',
        'quantization': 'int8',
        'target_device': 'ESP32'
    },
    'features': {
        'input_features': available_features,
        'num_features': len(available_features),
        'scaling': 'StandardScaler (zero mean, unit variance)'
    },
    'labels': {
        'num_classes': num_classes,
        'classes': label_encoder.classes_.tolist()
    },
    'model_architecture': {
        'layers': f'Dense(24) → Dropout(0.2) → Dense({num_classes})',
        'total_parameters': int(total_params),
        'activation': 'ReLU',
        'regularization': 'L2(0.001)'
    },
    'training_data': {
        'original_samples': len(df),
        'augmented_samples': len(df_augmented),
        'augmentation_multiplier': actual_multiplier,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test)
    },
    'performance': {
        'keras_accuracy': float(test_accuracy),
        'keras_loss': float(test_loss),
        'tflite_accuracy': float(tflite_accuracy),
        'quantization_drop': float(abs(test_accuracy - tflite_accuracy))
    },
    'model_files': {
        'keras_h5': 'behavior_model_full.h5',
        'tflite': tflite_filename,
        'tflite_size_kb': float(tflite_size),
        'c_header': 'model_data.h',
        'scaler_params': 'scaler_params.json'
    },
    'quantization_params': {
        'input_scale': float(input_scale),
        'input_zero_point': int(input_zero_point),
        'output_scale': float(output_scale),
        'output_zero_point': int(output_zero_point)
    }
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✓ Metadata saved: model_metadata.json")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*70)
print("MODEL TRAINING & CONVERSION COMPLETE!")
print("="*70)

print("\n📁 Generated Files:")
print("  1. behavior_model_full.h5      - Full Keras model")
print("  2. behavior_model_int8.tflite  - Quantized TFLite model")
print("  3. model_data.h                - C header for ESP32")
print("  4. scaler_params.json          - Feature scaling parameters")
print("  5. model_metadata.json         - Complete model information")
print("  6. training_history.png        - Training plots")

print("\n📊 Model Performance:")
print(f"  Keras Accuracy:  {test_accuracy*100:.2f}%")
print(f"  TFLite Accuracy: {tflite_accuracy*100:.2f}%")
print(f"  Model Size:      {tflite_size:.2f} KB (int8 quantized)")
print(f"  Parameters:      {total_params:,}")

print("\n🎯 ESP32 Deployment Info:")
print(f"  Input type:  int8 (scale={input_scale:.6f}, zero={input_zero_point})")
print(f"  Output type: int8 (scale={output_scale:.6f}, zero={output_zero_point})")
print(f"  Features:    {len(available_features)}")
print(f"  Classes:     {num_classes}")

print("\n💡 Next Steps:")
print("  1. Copy model_data.h to ESP32 project")
print("  2. Install TensorFlow Lite Micro for ESP32")
print("  3. Implement on-device scaling using scaler_params.json")
print("  4. Add EMA-based baseline adaptation (recommended)")
print("  5. Test with real sensor readings")

print("\n⚠️  Important Notes:")
print("  • Use precomputed sin/cos lookup table on ESP32 (24 values)")
print("  • Implement exponential moving average (EMA) for baseline adaptation")
print("  • Monitor RAM usage (~50-100KB for inference)")
print("  • Test quantization accuracy meets requirements")

print("\n" + "="*70)
print("✅ Ready for ESP32 TinyML deployment!")
print("="*70)
