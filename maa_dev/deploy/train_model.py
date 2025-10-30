importpandasaspd
importnumpyasnp
importtensorflowastf
fromtensorflowimportkeras
fromsklearn.model_selectionimporttrain_test_split
fromsklearn.preprocessingimportStandardScaler,LabelEncoder
fromsklearn.utilsimportshuffle
importmatplotlib.pyplotasplt
importjson

print("╔════════════════════════════════════════╗")
print("║  TinyML Behavioral Model Training      ║")
print("║  Enhanced & Corrected Version          ║")
print("╚════════════════════════════════════════╝\n")




print("[1/9] Loading training data...")
df=pd.read_csv('behavioral_training_data_natural.csv')
print(f"✓ Loaded {len(df):,} samples")

print(f"\nDataset shape: {df.shape}")
print(f"Labels: {df['label'].unique()}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())




print("\n[2/9] Applying data augmentation...")

defaugment_data(df,num_augmentations=3):
    
augmented_dfs=[df]

foriinrange(num_augmentations):
        df_aug=df.copy()


df_aug['temperature']+=np.random.normal(0,0.3,len(df))
df_aug['aqi']+=np.random.normal(0,4,len(df))
df_aug['pressure']+=np.random.normal(0,0.3,len(df))


df_aug['temp_change_rate']+=np.random.normal(0,0.05,len(df))
df_aug['aqi_change_rate']+=np.random.normal(0,1,len(df))


time_shift_minutes=np.random.randint(-5,6,len(df))


new_minutes=df_aug['minute']+time_shift_minutes


hour_carry=(new_minutes//60)
df_aug['minute']=new_minutes%60
df_aug['hour']=(df_aug['hour']+hour_carry)%24


if'hour_sin'indf_aug.columns:
            hour_decimal=df_aug['hour']+df_aug['minute']/60.0
df_aug['hour_sin']=np.sin(2*np.pi*hour_decimal/24)
df_aug['hour_cos']=np.cos(2*np.pi*hour_decimal/24)


df_aug['is_night']=df_aug['hour'].apply(lambdax:1if(x>=22orx<=6)else0)
df_aug['is_morning']=df_aug['hour'].apply(lambdax:1if(6<=x<=9)else0)
df_aug['is_cooking_hours']=df_aug['hour'].apply(lambdax:1if(19<=x<=21)else0)

augmented_dfs.append(df_aug)
print(f"  Augmentation {i+1}/{num_augmentations} complete")


df_combined=pd.concat(augmented_dfs,ignore_index=True)
df_combined=shuffle(df_combined,random_state=42)

actual_multiplier=len(df_combined)/len(df)
print(f"\n✓ Multiplier: {actual_multiplier:.1f}x (original + {num_augmentations} augmented copies)")

returndf_combined,num_augmentations+1


df_augmented,actual_multiplier=augment_data(df,num_augmentations=3)
print(f"✓ Final dataset: {len(df_augmented):,} samples")




print("\n[3/9] Preparing features...")


feature_columns=[
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
'hour_sin',
'hour_cos'
]


available_features=[fforfinfeature_columnsiffindf_augmented.columns]
print(f"Using {len(available_features)} features:")
fori,finenumerate(available_features,1):
    print(f"  {i}. {f}")

X=df_augmented[available_features].values
y=df_augmented['label'].values


label_encoder=LabelEncoder()
y_encoded=label_encoder.fit_transform(y)
num_classes=len(label_encoder.classes_)

print(f"\nFeatures shape: {X.shape}")
print(f"Labels: {label_encoder.classes_}")
print(f"Number of classes: {num_classes}")


print("\nClass distribution after augmentation:")
unique,counts=np.unique(y_encoded,return_counts=True)
forlabel_idx,countinzip(unique,counts):
    label_name=label_encoder.classes_[label_idx]
percentage=count/len(y_encoded)*100
print(f"  {label_name:15s}: {count:6,} ({percentage:5.1f}%)")




print("\n[4/9] Splitting data...")


X_train,X_temp,y_train,y_temp=train_test_split(
X,y_encoded,test_size=0.3,random_state=42,stratify=y_encoded
)

X_val,X_test,y_val,y_test=train_test_split(
X_temp,y_temp,test_size=0.5,random_state=42,stratify=y_temp
)

print(f"Train set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation set: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")




print("\n[5/9] Scaling features...")

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_val_scaled=scaler.transform(X_val)
X_test_scaled=scaler.transform(X_test)

print("✓ Features normalized (zero mean, unit variance)")


scaler_params={
'mean':scaler.mean_.tolist(),
'scale':scaler.scale_.tolist(),
'feature_names':available_features,
'num_features':len(available_features)
}

withopen('scaler_params.json','w')asf:
    json.dump(scaler_params,f,indent=2)

print("✓ Scaler parameters saved to scaler_params.json")




print("\n[6/9] Building optimized TinyML model...")

defcreate_tiny_model(input_shape,num_classes):
    
model=keras.Sequential([
keras.layers.Input(shape=input_shape),


keras.layers.Dense(
24,
activation='relu',
kernel_regularizer=keras.regularizers.l2(0.001),
name='dense_hidden'
),

keras.layers.Dropout(0.2),


keras.layers.Dense(num_classes,activation='softmax',name='output')
])

returnmodel

model=create_tiny_model(
input_shape=(len(available_features),),
num_classes=num_classes
)


model.compile(
optimizer=keras.optimizers.Adam(learning_rate=0.001),
loss='sparse_categorical_crossentropy',
metrics=['accuracy']
)

print("\n"+"="*50)
model.summary()
print("="*50)

total_params=model.count_params()
print(f"\nTotal parameters: {total_params:,}")
print(f"Estimated float32 size: ~{total_params * 4 / 1024:.1f} KB")
print(f"Expected int8 size: ~{total_params / 1024:.1f} KB")




print("\n[7/9] Training model...")

callbacks=[
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

history=model.fit(
X_train_scaled,y_train,
validation_data=(X_val_scaled,y_val),
epochs=150,
batch_size=128,
callbacks=callbacks,
verbose=1
)

print("\n✓ Training complete!")




print("\n[8/9] Evaluating model...")


test_loss,test_accuracy=model.evaluate(X_test_scaled,y_test,verbose=0)
print(f"\nKeras Model Performance:")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy*100:.2f}%")


y_pred=model.predict(X_test_scaled,verbose=0)
y_pred_classes=np.argmax(y_pred,axis=1)

print("\n--- Per-Class Accuracy (Keras) ---")
fori,labelinenumerate(label_encoder.classes_):
    mask=y_test==i
ifmask.sum()>0:
        class_acc=(y_pred_classes[mask]==y_test[mask]).mean()
print(f"{label:15s}: {class_acc*100:.1f}% ({mask.sum():,} samples)")




print("\n[9/9] Converting to TensorFlow Lite...")
print("="*50)


model.save('behavior_model_full.h5')
print("✓ Saved Keras model: behavior_model_full.h5")


defrepresentative_dataset():
    
num_samples=min(500,len(X_train_scaled))
indices=np.random.choice(len(X_train_scaled),num_samples,replace=False)

foriinindices:
        yield[X_train_scaled[i:i+1].astype(np.float32)]

print("\nQuantizing to int8...")

converter=tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations=[tf.lite.Optimize.DEFAULT]
converter.representative_dataset=representative_dataset


converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type=tf.int8
converter.inference_output_type=tf.int8


tflite_model_quant=converter.convert()


tflite_filename='behavior_model_int8.tflite'
withopen(tflite_filename,'wb')asf:
    f.write(tflite_model_quant)

tflite_size=len(tflite_model_quant)/1024
print(f"✓ TFLite model saved: {tflite_filename}")
print(f"✓ TFLite model size: {tflite_size:.2f} KB")




print("\n"+"="*50)
print("TESTING TFLITE MODEL LOCALLY")
print("="*50)


interpreter=tf.lite.Interpreter(model_path=tflite_filename)
interpreter.allocate_tensors()


input_details=interpreter.get_input_details()
output_details=interpreter.get_output_details()

print("\nInput details:")
print(f"  Shape: {input_details[0]['shape']}")
print(f"  Type: {input_details[0]['dtype']}")
print(f"  Quantization: scale={input_details[0]['quantization'][0]:.6f}, zero_point={input_details[0]['quantization'][1]}")

print("\nOutput details:")
print(f"  Shape: {output_details[0]['shape']}")
print(f"  Type: {output_details[0]['dtype']}")
print(f"  Quantization: scale={output_details[0]['quantization'][0]:.6f}, zero_point={output_details[0]['quantization'][1]}")


print("\nTesting TFLite accuracy on test set...")


input_scale=input_details[0]['quantization'][0]
input_zero_point=input_details[0]['quantization'][1]
output_scale=output_details[0]['quantization'][0]
output_zero_point=output_details[0]['quantization'][1]

defquantize_input(x,scale,zero_point):
    
returnnp.clip(np.round(x/scale)+zero_point,-128,127).astype(np.int8)

defdequantize_output(x_quant,scale,zero_point):
    
return(x_quant.astype(np.float32)-zero_point)*scale


num_test_samples=min(1000,len(X_test_scaled))
tflite_predictions=[]

foriinrange(num_test_samples):

    input_data=quantize_input(
X_test_scaled[i:i+1],
input_scale,
input_zero_point
)


interpreter.set_tensor(input_details[0]['index'],input_data)


interpreter.invoke()


output_data=interpreter.get_tensor(output_details[0]['index'])


output_probs=dequantize_output(output_data[0],output_scale,output_zero_point)


output_probs=np.exp(output_probs)/np.sum(np.exp(output_probs))

tflite_predictions.append(np.argmax(output_probs))

tflite_predictions=np.array(tflite_predictions)
tflite_accuracy=(tflite_predictions==y_test[:num_test_samples]).mean()

print(f"✓ TFLite accuracy: {tflite_accuracy*100:.2f}% (on {num_test_samples} samples)")
print(f"  Keras accuracy: {test_accuracy*100:.2f}%")
print(f"  Accuracy drop: {(test_accuracy - tflite_accuracy)*100:.2f}%")

ifabs(test_accuracy-tflite_accuracy)<0.05:
    print("✓ Quantization quality: GOOD (< 5% drop)")
elifabs(test_accuracy-tflite_accuracy)<0.10:
    print("⚠ Quantization quality: ACCEPTABLE (5-10% drop)")
else:
    print("⚠ Quantization quality: POOR (> 10% drop) - consider adjusting")




print("\n"+"="*50)
print("GENERATING C HEADER FOR ESP32")
print("="*50)

defconvert_to_c_array(data,var_name):
    c_str=f'// TinyML Behavioral Model\n'
c_str+=f'// Model size: {len(data)} bytes ({len(data)/1024:.2f} KB)\n'
c_str+=f'// Quantization: int8\n\n'
c_str+=f'#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n'
c_str+=f'const unsigned char {var_name}[] = {{\n'

fori,byteinenumerate(data):
        ifi%12==0:
            c_str+='  '
c_str+=f'0x{byte:02x}'
ifi<len(data)-1:
            c_str+=', '
if(i+1)%12==0:
            c_str+='\n'

c_str+='\n};\n\n'
c_str+=f'const unsigned int {var_name}_len = {len(data)};\n\n'

returnc_str

header_content=convert_to_c_array(tflite_model_quant,'model_data')


header_content+='// Label mappings\n'
header_content+='const char* behavior_labels[] = {\n'
forlabelinlabel_encoder.classes_:
    header_content+=f'  "{label}",\n'
header_content+='};\n\n'
header_content+=f'const int num_labels = {num_classes};\n\n'


header_content+='// Quantization parameters\n'
header_content+=f'const float input_scale = {input_scale:.8f}f;\n'
header_content+=f'const int input_zero_point = {input_zero_point};\n'
header_content+=f'const float output_scale = {output_scale:.8f}f;\n'
header_content+=f'const int output_zero_point = {output_zero_point};\n\n'
header_content+='#endif // MODEL_DATA_H\n'

withopen('model_data.h','w')asf:
    f.write(header_content)

print("✓ C header saved: model_data.h")




print("\nGenerating training plots...")

plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],label='Train Accuracy',linewidth=2)
plt.plot(history.history['val_accuracy'],label='Val Accuracy',linewidth=2)
plt.title('Model Accuracy',fontsize=14,fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True,alpha=0.3)

plt.subplot(1,2,2)
plt.plot(history.history['loss'],label='Train Loss',linewidth=2)
plt.plot(history.history['val_loss'],label='Val Loss',linewidth=2)
plt.title('Model Loss',fontsize=14,fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True,alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png',dpi=150,bbox_inches='tight')
print("✓ Saved training_history.png")




metadata={
'model_info':{
'type':'TinyML Behavioral Classifier',
'framework':'TensorFlow Lite',
'quantization':'int8',
'target_device':'ESP32'
},
'features':{
'input_features':available_features,
'num_features':len(available_features),
'scaling':'StandardScaler (zero mean, unit variance)'
},
'labels':{
'num_classes':num_classes,
'classes':label_encoder.classes_.tolist()
},
'model_architecture':{
'layers':f'Dense(24) → Dropout(0.2) → Dense({num_classes})',
'total_parameters':int(total_params),
'activation':'ReLU',
'regularization':'L2(0.001)'
},
'training_data':{
'original_samples':len(df),
'augmented_samples':len(df_augmented),
'augmentation_multiplier':actual_multiplier,
'train_samples':len(X_train),
'val_samples':len(X_val),
'test_samples':len(X_test)
},
'performance':{
'keras_accuracy':float(test_accuracy),
'keras_loss':float(test_loss),
'tflite_accuracy':float(tflite_accuracy),
'quantization_drop':float(abs(test_accuracy-tflite_accuracy))
},
'model_files':{
'keras_h5':'behavior_model_full.h5',
'tflite':tflite_filename,
'tflite_size_kb':float(tflite_size),
'c_header':'model_data.h',
'scaler_params':'scaler_params.json'
},
'quantization_params':{
'input_scale':float(input_scale),
'input_zero_point':int(input_zero_point),
'output_scale':float(output_scale),
'output_zero_point':int(output_zero_point)
}
}

withopen('model_metadata.json','w')asf:
    json.dump(metadata,f,indent=2)

print("✓ Metadata saved: model_metadata.json")




print("\n"+"="*70)
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

print("\n"+"="*70)
print("✅ Ready for ESP32 TinyML deployment!")
print("="*70)
