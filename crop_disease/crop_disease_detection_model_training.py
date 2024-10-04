import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. Set Paths and Parameters
data_dir = 'F:/Knowledge/Skills/All_Projects/NASA Space Apps Challenge 2024/CropDisease/Crop___DIsease'
image_size = (128, 128)
batch_size = 16
epochs = 30
num_classes = 15
learning_rate = 0.0001

# 2. Data Preprocessing
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 3. CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# 4. Compile Model
optimizer = optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Save the best model during training as .keras
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

# 6. Train the Model
history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=[early_stopping, model_checkpoint]
)

# 7. Save the Model as .h5 Manually After Training
model.save('crop_disease_model.h5')
print("Model saved as crop_disease_model.h5")

# Save the model in TensorFlow's native Keras format (instead of using save_format='tf')
model.save('crop_disease_model.keras')
print("Model saved in Keras format (crop_disease_model.keras)")

# 8. Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('crop_disease_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("TFLite model saved as crop_disease_model.tflite") 