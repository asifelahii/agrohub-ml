import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

# 1. Set Paths and Parameters
data_dir = 'F:/Knowledge/Skills/All_Projects/NASA Space Apps Challenge 2024/CropDisease/Crop___DIsease'
image_size = (128, 128)  # Resize images to 128x128 pixels
batch_size = 32
epochs = 10
num_classes = 15  # Number of crop disease categories (15 folders)

# 2. Data Preprocessing: Loading and Augmenting Images
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values between 0 and 1
    validation_split=0.2,  # Split 20% of the data for validation
)

# Load training data
train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Use as training data
)

# Load validation data
val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Use as validation data
)

# 3. CNN Model Definition
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # To prevent overfitting
    layers.Dense(num_classes, activation='softmax')  # Output layer with 15 categories
])

# 4. Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train the Model
history = model.fit(train_gen, epochs=epochs, validation_data=val_gen)

# 6. Evaluate the Model on Validation Data
val_loss, val_acc = model.evaluate(val_gen)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_acc}')

# 7. Convert the Model to TensorFlow Lite for Mobile Deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
model_file = 'crop_disease_model.tflite'
with open(model_file, 'wb') as f:
    f.write(tflite_model)

print(f'TFLite model saved to {model_file}') 