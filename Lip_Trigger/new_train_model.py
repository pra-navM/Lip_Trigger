import os
import tensorflow as tf
from tensorflow.keras import layers, models


dataset_dir = 'dataset'
model_output_path = 'new_new_mouth_state_model.h5'


batch_size = 32
num_epochs = 10
img_height = 224
img_width = 224

# preprocess 
def load_and_preprocess_image(image_path, target_size=(img_height, img_width)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def load_and_preprocess_from_path_label(image_path, label):
    return load_and_preprocess_image(image_path), label

def create_dataset(dataset_dir, batch_size=32):
    #list of file paths and labels
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(dataset_dir))
    class_indices = {name: index for index, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(file_path)
                    labels.append(class_indices[class_name])
    
    # TF dataset
    path_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    image_label_ds = path_ds.map(load_and_preprocess_from_path_label, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Shuffle and batch
    dataset = image_label_ds.shuffle(buffer_size=len(image_paths))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset, class_names

# Build
def build_model(input_shape=(img_height, img_width, 3), num_classes=2):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # For binary classification
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

train_dataset, class_names = create_dataset(dataset_dir, batch_size)

# train
num_classes = len(class_names)
model = build_model(input_shape=(img_height, img_width, 3), num_classes=num_classes)

print("Starting training...")
model.fit(train_dataset, epochs=num_epochs)

# save
model.save(model_output_path)
print(f"Model saved to {model_output_path}")
