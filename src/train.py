import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from src.data_loader import load_data, preprocess_image
from src.model import build_model


def main():
    data_dir = 'data/processed'
    batch_size = 32
    initial_epochs = 10
    fine_tune_epochs = 5
    num_classes = 2

    # Load datasets
    train_ds = load_data(os.path.join(data_dir, 'train'), batch_size=batch_size)
    val_ds = load_data(os.path.join(data_dir, 'val'), batch_size=batch_size, shuffle=False)

    # Step 1: Data Augmentation Enhancements
    def extra_augment(image, label):
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        return image, label

    train_ds = train_ds.map(extra_augment, num_parallel_calls=tf.data.AUTOTUNE)

    # Step 1b: Hard-Example Mining (Inject specific down example)
    hard_example_path = os.path.join(data_dir, 'val', 'down', '170.jpg')
    if os.path.exists(hard_example_path):
        hard_ds = tf.data.Dataset.from_tensor_slices(([hard_example_path], [1]))
        hard_ds = hard_ds.map(lambda x, y: preprocess_image(x, y), num_parallel_calls=tf.data.AUTOTUNE)
        hard_ds = hard_ds.batch(batch_size)
        train_ds = train_ds.concatenate(hard_ds).shuffle(buffer_size=1000)

    # Step 3: Convert labels to one-hot for label smoothing
    train_ds = train_ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)), num_parallel_calls=tf.data.AUTOTUNE)

    # Build and compile model with label smoothing
    model = build_model(num_classes=num_classes)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    # Prepare checkpoint callback
    os.makedirs('models', exist_ok=True)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        'models/best_weights.weights.h5',
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss'
    )

    # Initial training
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        callbacks=[checkpoint_cb]
    )

    # Step 2: Fine-tuning the backbone
    base_model = model.layers[1]
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    # Recompile with lower LR
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    # Fine-tune
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
        callbacks=[checkpoint_cb]
    )

    # Step 4: Pose-Based Hybrid Check
    # Note: Pose-based hybrid logic (e.g., angle override) should be implemented in the inference script (predict.py).


if __name__ == '__main__':
    main()
