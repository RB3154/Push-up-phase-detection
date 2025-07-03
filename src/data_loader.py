import tensorflow as tf
import os


def preprocess_image(image_path, label, img_size=(224,224)):
    """
    Reads an image from disk, decodes, resizes, normalizes and applies augmentations.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = image / 255.0
    return image, label


def load_data(data_dir, batch_size=32, img_size=(224,224), shuffle=True):
    """
    Creates a tf.data.Dataset from a directory structured as:
    data_dir/class_name/*.jpg
    """
    classes = sorted(os.listdir(data_dir))
    all_image_paths = []
    all_labels = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('jpg','jpeg','png')):
                all_image_paths.append(os.path.join(cls_dir, fname))
                all_labels.append(idx)
    dataset = tf.data.Dataset.from_tensor_slices((all_image_paths, all_labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(all_image_paths))
    dataset = dataset.map(
        lambda x, y: preprocess_image(x, y, img_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset