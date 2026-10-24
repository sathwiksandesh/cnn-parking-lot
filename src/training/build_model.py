
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(
    image_height: int = 128,
    image_width: int = 128,
    channels: int = 3,
    learning_rate: float = 0.001,
) -> keras.Model:
    inputs = keras.Input(
        shape=(image_height, image_width, channels),
        name="parking_slot_image",
    )

    # Mild augmentation helps the model handle visual variation.
    x = layers.RandomFlip("horizontal", name="random_flip")(inputs)
    x = layers.RandomRotation(
        0.03,
        name="random_rotation",
    )(x)
    x = layers.RandomZoom(
        0.08,
        name="random_zoom",
    )(x)

    # Convolution block 1
    x = layers.Conv2D(
        32, 3, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    # Convolution block 2
    x = layers.Conv2D(
        64, 3, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    # Convolution block 3
    x = layers.Conv2D(
        128, 3, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.40)(x)

    # Class order is explicitly [vacant, occupied].
    outputs = layers.Dense(
        2,
        activation="softmax",
        name="occupancy_probabilities",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="parking_occupancy_cnn",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(
                class_id=1,
                name="occupied_precision",
            ),
            keras.metrics.Recall(
                class_id=1,
                name="occupied_recall",
            ),
        ],
    )

    return model


if __name__ == "__main__":
    model = build_cnn()
    model.summary()
    print("\nCNN architecture created successfully.")
