import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from termcolor import cprint
from PIL import Image


class emnist:
    def __init__(self):
        (self.train_ds, self.test_ds), self.ds_info = tfds.load(
            "emnist",
            split=["train", "test"],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
        )

        self.labels = self.ds_info.features["label"].names
        cprint(f"Class labels: {self.labels}", "cyan")

        self.num_classes = self.ds_info.features["label"].num_classes
        self.labels = self.ds_info.features["label"].names  # Load class names

        self.train_ds = self.__prepare_dataset(self.train_ds)
        self.test_ds = self.__prepare_dataset(self.test_ds)

    def __prepare_dataset(self, dataset, batch_size=32):
        dataset = dataset.map(lambda img, lbl: (tf.cast(img, tf.float32) / 255.0, lbl))
        dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def __compile(self):
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def build_model(self):
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(self.num_classes, activation="softmax"),
            ]
        )
        self.__compile()
        self.model.fit(self.train_ds, epochs=3)

    def save(self):
        self.model.save("emnist.h5")

    def load(self, filename="emnist.h5"):
        self.model = tf.keras.models.load_model(filename)
        self.__compile()

    def evaluate(self):
        test_loss, test_accuracy = self.model.evaluate(self.test_ds)
        cprint(f"Test accuracy: {test_accuracy}", "red")
        cprint(f"Test loss: {test_loss}", "red")

    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert("L")  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28 pixels
        img = np.array(img)  # Convert to numpy array
        img = img / 255.0  # Normalize pixel values
        img = img.reshape(1, 28, 28, 1)  # Reshape for model input
        return img

    def predict_image(self, image_path):
        img = self.preprocess_image(image_path)
        prediction = self.model.predict(img)
        predicted_class_index = np.argmax(prediction, axis=1)
        self.labels[predicted_class_index[0]]

        return chr(int(self.labels[predicted_class_index[0]]) + 39)  # Return class name


def main():
    t = emnist()
    t.load()

    image_paths = [
        "assets\letterA.png",
        "assets\letterB.png",
        "assets\letterC.png",
        "assets\letterD.png",
        "assets\letterE.png",
    ]

    for image_path in image_paths:
        predicted_class_name = t.predict_image(image_path)
        cprint(predicted_class_name, "red")


if __name__ == "__main__":
    main()
