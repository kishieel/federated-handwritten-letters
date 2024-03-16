import argparse
import keras
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Tuple

class_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']


def get_test_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    dataset = keras.utils.image_dataset_from_directory(
        directory=data_dir,
        color_mode="grayscale",
        image_size=(100, 100),
        shuffle=True,
        batch_size=32,
        seed=522437,
    )
    test_images, test_labels = zip(*dataset.unbatch().as_numpy_iterator())
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    return test_images, test_labels


def main(model_path: str, data_dir: str) -> None:
    model = keras.models.load_model(model_path)
    test_images, test_labels = get_test_data(data_dir)

    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    matrix = confusion_matrix(test_labels, predicted_labels, labels=range(10))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[s[-1] for s in class_names])
    display.plot(cmap=plt.cm.Blues, values_format='.4g')

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validator')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    args = parser.parse_args()

    main(args.model_path, args.data_dir)
