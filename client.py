import flwr as fl
import tensorflow as tf
import keras
import argparse

from typing import Dict, Tuple
from flwr.common import Scalar, NDArrays
from model import get_model


class Client(fl.client.NumPyClient):
    def __init__(self, model: keras.Model, trainset: tf.data.Dataset, validset: tf.data.Dataset):
        self.model = model
        self.trainset = trainset
        self.validset = validset

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return self.model.get_weights()

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.model.set_weights(parameters)
        self.model.fit(self.trainset, epochs=1, batch_size=32)
        return self.model.get_weights(), len(self.trainset), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.validset)
        return loss, len(self.validset), {"accuracy": accuracy}


def get_datasets(data_dir: str) -> tf.data.Dataset:
    datasets = keras.utils.image_dataset_from_directory(
        directory=data_dir,
        validation_split=0.1,
        subset="both",
        color_mode="grayscale",
        image_size=(100, 100),
        shuffle=True,
        batch_size=32,
        seed=522437,
    )
    return datasets


def main(server_address: str, data_dir: str) -> None:
    trainset, validset = get_datasets(data_dir)
    model = get_model()

    fl.client.start_numpy_client(
        server_address=server_address,
        client=Client(model, trainset, validset)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Client')
    parser.add_argument('--server-address', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    args = parser.parse_args()

    main(server_address=args.server_address, data_dir=args.data_dir)
