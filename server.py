import numpy as np
import flwr as fl
import argparse
import matplotlib.pyplot as plt

from flwr.common import EvaluateRes, Scalar, FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from logging import INFO
from typing import List, Tuple, Union, Optional, Dict
from model import get_model


class CustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        accuracy = sum(accuracies) / sum(examples)
        log(INFO, f"Round {server_round} accuracy aggregated from {len(results)} clients: {accuracy}")

        return loss, {"accuracy": accuracy}

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if parameters is not None:
            model = get_model()
            model.set_weights(fl.common.parameters_to_ndarrays(parameters))
            model.save(f"models/model-round-{server_round}.keras")
            log(INFO, f"Saving round {server_round} model")

        return parameters, metrics


def plot(history: fl.server.history.History) -> None:
    accuracy = history.metrics_distributed["accuracy"]
    accuracy_index = [data[0] for data in accuracy]
    accuracy_value = [100.0 * data[1] for data in accuracy]

    loss = history.losses_distributed
    loss_index = [data[0] for data in loss]
    loss_value = [data[1] for data in loss]

    plt.plot(accuracy_index, accuracy_value, "r-", label="Accuracy")
    plt.plot(loss_index, loss_value, "b-", label="Loss")
    plt.grid()
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.title("Handwritten Letters Classifier - Federated Accuracy")
    plt.show()


def main(server_address: str, num_rounds: int) -> None:
    history = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=CustomStrategy(min_available_clients=3, min_fit_clients=3, min_evaluate_clients=3),
    )

    plot(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Server')
    parser.add_argument('--server-address', type=str, required=True)
    parser.add_argument('--num-rounds', type=int, required=True)
    args = parser.parse_args()

    main(server_address=args.server_address, num_rounds=args.num_rounds)
