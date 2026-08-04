"""Small, dependency-light Q networks implemented with NumPy.

The network intentionally implements only the operations needed by the lab's
value-based agents.  Hidden layers use ReLU activations and the output layer is
linear, so the returned values are unconstrained action-value estimates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]


def _positive_dimension(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class MLPQNetwork:
    """A two- or three-layer fully connected Q network.

    ``hidden_sizes`` must contain one or two entries.  In the conventional
    layer-counting terminology this gives either a two-layer network (one
    hidden layer plus the output layer) or a three-layer network.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        *,
        seed: int = 0,
    ) -> None:
        self.input_dim = _positive_dimension(input_dim, "input_dim")
        self.output_dim = _positive_dimension(output_dim, "output_dim")
        if len(hidden_sizes) not in (1, 2):
            raise ValueError("hidden_sizes must describe one or two hidden layers")
        self.hidden_sizes = tuple(
            _positive_dimension(size, f"hidden_sizes[{index}]")
            for index, size in enumerate(hidden_sizes)
        )
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        rng = np.random.default_rng(seed)
        layer_sizes = (self.input_dim, *self.hidden_sizes, self.output_dim)
        self._weights: list[FloatArray] = []
        self._biases: list[FloatArray] = []
        for index, (fan_in, fan_out) in enumerate(pairwise(layer_sizes)):
            # He initialization for ReLU layers and Xavier-like scaling for the
            # final linear layer keep initial values in a useful numeric range.
            scale = np.sqrt((2.0 if index < len(self.hidden_sizes) else 1.0) / fan_in)
            weights = rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float64)
            self._weights.append(weights)
            self._biases.append(np.zeros(fan_out, dtype=np.float64))

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        """Return all layer widths, including input and output widths."""

        return (self.input_dim, *self.hidden_sizes, self.output_dim)

    def predict(self, inputs: npt.ArrayLike) -> FloatArray:
        """Predict Q values for one state or a batch of states.

        A one-dimensional input produces a one-dimensional result; a batch
        preserves its leading batch dimension.
        """

        batch, single = self._validated_inputs(inputs)
        output, _, _ = self._forward(batch)
        return output[0].copy() if single else output.copy()

    def train_batch(
        self,
        inputs: npt.ArrayLike,
        targets: npt.ArrayLike,
        *,
        learning_rate: float = 1e-3,
        max_grad_norm: float | None = None,
    ) -> float:
        """Take one gradient step on mean-squared error for full Q targets."""

        batch, _ = self._validated_inputs(inputs)
        target_batch = np.asarray(targets, dtype=np.float64)
        if target_batch.ndim == 1 and batch.shape[0] == 1:
            target_batch = target_batch.reshape(1, -1)
        expected = (batch.shape[0], self.output_dim)
        if target_batch.shape != expected:
            raise ValueError(f"targets must have shape {expected}, got {target_batch.shape}")
        self._require_finite(target_batch, "targets")

        predictions, activations, preactivations = self._forward(batch)
        difference = predictions - target_batch
        with np.errstate(over="ignore", invalid="ignore"):
            loss = float(np.mean(np.square(difference)))
        if not np.isfinite(loss):
            raise FloatingPointError("loss is not finite")
        output_gradient = (2.0 / difference.size) * difference
        gradients = self._backward(output_gradient, activations, preactivations)
        self._apply_gradients(gradients, learning_rate, max_grad_norm)
        return loss

    def train_td_batch(
        self,
        states: npt.ArrayLike,
        actions: npt.ArrayLike,
        td_targets: npt.ArrayLike,
        *,
        learning_rate: float = 1e-3,
        max_grad_norm: float | None = None,
    ) -> float:
        """Take one TD/MSE step for the selected action in each state."""

        batch, _ = self._validated_inputs(states)
        action_array = np.asarray(actions)
        if action_array.ndim != 1 or action_array.shape[0] != batch.shape[0]:
            raise ValueError(f"actions must have shape ({batch.shape[0]},)")
        if action_array.dtype.kind not in "iu":
            raise TypeError("actions must contain integer action indices")
        action_indices = action_array.astype(np.intp, copy=False)
        if np.any(action_indices < 0) or np.any(action_indices >= self.output_dim):
            raise ValueError(f"actions must be in [0, {self.output_dim})")

        target_array = np.asarray(td_targets, dtype=np.float64)
        if target_array.shape != (batch.shape[0],):
            raise ValueError(f"td_targets must have shape ({batch.shape[0]},)")
        self._require_finite(target_array, "td_targets")

        predictions, activations, preactivations = self._forward(batch)
        rows = np.arange(batch.shape[0])
        difference = predictions[rows, action_indices] - target_array
        with np.errstate(over="ignore", invalid="ignore"):
            loss = float(np.mean(np.square(difference)))
        if not np.isfinite(loss):
            raise FloatingPointError("loss is not finite")

        output_gradient = np.zeros_like(predictions)
        output_gradient[rows, action_indices] = (2.0 / batch.shape[0]) * difference
        gradients = self._backward(output_gradient, activations, preactivations)
        self._apply_gradients(gradients, learning_rate, max_grad_norm)
        return loss

    def copy(self) -> MLPQNetwork:
        """Return a fully independent network with identical parameters."""

        clone = type(self)(self.input_dim, self.output_dim, self.hidden_sizes, seed=0)
        clone._weights = [weights.copy() for weights in self._weights]
        clone._biases = [biases.copy() for biases in self._biases]
        return clone

    def soft_update_from(self, source: MLPQNetwork, tau: float) -> None:
        """Move this network's parameters toward ``source`` by fraction ``tau``."""

        self._require_compatible(source)
        if isinstance(tau, bool) or not np.isfinite(tau) or not 0.0 <= tau <= 1.0:
            raise ValueError("tau must be finite and in [0, 1]")
        blend = float(tau)
        self._weights = [
            (1.0 - blend) * target + blend * origin
            for target, origin in zip(self._weights, source._weights, strict=True)
        ]
        self._biases = [
            (1.0 - blend) * target + blend * origin
            for target, origin in zip(self._biases, source._biases, strict=True)
        ]

    def soft_update(self, source: MLPQNetwork, tau: float) -> None:
        """Alias for :meth:`soft_update_from`."""

        self.soft_update_from(source, tau)

    def state_dict(self) -> dict[str, object]:
        """Return parameters and architecture using JSON-serializable values."""

        return {
            "format_version": 1,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_sizes": list(self.hidden_sizes),
            "weights": [weights.tolist() for weights in self._weights],
            "biases": [biases.tolist() for biases in self._biases],
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Load a state produced by :meth:`state_dict` after strict validation."""

        input_dim, output_dim, hidden_sizes = self._architecture_from_state(state)
        raw_weights = state.get("weights")
        raw_biases = state.get("biases")
        if (input_dim, output_dim, hidden_sizes) != (
            self.input_dim,
            self.output_dim,
            self.hidden_sizes,
        ):
            raise ValueError("network state architecture does not match this network")
        if not isinstance(raw_weights, Sequence) or not isinstance(raw_biases, Sequence):
            raise ValueError("weights and biases must be sequences")
        if len(raw_weights) != len(self._weights) or len(raw_biases) != len(self._biases):
            raise ValueError("network state has an incorrect number of layers")

        weights = [np.asarray(value, dtype=np.float64) for value in raw_weights]
        biases = [np.asarray(value, dtype=np.float64) for value in raw_biases]
        for index, (candidate, current) in enumerate(
            zip(weights, self._weights, strict=True)
        ):
            if candidate.shape != current.shape:
                raise ValueError(
                    f"weights[{index}] has shape {candidate.shape}, expected {current.shape}"
                )
            self._require_finite(candidate, f"weights[{index}]")
        for index, (candidate, current) in enumerate(
            zip(biases, self._biases, strict=True)
        ):
            if candidate.shape != current.shape:
                raise ValueError(
                    f"biases[{index}] has shape {candidate.shape}, expected {current.shape}"
                )
            self._require_finite(candidate, f"biases[{index}]")
        self._weights = [value.copy() for value in weights]
        self._biases = [value.copy() for value in biases]

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> MLPQNetwork:
        """Construct a network from a serialized state dictionary."""

        input_dim, output_dim, hidden_sizes = cls._architecture_from_state(state)
        network = cls(input_dim, output_dim, hidden_sizes, seed=0)
        network.load_state_dict(state)
        return network

    @staticmethod
    def _architecture_from_state(
        state: Mapping[str, object],
    ) -> tuple[int, int, tuple[int, ...]]:
        raw_input = state.get("input_dim")
        raw_output = state.get("output_dim")
        raw_hidden = state.get("hidden_sizes")
        if (
            isinstance(raw_input, bool)
            or not isinstance(raw_input, int)
            or isinstance(raw_output, bool)
            or not isinstance(raw_output, int)
            or not isinstance(raw_hidden, Sequence)
            or isinstance(raw_hidden, (str, bytes))
        ):
            raise ValueError("invalid network state architecture")
        hidden: list[int] = []
        for value in raw_hidden:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("hidden_sizes in network state must contain integers")
            hidden.append(value)
        return raw_input, raw_output, tuple(hidden)

    def _validated_inputs(self, inputs: npt.ArrayLike) -> tuple[FloatArray, bool]:
        batch = np.asarray(inputs, dtype=np.float64)
        single = batch.ndim == 1
        if single:
            if batch.shape != (self.input_dim,):
                raise ValueError(f"input must have shape ({self.input_dim},), got {batch.shape}")
            batch = batch.reshape(1, -1)
        elif batch.ndim == 2:
            if batch.shape[1] != self.input_dim:
                raise ValueError(
                    f"input batch must have {self.input_dim} features, got {batch.shape[1]}"
                )
            if batch.shape[0] == 0:
                raise ValueError("input batch must not be empty")
        else:
            raise ValueError("inputs must be a state vector or a two-dimensional batch")
        self._require_finite(batch, "inputs")
        return batch, single

    def _forward(
        self, batch: FloatArray
    ) -> tuple[FloatArray, list[FloatArray], list[FloatArray]]:
        activations = [batch]
        preactivations: list[FloatArray] = []
        current = batch
        with np.errstate(over="ignore", invalid="ignore"):
            for weights, biases in zip(
                self._weights[:-1], self._biases[:-1], strict=True
            ):
                linear = current @ weights + biases
                preactivations.append(linear)
                current = np.maximum(linear, 0.0)
                activations.append(current)
            output = current @ self._weights[-1] + self._biases[-1]
        if not np.all(np.isfinite(output)):
            raise FloatingPointError("network prediction is not finite")
        return output, activations, preactivations

    def _backward(
        self,
        output_gradient: FloatArray,
        activations: list[FloatArray],
        preactivations: list[FloatArray],
    ) -> list[tuple[FloatArray, FloatArray]]:
        gradients: list[tuple[FloatArray, FloatArray]] = []
        gradient = output_gradient
        for layer in range(len(self._weights) - 1, -1, -1):
            weight_gradient = activations[layer].T @ gradient
            bias_gradient = gradient.sum(axis=0)
            gradients.append((weight_gradient, bias_gradient))
            if layer > 0:
                gradient = (gradient @ self._weights[layer].T) * (
                    preactivations[layer - 1] > 0.0
                )
        gradients.reverse()
        return gradients

    def _apply_gradients(
        self,
        gradients: list[tuple[FloatArray, FloatArray]],
        learning_rate: float,
        max_grad_norm: float | None,
    ) -> None:
        if isinstance(learning_rate, bool) or not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if max_grad_norm is not None:
            if (
                isinstance(max_grad_norm, bool)
                or not np.isfinite(max_grad_norm)
                or max_grad_norm <= 0
            ):
                raise ValueError("max_grad_norm must be finite and positive")
            squared_norm = sum(
                float(np.sum(weight_grad * weight_grad) + np.sum(bias_grad * bias_grad))
                for weight_grad, bias_grad in gradients
            )
            norm = np.sqrt(squared_norm)
            if not np.isfinite(norm):
                raise FloatingPointError("gradient norm is not finite")
            if norm > max_grad_norm:
                scale = float(max_grad_norm / norm)
                gradients = [(weight * scale, bias * scale) for weight, bias in gradients]

        new_weights = [
            weights - float(learning_rate) * weight_gradient
            for weights, (weight_gradient, _) in zip(
                self._weights, gradients, strict=True
            )
        ]
        new_biases = [
            biases - float(learning_rate) * bias_gradient
            for biases, (_, bias_gradient) in zip(
                self._biases, gradients, strict=True
            )
        ]
        if not all(np.all(np.isfinite(value)) for value in (*new_weights, *new_biases)):
            raise FloatingPointError("parameter update produced non-finite values")
        self._weights = new_weights
        self._biases = new_biases

    def _require_compatible(self, other: MLPQNetwork) -> None:
        if not isinstance(other, MLPQNetwork) or other.layer_sizes != self.layer_sizes:
            raise ValueError("networks must have identical architectures")

    @staticmethod
    def _require_finite(values: FloatArray, name: str) -> None:
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite values")


# A concise alias is convenient for agents while the explicit class name keeps
# documentation clear.
QNetwork = MLPQNetwork

__all__ = ["MLPQNetwork", "QNetwork"]
