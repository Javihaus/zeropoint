"""The energy function and the decision rule, in NumPy.

Each option y is a well in a learned space:

    E_y(z) = -A_y * exp(-||z - c_y||^2 / s_y^2)

The well is deep (A_y) near its centre c_y and flat, at exactly zero, far away. Zero is the energy of the
vacuum, the state "none of the options". Turning energies into probabilities with Boltzmann's rule and
measuring every option against the vacuum gives

    w_y = exp(-E_y / T) - 1,     w_none = 1,     p(y) = w_y / (w_none + sum w),

so a text that falls in no well gives every option weight close to 0 and leaves the probability with "none".
"""

from __future__ import annotations

import numpy as np

_SQRT_2_OVER_PI = 0.7978845608028654


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(_SQRT_2_OVER_PI * (x + 0.044715 * x ** 3)))


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def mlp(x: np.ndarray, w: dict, name: str) -> np.ndarray:
    h = gelu(x @ w[f"{name}.0.weight"].T + w[f"{name}.0.bias"])
    return h @ w[f"{name}.2.weight"].T + w[f"{name}.2.bias"]


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def wells(option_emb: np.ndarray, w: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre, depth and squared width of the well of each option, from the option's embedding."""
    centre = unit(mlp(option_emb, w, "g"))
    depth = softplus(option_emb @ w["depth.weight"].T + w["depth.bias"])[:, 0] + 0.05
    width2 = softplus(option_emb @ w["width.weight"].T + w["width.bias"])[:, 0] + 0.05
    return centre, depth, width2


def energies(text_emb: np.ndarray, centre: np.ndarray, depth: np.ndarray, width2: np.ndarray,
             w: dict) -> np.ndarray:
    """Energy of every text (rows) in every well (columns). Always <= 0; 0 is the vacuum."""
    z = unit(mlp(text_emb, w, "f"))
    d2 = ((z[:, None, :] - centre[None, :, :]) ** 2).sum(-1)
    return -depth[None, :] * np.exp(-d2 / width2[None, :])


def log_weights(energy: np.ndarray, temperature: float) -> np.ndarray:
    """log(exp(-E/T) - 1), computed without overflow."""
    x = np.maximum(-energy / temperature, 1e-6)
    return x + np.log(-np.expm1(-x))


def probabilities(energy: np.ndarray, temperature: float, log_vacuum: np.ndarray) -> np.ndarray:
    """Columns: one per option, then "none" last. Rows sum to 1."""
    logits = np.concatenate([log_weights(energy, temperature), log_vacuum[:, None]], axis=1)
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def distance_beyond_edge(text_emb: np.ndarray, means: np.ndarray, precision: np.ndarray,
                         q50: float, q99: float) -> np.ndarray:
    """How far each text is beyond the edge of the data the model was calibrated on.

    Mahalanobis distance to the nearest known class, in the encoder's own space, rescaled so that 0 is the
    99th percentile of in-domain texts and 1 is one more (q99 - q50) beyond it. Negative means inside.
    """
    zp = text_emb @ precision
    d = (zp * text_emb).sum(1)[:, None] - 2 * zp @ means.T + ((means @ precision) * means).sum(1)[None, :]
    return (d.min(1) - q99) / (q99 - q50)
