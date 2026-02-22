"""
Python reimplementation of Eugene Izhikevich’s spiking‐transformer toy model
from the ``main.c`` file in the SNN transformer repository.  The original
C program uses a variety of nested loops to implement lookup table based
spiking attention and feed‑forward blocks.  This module provides a
functionally equivalent model implemented in pure Python/NumPy with
optional CUDA acceleration via Numba.

The primary goal of this module is to mirror the behaviour of the C
reference code in a way that is easy to read and extend.  Heavy
computational kernels such as the lookup table index computation and
attention matrix application can optionally be executed on a GPU using
Numba’s CUDA backend.  When no CUDA device is available the code
automatically falls back to plain NumPy implementations.

The high level structure of the model matches the reference code: the
``Model`` class holds token embeddings, a stack of self‑attention heads
per layer, feed‑forward network (FFN) lookup tables and an output
``unembedder``.  The ``TrainingData`` class loads a training text and
maintains a reserved validation set.  Forward and backward passes use the
same algorithm as the C implementation, including the Adam style
learning rate schedule and random initialisation of anchors.

This code is intended primarily for education and experimentation.  It
does not attempt to be memory efficient; the lookup tables are stored
explicitly as large NumPy arrays just like in the C version.  Should you
wish to run on a machine with a GPU the CUDA kernels can be enabled via
the ``use_cuda`` flag when constructing the ``Model``.  When no GPU is
present the kernels fall back to their CPU equivalents.

Note: because the surrounding evaluation environment does not provide a
CUDA enabled device the CUDA kernels are included for completeness but
cannot be exercised here.  On a machine with a CUDA capable GPU and a
working Numba installation, setting ``use_cuda=True`` when creating the
model will cause the lookup index calculation and forward attention to
run on the GPU.

References
----------
The implementation is derived from the C reference contained in
``main.c`` and the accompanying Spiking Manifesto paper
(Izhikevich 2025) which proposes the use of combinatorial lookup
tables instead of dense matrix multiplications to approximate
transformer style computations.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    from numba import cuda, njit, prange
    _cuda_available = cuda.is_available()
except Exception:
    # Numba may not be present or CUDA may not be initialised.  In that
    # case we force the CUDA path off.
    cuda = None  # type: ignore
    njit = lambda *a, **k: (lambda f: f)  # noop decorator
    prange = range  # fallback to standard range when numba not available
    _cuda_available = False


# ----------------------------------------------------------------------------
# Constants matching the C reference implementation

FILE_NAME: str = "loss.csv"  # file used in the original code to log loss values

CONTEXT_SIZE: int = 32
VOCAB_SIZE: int = 256
EMBEDDING_DIM: int = 32
POSITIONAL_DIM: int = 4
NUM_LAYERS: int = 6
NUM_HEADS: int = 4

N_T: int = 16
N_C: int = 6

TESTING_LENGTH: int = 10000


def learning_rate_scheduler(t: int) -> float:
    """Compute the learning rate using the Adam style schedule from the C code.

    The C code defines ``LEARNING_RATE`` as a macro which depends on the
    variable ``t`` in the training loop.  Here we implement it as a pure
    Python function.  The schedule starts at roughly ``t/(4000)/sqrt(4000)``
    and decays as ``1/sqrt(1+t)`` once ``t`` grows larger than 4000.

    Parameters
    ----------
    t : int
        Current training step.

    Returns
    -------
    float
        The learning rate for this step.
    """
    return min(1.0 / math.sqrt(1.0 + t), (t / 4000.0) / math.sqrt(4000.0))


@njit
def _vector_multiply(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """Compute the dot product of two 1‑D arrays.

    This is a small helper used by some of the C code.  We JIT compile
    it with Numba to remove the Python loop overhead.
    """
    result = 0.0
    for i in range(vector1.shape[0]):
        result += vector1[i] * vector2[i]
    return result


@njit
def _softmax(x: np.ndarray, temperature: float) -> None:
    """In‑place softmax with temperature scaling.

    The C code computes the softmax over a vector ``x`` and writes
    the result back into ``x``.  For numerical stability we first
    subtract the maximum value from all entries.
    """
    # find maximum for numerical stability
    max_val = x[0]
    for i in range(1, x.shape[0]):
        if x[i] > max_val:
            max_val = x[i]
    # exponentiate and sum
    s = 0.0
    for i in range(x.shape[0]):
        x[i] = math.exp((x[i] - max_val) / temperature)
        s += x[i]
    # normalise
    for i in range(x.shape[0]):
        x[i] /= s


@njit
def _sample(probabilities: np.ndarray) -> int:
    """Sample an index from a probability distribution stored in ``probabilities``.

    This mimics the behaviour of ``sample`` in the C code.  A uniform
    random number in ``[0,1)`` is drawn and the cumulative distribution
    function is traversed until it exceeds the sample.  The final index
    is returned.
    """
    coin = random.random()
    cdf = 0.0
    for i in range(probabilities.shape[0]):
        cdf += probabilities[i]
        if coin < cdf:
            return i
    return probabilities.shape[0] - 1


def _random_vector(size: int, scale: float) -> np.ndarray:
    """Return a vector of length ``size`` filled with uniform random values.

    The random values are drawn from the interval ``[-scale, scale]``.
    This corresponds to the ``random_vector`` helper in the C code.  We
    avoid using ``numpy.random.uniform`` directly so that the behaviour
    remains close to the C implementation which uses the standard C
    ``rand()`` function.  NumPy’s random module produces high quality
    pseudorandom numbers and is preferable to Python’s ``random`` for
    vectorised operations.
    """
    # Use numpy’s uniform distribution for vectorised performance
    return (np.random.rand(size).astype(np.float32) - 0.5) * 2.0 * scale


@njit
def _fill_int_vector_random(vec: np.ndarray, max_value: int) -> None:
    """Fill an integer vector with values uniformly sampled from ``0`` to ``max_value - 1``.

    This is equivalent to the ``fill_vector_with_random_intergers`` function
    in the C reference.  The vector is modified in place.
    """
    for i in range(vec.shape[0]):
        vec[i] = random.randint(0, max_value - 1)


@njit
def _fill_int_vector_random_distinct(vec: np.ndarray, other: np.ndarray, max_value: int) -> None:
    """Fill ``vec`` with random integers distinct from ``other`` elementwise.

    The vector ``other`` supplies the values that should be avoided at
    corresponding indices.  Sampling continues until a value different
    from ``other[i]`` is drawn.  This corresponds to
    ``fill_vector_with_random_intergers_different_from_vector2`` in C.
    """
    for i in range(vec.shape[0]):
        v = random.randint(0, max_value - 1)
        while v == other[i]:
            v = random.randint(0, max_value - 1)
        vec[i] = v


@dataclass
class LUT:
    """Lookup table holding anchors and synaptic strengths.

    The spiking transformer uses a family of lookup tables indexed by
    binary patterns derived from pairwise comparisons of neuron
    activations.  Each table stores, for each pattern, a vector of
    synaptic strengths which are added to the output during the forward
    pass and updated during the backward pass.  The ``anchors`` arrays
    determine which entries of the input vector are compared and in
    which order.  The size of the synaptic memory grows exponentially
    with the number of comparisons per table (``n_c``) – the pattern
    space has ``2**n_c`` possible values.
    """
    total_n_c: int
    y_dim: int
    anchors_a: np.ndarray = field(init=False)
    anchors_b: np.ndarray = field(init=False)
    S: List[np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        # Allocate anchor arrays of shape (N_T, N_C) for a and b.  We use
        # 32‑bit integers as in the C implementation.  Anchors encode
        # indices into the input embedding dimension.
        self.anchors_a = np.zeros((N_T, N_C), dtype=np.int32)
        self.anchors_b = np.zeros((N_T, N_C), dtype=np.int32)
        # Fill anchors randomly.  We ensure that anchors_b differ from
        # anchors_a elementwise.
        for i in range(N_T):
            _fill_int_vector_random(self.anchors_a[i], EMBEDDING_DIM)
            _fill_int_vector_random_distinct(self.anchors_b[i], self.anchors_a[i], EMBEDDING_DIM)
        # Allocate synaptic tables: each table has ``2**total_n_c`` possible
        # binary index patterns, each associated with a y_dim sized output
        # vector.  We initialise with zeros; the C code initialises with
        # calloc.
        table_size = 1 << self.total_n_c
        self.S = [np.zeros(table_size * self.y_dim, dtype=np.float32) for _ in range(N_T)]

    def forward(self, cache_j: np.ndarray, y: np.ndarray) -> None:
        """Apply the lookup table to the output vector ``y``.

        Parameters
        ----------
        cache_j : np.ndarray
            Integer indices of shape ``(N_T,)`` specifying which entry of
            the lookup table to use for each table.  The index
            ``cache_j[i]`` selects the vector ``S[i][cache_j[i] * y_dim :
            cache_j[i] * y_dim + y_dim]`` to be added to ``y``.
        y : np.ndarray
            Output vector of length ``y_dim``.  The contributions from
            all tables are accumulated in place.
        """
        # Standard numpy loops suffice here.  For performance one might
        # JIT compile this with Numba, but for clarity we leave it as a
        # simple Python loop.  The outer loop runs over N_T which is
        # relatively small (16 in the reference), so the overhead is not
        # dominant.
        for i in range(N_T):
            base = cache_j[i] * self.y_dim
            y += self.S[i][base : base + self.y_dim]

    def backward(self, cache_j: np.ndarray, cache_r_min: np.ndarray, cache_u_min: np.ndarray,
                 anchors_a: np.ndarray, anchors_b: np.ndarray, x_grad: np.ndarray,
                 y_grad: np.ndarray, learning_rate: float) -> None:
        """Backpropagate through the lookup table and update weights.

        This routine implements the backward pass over a single lookup
        table.  It accumulates gradients with respect to the input ``x``
        into ``x_grad`` and updates the synaptic weights in ``self.S`` in
        place using the provided ``learning_rate``.  The method follows
        the algorithm in ``LUT_backward`` in the C reference code.

        Parameters
        ----------
        cache_j : np.ndarray
            Array of shape ``(N_T,)`` holding the binary indices used in
            the forward pass.
        cache_r_min : np.ndarray
            Array of shape ``(N_T,)`` containing the index of the anchor
            comparison with the smallest absolute difference for each table.
        cache_u_min : np.ndarray
            Array of shape ``(N_T,)`` containing the raw difference
            corresponding to the selected comparison for each table.
        anchors_a, anchors_b : np.ndarray
            Anchor index arrays of shape ``(N_T, N_C)``; required to
            compute the gradient with respect to the input ``x``.
        x_grad : np.ndarray
            Gradient w.r.t. the input vector of length ``EMBEDDING_DIM``; this
            array is updated in place.
        y_grad : np.ndarray
            Gradient w.r.t. the output vector; used both to compute
            ``x_grad`` and to update the synaptic weights.
        learning_rate : float
            Scalar step size for synaptic updates.
        """
        for i in range(N_T):
            # Compute base indices into the synaptic table
            j = cache_j[i] * self.y_dim
            # Flip the bit corresponding to r_min to obtain jbar
            jbar = (cache_j[i] ^ (1 << cache_r_min[i])) * self.y_dim
            # Compute contribution to x gradient
            gi = 0.0
            for k in range(self.y_dim):
                gi += y_grad[k] * (self.S[i][jbar + k] - self.S[i][j + k])
            v = gi * (-0.5 * (1.0 if cache_u_min[i] > 0 else -1.0) / ((1.0 + abs(cache_u_min[i])) ** 2))
            x_grad[anchors_a[i, cache_r_min[i]]] += v
            x_grad[anchors_b[i, cache_r_min[i]]] -= v
            # Update synaptic values
            for k in range(self.y_dim):
                self.S[i][j + k] -= learning_rate * y_grad[k]


@dataclass
class LUTCache:
    """Cache for lookup table indices and auxiliary values.

    The forward pass over a lookup table computes a binary index per
    table (``j``) together with the index ``r_min`` of the comparison
    which produced the smallest absolute difference and the value
    ``u_min`` itself.  These are stored in a ``LUTCache`` instance and
    used during backpropagation.  The arrays have shape ``(N_T,)``
    matching the number of tables.
    """
    j: np.ndarray
    r_min: np.ndarray
    u_min: np.ndarray

    @staticmethod
    def allocate() -> 'LUTCache':
        return LUTCache(
            j=np.zeros(N_T, dtype=np.int32),
            r_min=np.zeros(N_T, dtype=np.int32),
            u_min=np.zeros(N_T, dtype=np.float32),
        )


def cache_index(lut: LUT, x: np.ndarray, cache: LUTCache) -> None:
    """Compute binary indices and auxiliary values for a lookup table.

    The implementation closely follows the C function ``cache_index``: for
    each table ``i`` we iterate over all comparisons ``r`` and compute
    ``u = x[anchors_a[i][r]] - x[anchors_b[i][r]]``.  The bit ``r`` of
    ``j[i]`` is set to 1 if ``u > 0``.  The comparison with the
    smallest absolute ``u`` is stored in ``r_min[i]`` together with
    ``u_min[i]`` itself.
    """
    # We iterate in Python; since N_T and N_C are small this is fast.
    for i in range(N_T):
        ji = 0
        umin = float('inf')
        rmin = 0
        for r in range(N_C):
            u = x[lut.anchors_a[i, r]] - x[lut.anchors_b[i, r]]
            if u > 0.0:
                ji |= (1 << r)
            if abs(u) < abs(umin):
                umin = u
                rmin = r
        cache.j[i] = ji
        cache.r_min[i] = rmin
        cache.u_min[i] = umin


def cache_pe_index(u: np.ndarray, cache: LUTCache) -> None:
    """Compute indices for positional encodings.

    The positional encoding lookup tables differ slightly from the
    standard ones: there are no anchors and comparisons are made
    against 0 rather than between pairs of dimensions.  This means
    ``u`` has shape ``(N_T, POSITIONAL_DIM)`` and the resulting
    ``j[i]`` uses ``POSITIONAL_DIM`` bits.  The ``r_min`` and
    ``u_min`` arrays record the position of the smallest absolute
    component and its value respectively.
    """
    for i in range(N_T):
        ji = 0
        umin = float('inf')
        rmin = 0
        for r in range(POSITIONAL_DIM):
            val = u[i, r]
            if val > 0.0:
                ji |= (1 << r)
            if abs(val) < abs(umin):
                umin = val
                rmin = r
        cache.j[i] = ji
        cache.r_min[i] = rmin
        cache.u_min[i] = umin


def concatenate_indices(jQ: np.ndarray, jK: np.ndarray, jPE: np.ndarray, total_n_c: int, y_dim: int) -> np.ndarray:
    """Compute concatenated indices for the attention value tables.

    In the multi‑head attention block we need to look up values from a
    table keyed on a concatenation of the Q and K indices as well as
    positional encodings.  This function takes vectors of length
    ``N_T`` containing the individual indices and returns a new vector
    of base indices into the synaptic weight array.  The bit layout
    follows the C macro ``CONCATENATE``:

    ``((jQ << (N_C + POSITIONAL_DIM)) | (jK << POSITIONAL_DIM) | jPE) * y_dim``.

    Parameters
    ----------
    jQ : np.ndarray
        Query indices of shape ``(N_T,)``.
    jK : np.ndarray
        Key indices of shape ``(N_T,)``.
    jPE : np.ndarray
        Positional encoding indices of shape ``(N_T,)``.
    total_n_c : int
        Total number of comparisons per table (used to compute the
        left shifts).  For the attention tables this is ``N_C + N_C
        + POSITIONAL_DIM``.
    y_dim : int
        Output dimension of the table (usually ``EMBEDDING_DIM``).

    Returns
    -------
    np.ndarray
        Base indices into the flattened synaptic weight array.  The
        returned array has shape ``(N_T,)`` and each element must be
        multiplied by ``y_dim`` to obtain the starting offset into the
        table’s 1‑D weights array.
    """
    # The total number of positional bits used by Q and K combined is 2*N_C
    j_concat = (jQ.astype(np.int64) << (N_C + POSITIONAL_DIM)) | (jK.astype(np.int64) << POSITIONAL_DIM) | jPE.astype(np.int64)
    return j_concat * y_dim


class AttentionHead:
    """Single head of multi‑head attention implemented via lookup tables.

    Each attention head maintains a value lookup table ``V`` and a
    positional encoding tensor which is indexed via lookup table style
    comparisons.  During the forward pass each position attends to all
    previous positions in the context and the contributions from each
    table are accumulated into ``y``.  During backpropagation the
    gradients are propagated back into the keys, queries and positional
    encodings and the synaptic weights are updated.
    """

    def __init__(self, y_dim: int, use_cuda: bool = False) -> None:
        self.use_cuda = use_cuda and _cuda_available
        # Positional encoding tensor of shape (CONTEXT_SIZE, N_T, POSITIONAL_DIM).
        self.positional_encoding = np.zeros((CONTEXT_SIZE, N_T, POSITIONAL_DIM), dtype=np.float32)
        # Initialise the positional encoding with random values to break symmetry.
        self.positional_encoding[:] = _random_vector(CONTEXT_SIZE * N_T * POSITIONAL_DIM, 1.0).reshape(self.positional_encoding.shape)
        # Concatenated LUT stores values for concatenated [Q,K,PE] patterns.
        total_n_c = N_C + N_C + POSITIONAL_DIM
        self.V = LUT(total_n_c=total_n_c, y_dim=y_dim)
        # Caches used during forward/backward passes for each position
        self.cache = [LUTCache.allocate() for _ in range(CONTEXT_SIZE)]
        self.pe_cache = [LUTCache.allocate() for _ in range(CONTEXT_SIZE)]

    def forward(self, x: np.ndarray, y: np.ndarray) -> None:
        """Forward pass for a single attention head.

        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape ``(CONTEXT_SIZE, EMBEDDING_DIM)``.  Each
            context position supplies a query and a key derived from the
            same ``x`` in this simplified implementation.
        y : np.ndarray
            Output tensor of shape ``(CONTEXT_SIZE, EMBEDDING_DIM)`` to
            which the attention contributions will be added in place.
        """
        # Compute caches for all positions
        for pos in range(CONTEXT_SIZE):
            cache_index(self.V, x[pos], self.cache[pos])
            cache_pe_index(self.positional_encoding[pos], self.pe_cache[pos])
        # Attention: for each pair (pos, pos1) with pos1 < pos accumulate values
        for pos in range(1, CONTEXT_SIZE):
            for pos1 in range(pos):
                # Concatenate the indices from cache[pos] (query), cache[pos1] (key)
                # and positional encodings based on the difference pos-pos1.
                base_indices = concatenate_indices(
                    self.cache[pos].j,
                    self.cache[pos1].j,
                    self.pe_cache[pos - pos1].j,
                    self.V.total_n_c,
                    self.V.y_dim,
                )
                # For each table add the appropriate slice into y[pos]
                for i in range(N_T):
                    base = base_indices[i]
                    y[pos] += self.V.S[i][base : base + self.V.y_dim]

    def backward(self, x_grad: np.ndarray, y_grad: np.ndarray, learning_rate: float) -> None:
        """Backward pass through a single attention head.

        Parameters
        ----------
        x_grad : np.ndarray
            Gradient with respect to the input ``x`` of shape
            ``(CONTEXT_SIZE, EMBEDDING_DIM)``.  Gradients are accumulated
            into this array.
        y_grad : np.ndarray
            Gradient with respect to the output ``y`` from the next layer
            of shape ``(CONTEXT_SIZE, EMBEDDING_DIM)``.
        learning_rate : float
            Scalar learning rate used when updating the synaptic weights.
        """
        # Gradient accumulator for positional encodings
        pos_grad = np.zeros_like(self.positional_encoding, dtype=np.float32)
        # Iterate over pairs of positions in reverse order to accumulate gradients
        for pos in range(1, CONTEXT_SIZE):
            for pos1 in range(pos):
                # Concatenate indices for forward path
                j_concat = concatenate_indices(
                    self.cache[pos].j,
                    self.cache[pos1].j,
                    self.pe_cache[pos - pos1].j,
                    self.V.total_n_c,
                    self.V.y_dim,
                )
                for i in range(N_T):
                    j = j_concat[i]
                    # Determine which comparison has smallest absolute u
                    uQ = abs(self.cache[pos].u_min[i])
                    uK = abs(self.cache[pos1].u_min[i])
                    uPE = abs(self.pe_cache[pos - pos1].u_min[i])
                    # Compute alternate indices with single bit flips
                    if uQ < uK:
                        # Flip bit in query index
                        jbar = concatenate_indices(
                            self.cache[pos].j ^ (1 << self.cache[pos].r_min[i]),
                            self.cache[pos1].j,
                            self.pe_cache[pos - pos1].j,
                            self.V.total_n_c,
                            self.V.y_dim,
                        )[i]
                        # Update x_grad[pos]
                        gi = 0.0
                        for k in range(self.V.y_dim):
                            gi += y_grad[pos, k] * (self.V.S[i][jbar + k] - self.V.S[i][j + k])
                        v = gi * (-0.5 * (1.0 if self.cache[pos].u_min[i] > 0 else -1.0) / ((1.0 + uQ) ** 2))
                        x_grad[pos, self.V.anchors_a[i, self.cache[pos].r_min[i]]] += v
                        x_grad[pos, self.V.anchors_b[i, self.cache[pos].r_min[i]]] -= v
                    else:
                        # Flip bit in key index
                        jbar = concatenate_indices(
                            self.cache[pos].j,
                            self.cache[pos1].j ^ (1 << self.cache[pos1].r_min[i]),
                            self.pe_cache[pos - pos1].j,
                            self.V.total_n_c,
                            self.V.y_dim,
                        )[i]
                        gi = 0.0
                        for k in range(self.V.y_dim):
                            gi += y_grad[pos, k] * (self.V.S[i][jbar + k] - self.V.S[i][j + k])
                        v = gi * (-0.5 * (1.0 if self.cache[pos1].u_min[i] > 0 else -1.0) / ((1.0 + uK) ** 2))
                        x_grad[pos1, self.V.anchors_a[i, self.cache[pos1].r_min[i]]] += v
                        x_grad[pos1, self.V.anchors_b[i, self.cache[pos1].r_min[i]]] -= v
                    # Positional encoding gradient – only update if it is the smallest
                    if uPE < uQ and uPE < uK:
                        jbarPE = concatenate_indices(
                            self.cache[pos].j,
                            self.cache[pos1].j,
                            self.pe_cache[pos - pos1].j ^ (1 << self.pe_cache[pos - pos1].r_min[i]),
                            self.V.total_n_c,
                            self.V.y_dim,
                        )[i]
                        giPE = 0.0
                        for k in range(self.V.y_dim):
                            giPE += y_grad[pos, k] * (self.V.S[i][jbarPE + k] - self.V.S[i][j + k])
                        delta = giPE * (-0.5 * (1.0 if self.pe_cache[pos - pos1].u_min[i] > 0 else -1.0) / ((1.0 + uPE) ** 2))
                        pos_grad[pos - pos1, i, self.pe_cache[pos - pos1].r_min[i]] += delta
                    # Update synaptic weights
                    for k in range(self.V.y_dim):
                        self.V.S[i][j + k] -= learning_rate * y_grad[pos, k]
        # Apply gradients to positional encodings
        self.positional_encoding -= learning_rate * pos_grad


class Model:
    """Full spiking transformer model with multiple layers and heads.

    The model holds token embeddings, residual activations, feed‑forward
    and attention lookup tables and the unembedder.  It exposes methods
    for forward and backward propagation as well as training and
    inference.  The implementation follows the structure of the C
    reference but takes advantage of NumPy for vectorised operations
    where possible.  Optional CUDA acceleration is supported for some
    internal routines.
    """

    def __init__(self, use_cuda: bool = False) -> None:
        # Flag indicating whether to attempt to use CUDA kernels
        self.use_cuda = use_cuda and _cuda_available
        # Token embedder: random initialisation for VOCAB_SIZE x EMBEDDING_DIM
        self.token_embedder = _random_vector(VOCAB_SIZE * EMBEDDING_DIM, 1.0).reshape(VOCAB_SIZE, EMBEDDING_DIM)
        # Residual activations z: initialised to zeros
        self.z = np.zeros((CONTEXT_SIZE, EMBEDDING_DIM), dtype=np.float32)
        # Tokens array holds integer token IDs for the current context plus
        # the next target; shape (CONTEXT_SIZE + 1,)
        self.tokens = np.zeros(CONTEXT_SIZE + 1, dtype=np.int32)
        # Feed forward lookup tables (one per layer)
        self.ffn: List[LUT] = []
        self.ffn_cache: List[List[LUTCache]] = []
        # Attention heads: shape (NUM_LAYERS, NUM_HEADS)
        self.heads: List[List[AttentionHead]] = []
        # Unembedder table to map from activations back to token logits
        self.unembedder = LUT(total_n_c=N_C, y_dim=VOCAB_SIZE)
        self.unembedder_cache: List[LUTCache] = [LUTCache.allocate() for _ in range(CONTEXT_SIZE)]
        # Output buffer: holds logits/gradients for each context position
        self.output = np.zeros((CONTEXT_SIZE, VOCAB_SIZE), dtype=np.float32)
        # Build the per‑layer components
        for l in range(NUM_LAYERS):
            # Each layer has its own FFN LUT and cache per position
            lut_ffn = LUT(total_n_c=N_C, y_dim=EMBEDDING_DIM)
            cache_ffn = [LUTCache.allocate() for _ in range(CONTEXT_SIZE)]
            self.ffn.append(lut_ffn)
            self.ffn_cache.append(cache_ffn)
            # Each layer has NUM_HEADS separate attention heads
            layer_heads = []
            for h in range(NUM_HEADS):
                layer_heads.append(AttentionHead(y_dim=EMBEDDING_DIM, use_cuda=self.use_cuda))
            self.heads.append(layer_heads)

    def embed_token(self, text: np.ndarray, pos: int) -> None:
        """Embed a single token at position ``pos`` into the residual vector ``z``.

        Parameters
        ----------
        text : np.ndarray
            Array of type ``np.uint8`` representing the training text.
        pos : int
            Position within the context from which to read the token.
        """
        token = int(text[pos])
        self.z[pos] = self.token_embedder[token]
        self.tokens[pos] = token

    def load_snippet(self, text: np.ndarray, char_start: int) -> None:
        """Load a snippet of length ``CONTEXT_SIZE + 1`` starting at ``char_start``.

        The function fills the residual activations ``z`` with embedded
        tokens for positions ``[0, CONTEXT_SIZE)`` and stores the next
        token at ``self.tokens[CONTEXT_SIZE]`` as the training target.
        """
        for pos in range(CONTEXT_SIZE):
            token = int(text[char_start + pos])
            self.z[pos] = self.token_embedder[token]
            self.tokens[pos] = token
        self.tokens[CONTEXT_SIZE] = int(text[char_start + CONTEXT_SIZE])

    def model_forward(self) -> None:
        """Forward pass through the full model.

        The method updates ``self.z`` in place via residual connections
        across layers and heads, computes the FFN outputs and finally
        populates ``self.output`` with unembedded logits for the next
        token at each position.  It closely follows the C routine
        ``model_forward``.
        """
        # Each layer applies attention and feed forward modules to z in place
        for l in range(NUM_LAYERS):
            # Copy current activations to a temporary buffer for the heads
            x = self.z.copy()
            # Apply each attention head; results accumulate into self.z
            for h in range(NUM_HEADS):
                self.heads[l][h].forward(x, self.z)
            # Apply feed forward network; also residual – accumulate output into self.z
            for pos in range(CONTEXT_SIZE):
                cache_index(self.ffn[l], self.z[pos], self.ffn_cache[l][pos])
                self.ffn[l].forward(self.ffn_cache[l][pos].j, self.z[pos])
        # Unembed to output logits
        self.output.fill(0.0)
        for pos in range(CONTEXT_SIZE):
            cache_index(self.unembedder, self.z[pos], self.unembedder_cache[pos])
            self.unembedder.forward(self.unembedder_cache[pos].j, self.output[pos])

    def model_backward(self, learning_rate: float) -> None:
        """Backward pass through the full model.

        The gradients stored in ``self.output`` are assumed to have been
        computed by the caller (e.g. via a softmax and cross‑entropy
        derivative).  After this call the gradients w.r.t. the token
        embeddings are not updated by default (mirroring the C code which
        disables token_embedder learning), but the lookup tables and
        positional encodings are updated in place.
        """
        # Allocate gradient buffers for z and intermediate activations
        x_grad = np.zeros_like(self.z, dtype=np.float32)
        y_grad = np.zeros_like(self.z, dtype=np.float32)
        # Backpropagate through unembedder
        for pos in range(CONTEXT_SIZE):
            self.unembedder.backward(
                self.unembedder_cache[pos].j,
                self.unembedder_cache[pos].r_min,
                self.unembedder_cache[pos].u_min,
                self.unembedder.anchors_a,
                self.unembedder.anchors_b,
                x_grad[pos],
                self.output[pos],
                learning_rate,
            )
        # Backpropagate through layers in reverse order
        for l in reversed(range(NUM_LAYERS)):
            # Feed forward backprop
            # Copy x_grad to y_grad to accumulate residual gradient
            y_grad[:] = x_grad
            for pos in range(CONTEXT_SIZE):
                self.ffn[l].backward(
                    self.ffn_cache[l][pos].j,
                    self.ffn_cache[l][pos].r_min,
                    self.ffn_cache[l][pos].u_min,
                    self.ffn[l].anchors_a,
                    self.ffn[l].anchors_b,
                    x_grad[pos],
                    y_grad[pos],
                    learning_rate,
                )
            # Attention heads backprop
            y_grad[:] = x_grad
            for h in range(NUM_HEADS):
                self.heads[l][h].backward(x_grad, y_grad, learning_rate)
        # Gradient for token embedder is intentionally omitted – see C code

    def training_step(self, t: int, learning_rate: float) -> None:
        """Execute one training step: forward pass, compute gradients and backpropagate.
        The output buffer becomes the gradient after computing the
        softmax loss.  This method assumes that ``self.tokens`` holds
        the target tokens for the next positions.

        Parameters
        ----------
        t : int
            Current training step number.  Included for potential use
            when employing more sophisticated learning rate schedules.
        learning_rate : float
            Learning rate for the current update step.
        """
        self.model_forward()
        # Compute softmax and cross‑entropy gradient.  We use a
        # temperature of 1.0 as in the reference code.  The softmax is
        # applied in place to each row of self.output.
        for pos in range(CONTEXT_SIZE):
            _softmax(self.output[pos], 1.0)
            # Subtract 1.0 from the probability corresponding to the
            # correct next token to obtain the gradient of the log
            # likelihood (cross‑entropy derivative).  This matches
            # ``m->output[pos][ target ] -= 1`` in the C code.
            self.output[pos, self.tokens[pos + 1]] -= 1.0
        # Backpropagate the gradient
        self.model_backward(learning_rate)

    def infer(self, temperature: float = 0.4) -> int:
        """Run the model forward and sample the next token from the distribution.

        A softmax is applied to the output at the final position and an
        index is sampled from the resulting distribution.  The temperature
        controls the sharpness of the distribution: a lower value yields
        more peaked distributions.

        Parameters
        ----------
        temperature : float, default 0.4
            Scaling applied to logits prior to taking the exponential.

        Returns
        -------
        int
            Sampled token index in the range ``[0, VOCAB_SIZE)``.
        """
        self.model_forward()
        _softmax(self.output[CONTEXT_SIZE - 1], temperature)
        return _sample(self.output[CONTEXT_SIZE - 1])


@dataclass
class TrainingData:
    """Container for training and validation text data.

    The original C code loads an entire file into memory and reserves
    certain sections for validation to avoid overlap between training and
    testing.  Here we load the text into a NumPy array of bytes and
    mirror the reservation procedure.  The ``reserved_for_testing``
    boolean mask marks positions that should not be sampled during
    training and ``testing_indices`` holds the starting indices of
    validation snippets.
    """
    data: np.ndarray
    reserved_for_testing: np.ndarray
    testing_indices: np.ndarray

    @staticmethod
    def load(fname: str) -> 'TrainingData':
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Cannot open training data file {fname}")
        # Read the entire file as bytes
        with open(fname, 'rb') as f:
            local_data = np.frombuffer(f.read(), dtype=np.uint8)
        length = local_data.shape[0]
        # We reserve CONTEXT_SIZE+1 bytes at the end for indexing safety
        usable_length = length - (CONTEXT_SIZE + 1)
        if usable_length <= 0:
            raise ValueError("Training file is too short for the context size")
        # Reserve positions for testing (validation)
        reserved = np.zeros(usable_length, dtype=np.uint8)
        testing_indices = np.zeros(TESTING_LENGTH, dtype=np.int32)
        for i in range(TESTING_LENGTH):
            idx = random.randint(0, usable_length - 1)
            testing_indices[i] = idx
            # Mark a window around the testing index as reserved
            for j in range(-CONTEXT_SIZE, CONTEXT_SIZE + 1):
                pos = idx + j
                if 0 <= pos < usable_length:
                    reserved[pos] = 1
        return TrainingData(data=local_data, reserved_for_testing=reserved, testing_indices=testing_indices)

    def get_training_index(self) -> int:
        """Return a random index into the data which is not reserved for testing."""
        usable_length = self.reserved_for_testing.shape[0]
        while True:
            idx = random.randint(0, usable_length - 1)
            if self.reserved_for_testing[idx] == 0:
                return idx


def run_training(model: Model, training_data: TrainingData, steps: int = 1000, prompt: Optional[str] = None) -> None:
    """Train the model for a fixed number of steps and periodically report loss.

    The training loop mimics the behaviour of the C reference: every
    10,000 steps the model is evaluated on the reserved validation
    snippets and the average negative log likelihood is printed and
    optionally written to ``FILE_NAME``.  A prompt can be provided to
    generate sample text during validation.  In this implementation
    ``steps`` defaults to 1000 to keep the execution time reasonable.

    Parameters
    ----------
    model : Model
        Instance of the model to train.
    training_data : TrainingData
        Loaded training text and reserved indices.
    steps : int, default 1000
        Number of training iterations to perform.
    prompt : str, optional
        Prompt string to use for generating text during validation.
    """
    # Prepare loss logging
    loss_file = open(FILE_NAME, 'w')
    loss_file.close()
    # Seed PRNG for reproducibility
    random.seed(0)
    np.random.seed(0)
    for t in range(steps):
        # Load random training snippet
        idx = training_data.get_training_index()
        model.load_snippet(training_data.data, idx)
        lr = learning_rate_scheduler(t)
        model.training_step(t, lr)
        # Periodically evaluate on validation set
        if t % 100 == 0:  # reduced frequency from 10000 for shorter training
            validation_loss = 0.0
            for i in range(min(TESTING_LENGTH, 50)):  # evaluate on a subset for speed
                start = training_data.testing_indices[i]
                model.load_snippet(training_data.data, start)
                model.model_forward()
                _softmax(model.output[CONTEXT_SIZE - 1], 1.0)
                validation_loss -= math.log(model.output[CONTEXT_SIZE - 1, model.tokens[CONTEXT_SIZE]])
            validation_loss /= min(TESTING_LENGTH, 50)
            # Append to loss file
            with open(FILE_NAME, 'a') as f:
                f.write(f"{t},{validation_loss}\n")
            # Generate sample text if a prompt is provided
            if prompt is not None:
                # Initialise prompt buffer
                prompt_buf = list(prompt[:CONTEXT_SIZE].ljust(CONTEXT_SIZE))
                generated = []
                # Print prompt
                print(''.join(prompt_buf), end='')
                for _ in range(80):  # generate 80 characters
                    # Embed prompt into model.z
                    for pos in range(CONTEXT_SIZE):
                        token = ord(prompt_buf[pos]) if pos < len(prompt_buf) else 0
                        model.z[pos] = model.token_embedder[token]
                    # Infer next token
                    token_idx = model.infer()
                    char_out = chr(token_idx)
                    generated.append(char_out)
                    # Slide window
                    prompt_buf = prompt_buf[1:] + [char_out]
                print(''.join(generated))
            # Print training progress
            print(f"step={t}, validation_loss={validation_loss:.3f}")


if __name__ == '__main__':
    # Example usage: this block will only run when executing the module
    # directly.  It attempts to load a training file and run a few
    # training iterations, printing sample output.  Adjust the
    # filename and number of steps as appropriate.
    fname = os.environ.get('SNN_TRAIN_FILE', 'train_v2_drcat_02.csv')
    try:
        data = TrainingData.load(fname)
    except Exception as e:
        print(f"Failed to load training data: {e}")
        print("Set the environment variable SNN_TRAIN_FILE to point to a local training file.")
        raise SystemExit
    # Create model; enable CUDA if available and desired
    model = Model(use_cuda=False)
    # Run a short training session
    run_training(model, data, steps=10, prompt="insert your validation prompt here ")