"""Place only traced callback constants on a multi-device sampler mesh.

With simplified JAXPR constants, committed array-valued literals retain their
original device assignment. A likelihood built on one device therefore needs
its captured constants replicated before compiling a sampler over a mesh.
Native detector attributes that the callback never reads are not traversed.
"""

import logging
from functools import wraps

import jax
import numpy as np
from jax.extend import core

from jimgw.samplers.blackjax.sharding import replicated_sharding

logger = logging.getLogger(__name__)


def with_replicated_constants(function, *example_args, mesh):
    """Trace a pure sampler boundary and replicate its captured array buffers.

    The returned callable preserves the input/output pytrees and every JAXPR
    operation. Only constant placement changes. Array-valued literals inside
    scan/while/cond/jit/shard_map bodies and constant-return outputs are covered,
    alongside traditional constant arguments. Shared buffers are copied once.
    Like an AOT executable, this callable specializes to the example shapes
    and dtypes; each compilation boundary owns its replicated constant set.
    """
    closed, output_shape = jax.make_jaxpr(function, return_shape=True)(*example_args)
    destination = replicated_sharding(mesh)
    buffers = {}
    jaxprs = {}
    copied_bytes = 0

    def place(value):
        nonlocal copied_bytes
        # JAX hoists non-scalar array literals; scalar literals remain embedded
        # and do not impose a committed constant-argument device assignment.
        if not isinstance(value, (jax.Array, np.ndarray)) or value.ndim == 0:
            return value
        identity = id(value)
        if identity not in buffers:
            buffers[identity] = jax.device_put(value, destination)
            copied_bytes += value.nbytes
        return buffers[identity]

    def atom(value):
        if isinstance(value, core.Literal):
            replacement = place(value.val)
            if replacement is not value.val:
                return core.Literal(replacement, value.aval)
        return value

    def rewrite(value):
        # Since JAX 0.11 ClosedJaxpr is an alias of Jaxpr and .jaxpr is self.
        # Handle Jaxpr first to avoid recursing through that legacy accessor.
        if isinstance(value, core.Jaxpr):
            identity = id(value)
            if identity not in jaxprs:
                updates = {
                    "eqns": [
                        equation.replace(
                            invars=[atom(v) for v in equation.invars],
                            params=rewrite(equation.params),
                        )
                        for equation in value.eqns
                    ],
                    "outvars": [atom(v) for v in value.outvars],
                }
                if hasattr(value, "consts"):
                    updates["consts"] = [place(v) for v in value.consts]
                jaxprs[identity] = value.replace(**updates)
            return jaxprs[identity]
        if isinstance(value, core.ClosedJaxpr):
            return core.ClosedJaxpr(
                rewrite(value.jaxpr), [place(v) for v in value.consts]
            )
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if type(value) is tuple:
            return tuple(rewrite(item) for item in value)
        return value

    placed = rewrite(closed)
    flat_function = core.jaxpr_as_fun(placed)
    input_tree = jax.tree.structure(example_args)
    output_tree = jax.tree.structure(output_shape)

    @wraps(function)
    def converted(*args):
        leaves, tree = jax.tree.flatten(args)
        if tree != input_tree:
            raise TypeError("Replicated sampler callback input pytree changed")
        return jax.tree.unflatten(output_tree, flat_function(*leaves))

    converted.constant_placement = {
        "unique_array_buffers": len(buffers),
        "logical_bytes_per_replica": copied_bytes,
        "devices": len(mesh.devices.flat),
    }
    logger.info(
        "Placed %d traced sampler constants (%d bytes per replica) on %d devices",
        len(buffers),
        copied_bytes,
        len(mesh.devices.flat),
    )
    return converted
