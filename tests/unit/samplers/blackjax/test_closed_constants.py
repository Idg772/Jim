"""Four-device placement of closed arrays without embedding them into HLO."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


def _environment():
    return {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_NUM_CPU_DEVICES": "4",
        "JAX_ENABLE_X64": "true",
        "JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS": "True",
        "JAX_EMBEDDED_CONSTANTS_MAX_BYTES": "32",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


@pytest.mark.parametrize("simplified", [True, False])
def test_nested_arrays_and_literal_outputs_keep_values_and_runtime_buffers(simplified):
    source = r"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jimgw.samplers.blackjax.closed_constants import with_replicated_constants
mesh = Mesh(np.asarray(jax.devices()), ('replacement',))
sharding = NamedSharding(mesh, P('replacement'))
c = jax.device_put(jnp.arange(16., dtype=jnp.float64), jax.devices()[0])
scalar = jax.device_put(jnp.asarray(2., dtype=jnp.float64), jax.devices()[0])
@jax.jit
def inner(x):
    return jax.lax.cond(x[0] > 0, lambda x: x * c, lambda x: x + c, x)
def one(x):
    _, values = jax.lax.scan(lambda state, _: (state + 1, inner(x)), 0, None, length=2)
    _, result = jax.lax.while_loop(lambda t: t[0] < 2,
        lambda t: (t[0] + 1, t[1] + jnp.sum(values)), (0, 0.))
    return result * scalar
def batch(x):
    return {'scores': jax.lax.map(one, x, batch_size=8), 'constant': c}
x = jax.device_put(jnp.ones((32, 16)), sharding)
if jax.config.jax_use_simplified_jaxpr_constants:
    try:
        jax.jit(batch).lower(x).compile()
    except ValueError as error:
        assert 'incompatible devices' in str(error)
    else:
        raise AssertionError('Missing original-device-conflict reproduction')
converted = with_replicated_constants(batch, x, mesh=mesh)
compiled = jax.jit(converted).lower(x).compile()
actual = compiled(x)
np.testing.assert_array_equal(actual['scores'], np.full(32, 960.))
np.testing.assert_array_equal(actual['constant'], np.arange(16.))
assert c.devices() == {jax.devices()[0]} and c.committed
assert converted.constant_placement['unique_array_buffers'] >= 1
assert converted.constant_placement['logical_bytes_per_replica'] >= 128
if jax.config.jax_use_simplified_jaxpr_constants:
    text = compiled.as_text()
    assert any('f64[16]' in line and 'parameter(' in line for line in text.splitlines()), text
    assert not any('f64[16]' in line and 'constant({' in line for line in text.splitlines()), text
print('nested constant placement passed', converted.constant_placement)
"""
    env = _environment()
    env["JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS"] = str(simplified)
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
