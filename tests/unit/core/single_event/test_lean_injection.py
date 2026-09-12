"""Host-resident, frequency-domain-only injections.

Long XG injections must not materialise time-domain arrays or keep full
native strain/PSD/grid copies on an accelerator. The lean path stores one
host frequency-domain array per channel, slices it by index (views, not
copies), and reproduces the legacy device injection bit for bit for the same
random key.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jimgw.core.single_event.data import Data, PowerSpectrum
from jimgw.core.single_event.detector import get_H1
from jimgw.core.single_event.waveform import RippleIMRPhenomD

FIXTURES_DIR = Path(__file__).parent.parent.parent.parent / "fixtures"
GPS_TIME = 1126259462.0
DURATION = 4.0
F_MIN, F_MAX = 20.0, 1024.0
SAMPLING_FREQUENCY = F_MAX * 2
PARAMS = {
    "M_c": 28.0,
    "eta": 0.24,
    "s1_x": 0.3,
    "s1_y": 0.2,
    "s1_z": 0.1,
    "s2_x": -0.1,
    "s2_y": 0.2,
    "s2_z": -0.3,
    "d_L": 440.0,
    "phase_c": 0.0,
    "iota": 0.0,
    "ra": 1.5,
    "dec": 0.5,
    "psi": 0.3,
    "t_c": 0.0,
}


def _inject(lean, zero_noise=False, seed=7):
    det = get_H1()
    det.set_psd(PowerSpectrum.from_file(str(FIXTURES_DIR / "GW150914_psd_H1.npz")))
    det.inject_signal(
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        trigger_time=GPS_TIME,
        waveform_model=RippleIMRPhenomD(f_ref=20.0),
        parameters=PARAMS,
        f_min=F_MIN,
        f_max=F_MAX,
        zero_noise=zero_noise,
        rng_key=jax.random.key(seed),
        waveform_chunk_size=1000,
        host_resident=lean,
    )
    return det


def test_host_fd_data_slices_like_eager_data_without_time_domain_arrays():
    delta_t = 1.0 / 64.0
    n_time = 256
    frequencies = np.fft.rfftfreq(n_time, delta_t)
    rng = np.random.default_rng(3)
    fd = rng.normal(size=frequencies.size) + 1j * rng.normal(size=frequencies.size)
    eager = Data.from_fd(jnp.asarray(fd), jnp.asarray(frequencies), start_time=10.0)
    lean = Data.from_host_fd(fd, delta_t=delta_t, start_time=10.0)

    assert lean.has_fd and lean.n_time == n_time and len(lean) == n_time
    assert lean.duration == pytest.approx(eager.duration)
    lean_fd, lean_f = lean.frequency_slice(5.0, 20.0)
    eager_fd, eager_f = eager.frequency_slice(5.0, 20.0)
    np.testing.assert_array_equal(np.asarray(lean_fd), np.asarray(eager_fd))
    np.testing.assert_array_equal(np.asarray(lean_f), np.asarray(eager_f))
    assert isinstance(lean_fd, np.ndarray) and np.shares_memory(lean_fd, lean.fd)
    assert lean.time_domain_materialised is False
    np.testing.assert_allclose(
        np.asarray(lean.td), np.asarray(eager.td), rtol=0, atol=1e-12
    )
    assert lean.time_domain_materialised is False  # on-demand TD is not retained


def test_lean_injection_reproduces_device_injection_bitwise():
    legacy = _inject(lean=False)
    lean = _inject(lean=True)
    np.testing.assert_array_equal(
        np.asarray(lean.sliced_frequencies), np.asarray(legacy.sliced_frequencies)
    )
    np.testing.assert_array_equal(
        np.asarray(lean.sliced_fd_data), np.asarray(legacy.sliced_fd_data)
    )
    np.testing.assert_array_equal(
        np.asarray(lean.sliced_psd), np.asarray(legacy.sliced_psd)
    )
    assert float(lean.optimal_snr) == pytest.approx(
        float(legacy.optimal_snr), rel=1e-12
    )
    legacy_mf = complex(legacy.match_filtered_snr)
    assert abs(complex(lean.match_filtered_snr) - legacy_mf) <= 1e-12 * abs(legacy_mf)


def test_lean_injection_keeps_host_views_and_no_time_domain_copy():
    lean = _inject(lean=True, zero_noise=True)
    assert isinstance(lean.data.fd, np.ndarray)
    assert lean.data.time_domain_materialised is False
    assert isinstance(lean.sliced_fd_data, np.ndarray)
    assert np.shares_memory(lean.sliced_fd_data, lean.data.fd)
    assert isinstance(lean.sliced_psd, np.ndarray)
    assert np.shares_memory(lean.sliced_psd, lean.psd.values)
    assert isinstance(lean.sliced_frequencies, np.ndarray)
    assert lean.sliced_frequencies[0] >= F_MIN and lean.sliced_frequencies[-1] <= F_MAX
