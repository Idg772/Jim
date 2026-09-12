"""Unit tests for the CLI output writer."""

import json
from pathlib import Path

import numpy as np
import pytest

from jimgw.cli._config import PipelineConfig
from jimgw.cli._output import write_outputs

_RAW = {
    "data": {
        "type": "gwosc",
        "detectors": ["H1", "L1"],
        "trigger_time": 1126259462.4,
        "duration": 4.0,
        "psd_duration": 1024.0,
    },
    "waveform": {"approximant": "IMRPhenomXAS"},
    "prior": {
        "M_c": {"type": "uniform", "min": 10.0, "max": 80.0},
        "q": {"type": "uniform", "min": 0.125, "max": 1.0},
    },
    "likelihood": {"f_min": 20.0, "f_max": 1024.0},
    "sampler": {
        "type": "blackjax-swig",
        "blocks": [["M_c"], ["q"]],
        "n_live": 4,
        "n_delete_frac": 0.5,
    },
}


class _FakeJim:
    def __init__(self, diagnostics: dict):
        self._diagnostics = diagnostics

    def get_samples(self, n_samples: int = 0):
        return {
            "M_c": np.linspace(10.0, 80.0, 8),
            "q": np.linspace(0.2, 1.0, 8),
            "log_likelihood": np.zeros(8),
        }

    def get_diagnostics(self):
        return dict(self._diagnostics)


def test_output_retains_construction_and_complete_inference_timing(tmp_path):
    from types import SimpleNamespace

    jim = _FakeJim({"sample_phase_seconds": {"ns_loop": 2.0}})
    jim.likelihood = SimpleNamespace(
        construction_diagnostics={
            "summary_backend": "jax",
            "wall_seconds_including_compilation": 3.0,
            "detectors": {
                "H1": {"chunks": 2, "wall_seconds_including_compilation": 2.5}
            },
        }
    )
    jim.pipeline_timing = {
        "inference_including_construction_and_compilation_seconds": 7.0,
        "total_preparation_and_inference_seconds": 9.0,
    }
    out_dir = tmp_path / "timed"
    write_outputs(jim, _config(out_dir))
    report = json.loads((out_dir / "diagnostics.json").read_text())
    assert report["likelihood_construction"] == jim.likelihood.construction_diagnostics
    assert report["pipeline_timing"] == jim.pipeline_timing
    assert report["sample_phase_seconds"]["ns_loop"] == 2.0


def test_output_records_storage_without_materializing_lazy_data(tmp_path):
    from types import SimpleNamespace

    from jimgw.core.single_event.data import Data, PowerSpectrum
    from jimgw.core.single_event.native_storage import allocate_native_strain

    array, owner = allocate_native_strain(17, "mmap")
    data = Data.from_host_fd(array, 1 / 64)
    data._host_storage_owner = owner
    psd = PowerSpectrum(np.ones(17), data.frequencies)
    detector = SimpleNamespace(
        name="H1",
        data=data,
        psd=psd,
        data_preparation_diagnostics={
            "noise_generation": "indexed-v1",
            "host_storage": "mmap",
        },
    )
    jim = _FakeJim({})
    jim.likelihood = SimpleNamespace(detectors=[detector])
    out_dir = tmp_path / "storage"
    write_outputs(jim, _config(out_dir))
    report = json.loads((out_dir / "diagnostics.json").read_text())
    assert report["native_data_storage"]["mapped_file_logical_bytes"] == 17 * 16
    assert report["native_data_storage"]["host_allocation_bytes"] == 17 * 8 * 2
    assert report["data_preparation"]["H1"]["noise_generation"] == "indexed-v1"
    assert data.time_domain_materialised is False and data._window is None


def _config(out_dir: Path, overwrite: bool = False) -> PipelineConfig:
    raw = {
        **_RAW,
        "output": {"dir": str(out_dir), "overwrite": overwrite, "save_corner": False},
    }
    return PipelineConfig.model_validate(raw)


def test_write_outputs_serialises_nested_and_null_diagnostics(tmp_path):
    """SwiG reports dict-valued phase timings and ``None`` rates; neither may abort
    the output stage after a completed run."""
    diagnostics = {
        "log_Z": np.float64(-12.5),
        "log_Z_error": 0.1,
        "n_iterations": 42,
        "sample_phase_seconds": {"ns_loop": 3.5, "finalise": None},
        "de_jump_acceptance_rate": None,
        "acceptance_history": np.ones(3),
    }
    out_dir = tmp_path / "out"
    write_outputs(_FakeJim(diagnostics), _config(out_dir))

    written = json.loads((out_dir / "diagnostics.json").read_text())
    assert written["log_Z"] == pytest.approx(-12.5)
    assert written["n_iterations"] == 42
    assert written["sample_phase_seconds"] == {"ns_loop": 3.5, "finalise": None}
    assert written["de_jump_acceptance_rate"] is None
    with np.load(out_dir / "diagnostics.npz") as arrays:
        assert arrays["acceptance_history"].shape == (3,)


def test_write_outputs_accepts_own_checkpoint_artifacts_without_overwrite(tmp_path):
    """The CLI points the sampler checkpoint at ``output.dir``; the JAX cache and
    checkpoint the sampler writes there must not count as a pre-existing output."""
    out_dir = tmp_path / "out"
    (out_dir / "jax_cache").mkdir(parents=True)
    (out_dir / "jax_cache" / "compiled.bin").write_bytes(b"\x00")
    (out_dir / "checkpoint.pkl").write_bytes(b"\x00")

    write_outputs(_FakeJim({"log_Z": 1.0}), _config(out_dir, overwrite=False))

    assert (out_dir / "samples.npz").exists()
    assert (out_dir / "diagnostics.json").exists()


def test_write_outputs_still_refuses_foreign_content_without_overwrite(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "samples.npz").write_bytes(b"\x00")

    with pytest.raises(FileExistsError):
        write_outputs(_FakeJim({"log_Z": 1.0}), _config(out_dir, overwrite=False))


def test_write_outputs_saves_nested_points_when_available(tmp_path):
    """Nested backends expose the raw dead points with birth contours; saving
    them lets a failed run's population history be reconstructed offline."""

    class _NestedJim(_FakeJim):
        def get_weighted_samples(self, space="prior"):
            assert space == "prior"
            return {
                "M_c": np.linspace(10.0, 80.0, 5),
                "q": np.linspace(0.2, 1.0, 5),
                "log_likelihood": np.arange(5.0),
                "log_likelihood_birth": np.arange(5.0) - 1.0,
                "log_weights": np.full(5, -np.log(5.0)),
            }

    out_dir = tmp_path / "out"
    write_outputs(_NestedJim({"log_Z": 1.0}), _config(out_dir))

    with np.load(out_dir / "nested_samples.npz") as nested:
        assert set(nested.files) == {
            "M_c",
            "q",
            "log_likelihood",
            "log_likelihood_birth",
            "log_weights",
        }
        assert nested["log_likelihood_birth"].shape == (5,)


def test_write_outputs_tolerates_backends_without_nested_points(tmp_path):
    class _NoNested(_FakeJim):
        def get_weighted_samples(self, space="prior"):
            raise NotImplementedError("flowMC keeps no weighted collection")

    out_dir = tmp_path / "out"
    write_outputs(_NoNested({"log_Z": 1.0}), _config(out_dir))
    assert not (out_dir / "nested_samples.npz").exists()
    assert (out_dir / "samples.npz").exists()
