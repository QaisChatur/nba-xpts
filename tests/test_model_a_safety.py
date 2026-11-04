import os

import joblib
import numpy as np
import pytest

from pipeline.legacy import model_a
from pipeline.utils import DataValidationError


class _Exploit:
    def __reduce__(self):
        return (os.system, ("touch PWNED_BY_PICKLE",))


def test_restricted_loader_refuses_arbitrary_callables(tmp_path):
    p = tmp_path / "evil.pkl"
    joblib.dump({"x": _Exploit()}, p)
    marker = tmp_path / "PWNED_BY_PICKLE"
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(DataValidationError, match="non-allowlisted"):
            model_a.restricted_joblib_load(p)
    finally:
        os.chdir(cwd)
    assert not marker.exists()


def test_numpy_module_is_not_blanket_allowed():
    assert not model_a.is_allowed_global("numpy.testing", "runstring")
    assert not model_a.is_allowed_global("builtins", "eval")
    assert model_a.is_allowed_global("xgboost.sklearn", "XGBClassifier")


def test_plain_numpy_array_roundtrip(tmp_path):
    p = tmp_path / "arr.pkl"
    joblib.dump(np.arange(5.0), p)
    obj, seen = model_a.restricted_joblib_load(p)
    assert obj.tolist() == [0, 1, 2, 3, 4]


def test_model_a_hash_pin_and_schema(have_model_a):
    m = model_a.load_model_a()
    prov = m._xpts_provenance
    assert prov["sha256"].startswith(model_a.MODEL_A_SHA256)
    assert all(model_a.is_allowed_global(*g.rsplit(".", 1)) for g in prov["pickle_globals"])


def test_hash_mismatch_refuses(have_model_a):
    with pytest.raises(DataValidationError, match="sha256"):
        model_a.load_model_a(expected_sha256_prefix="0000000000000000")
