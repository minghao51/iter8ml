"""Property-based tests for PSI drift detection invariants."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from iter8ml.analysis.psi import classify_drift, compute_psi

pytestmark = pytest.mark.property


class TestPropertyPSI:
    """Property: compute_psi is non-negative, zero for identical distributions."""

    @given(
        ref=arrays(
            dtype=np.float64,
            shape=st.integers(10, 500),
            elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
        ),
        live=arrays(
            dtype=np.float64,
            shape=st.integers(10, 500),
            elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=100)
    def test_psi_non_negative(self, ref, live):
        psi = compute_psi(ref, live)
        assert psi >= 0.0

    @given(
        data=arrays(
            dtype=np.float64,
            shape=st.integers(10, 200),
            elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=50)
    def test_psi_self_zero(self, data):
        psi = compute_psi(data, data)
        assert psi < 1e-10

    @given(
        ref=arrays(
            dtype=np.float64,
            shape=st.integers(10, 100),
            elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=50)
    def test_psi_empty_ref_returns_zero(self, ref):
        empty = np.array([])
        psi = compute_psi(empty, ref)
        assert psi == 0.0

    @given(
        ref=arrays(
            dtype=np.float64,
            shape=st.integers(10, 100),
            elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=50)
    def test_psi_empty_live_returns_zero(self, ref):
        empty = np.array([])
        psi = compute_psi(ref, empty)
        assert psi == 0.0


class TestPropertyClassifyDrift:
    """Property: classify_drift returns one of three valid levels."""

    @given(psi=st.floats(-1.0, 5.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_classify_drift_valid_level(self, psi):
        level = classify_drift(psi)
        assert level in ("none", "moderate", "severe")

    @given(psi=st.floats(0.0, 0.19, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_low_psi_is_none(self, psi):
        assert classify_drift(psi) == "none"

    @given(psi=st.floats(0.21, 0.29, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_moderate_psi(self, psi):
        assert classify_drift(psi) == "moderate"

    @given(psi=st.floats(0.31, 5.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_severe_psi(self, psi):
        assert classify_drift(psi) == "severe"
