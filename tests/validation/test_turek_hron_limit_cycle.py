from __future__ import annotations
import numpy as np
import pytest
from src.refactored.validation.turek_hron_fsi.limit_cycle import LimitCycleValidationError, analyze_limit_cycle

def _signals(t):
    return (.3+.4*np.cos(2*np.pi*t), 2*np.sin(2*np.pi*(t-.25))+.5, 10+3*np.sin(2*np.pi*t), -1+4*np.cos(2*np.pi*t))

def test_extrema_differ_from_mean_rms_and_last_three_stability():
    t=np.arange(0.,5.1,.05); ux,uy,drag,lift=_signals(t); r=analyze_limit_cycle(t,ux,uy,drag,lift)
    assert r.primary_frequency_hz == pytest.approx(1.)
    assert r.cycles[-1].signals["point_a_uy_m"].midrange == pytest.approx(.5)
    assert r.cycles[-1].signals["point_a_uy_m"].amplitude == pytest.approx(2.)
    assert r.cycles[-1].signals["total_drag_n"].midrange == pytest.approx(10.)
    assert r.stability.period_spread_rel == pytest.approx(0.)
    assert r.stability.point_a_uy_amplitude_spread_rel == pytest.approx(0.)
    assert r.stability.point_a_uy_midrange_range_rel == pytest.approx(0.)
    assert r.stability.total_drag_amplitude_growth_rel == pytest.approx(0.)
    assert r.stability.total_lift_amplitude_growth_rel == pytest.approx(0.)
    assert r.stability.passed and all(r.stability.checks.values())

def test_rising_crossing_interpolates():
    t=np.arange(0.,5.,.2); uy=np.mod(t, 1.)-.5; z=np.zeros_like(t)
    assert analyze_limit_cycle(t,z,uy,z,z).crossing_times_s[:4] == pytest.approx((.5,1.5,2.5,3.5))

def test_zero_plateau_creates_one_event():
    t=np.arange(0.,1.7,.1); uy=np.array([-1.,0.,0.,1.]*4+[-1.]); z=np.zeros_like(t)
    assert analyze_limit_cycle(t,z,uy,z,z).crossing_times_s == pytest.approx((.1,.5,.9,1.3))

def test_partial_tail_is_ignored():
    t=np.arange(0.,5.7,.05); ux,uy,drag,lift=_signals(t); uy[t>5.3]+=100
    r=analyze_limit_cycle(t,ux,uy,drag,lift)
    assert len(r.cycles)==3 and r.cycles[-1].end_s < 5.3
    assert r.cycles[-1].signals["point_a_uy_m"].amplitude == pytest.approx(2.)

def test_fewer_than_four_crossings_fails():
    t=np.arange(0.,3.1,.05); ux,uy,drag,lift=_signals(t)
    with pytest.raises(LimitCycleValidationError, match="four rising"): analyze_limit_cycle(t,ux,uy,drag,lift)

def test_fft_is_deterministic_and_ties_choose_lower_frequency():
    t=np.arange(0.,5.05,.01); uy=np.sin(2*np.pi*(t-.25)); ux=np.sin(2*np.pi*t)+np.sin(4*np.pi*t); z=np.zeros_like(t)
    a=analyze_limit_cycle(t,ux,uy,z,z); b=analyze_limit_cycle(t,ux,uy,z,z)
    assert a.spectral_frequency_hz == b.spectral_frequency_hz
    assert a.spectral_frequency_hz["point_a_ux_m"] < 1.5
