import pytest
from LAMMPS_IO import run
import numpy as np
import os
from math import sqrt, pi
from scipy.special import erfc, erf

run.set_script("pair")

config_orig = {
    "Kstyle": '"pppm 1e-4"',
    "Tablebits": 0,
    "Pstyle": '"lj/cut/coul/long 10 10"',
}
result = run.run(config_orig)
assert len(result) == 13
result0 = result[:, [2, 12]]
F_const = sqrt(2 / pi)
const_F = 2 / sqrt(pi)


def test_binmat():
    run.set_script("pair")
    if os.path.exists("a.binmat"):
        os.remove("a.binmat")


def pot(x, alpha):
    return erf(x * alpha) / x


def force(x, alpha):
    return (
        erfc(x * alpha) / (x * x) + const_F * alpha * np.exp(-x * x * alpha * alpha) / x
    )


def force_g(x, alpha):
    return (
        erf(x * alpha) / (x * x) - const_F * alpha * np.exp(-x * x * alpha * alpha) / x
    )


class TestPair:
    def test_ini_read(self):
        # print(LAMMPS_IO.config)
        assert int(run.config["CORE_NUM"]) > 0

    @pytest.mark.parametrize(
        "Kstyle", ['"pppm_la_conp/GA 1e-4"', '"pppm_la_conp/GA 1e-4"']
    )
    def test_pppm(self, Kstyle):
        config = config_orig.copy()
        config["Kstyle"] = Kstyle
        result = run.run(config, 13)
        data = result[:, [2, 12]]
        assert np.allclose(data, result0)

    def test_point(self):
        config = config_orig.copy()

        config["Pstyle"] = '"lj/cut/point/long 10 10 point 1 0 point 2 0"'
        result = run.run(config, 13)
        data = result[:, [2, 12]]
        assert np.allclose(data, result0)

        Eta_self = pi / 3
        config["Pstyle"] = '"lj/cut/point/long 10 10 point 1 {0} point 2 {0}"'.format(
            Eta_self
        )
        config["Tablebits"] = 12
        result = run.run(config, 13)
        data = result[:, [2, 12]]
        # U^LATT = \sqrt{2 / PI} \alpha_i
        # Energy = 1/2 * U^LATT * q_i^2
        data[:, 1] -= F_const * Eta_self
        assert np.allclose(data, result0)

    @pytest.mark.parametrize("bits", [0, 12, 16])
    def test_gauss(self, bits):
        config = config_orig.copy()
        config["Tablebits"] = bits
        Eta_self = pi / 5
        config["Pstyle"] = '"lj/cut/point/long 10 10 gauss 1 {0} gauss 2 {0}"'.format(
            Eta_self
        )

        result = run.run(config, lw=13)
        data = result[:, [2, 12]]

        pos = result[:, 1]
        alpha = Eta_self / sqrt(2)

        Fz = force_g(pos, alpha)
        Fz_error = force_g(pos, 1e8) - result0[:, 0]
        assert np.allclose(data[:, 0] + Fz_error, Fz)

        energy = pot(pos, alpha)
        # pot(pos, 1e10) - result0[:, 1]
        energy_error = 1 / pos + result0[:, 1] + F_const * Eta_self
        energy_error -= data[:, 1]
        assert np.allclose(energy_error, energy)

    @pytest.mark.parametrize("bits", [0, 12, 16])
    def test_mixture(self, bits):
        config = config_orig.copy()
        config["Tablebits"] = bits
        Eta_self = 10  # pi / 3
        config["Pstyle"] = '"lj/cut/point/long 10 10 gauss 1 {0} point 2 {0}"'.format(
            Eta_self
        )

        result = run.run(config)
        data = result[:, [2, 12]]
        assert len(result) == 13

        pos = result[:, 1]
        alpha = Eta_self / sqrt(2)

        Fz = force_g(pos, alpha)
        Fz_error = force_g(pos, 1e8) - result0[:, 0]
        assert np.allclose(data[:, 0] + Fz_error, Fz)

        energy = pot(pos, alpha)
        energy_error = pot(pos, 1e8) - result0[:, 1] - F_const * Eta_self
        assert np.allclose(data[:, 1] + energy_error, energy)

    def test_swith(self):
        # CHECK if alpha_ij > 1e30 switch to point style.
        config = config_orig.copy()
        Eta_ij = 2e30
        Eta_self = pi / 3
        config[
            "Pstyle"
        ] = '"lj/cut/point/long 10 10 eta 1 {0} {1} eta 2 {0} {1}"'.format(
            Eta_ij, Eta_self
        )
        result = run.run(config)
        data = result[:, [2, 12]]
        data[:, 1] -= F_const * Eta_self
        assert np.allclose(data, result0)

    @pytest.mark.parametrize("bits", [0, 12, 16])
    def test_eta(self, bits):
        config = config_orig.copy()
        config["Tablebits"] = bits
        Eta_ij = pi / 5  # pi / 3
        Eta_self = pi / 10  # pi / 3
        config[
            "Pstyle"
        ] = '"lj/cut/point/long 10 10 eta 1 {0} {1} eta 2 {0} {1}"'.format(
            Eta_ij, Eta_self
        )

        result = run.run(config, 13)
        data = result[:, [2, 12]]

        pos = result[:, 1]
        alpha = Eta_ij / sqrt(2)

        Fz = force_g(pos, alpha)
        Fz_error = force_g(pos, 1e8) - result0[:, 0]
        assert np.allclose(data[:, 0] + Fz_error, Fz)

        energy = pot(pos, alpha)
        # pot(pos, 1e10) - result0[:, 1]
        energy_error = 1 / pos + result0[:, 1] + F_const * Eta_self
        energy_error -= data[:, 1]
        assert np.allclose(energy_error, energy)
        # np.allclose(energy_error, energy)


if __name__ == "__main__":
    pass
