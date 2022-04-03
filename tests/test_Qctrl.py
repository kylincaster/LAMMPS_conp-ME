import pytest
from LAMMPS_IO import run
import numpy as np
import os

config_save = {
    "NX": 6 + 5,
    "NY": 11 + 5,
    "tol": 5e-9,
    "multiRun_times": 3 + 3,
    "neutral": "on",
    "pair": "on",
    "newton": "on",
    "first": "on",
    "kspace": '"pppm 1e-4"',
    "selfGG": "on",
    "cg_style": "cg",
    "tol_style": "rel_B",
    "pcg_mat": '"DC 0.079 1e-6"',
    "fixExtra": '""',
    "pair_extra": '""',
}

OnOFF = ["on", "off"]


def test_binmat():
    run.set_script("Qctrl")
    if os.path.exists("a.binmat"):
        os.remove("a.binmat")


class Test_INV:
    def test_inv(self):
        config = config_save.copy()
        config["minimizer"] = "inv"
        result = run.run(config, 13)
        assert np.all(result[0] == result[12])


class Test_CG:
    @pytest.mark.parametrize("tol_style", ["max_Q", "rel_B"])
    def test_CG(self, tol_style):
        config = config_save.copy()
        config["minimizer"] = "cg"
        config["tol_style"] = tol_style
        result = run.run(config, 13)
        assert np.all(result[0] == result[12])

    @pytest.mark.parametrize("tol_style", ["max_Q", "rel_B"])
    def test_PCG(self, tol_style):
        config = config_save.copy()
        config["minimizer"] = "pcg"
        config["tol_style"] = tol_style
        result = run.run(config, 13)
        assert np.all(result[0] == result[12])


if __name__ == "__main__":
    pass
