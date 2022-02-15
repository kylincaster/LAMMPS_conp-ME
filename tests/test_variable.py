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
    "cg_style": "pcg",
    "tol_style": "rel_B",
    "pcg_mat": '"DC 0.079 1e-6"',
    "fixExtra": '""',
    "pair_extra": '""',
    "U_right": '"v_U_cap"',
    "Vext_update": "off",
}

OnOFF = ["on", "off"]


def test_binmat():
    run.set_script("cg")
    if os.path.exists("a.binmat"):
        os.remove("a.binmat")


class TestVAR:
    @pytest.mark.parametrize("U_right", ["v_U_cap", "v_U_one"])
    @pytest.mark.parametrize("Vext_update", ["off", "on"])
    def test_variable(self, U_right, Vext_update):
        config = config_save.copy()
        config["U_right"] = U_right
        config["Vext_update"] = Vext_update
        result = run.run(config, 13)
        assert np.all(result[0] == result[12])


if __name__ == "__main__":
    pass
