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
    run.set_script("restart")
    if os.path.exists("a.binmat"):
        os.remove("a.binmat")


class TestCG:
    def test_cg(self):
        config = config_save.copy()
        result = run.run(config, 13)
        assert np.all(result[0] == result[12])


# ToDo: the pe style is removed due to lower rate of covergent
class _TestPCG:
    @pytest.mark.parametrize("newton", ["on", "off"])
    @pytest.mark.parametrize("selfGG", ["off", "on"])
    @pytest.mark.parametrize(
        "tol_style", ["res", "max_Q", "rel_B", "std_Q", "rel_Q", "abs_Q"]
    )
    def test_pcg(self, newton, selfGG, tol_style):
        config = config_save.copy()
        if tol_style == "pe":
            config["tol"] = 1e-4
        config["cg_style"] = "pcg"
        config["selfGG"] = selfGG
        config["tol_style"] = tol_style
        config["newton"] = newton
        config["kspace"] = '"pppm_conp/ME 1e-4"'
        result = run.run(config, 13)
        assert np.all(result[0] == result[12])


if __name__ == "__main__":
    pass
