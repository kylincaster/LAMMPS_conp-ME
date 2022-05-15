import pytest
from LAMMPS_IO import run
import numpy as np
import os

config_save = {
    "NX": 6 + 5,
    "NY": 11 + 5,
    "tol": 1e-9,
    "multiRun_times": 3 + 3,
    "neutral": "on",
    "pair": "on",
    "newton": "on",
    "first": "on",
    "kspace": '"pppm_conp/ME 1e-4"',
    "selfGG": "on",
    "cg_style": "pcg",
    "tol_style": "rel_B",
    "pcg_mat": "full",
    "fixExtra": '""',
    "pair_extra": '""',
}

OnOFF = ["on", "off"]


def test_binmat():
    run.set_script("cg")
    if os.path.exists("a.binmat"):
        os.remove("a.binmat")


class TestPCG:
    @pytest.mark.parametrize("newton", ["on", "off"])
    @pytest.mark.parametrize("pcg_mat", ["full", "normal"])
    @pytest.mark.parametrize("selfGG", ["on", "off"])
    def test_pcg(self, newton, pcg_mat, selfGG):
        config = config_save.copy()
        config["pcg_mat"] = pcg_mat
        config["newton"] = newton
        config["selfGG"] = selfGG
        result = run.run(config, 13)
        assert np.all(result[0] == result[12])


# TOdo: something wrong with newton = on and selfGG = off
class TestPPCG:
    @pytest.mark.parametrize("newton", ["on", "off"])
    @pytest.mark.parametrize("selfGG", ["off", "on"])
    @pytest.mark.parametrize("tol_style", ["res", "max_Q", "rel_B"])
    def test_ppcg(self, newton, selfGG, tol_style):
        config = config_save.copy()
        config["cg_style"] = "ppcg"
        config["tol_style"] = "res"
        config["newton"] = newton
        config["selfGG"] = selfGG
        config["tol_style"] = tol_style
        config["fixExtra"] = '"ppcg_block 4 ppcg_cut 1e-6 pshift_scale 1.0"'
        result = run.run(config, 13)
        assert np.all(result[0] == result[12])

    def _test(self):
        config = config_save.copy()
        config["cg_style"] = "ppcg"
        config["tol_style"] = "rel_B"
        config["selfGG"] = "off"
        config["fixExtra"] = '"ppcg_block 4 ppcg_cut 1e-6 pshift_scale 1.0"'
        result = run.run(config, 13)


# TO DO fix rel_B for ppcg


if __name__ == "__main__":
    pass
