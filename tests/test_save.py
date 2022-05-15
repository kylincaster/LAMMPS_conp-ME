import pytest
from LAMMPS_IO import run
import numpy as np
import os

config_save = {
    "NX": 6 + 5,
    "NY": 11 + 5,
    "tol": 9e-9,
    "multiRun_times": 3 + 3,
    "neutral": "on",
    "pair": "on",
    "newton": "on",
    "first": "on",
    "kspace": '"pppm_conp/ME 1e-4"',
    "selfGG": "on",
    "cg_style": "pcg",
    "tol_style": "rel_B",
    "pcg_mat": '"DC 0.079 1e-6"',
    "fixExtra": '""',
    "pair_extra": '""',
}

OnOFF = ["on", "off"]


def test_binmat():
    run.set_script("cg")
    if os.path.exists("a.binmat"):
        os.remove("a.binmat")


class TestSaveCG:
    @pytest.mark.parametrize("newton", OnOFF)
    @pytest.mark.parametrize("selfGG", OnOFF)
    def test_pppm_conp(self, newton, selfGG):
        config = config_save.copy()
        config["cg"] = "cg"
        config["newton"] = newton
        config["selfGG"] = selfGG
        config["kspace"] = '"pppm_conp/ME 1e-4 sv_SOL on"'
        config["pair_extra"] = '"sv_SOL on sv_ME on"'
        result = run.run(config, 13)
        assert np.all(result[0] == result[12])


class TestSavePCG:
    @pytest.mark.parametrize("newton", OnOFF)
    @pytest.mark.parametrize("selfGG", OnOFF)
    def test_pppm_conp(self, newton, selfGG):
        config = config_save.copy()
        config["cg"] = "pcg"
        config["newton"] = newton
        config["selfGG"] = selfGG
        config["kspace"] = '"pppm_conp/ME 1e-4 sv_SOL on"'
        config["pair_extra"] = '"sv_SOL on sv_ME on"'
        result = run.run(config, 13)
        assert np.all(result[0] == result[12])
