import pytest
from LAMMPS_IO import run
import numpy as np
import os

config_save = {
    "NX": 6 + 5,
    "NY": 11 + 5,
    "tol": 1e-9,
    "scalapack_num": 100000000,
    "multiRun_times": 3 + 3,
    "mat_name": '"a.binmat auto"',
    "neutral": "on",
    "pair": "on",
    "newton": "on",
    "first": "on",
    "kspace": '"pppm 1e-4"',
    "selfGG": "on",
    "pair_extra": '""',
    "Smat": "off",
}
OnOFF = ["on", "off"]

check_once = True


def test_binmat():
    run.set_script("inv")
    if os.path.exists("a.binmat"):
        os.remove("a.binmat")


# TODO fixed the bug on the initial begin
@pytest.mark.parametrize("pair", OnOFF)
@pytest.mark.parametrize("first", OnOFF)
@pytest.mark.parametrize("newton", OnOFF)
def test_Smat(first, pair, newton):
    config = config_save.copy()
    config["first"] = first
    config["newton"] = newton
    config["pair"] = pair
    config["Smat"] = "on"
    config["kspace"] = '"pppm_conp/GA 1e-4"'
    result = run.run(config, 16)
    # assert np.all(result[0] == result[15])


class TestINV:
    @pytest.mark.parametrize("pair", OnOFF)
    @pytest.mark.parametrize("neutral", OnOFF)
    @pytest.mark.parametrize("newton", OnOFF)
    def test_inv(self, neutral, pair, newton):
        config = config_save.copy()
        config["neutral"] = neutral
        config["pair"] = pair
        if pair == "off":
            config["kspace"] = '"pppm_conp/GA 1e-4"'
        config["newton"] = newton
        result = run.run(config, 13)
        assert np.all(result[0] == result[12])

    @pytest.mark.parametrize("pair", OnOFF)
    @pytest.mark.parametrize("first", OnOFF)
    @pytest.mark.parametrize("newton", OnOFF)
    def _test_Smat(self, first, pair, newton):
        config = config_save.copy()
        config["first"] = first
        config["newton"] = newton
        config["pair"] = pair
        config["Smat"] = "on"
        result = run.run(config, 13)
        assert np.all(result[0] == result[12])

    @pytest.mark.parametrize("newton", OnOFF)
    @pytest.mark.parametrize("first", OnOFF)
    def test_first(self, newton, first):
        config = config_save.copy()
        config["first"] = first
        config["newton"] = newton
        result = run.run(config, 13)
        assert np.all(result[0] == result[12])

    @pytest.mark.parametrize("newton", OnOFF)
    @pytest.mark.parametrize("selfGG", OnOFF)
    def test_pppm_conp(self, newton, selfGG):
        config = config_save.copy()
        config["newton"] = newton
        config["selfGG"] = selfGG
        config["kspace"] = '"pppm_conp/GA 1e-4 sv_SOL off"'
        result = run.run(config, 13)
        assert np.all(result[0] == result[12])

        # config["kspace"] = '"pppm_conp/GA 1e-4 sv_SOL on"'
        # result = run.run(config, 13)
        # assert np.all(result[0] == result[12])

    @pytest.mark.parametrize("newton", OnOFF)
    @pytest.mark.parametrize("selfGG", OnOFF)
    def test_pppm_conp(self, newton, selfGG):
        config = config_save.copy()
        config["newton"] = newton
        config["selfGG"] = selfGG
        config["kspace"] = '"pppm_conp/GA 1e-4 sv_SOL on"'
        # config["pair_extra"] = '"sv_SOL on sv_GA on"'
        result = run.run(config, 13)
        assert np.all(result[0] == result[12])

    def _test(self):
        config = config_save.copy()
        config["selfGG"] = "on"
        config["kspace"] = '"pppm_conp/GA 1e-4 sv_SOL on"'
        result = run.run(config, 13)
        assert np.all(result[0] == result[12])

    @pytest.mark.parametrize("newton", OnOFF)
    @pytest.mark.parametrize("selfGG", OnOFF)
    def test_pppm_la_conp(self, newton, selfGG):
        config = config_save.copy()
        config["mat_name"] = '"a.binmat none"'
        config["newton"] = newton
        config["selfGG"] = selfGG
        config["kspace"] = '"pppm_la_conp/GA 1e-4 sv_SOL off sv_Xuv on"'
        config["pair_extra"] = '"sv_SOL on sv_GA on"'
        result = run.run(config, 13)
        assert np.all(result[0] == result[12])


if __name__ == "__main__":
    pass
