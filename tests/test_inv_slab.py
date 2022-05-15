import pytest
from LAMMPS_IO import run
import numpy as np
import os

run.set_script("matrix_IO")
config_save = {
    "NX": 6 + 5,
    "NY": 11 + 5,
    "tol": 1e-9,
    "scalapack_num": 100000000,
    "multiRun_times": 3 + 3,
    "mat_name": '"a.binmat save"',
    "neutral": "on",
    "pair": "on",
    "newton": "on",
    "first": "on",
    "kspace_style": '"pppm_conp/ME 1e-4"',
}
OnOFF = ["on", "off"]


def test_binmat():
    run.set_script("inv_slab")
    if os.path.exists("a.binmat"):
        os.remove("a.binmat")


class TestINVSlab:
    @pytest.mark.parametrize("newton", OnOFF)
    @pytest.mark.parametrize("first", OnOFF)
    def test_first(self, newton, first):
        config = config_save.copy()
        config["first"] = first
        config["newton"] = newton
        result = run.run(config, 13)
        assert np.all(result[0] == result[12])

    def test_pppm(self):
        config = config_save.copy()
        config["kspace_style"] = '"pppm 1e-4"'
        result = run.run(config, 13)
        assert np.all(result[0] == result[12])

    def test_pppm_la(self):
        config = config_save.copy()
        config["kspace_style"] = '"pppm_la_conp/ME 1e-4"'
        result = run.run(config, 13)
        assert np.all(result[0] == result[12])


if __name__ == "__main__":
    pass
