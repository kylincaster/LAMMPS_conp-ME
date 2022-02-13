import pytest
from LAMMPS_IO import run
import numpy as np
import os

config_orig = {
    "NX": 6 + 5,
    "NY": 11 + 5,
    "tol": 1e-10,
    "multiRun_times": 3 + 3,
    "mat_name": '"a.txtmat save"',
    "scalapack_num": 100000000,
}


def test_binmat():
    run.set_script("matrix_IO")
    if os.path.exists("a.binmat"):
        os.remove("a.binmat")


class TestMatrix:
    @pytest.mark.parametrize("mat", ["a.txtmat", "a.binmat"])
    def test_save_IO(self, mat):
        core_num = int(run.config["CORE_NUM"])
        if (core_num) <= 3:
            core_num += 1
        else:
            core_num = core_num // 2

        config = config_orig.copy()

        config["mat_name"] = '"{} save"'.format(mat)
        lw = config["multiRun_times"] * 3 + 1
        result = run.run(config, lw=lw)

        config["mat_name"] = '"{} load"'.format(mat)
        result = run.run(config, core_num=core_num, lw=lw)
        assert len(result) == config["multiRun_times"] * 3 + 1
        os.remove(mat)

    @pytest.mark.parametrize("mat", ["a.txtmat", "a.binmat"])
    def test_none(self, mat):
        config = config_orig.copy()
        config["mat_name"] = '"{} none"'.format(mat)
        config["scalapack_num"] = 1
        lw = config["multiRun_times"] * 3 + 1
        result = run.run(config, lw=lw)
        # print(LAMMPS_IO.config)


if __name__ == "__main__":
    pass
