import configparser
import os
import numpy as np


class lammps_run:
    def __init__(self, ini_file):
        config = configparser.ConfigParser()
        config.optionxform = lambda option: option  # preserve case for letters
        config.read(ini_file)
        ini = my_config_parser_dict = {
            s: dict(config.items(s)) for s in config.sections()
        }
        print(ini)
        self.config = ini["default"]
        # print(self.config)

    def set_script(self, script):
        self.script = "./tests/samples/{}.txt".format(script)

    def run(
        self,
        config,
        lw=None,
        core_num=None,
    ):
        if core_num is None:
            CMD = "{MPI} -np {CORE_NUM} {LAMMPS} -in ".format(**self.config)
        else:
            mpi_config = self.config.copy()
            mpi_config["CORE_NUM"] = core_num
            CMD = "{MPI} -np {CORE_NUM} {LAMMPS} -in ".format(**mpi_config)

        CMD += self.script + " -log none "
        CMD_extra = ["-v {} {} ".format(k, v) for k, v in config.items()]
        CMD += "".join(CMD_extra)
        # CMD += '| tee screen.txt | grep -E "^0 " | uniq > log.txt '
        CMD += '| grep -E "^0 " | uniq > log.txt '
        os.system(CMD)
        print(CMD)
        data = np.loadtxt("log.txt")
        print(data, len(data))
        if lw is not None:
            assert len(data) == lw
        os.remove("log.txt")
        return data


run = lammps_run("./tests/config.ini")

if __name__ == "__main__":
    pass
