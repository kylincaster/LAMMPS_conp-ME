DIR=./src/Obj_intel_cpu_intelmpi
LMP_MAKE=intel_cpu_intelmpi
SCALAPACK_LIB=-lmkl_scalapack_ilp64 -lmkl_blacs_intelmpi_ilp64

SRC=./src_mod/
HEADER=../src/

PY=python
CPP= mpiicpc
CONP_MACRO=-DFFT_MKL -DFFT_DOUBLE -DMKL_ILP64 -DSCALAPACK -DINTEL_MPI
CONP_MACRO_RELEASE=$(CONP_MACRO) -DCONP_NO_DEBUG
CFLAG=-g -O1 -std=c++11 -qopenmp -qno-offload -fno-alias -ansi-alias -restrict -DLMP_INTEL_USELRT -DLMP_USE_MKL_RNG -DMKL_ILP64 -xHost -O2 -fp-model fast=2 -no-prec-div -qoverride-limits -qopt-zmm-usage=high  -DLAMMPS_GZIP  -DMPICH_SKIP_MPICXX -DOMPI_SKIP_MPICXX=1 $(CONP_MACRO)



all: new
	
test:
	pytest
	
build:debug_func compute print spmat pair spmat pair electro pppm pppm_la conp all


debug: rsync
	cd ../src; make $(LMP_MAKE) -j4 FFT_INC="$(CONP_MACRO)" FFT_LIB="$(SCALAPACK_LIB)"

new: rsync
	cd ../src; make -j4 $(LMP_MAKE) FFT_INC="$(CONP_MACRO_RELEASE)" FFT_LIB="$(SCALAPACK_LIB)"

debug_func:
	$(CPP) $(CFLAG) -I../src/ -c $(SRC)/debug_func.cpp -o $(DIR)/debug_func.o

compute:
	$(CPP) $(CFLAG) -I../src/ -c $(SRC)/compute_pe_atom_GA.cpp -o $(DIR)/compute_pe_atom_GA.o

print:
	$(CPP) $(CFLAG) -I../src/ -c $(SRC)/fix_null_print.cpp -o $(DIR)/fix_null_print.o

rsync:
	rsync -u ./src_mod/* ../src/

shake:
	$(CPP) $(CFLAG) -I../src/ -c $(SRC)/fix_shake_water.cpp -o $(DIR)/fix_shake_water.o

spmat:
	$(CPP) $(CFLAG) -I../src/ -c $(SRC)/spmat_class.cpp -o $(DIR)/spmat_class.o

pair: 
	$(CPP) $(CFLAG) -I../src/ -c $(SRC)/pair_conp_GA.cpp -o $(DIR)/pair_conp_GA.o
	$(CPP) $(CFLAG) -I../src/ -c $(SRC)/pair_lj_cut_point_long.cpp -o $(DIR)/pair_lj_cut_point_long.o
#	$(CPP) $(CFLAG) -I../src/ -c $(SRC)/pair_lj_cut_gauss_long.cpp -o $(DIR)/pair_lj_cut_gauss_long.o
#	$(CPP) $(CFLAG) -I../src/ -c $(SRC)/pair_lj_cut_tip4p_long_M.cpp -o $(DIR)/pair_lj_cut_tip4p_long_M.o

electro:
	$(CPP) $(CFLAG) -I../src/ -c $(SRC)/electro_control_GA.cpp -o $(DIR)/electro_control_GA.o

pppm0: 
	$(CPP) $(CFLAG) -I../src/ -c $(SRC)/pppm.cpp -o $(DIR)/pppm.o
	
pppm: 
	$(CPP) $(CFLAG) -I../src/ -c $(SRC)/pppm_conp_GA.cpp -o $(DIR)/pppm_conp_GA.o

pppm_la:
	$(CPP) $(CFLAG) -I../src/ -c $(SRC)/pppm_la_conp_GA.cpp -o $(DIR)/pppm_la_conp_GA.o


conp:
	$(CPP) $(CFLAG) -I$(SRC) -I../src/ -c $(SRC)/fix_conp_GA.cpp -o $(DIR)/fix_conp_GA.o
	cp $(SRC)/fix_conp_GA.h ../src/fix_conp_GA.h
	#echo '#include "fix_conp_GA.h"' >> ./src/style_fix.h

atom:
	$(CPP) $(CFLAG) -I$(SRC) -I../src/ -c $(SRC)/atom.cpp -o $(DIR)/atom.o

clean-cache:
	find . | grep -E "\(__pycache__|\.pytest_cache|\.pyc|\.pyo$\)" | xargs rm -rf