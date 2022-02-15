DIR=../src/Obj_intel_cpu_intelmpi
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
	./rename.tcl $(LMP_MAKE)

test:
	pytest


build:
	cd $(DIR); $(CPP) -std=c++11 -qopenmp -xHost -O2 -fp-model fast=2 -no-prec-div -qoverride-limits -qopt-zmm-usage=high -L/opt/intel/oneapi/mkl/2021.3.0/lib/intel64/ main.o     math_special.o pair_coul_cut.o image.o npair_skip_respa.o fix_ave_histo.o npair_half_multi_old_newton_tri.o pppm_slab.o bond_fene.o displace_atoms.o reader_xyz.o pair_deprecated.o angle_deprecated.o fix_dt_reset.o rerun.o atom.o fix_wall_harmonic.o ewald_dipole.o compute_chunk_atom.o math_extra.o pair_lj_charmm_coul_charmm.o pair_buck.o atom_vec_charge.o compute_deprecated.o compute_rdf.o npair_full_multi.o lattice.o fix_rigid_nve.o write_dump.o npair_full_nsq.o pair_buck_coul_long.o input.o npair_full_bin_atomonly.o compute_angle_local.o neigh_list.o compute_dipole.o kspace_deprecated.o pppm.o fix_nph_sphere.o fix_lineforce.o fix_wall_region.o min_fire.o create_box.o compute_com.o pair_lj_expand.o angle_harmonic.o fix_respa.o fix_accelerate_cos.o compute_vcm_chunk.o compute_gyration_chunk.o pair_hbond_dreiding_morse.o fix_bond_create.o atom_vec_line.o fix_wall.o rcb.o pair_zero.o pair_buck_coul_cut.o velocity.o angle_cosine_squared.o compute_pair_local.o pppm_conp_GA.o pppm_cg.o my_pool_chunk.o compute_viscosity_cos.o imbalance_var.o fix_minimize.o fix_wall_lj1043.o ntopo_improper_all.o change_box.o compute_ke_rigid.o region_prism.o min_quickmin.o pair_coul_streitz.o fix_spring_chunk.o npair_full_bin_ghost.o atom_vec_bond.o nstencil_full_multi_2d.o npair_half_size_multi_newton.o region_cylinder.o npair_half_size_nsq_newton.o npair_half_respa_nsq_newton.o pair_conp_GA.o atom_vec_sphere.o fix_nph.o run.o group.o dump_xyz.o nstencil_half_multi_3d.o atom_vec_ellipsoid.o fix_nvt.o pair_coul_wolf.o fix_heat.o npair_half_multi_old_newtoff.o fix_nve_limit.o fix_temp_berendsen.o fix_rigid_nph_small.o fix_rattle.o compute_temp_profile.o fix_null_print.o pair_born.o fix_shake_water.o fix_ave_chunk.o error.o fix_aveforce.o fix_ave_atom.o pppm_disp.o pair_born_coul_long.o npair_half_multi_newton_tri.o dump_image.o compute_improper.o fix_move.o fix_nve_sphere.o bond_harmonic.o remap.o nstencil_half_bin_2d.o fix_neigh_history.o fix_ehex.o pair_lj_charmmfsw_coul_charmmfsh.o fix_bond_break.o fix_imd.o pair_lj_cut_tip4p_cut.o dihedral_harmonic.o fix_rigid_small.o create_atoms.o ntopo.o gridcomm.o atom_map.o replicate.o ntopo_angle_partial.o fix_nve.o irregular.o fix_read_restart.o timer.o improper_hybrid.o npair_full_nsq_ghost.o npair_half_nsq_newtoff_ghost.o fix_store.o compute_msd.o atom_vec_molecular.o fix_recenter.o dihedral_charmm.o thermo.o compute_rigid_local.o improper_zero.o dihedral_charmmfsw.o pair_list.o fix_restrain.o nstencil_full_multi_old_2d.o fix_langevin.o compute_omega_chunk.o body.o npair_full_bin.o fix_planeforce.o fix_ave_correlate.o bond.o compute_coord_atom.o compute_slice.o ntopo_angle_template.o compute_reduce.o fix_rigid_npt_small.o npair_half_size_multi_old_newton.o dihedral.o region.o ntopo_dihedral_partial.o fft3d.o fix_deprecated.o region_union.o fix_temp_rescale.o neighbor.o fix_nvt_sphere.o pair_table.o fix_vector.o dump_deprecated.o read_dump.o write_restart.o dihedral_multi_harmonic.o compute_temp_deform.o compute_cna_atom.o write_coeff.o pppm_dipole.o min_cg.o procmap.o compute_stress_atom.o compute_cluster_atom.o fix_rigid_nvt.o npair_half_size_bin_newtoff.o npair_half_respa_bin_newtoff.o pppm_stagger.o imbalance_neigh.o compute_erotate_sphere.o fix_ave_histo_weight.o npair_skip.o fix_external.o fix_group.o atom_vec_angle.o fix_npt.o lmppython.o min_fire_old.o electro_control_GA.o pair_coul_long.o compute_property_atom.o text_file_reader.o compute.o pair_lj_charmm_coul_charmm_implicit.o atom_vec_full.o min_sd.o verlet.o kspace.o compute_temp_chunk.o npair_half_multi_old_newton.o npair_half_size_multi_old_newtoff.o nstencil_half_multi_2d.o arg_info.o compute_temp.o pair_born_coul_msm.o fix_rigid_nph.o fix_adapt.o compute_msd_chunk.o fix_bond_swap.o fmtlib_os.o table_file_reader.o pair_yukawa.o compute_temp_ramp.o imbalance_time.o pair_hybrid_scaled.o bond_hybrid.o improper_deprecated.o npair_half_multi_newton.o compute_temp_partial.o atom_vec_atomic.o npair_half_size_multi_newton_tri.o npair_half_nsq_newton.o dihedral_hybrid.o reset_atom_ids.o compute_reduce_region.o dump_atom.o compute_orientorder_atom.o pair_lj_cut_coul_long.o fix_rigid_nh_small.o bond_zero.o deprecated.o region_plane.o nstencil_half_multi_3d_tri.o npair_halffull_newton.o dump_cfg.o angle.o dihedral_zero.o nbin.o fix_halt.o fix_rigid.o fix_wall_lj126.o variable.o info.o atom_vec_tri.o bond_gromos.o fix_gravity.o fix_ipi.o msm.o fix_thermal_conductivity.o memory.o npair_skip_size_off2on_oneside.o fix_tfmc.o npair_half_bin_newtoff_ghost.o compute_temp_com.o fix_setforce.o nstencil_half_bin_2d_tri.o min.o compute_angmom_chunk.o fix_tune_kspace.o minimize.o fix_nh_sphere.o tabular_function.o fix_wall_lj93.o reader.o pair_coul_dsf.o fix_box_relax.o read_restart.o imbalance.o compute_centro_atom.o compute_dipole_chunk.o potential_file_reader.o fix_nh.o pair_morse.o fix_rigid_npt.o output.o comm.o npair_halffull_newtoff.o atom_vec_body.o pair_lj_cut_tip4p_long.o neigh_request.o fix_indent.o fix_momentum.o domain.o improper_harmonic.o read_data.o fix_enforce2d.o lammps.o fix_bond_create_angle.o nstencil_full_bin_3d.o utils.o dump_custom.o pair_lj_cut_coul_cut.o ewald_disp.o finish.o fix_evaporate.o compute_pe.o dihedral_opls.o imbalance_group.o nstencil.o bond_morse.o npair_skip_size.o npair_half_size_multi_old_newton_tri.o molecule.o npair_half_bin_newtoff.o fix_deform.o remap_wrap.o nstencil_half_multi_old_3d_tri.o ntopo_bond_partial.o angle_table.o nstencil_full_ghost_bin_2d.o comm_brick.o compute_global_atom.o compute_fragment_atom.o bond_special.o fix_rigid_nvt_small.o bond_fene_expand.o fix_srp.o ntopo_improper_partial.o region_cone.o improper.o ewald.o compute_com_chunk.o math_eigen.o hashlittle.o min_linesearch.o nstencil_half_multi_2d_tri.o nbin_multi.o fix_press_berendsen.o fix_efield.o fix_print.o atom_vec_template.o ntopo_bond_template.o ntopo_dihedral_template.o region_sphere.o compute_pressure.o dump_movie.o compute_reduce_chunk.o fix_nvt_sllod.o ntopo_improper_template.o pair_agni.o comm_tiled.o compute_property_local.o pair_srp.o npair_half_bin_atomonly_newton.o pair_lj_cut.o force.o reset_mol_ids.o fix_rigid_nve_small.o compute_ke.o pair_soft.o nstencil_full_multi_3d.o library.o fix_rigid_nh.o pair_lj_cut_point_long.o pair_tracker.o fix_spring_self.o pair_lj_charmm_coul_msm.o pair_hybrid.o angle_hybrid.o compute_angle.o compute_pe_atom_GA.o ewald_dipole_spin.o fix_dummy.o pair_lj_long_tip4p_long.o balance.o compute_pe_atom.o pair_zbl.o npair_skip_size_off2on.o min_hftn.o compute_temp_sphere.o pair_lj_cut_coul_slab.o compute_heat_flux.o compute_temp_region.o angle_zero.o fft3d_wrap.o atom_vec.o spmat_class.o pair_lj_long_coul_long.o compute_aggregate_atom.o integrate.o compute_gyration.o fix_cmap.o fix_charge_regulation.o angle_cosine.o nstencil_full_bin_2d.o npair_half_size_bin_newton.o npair_half_respa_bin_newton.o compute_pair.o fix_wall_morse.o nbin_standard.o msm_cg.o npair_half_multi_newtoff.o fix_widom.o pair_lj_cut_coul_msm.o npair_half_nsq_newtoff.o delete_atoms.o nstencil_half_bin_3d.o pair_dsmc.o pppm_tip4p.o compute_dihedral.o delete_bonds.o nstencil_half_multi_old_2d_tri.o fmtlib_format.o compute_bond_local.o compute_group_group.o pair_coul_msm.o dihedral_deprecated.o nstencil_full_ghost_bin_3d.o pair_tip4p_cut.o compute_vacf.o citeme.o npair_half_bin_newton_tri.o dump_local.o create_bonds.o my_page.o write_data.o compute_centroid_stress_atom.o fix_store_force.o region_block.o pair_tip4p_long.o universe.o bond_deprecated.o pppm_dipole_spin.o compute_property_chunk.o fix_conp_GA.o fix_atom_swap.o special.o pair_hybrid_overlay.o compute_ke_atom.o respa.o ntopo_bond_all.o fix_viscous.o improper_umbrella.o region_deprecated.o npair_copy.o nstencil_full_multi_old_3d.o fix_ave_time.o compute_chunk_spread_atom.o bond_quartic.o ntopo_angle_all.o npair_half_size_multi_newtoff.o imbalance_store.o fix_wall_reflect.o npair_half_size_nsq_newtoff.o npair_half_respa_nsq_newtoff.o reader_native.o fix_deposit.o pair_buck_coul_msm.o modify.o compute_dihedral_local.o pppm_la_conp_GA.o fix_spring.o compute_erotate_sphere_atom.o nstencil_half_multi_old_2d.o pppm_disp_tip4p.o nstencil_half_multi_old_3d.o fix_property_atom.o fix.o nstencil_half_bin_3d_tri.o random_mars.o compute_displace_atom.o npair.o npair_half_bin_newton.o dump.o improper_cvff.o npair_half_size_bin_newton_tri.o npair_half_respa_bin_newton_tri.o tokenizer.o update.o angle_charmm.o compute_improper_local.o debug_func.o ntopo_dihedral_all.o pair_buck_long_coul_long.o fix_shake.o bond_table.o fix_addforce.o region_intersect.o atom_vec_hybrid.o random_park.o set.o pair_hbond_dreiding_lj.o fix_npt_sphere.o fix_nve_noforce.o fix_balance.o pair_lj_charmm_coul_long.o compute_erotate_rigid.o dihedral_table.o fix_gcmc.o compute_bond.o pair_lj_charmmfsw_coul_long.o npair_full_multi_old.o compute_torque_chunk.o pair.o fix_pair_tracker.o pair_coul_debye.o compute_inertia_chunk.o fix_store_state.o  -lmkl_scalapack_ilp64 -lmkl_blacs_intelmpi_ilp64   -ltbbmalloc -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core    -o ../lmp_intel_cpu_intelmpi 
	size ../src/lmp_intel_cpu_intelmpi
	./rename.tcl $(LMP_MAKE)

build-all:debug_func compute print spmat pair spmat pair electro pppm pppm_la conp build

rsync:
	rsync -u ./src_mod/* ../src/

# Debug version
debug: rsync
	cd ../src; make $(LMP_MAKE) -j4 FFT_INC="$(CONP_MACRO)" FFT_LIB="$(SCALAPACK_LIB)"

# Release version
new: rsync
	cd ../src; make -j4 $(LMP_MAKE) FFT_INC="$(CONP_MACRO_RELEASE)" FFT_LIB="$(SCALAPACK_LIB)"

debug_func:
	$(CPP) $(CFLAG) -I../src/ -c $(SRC)/debug_func.cpp -o $(DIR)/debug_func.o

compute:
	$(CPP) $(CFLAG) -I../src/ -c $(SRC)/compute_pe_atom_GA.cpp -o $(DIR)/compute_pe_atom_GA.o

print:
	$(CPP) $(CFLAG) -I../src/ -c $(SRC)/fix_null_print.cpp -o $(DIR)/fix_null_print.o

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

atom:
	$(CPP) $(CFLAG) -I$(SRC) -I../src/ -c $(SRC)/atom.cpp -o $(DIR)/atom.o

clean-cache:
	find . | grep -E "\(__pycache__|log.lammps|\.pytest_cache|\.pyc|\.pyo$\)" | xargs rm -rf {} \;