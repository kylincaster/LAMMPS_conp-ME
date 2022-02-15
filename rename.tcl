#!/usr/bin/env tclsh

set res [ exec grep FIX_CONPGA_VERSION ./src_mod/fix_conp_GA.cpp | head -n 1 ]
set res [split $res]
set VERSION [lindex $res 2]
set LMP_MAKE [lindex $argv 0]

puts "rename as ./bin/lmp_${LMP_MAKE}_conp_DOUBLE_ILP64_v${VERSION}"

exec mv ../src/lmp_${LMP_MAKE} ../bin/lmp_${LMP_MAKE}_conp_DOUBLE_ILP64_v${VERSION}

