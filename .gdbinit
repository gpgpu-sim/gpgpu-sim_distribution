# Provides some useful debugging macros.  To use this file, copy to your home
# directory or to your simulation directory then run GPGPU-Sim in gdb.

printf "\n  ** loading GPGPU-Sim debugging macros... ** \n\n"

set print pretty
set print array-indexes
set unwindonsignal on

define dp
        call dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
end

document dp
Usage: dp <index>
Display pipeline state.
<index>: index of shader core you would like to see the pipeline state of

This function displays the state of the pipeline on a single shader core
(setting different values for the first argument of the call to
dump_pipeline_impl will cause different information to be displayed--
see the source code for more details)
end

define dpc
        call dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
        continue
end

document dpc
Usage: dpc <index>
Display pipeline state, then continue to next breakpoint.
<index>: index of shader core you would like to see the pipeline state of

This version is useful if you set a breakpoint where gpu_sim_cycle is
incremented in gpu_sim_loop() in src/gpgpu-sim/gpu-sim.c
repeatly hitting enter will advance to show the pipeline contents on
the next cycle.
end

define ptxdis
	set $addr=$arg0
	printf "disassemble instructions from 0x%x to 0x%x\n", $arg0, $arg1
	call fflush(stdout)
	while ( $addr <= $arg1 )
	      printf "0x%04x (%4u)  : ", $addr, $addr
	      call ptx_print_insn( $addr, stdout )
	      call fflush(stdout)
	      set $addr = $addr + 1
	end
end

document ptxdis
Usage: ptxdis <start> <end>
Disassemble PTX instructions between <start> and <end> (PCs).
end

define ptxdis_func
	set $ptx_tinfo = (ptx_thread_info*)sc[$arg0]->thread[$arg1].ptx_thd_info
	set $finfo = $ptx_tinfo->m_func_info
	set $minpc = $finfo->m_start_PC
	set $maxpc = $minpc + $finfo->m_instr_mem_size
	printf "disassembly of function %s (min pc = %u, max pc = %u):\n", $finfo->m_name.c_str(), $minpc, $maxpc
	ptxdis $minpc $maxpc $arg0 $arg1
end

document ptxdis_func
Usage: ptxdis_func <shd_idx> <tid>
<shd_idx>: shader core number
<tid>: thread ID
end

define ptx_tids2pcs
	set $i = 0
	while ( $i < $arg1 )
		set $tid =  $arg0[$i]
  		set $addr = ((ptx_thread_info*)sc[$arg2]->thread[$tid].ptx_thd_info)->m_PC
		printf "%2u : tid = %3u  => pc = %d\n", $i, $tid, $addr
		set $i = $i + 1
	end
end

document ptx_tids2pcs
Usage: ptx_tids2pcs <tids> <tidslen> <shd_idx>
<tids>: array of tids
<tidslen>: length of <tids> array
<shd_idx>: shader core number
end
