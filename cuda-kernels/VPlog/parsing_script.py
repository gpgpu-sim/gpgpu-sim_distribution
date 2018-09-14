import re
import time
from time import strftime
 
def main():
#    time_now = str(strftime("%Y-%m-%d %H-%M-%S", time.localtime()))
#    file = "\\" + "Parser Output " + time_now + ".txt"
 
    regexlist=[]
    regexlist.append('gpu_sim_cycle.*')
    regexlist.append('gpu_ipc.*')
    regexlist.append('gpu_tot_issued_cta.*')
    regexlist.append('L1I_total_cache_accesses.*' )
    regexlist.append('L1I_total_cache_misses.*')
    regexlist.append('L1C_total_cache_accesses.*')
    regexlist.append('L1C_total_cache_misses.*')
    regexlist.append('Total_core_cache_stats_breakdown\[CONST_ACC_R\]\[HIT\].*')
    regexlist.append('Total_core_cache_stats_breakdown\[CONST_ACC_R\]\[MISS\].*')
    regexlist.append('Total_core_cache_stats_breakdown\[INST_ACC_R\]\[HIT\].*')
    regexlist.append('Total_core_cache_stats_breakdown\[INST_ACC_R\]\[MISS\].*')
    regexlist.append('gpgpu_n_tot_thrd_icount.*')
    regexlist.append('gpgpu_n_tot_w_icount.*')
    regexlist.append('gpgpu_n_stall_shd_mem.*')
    regexlist.append('gpgpu_n_mem_read_global.*')
    regexlist.append('gpgpu_n_mem_write_global.*')
    regexlist.append('gpgpu_n_mem_const.*')
    regexlist.append('gpgpu_n_load_insn.*')
    regexlist.append('gpgpu_n_store_insn.*')
    regexlist.append('gpgpu_n_param_mem_insn.*')
    regexlist.append('traffic_breakdown_coretomem\[CONST_ACC_R\].*')
    regexlist.append('traffic_breakdown_coretomem\[GLOBAL_ACC_R\].*')
    regexlist.append('traffic_breakdown_coretomem\[GLOBAL_ACC_W\].*')
    regexlist.append('traffic_breakdown_coretomem\[INST_ACC_R\].*')
    regexlist.append('traffic_breakdown_memtocore\[CONST_ACC_R\].*')
    regexlist.append('traffic_breakdown_memtocore\[GLOBAL_ACC_R\].*')
    regexlist.append('traffic_breakdown_memtocore\[GLOBAL_ACC_W\].*')
    regexlist.append('traffic_breakdown_memtocore\[INST_ACC_R\].*')
    regexlist.append('L2_total_cache_accesses.*')
    regexlist.append('L2_total_cache_misses.*')
    regexlist.append('L2_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT\].*')
    regexlist.append('L2_cache_stats_breakdown\[GLOBAL_ACC_R\]\[MISS\].*')
    regexlist.append('L2_cache_stats_breakdown\[CONST_ACC_R\]\[HIT\].*')
    regexlist.append('L2_cache_stats_breakdown\[CONST_ACC_R\]\[MISS\].*')
    regexlist.append('L2_cache_stats_breakdown\[GLOBAL_ACC_W\]\[HIT\].*')
    regexlist.append('L2_cache_stats_breakdown\[GLOBAL_ACC_W\]\[MISS\].*')
    regexlist.append('L2_cache_stats_breakdown\[INST_ACC_R\]\[MISS\].*')
  
    #VP4
    log_file_path = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp4_16_summary"
    export_file = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp4_16_parsed"
    for regex in regexlist:  
    	parseData(log_file_path, export_file, regex, read_line=True)
    log_file_path = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp4_32_summary"
    export_file = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp4_32_parsed"
    for regex in regexlist:  
    	parseData(log_file_path, export_file, regex, read_line=True)
    log_file_path = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp4_64_summary"
    export_file = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp4_64_parsed"
    for regex in regexlist:  
    	parseData(log_file_path, export_file, regex, read_line=True)
    log_file_path = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp4_128_summary"
    export_file = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp4_128_parsed"
    for regex in regexlist:  
    	parseData(log_file_path, export_file, regex, read_line=True)
    log_file_path = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp4_256_summary"
    export_file = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp4_256_parsed"
    for regex in regexlist:  
    	parseData(log_file_path, export_file, regex, read_line=True)
   

    #VP8 
    log_file_path = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp8_16_summary"
    export_file = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp8_16_parsed"
    for regex in regexlist:  
    	parseData(log_file_path, export_file, regex, read_line=True)
    log_file_path = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp8_32_summary"
    export_file = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp8_32_parsed"
    for regex in regexlist:  
    	parseData(log_file_path, export_file, regex, read_line=True)
    log_file_path = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp8_64_summary"
    export_file = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp8_64_parsed"
    for regex in regexlist:  
    	parseData(log_file_path, export_file, regex, read_line=True)
    log_file_path = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp8_128_summary"
    export_file = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp8_128_parsed"
    for regex in regexlist:  
    	parseData(log_file_path, export_file, regex, read_line=True)
    log_file_path = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp8_256_summary"
    export_file = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp8_256_parsed"
    for regex in regexlist:  
    	parseData(log_file_path, export_file, regex, read_line=True)
   
    #VP16 
    log_file_path = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp16_16_summary"
    export_file = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp16_16_parsed"
    for regex in regexlist:  
    	parseData(log_file_path, export_file, regex, read_line=True)
    log_file_path = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp16_32_summary"
    export_file = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp16_32_parsed"
    for regex in regexlist:  
    	parseData(log_file_path, export_file, regex, read_line=True)
    log_file_path = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp16_64_summary"
    export_file = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp16_64_parsed"
    for regex in regexlist:  
    	parseData(log_file_path, export_file, regex, read_line=True)
    log_file_path = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp16_128_summary"
    export_file = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp16_128_parsed"
    for regex in regexlist:  
    	parseData(log_file_path, export_file, regex, read_line=True)
    log_file_path = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp16_256_summary"
    export_file = r"/home/araihan/negar-gpgpusim-tensorcore/cuda-kernels/VPlog/vp16_256_parsed"
    for regex in regexlist:  
    	parseData(log_file_path, export_file, regex, read_line=True)
 
def parseData(log_file_path, export_file, regex, read_line=True):
    with open(log_file_path, "r") as file:
        match_list = []
        if read_line == True:
            for line in file:
                for match in re.finditer(regex, line, re.S):
                    match_text = match.group()
		    match_text = re.sub('^.*=','',match_text.rstrip())
                    match_list.append(match_text+'\n')
                    print match_text
        else:
            data = file.read()
            for match in re.finditer(regex, data, re.S):
                match_text = match.group();
                match_list.append(match_text)
    file.close()
 
    with open(export_file, "a+") as file:
        match_list_clean = list(set(match_list))
        for item in xrange(0, len(match_list_clean)):
            print match_list_clean[item]
            file.write(match_list_clean[item] )#+ "\n")
    file.close()
 
if __name__ == '__main__':
    main() 
