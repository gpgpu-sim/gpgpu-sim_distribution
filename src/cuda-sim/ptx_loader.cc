/* 
 * Copyright (c) 2009 by Tor M. Aamodt, Wilson W. L. Fung, Ali Bakhoda, 
 * George L. Yuan, Dan O'Connor, Joey Ting, Henry Wong and the 
 * University of British Columbia
 * Vancouver, BC  V6T 1Z4
 * All Rights Reserved.
 * 
 * THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
 * TERMS AND CONDITIONS.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
 * are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
 * (property of NVIDIA).  The files benchmarks/BlackScholes/ and 
 * benchmarks/template/ are derived from the CUDA SDK available from 
 * http://www.nvidia.com/cuda (also property of NVIDIA).  The files from 
 * src/intersim/ are derived from Booksim (a simulator provided with the 
 * textbook "Principles and Practices of Interconnection Networks" available 
 * from http://cva.stanford.edu/books/ppin/). As such, those files are bound by 
 * the corresponding legal terms and conditions set forth separately (original 
 * copyright notices are left in files from these sources and where we have 
 * modified a file our copyright notice appears before the original copyright 
 * notice).  
 * 
 * Using this version of GPGPU-Sim requires a complete installation of CUDA 
 * which is distributed seperately by NVIDIA under separate terms and 
 * conditions.  To use this version of GPGPU-Sim with OpenCL requires a
 * recent version of NVIDIA's drivers which support OpenCL.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the University of British Columbia nor the names of
 * its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * 4. This version of GPGPU-SIM is distributed freely for non-commercial use only.  
 *  
 * 5. No nonprofit user may place any restrictions on the use of this software,
 * including as modified by the user, by any other authorized user.
 * 
 * 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung, 
 * Ali Bakhoda, George L. Yuan, at the University of British Columbia, 
 * Vancouver, BC V6T 1Z4
 */

#include "ptx_loader.h"
#include "ptx_ir.h"
#include "cuda-sim.h"
#include "ptx_parser.h"
#include <dirent.h>

/// globals

memory_space *g_global_mem;
memory_space *g_tex_mem;
memory_space *g_surf_mem;
memory_space *g_param_mem;
bool g_override_embedded_ptx = false;

struct ptx_info_t {
    char *str;
    char *fname;
    ptx_info_t *next;
};

/// extern prototypes

extern "C" int ptx_parse();
extern "C" int ptx__scan_string(const char*);
extern "C" FILE *ptx_in;

extern "C" const char *g_ptxinfo_filename = NULL;
extern "C" int ptxinfo_parse();
extern "C" int ptxinfo_debug;
extern "C" FILE *ptxinfo_in;

/// static functions

static int load_static_globals( symbol_table *symtab, unsigned min_gaddr, unsigned max_gaddr) 
{
   printf( "GPGPU-Sim PTX: loading globals with explicit initializers... \n" );
   fflush(stdout);
   int ng_bytes=0;
   symbol_table::iterator g=symtab->global_iterator_begin();

   for ( ; g!=symtab->global_iterator_end(); g++) {
      symbol *global = *g;
      if ( global->has_initializer() ) {
         printf( "GPGPU-Sim PTX:     initializing '%s' ... ", global->name().c_str() ); 
         unsigned addr=global->get_address();
         const type_info *type = global->type();
         type_info_key ti=type->get_key();
         size_t size;
         int t;
         ti.type_decode(size,t);
         int nbytes = size/8;
         int offset=0;
         std::list<operand_info> init_list = global->get_initializer();
         for ( std::list<operand_info>::iterator i=init_list.begin(); i!=init_list.end(); i++ ) {
            operand_info op = *i;
            ptx_reg_t value = op.get_literal_value();
            assert( (addr+offset+nbytes) < min_gaddr ); // min_gaddr is start of "heap" for cudaMalloc
            g_global_mem->write(addr+offset,nbytes,&value,NULL,NULL); // assuming little endian here
            offset+=nbytes;
            ng_bytes+=nbytes;
         }
         printf(" wrote %u bytes\n", offset ); 
      }
   }
   printf( "GPGPU-Sim PTX: finished loading globals (%u bytes total).\n", ng_bytes );
   fflush(stdout);
   return ng_bytes;
}

static int load_constants( symbol_table *symtab, addr_t min_gaddr ) 
{
   printf( "GPGPU-Sim PTX: loading constants with explicit initializers... " );
   fflush(stdout);
   int nc_bytes = 0;
   symbol_table::iterator g=symtab->const_iterator_begin();

   for ( ; g!=symtab->const_iterator_end(); g++) {
      symbol *constant = *g;
      if ( constant->is_const() && constant->has_initializer() ) {

         // get the constant element data size
         int basic_type;
         size_t num_bits;
         constant->type()->get_key().type_decode(num_bits,basic_type); 

         std::list<operand_info> init_list = constant->get_initializer();
         int nbytes_written = 0;
         for ( std::list<operand_info>::iterator i=init_list.begin(); i!=init_list.end(); i++ ) {
            operand_info op = *i;
            ptx_reg_t value = op.get_literal_value();
            int nbytes = num_bits/8;
            switch ( op.get_type() ) {
            case int_t: assert(nbytes >= 1); break;
            case float_op_t: assert(nbytes == 4); break;
            case double_op_t: assert(nbytes >= 4); break; // account for double DEMOTING
            default:
               abort();
            }
            unsigned addr=constant->get_address() + nbytes_written;
            assert( addr+nbytes < min_gaddr );

            g_global_mem->write(addr,nbytes,&value,NULL,NULL); // assume little endian (so u8 is the first byte in u32)
            nc_bytes+=nbytes;
            nbytes_written += nbytes;
         }
      }
   }
   printf( " done.\n");
   fflush(stdout);
   return nc_bytes;
}

static FILE *open_ptxinfo (const char* ptx_filename)
{
   const int ptx_fnamelen = strlen(ptx_filename);
   char *ptxi_fname = new char[ptx_fnamelen+5];
   strcpy (ptxi_fname, ptx_filename);
   strcpy (ptxi_fname+ptx_fnamelen, "info");

   //ptxinfo_debug=1;
   g_ptxinfo_filename = ptxi_fname;
   FILE *f = fopen (ptxi_fname, "rt");
   return f;
}

static int ptx_file_filter(
#if !defined(__APPLE__)
   const
#endif
   struct dirent *de )
{
   const char *tmp = strstr(de->d_name,".ptx");
   if ( tmp != NULL && tmp[4] == 0 ) {
      return 1;
   }
   return 0;
}

// global functions

static ptx_info_t *g_ptx_source_array = NULL;

void gpgpu_ptx_sim_load_gpu_kernels()
{
    static unsigned source_num = 0;
    ptx_in = NULL;
    if ( g_filename )
        ptx_in = fopen( g_filename, "r" );
    gpgpu_ptx_sim_init_memory();
    if (ptx_in) {
        symbol_table *symtab=init_parser(g_filename);
        ptx_parse();
        ptxinfo_in = open_ptxinfo(g_filename);
        ptxinfo_parse();
        load_static_globals(symtab,STATIC_ALLOC_LIMIT,0xFFFFFFFF);
        load_constants(symtab,STATIC_ALLOC_LIMIT);
    } else {
        if (!g_override_embedded_ptx) {
            printf("GPGPU-Sim PTX: USING EMBEDDED .ptx files...\n"); 
            ptx_info_t *s;
            for ( s=g_ptx_source_array; s!=NULL; s=s->next ) {
                 symbol_table *symtab=gpgpu_ptx_sim_load_ptx_from_string(s->str, ++source_num);
                 load_static_globals(symtab,STATIC_ALLOC_LIMIT,0xFFFFFFFF);
                 load_constants(symtab,STATIC_ALLOC_LIMIT);
            }
        } else {
            const char *filename = NULL;
            struct dirent **namelist;
            int n = scandir(".", &namelist, ptx_file_filter, alphasort);
            if (n < 0)
                perror("GPGPU-Sim PTX: no PTX files returned by scandir");
            else {
                while (n--) {
                    if ( filename != NULL ) {
                        printf("Loader error: support for multiple .ptx files not yet enabled\n");
                        abort();
                    }
                    filename = namelist[n]->d_name;
                    printf("Parsing %s..\n", filename);
                    ptx_in = fopen( filename, "r" );
                    free(namelist[n]);
                    symbol_table *symtab=init_parser(filename);
                    ptx_parse ();
                    ptxinfo_in = open_ptxinfo(filename);
                    ptxinfo_parse();
                    load_static_globals(symtab,STATIC_ALLOC_LIMIT,0xFFFFFFFF);
                    load_constants(symtab,STATIC_ALLOC_LIMIT);
                }
                free(namelist);
            }
        }
    }

   if ( ptx_in == NULL && g_override_embedded_ptx ) {
      printf("GPGPU-Sim PTX Simulator error: Could find/open .ptx file for reading\n");
      printf("    This means there are no .ptx files in the current directory.\n");
      printf("    Either place a .ptx file in the current directory, or ensure\n" );
      printf("    the PTX_SIM_KERNELFILE environment variable points to .ptx file.\n");  
      printf("    PTX_SIM_KERNELFILE=\"%s\"\n", g_filename );  
      exit(1);
   }

   if ( g_error_detected ) {
      printf( "GPGPU-Sim PTX: PTX parsing errors detected -- exiting.\n" );
      exit(1);
   }
   printf( "GPGPU-Sim PTX: Program parsing completed\n" );

   if ( g_kernel_name_to_function_lookup ) {
      for ( std::map<std::string,function_info*>::iterator f=g_kernel_name_to_function_lookup->begin();
          f != g_kernel_name_to_function_lookup->end(); f++ ) {
         gpgpu_ptx_assemble(f->first,f->second);
      }
   }
}

void gpgpu_ptx_sim_add_ptxstring( const char *ptx_string, const char *sourcefname )
{
    ptx_info_t *t = new ptx_info_t;
    t->next = NULL;
    t->str = strdup(ptx_string);
    t->fname = strdup(sourcefname);

    // put ptx source into a fifo
    if (g_ptx_source_array == NULL) {
        // first ptx source
        g_ptx_source_array = t;
    } else {
        // insert subsequent ptx source at the end of queue
        ptx_info_t *l_ptx_source = g_ptx_source_array;
        while (l_ptx_source->next != NULL) {
            l_ptx_source = l_ptx_source->next;
        }
        l_ptx_source->next = t;
    }
}

static bool g_save_embedded_ptx;
bool g_keep_intermediate_files;

void ptx_reg_options(option_parser_t opp)
{
   option_parser_register(opp, "-save_embedded_ptx", OPT_BOOL, &g_save_embedded_ptx, 
                "saves ptx files embedded in binary as <n>.ptx",
                "0");
   option_parser_register(opp, "-keep", OPT_BOOL, &g_keep_intermediate_files, 
                "keep intermediate files created by GPGPU-Sim when interfacing with external programs",
                "0");
}

void print_ptx_file( const char *p, unsigned source_num, const char *filename )
{
   printf("\nGPGPU-Sim PTX: file _%u.ptx contents:\n\n", source_num );
   char *s = strdup(p);
   char *t = s;
   unsigned n=1;
   while ( *t != '\0'  ) {
      char *u = t;
      while ( (*u != '\n') && (*u != '\0') ) u++;
      unsigned last = (*u == '\0');
      *u = '\0';
      const ptx_instruction *pI = ptx_instruction_lookup(filename,n);
      char pc[64];
      if( pI && pI->get_PC() )
         snprintf(pc,64,"%4u", pI->get_PC() );
      else 
         snprintf(pc,64,"    ");
      printf("    _%u.ptx  %4u (pc=%s):  %s\n", source_num, n, pc, t );
      if ( last ) break;
      t = u+1;
      n++;
   }
   free(s);
   fflush(stdout);
}

symbol_table *gpgpu_ptx_sim_load_ptx_from_string( const char *p, unsigned source_num ) 
{
    char buf[1024];
    snprintf(buf,1024,"_%u.ptx", source_num );
    if( g_save_embedded_ptx ) {
       FILE *fp = fopen(buf,"w");
       fprintf(fp,"%s",p);
       fclose(fp);
    }
    symbol_table *symtab=init_parser(buf);
    ptx__scan_string(p);
    int errors = ptx_parse ();
    if ( errors ) {
        char fname[1024];
        snprintf(fname,1024,"_ptx_XXXXXX");
        int fd=mkstemp(fname); 
        close(fd);
        printf("GPGPU-Sim PTX: parser error detected, exiting... but first extracting .ptx to \"%s\"\n", fname);
        FILE *ptxfile = fopen(fname,"w");
        fprintf(ptxfile,"%s", p );
        fclose(ptxfile);
        abort();
        exit(40);
    }

    if ( g_debug_execution >= 100 ) 
       print_ptx_file(p,source_num,buf);

    printf("GPGPU-Sim PTX: finished parsing EMBEDDED .ptx file %s\n",buf);

    char fname[1024];
    snprintf(fname,1024,"_ptx_XXXXXX");
    int fd=mkstemp(fname); 
    close(fd);

    printf("GPGPU-Sim PTX: extracting embedded .ptx to temporary file \"%s\"\n", fname);
    FILE *ptxfile = fopen(fname,"w");
    fprintf(ptxfile,"%s",p);
    fclose(ptxfile);

    char fname2[1024];
    snprintf(fname2,1024,"_ptx2_XXXXXX");
    fd=mkstemp(fname2); 
    close(fd);
    char commandline2[4096];
    snprintf(commandline2,4096,"cat %s | sed 's/.version 1.5/.version 1.4/' | sed 's/, texmode_independent//' | sed 's/\\(\\.extern \\.const\\[1\\] .b8 \\w\\+\\)\\[\\]/\\1\\[1\\]/' | sed 's/const\\[.\\]/const\\[0\\]/g' > %s", fname, fname2);
    int result = system(commandline2);
    if( result != 0 ) {
       printf("GPGPU-Sim PTX: ERROR ** while loading PTX (a) %d\n", result);
       printf("               Ensure you have write access to simulation directory\n");
       printf("               and have \'cat\' and \'sed\' in your path.\n");
       exit(1);
    }

    char tempfile_ptxinfo[1024];
    snprintf(tempfile_ptxinfo,1024,"%sinfo",fname);
    char commandline[1024];
    char extra_flags[1024];
    extra_flags[0]=0;
#if CUDART_VERSION >= 3000
    snprintf(extra_flags,1024,"--gpu-name=sm_20");
#endif
    snprintf(commandline,1024,"ptxas %s -v %s --output-file /dev/null 2> %s", 
             extra_flags, fname2, tempfile_ptxinfo);
    printf("GPGPU-Sim PTX: generating ptxinfo using \"%s\"\n", commandline);
    result = system(commandline);
    if( result != 0 ) {
       printf("GPGPU-Sim PTX: ERROR ** while loading PTX (b) %d\n", result);
       printf("               Ensure ptxas is in your path.\n");
       exit(1);
    }

    ptxinfo_in = fopen(tempfile_ptxinfo,"r");
    g_ptxinfo_filename = tempfile_ptxinfo;
    ptxinfo_parse();
    snprintf(commandline,1024,"rm -f %s %s %s", fname, fname2, tempfile_ptxinfo);
    printf("GPGPU-Sim PTX: removing ptxinfo using \"%s\"\n", commandline);
    result = system(commandline);
    if( result != 0 ) {
       printf("GPGPU-Sim PTX: ERROR ** while loading PTX (c) %d\n", result);
       exit(1);
    }
    return symtab;
}

