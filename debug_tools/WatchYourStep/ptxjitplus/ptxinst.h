/* ptxinst.h 
 * Jonathan Lew 
 * University of British Columbia
 */
 
#ifndef _PTXINST_H_
#define _PTXINST_H_

#include <string>

void* instrument_ptx_from_function(std::string function, std::string path);
void* instrument_ptx_from_string(std::string ptxcode);

#endif
