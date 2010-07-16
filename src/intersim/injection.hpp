#ifndef _INJECTION_HPP_
#define _INJECTION_HPP_

#include "config_utils.hpp"

typedef int (*tInjectionProcess)( int, double );

void InitializeInjectionMap( );

tInjectionProcess GetInjectionProcess( const Configuration& config );

#endif 
