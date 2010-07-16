#ifndef _TRAFFIC_HPP_
#define _TRAFFIC_HPP_

#include "config_utils.hpp"

typedef int (*tTrafficFunction)( int, int );

void InitializeTrafficMap( );

void ResetTraffic( );
void StepTrafficFunctions( );

tTrafficFunction GetTrafficFunction( const Configuration& config );

#endif
