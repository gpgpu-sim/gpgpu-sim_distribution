#ifndef _ROUTEFUNC_HPP_
#define _ROUTEFUNC_HPP_

#include "flit.hpp"
#include "router.hpp"
#include "outputset.hpp"
#include "config_utils.hpp"

typedef void (*tRoutingFunction)( const Router *, const Flit *, int in_channel, OutputSet *, bool );

void InitializeRoutingMap( );
tRoutingFunction GetRoutingFunction( const Configuration& config );

#endif
