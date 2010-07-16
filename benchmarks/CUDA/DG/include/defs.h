/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * defs.h
 *
 * This file contains constant definitions
 *
 * Started 8/27/94
 * George
 *
 * $Id: defs.h,v 1.4 2003/07/22 20:29:05 karypis Exp $
 *
 */


#define GLOBAL_DBGLVL			0
#define GLOBAL_SEED			15

#define MC_FLOW_BALANCE_THRESHOLD       0.2
#define MOC_GD_GRANULARITY_FACTOR       1.0
#define RIP_SPLIT_FACTOR                8
#define MAX_NPARTS_MULTIPLIER		20

#define STATIC_PARTITION        1
#define ADAPTIVE_PARTITION      2
#define REFINE_PARTITION        3
#define MESH_PARTITION		4

#define REDIST_WGT              2.0
#define MAXNVWGT_FACTOR         2.0

#define MAXNCON                 12
#define MAXNOBJ                 12
#define N_MOC_REDO_PASSES       10
#define N_MOC_GR_PASSES         8
#define NREMAP_PASSES           8
#define N_MOC_GD_PASSES         6
#define N_MOC_BAL_PASSES        4
#define NMATCH_PASSES           4

#define COUPLED			1
#define DISCOUPLED		2

#define MAX_NCON_FOR_DIFFUSION  2
#define SMALLGRAPH              10000

#define LTERM                   (void **) 0     /* List terminator for GKfree() */

#define NGD_PASSES		20

#define OPTION_IPART		1
#define OPTION_FOLDF		2
#define OPTION_DBGLVL		3

#define PMV3_OPTION_DBGLVL	1
#define PMV3_OPTION_SEED	2
#define PMV3_OPTION_IPART	3
#define PMV3_OPTION_PSR		3

#define XYZ_XCOORD		1
#define XYZ_SPFILL		2

/* Type of initial vertex separator algorithms */
#define ISEP_EDGE		1
#define ISEP_NODE		2

#define UNMATCHED		-1
#define MAYBE_MATCHED		-2
#define TOO_HEAVY		-3


#define HTABLE_EMPTY    	-1

#define NGR_PASSES		4	/* Number of greedy refinement passes */
#define NIPARTS			8	/* Number of random initial partitions */
#define NLGR_PASSES		5	/* Number of GR refinement during IPartition */

#define SMALLFLOAT		0.00001
/* #define KEEP_BIT        (idxtype)536870912    */    /* 1<<29 */
#define KEEP_BIT        ((idxtype)(1<<((sizeof(idxtype)*8)-2)))

#define MAX_PES			8192
#define MAX_NPARTS		67108864

#define COARSEN_FRACTION	0.75	/* Node reduction between succesive coarsening levels */
#define COARSEN_FRACTION2	0.55	/* Node reduction between succesive coarsening levels */
#define UNBALANCE_FRACTION		1.05
#define ORDER_UNBALANCE_FRACTION	1.05

#define MAXVWGT_FACTOR		1.4

#define MATCH_LOCAL		1
#define MATCH_GLOBAL		2

/* Debug Levels */
#define DBG_TIME	1		/* Perform timing analysis */
#define DBG_INFO	2		/* Perform timing analysis */
#define DBG_PROGRESS   	4		/* Show the coarsening progress */
#define DBG_REFINEINFO	8		/* Show info on communication during folding */
#define DBG_MATCHINFO	16		/* Show info on matching */
#define DBG_RMOVEINFO	32		/* Show info on communication during folding */
#define DBG_REMAP	64		/* Determines if remapping will take place */
