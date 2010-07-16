/*
 * Copyright 1997-2003, Regents of the University of Minnesota
 *
 * parmetis.h
 *
 * This file contains function prototypes and constrant definitions for 
 * ParMETIS
 *
 * Started 7/21/03
 * George
 *
 */

#ifndef __parmetis_h__
#define __parmetis_h__

#include <mpi.h>

#ifndef _MSC_VER
#define __cdecl
#endif


/*************************************************************************
* Data-structures
**************************************************************************/
/* Undefine the following #define in order to use short int as the idxtype */
#define IDXTYPE_INT

/* Indexes are as long as integers for now */
#ifdef IDXTYPE_INT
typedef int idxtype;
#else
typedef short idxtype;
#endif


/*************************************************************************
* Constants 
**************************************************************************/
#define PARMETIS_MAJOR_VERSION        3
#define PARMETIS_MINOR_VERSION        1


/*************************************************************************
* Function prototypes
**************************************************************************/
#ifdef __cplusplus
extern "C" {
#endif

/*-------------------------------------------------------------------
* API Introduced with Release 3.0 (current API) 
*--------------------------------------------------------------------*/
void __cdecl ParMETIS_V3_AdaptiveRepart(
             idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, 
	     idxtype *vsize, idxtype *adjwgt, int *wgtflag, int *numflag, int *ncon, 
	     int *nparts, float *tpwgts, float *ubvec, float *ipc2redist, 
	     int *options, int *edgecut, idxtype *part, MPI_Comm *comm);

void __cdecl ParMETIS_V3_PartGeomKway(
             idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, 
	     idxtype *adjwgt, int *wgtflag, int *numflag, int *ndims, float *xyz, 
	     int *ncon, int *nparts, float *tpwgts, float *ubvec, int *options, 
	     int *edgecut, idxtype *part, MPI_Comm *comm);

void __cdecl ParMETIS_V3_PartGeom(
             idxtype *vtxdist, int *ndims, float *xyz, idxtype *part, MPI_Comm *comm);

void __cdecl ParMETIS_V3_PartKway(
             idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, 
	     idxtype *adjwgt, int *wgtflag, int *numflag, int *ncon, int *nparts, 
	     float *tpwgts, float *ubvec, int *options, int *edgecut, idxtype *part, 
	     MPI_Comm *comm);

void __cdecl ParMETIS_V3_Mesh2Dual(
             idxtype *elmdist, idxtype *eptr, idxtype *eind, int *numflag, 
	     int *ncommonnodes, idxtype **xadj, idxtype **adjncy, MPI_Comm *comm);

void __cdecl ParMETIS_V3_PartMeshKway(
             idxtype *elmdist, idxtype *eptr, idxtype *eind, idxtype *elmwgt, 
	     int *wgtflag, int *numflag, int *ncon, int *ncommonnodes, int *nparts, 
	     float *tpwgts, float *ubvec, int *options, int *edgecut, idxtype *part, 
	     MPI_Comm *comm);

void __cdecl ParMETIS_V3_NodeND(
             idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, int *numflag, 
	     int *options, idxtype *order, idxtype *sizes, MPI_Comm *comm);

void __cdecl ParMETIS_V3_RefineKway(
             idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, 
	     idxtype *adjwgt, int *wgtflag, int *numflag, int *ncon, int *nparts, 
	     float *tpwgts, float *ubvec, int *options, int *edgecut, 
	     idxtype *part, MPI_Comm *comm);



/*------------------------------------------------------------------
* Backward compatibility routines with Release 2.0
*-------------------------------------------------------------------*/
void __cdecl ParMETIS_PartKway(
             idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, 
	     idxtype *adjwgt, int *wgtflag, int *numflag, int *nparts, int *options, 
             int *edgecut, idxtype *part, MPI_Comm *comm);

void __cdecl ParMETIS_PartGeomKway(
             idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, 
	     int *wgtflag, int *numflag, int *ndims, float *xyz, int *nparts, int *options, 
	     int *edgecut, idxtype *part, MPI_Comm *comm);

void __cdecl ParMETIS_PartGeom(
             idxtype *vtxdist, int *ndims, float *xyz, idxtype *part, MPI_Comm *comm);

void __cdecl ParMETIS_PartGeomRefine(
             idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, 
	     idxtype *adjwgt, int *wgtflag, int *numflag, int *ndims, float *xyz, 
	     int *options, int *edgecut, idxtype *part, MPI_Comm *comm);

void __cdecl ParMETIS_RefineKway(
             idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, 
	     idxtype *adjwgt, int *wgtflag, int *numflag, int *options, int *edgecut, 
	     idxtype *part, MPI_Comm *comm);

void __cdecl ParMETIS_RepartLDiffusion(
             idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, 
	     idxtype *adjwgt, int *wgtflag, int *numflag, int *options, int *edgecut, 
	     idxtype *part, MPI_Comm *comm);

void __cdecl ParMETIS_RepartGDiffusion(
             idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, 
	     idxtype *adjwgt, int *wgtflag, int *numflag, int *options, int *edgecut, 
	     idxtype *part, MPI_Comm *comm);

void __cdecl ParMETIS_RepartRemap(
             idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, 
	     int *wgtflag, int *numflag, int *options, int *edgecut, idxtype *part, 
	     MPI_Comm *comm);

void __cdecl ParMETIS_RepartMLRemap(
             idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, 
	     int *wgtflag, int *numflag, int *options, int *edgecut, idxtype *part, 
	     MPI_Comm *comm);

void __cdecl ParMETIS_NodeND(
             idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, int *numflag, int *options, 
	     idxtype *order, idxtype *sizes, MPI_Comm *comm);

void __cdecl ParMETIS_SerialNodeND(
             idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, int *numflag, int *options, 
	     idxtype *order, idxtype *sizes, MPI_Comm *comm);




/*-------------------------------------------------------------------
* Backward compatibility routines with Release 1.0 
*--------------------------------------------------------------------*/
void __cdecl PARKMETIS(
             idxtype *vtxdist, idxtype *xadj, idxtype *vwgt, idxtype *adjncy, idxtype *adjwgt, 
	     idxtype *part, int *options, MPI_Comm comm);

void __cdecl PARGKMETIS(
             idxtype *vtxdist, idxtype *xadj, idxtype *vwgt, idxtype *adjncy, idxtype *adjwgt,
             int ndims, float *xyz, idxtype *part, int *options, MPI_Comm comm);

void __cdecl PARGRMETIS(
             idxtype *vtxdist, idxtype *xadj, idxtype *vwgt, idxtype *adjncy, idxtype *adjwgt,
             int ndims, float *xyz, idxtype *part, int *options, MPI_Comm comm);

void __cdecl PARGMETIS(
             idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, int ndims, float *xyz,
             idxtype *part, int *options, MPI_Comm comm);

void __cdecl PARRMETIS(
             idxtype *vtxdist, idxtype *xadj, idxtype *vwgt, idxtype *adjncy, 
	     idxtype *adjwgt, idxtype *part, int *options, MPI_Comm comm);

void __cdecl PARUAMETIS(
             idxtype *vtxdist, idxtype *xadj, idxtype *vwgt, idxtype *adjncy, 
	     idxtype *adjwgt, idxtype *part, int *options, MPI_Comm comm);

void __cdecl PARDAMETIS(
             idxtype *vtxdist, idxtype *xadj, idxtype *vwgt, idxtype *adjncy, idxtype *adjwgt,
             idxtype *part, int *options, MPI_Comm comm);

#ifdef __cplusplus
}
#endif




#endif 
