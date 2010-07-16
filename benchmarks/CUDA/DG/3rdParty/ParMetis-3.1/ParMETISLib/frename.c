/*
 * frename.c
 *
 * This file contains some renaming routines to deal with different
 * Fortran compilers.
 *
 * Started 6/1/98
 * George
 *
 * $Id: frename.c,v 1.4 2003/07/30 18:37:58 karypis Exp $
 *
 */

#include <parmetislib.h>



/*************************************************************************
* Renaming macro (at least to save some typing :))  
**************************************************************************/
#define FRENAME(name0, name1, name2, name3, name4, dargs, cargs)   \
  void name1 dargs { name0 cargs; }                          \
  void name2 dargs { name0 cargs; }                          \
  void name3 dargs { name0 cargs; }                          \
  void name4 dargs { name0 cargs; }








/*************************************************************************
* Renames for Release 3.0 API
**************************************************************************/
FRENAME(ParMETIS_V3_AdaptiveRepart, 
        PARMETIS_V3_ADAPTIVEREPART,
	parmetis_v3_adaptiverepart,
	parmetis_v3_adaptiverepart_,
	parmetis_v3_adaptiverepart__,
	(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt,
	 idxtype *vsize, idxtype *adjwgt, int *wgtflag, int *numflag, int *ncon,
	 int *nparts, float *tpwgts, float *ubvec, float *ipc2redist,
	 int *options, int *edgecut, idxtype *part, MPI_Comm *comm),
	(vtxdist, xadj, adjncy, vwgt, vsize, adjwgt, wgtflag, numflag, ncon,
	 nparts, tpwgts, ubvec, ipc2redist, options, edgecut, part, comm)
)

FRENAME(ParMETIS_V3_PartGeomKway,
        PARMETIS_V3_PARTGEOMKWAY,
	parmetis_v3_partgeomkway,
	parmetis_v3_partgeomkway_,
	parmetis_v3_partgeomkway__,
        (idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, 
	 idxtype *adjwgt, int *wgtflag, int *numflag, int *ndims, float *xyz, 
	 int *ncon, int *nparts, float *tpwgts, float *ubvec, int *options, 
	 int *edgecut, idxtype *part, MPI_Comm *comm),
        (vtxdist, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, ndims, xyz, 
	 ncon, nparts, tpwgts, ubvec, options, edgecut, part, comm)
)
	 
FRENAME(ParMETIS_V3_PartGeom,
        PARMETIS_V3_PARTGEOM,
	parmetis_v3_partgeom,
	parmetis_v3_partgeom_,
	parmetis_v3_partgeom__,
	(idxtype *vtxdist, int *ndims, float *xyz, idxtype *part, MPI_Comm *comm),
	(vtxdist, ndims, xyz, part, comm)
)

FRENAME(ParMETIS_V3_PartKway,
        PARMETIS_V3_PARTKWAY,
	parmetis_v3_partkway,
	parmetis_v3_partkway_,
	parmetis_v3_partkway__,
	(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, 
	 int *wgtflag, int *numflag, int *ncon, int *nparts, float *tpwgts, float *ubvec, 
	 int *options, int *edgecut, idxtype *part, MPI_Comm *comm),
	(vtxdist, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, ncon, nparts, tpwgts, 
	 ubvec, options, edgecut, part, comm)
)

FRENAME(ParMETIS_V3_Mesh2Dual,
        PARMETIS_V3_MESH2DUAL,
	parmetis_v3_mesh2dual,
	parmetis_v3_mesh2dual_,
	parmetis_v3_mesh2dual__,
	(idxtype *elmdist, idxtype *eptr, idxtype *eind, int *numflag, int *ncommonnodes, 
	 idxtype **xadj, idxtype **adjncy, MPI_Comm *comm),
	(elmdist, eptr, eind, numflag, ncommonnodes, xadj, adjncy, comm)
)

FRENAME(ParMETIS_V3_PartMeshKway,
        PARMETIS_V3_PARTMESHKWAY, 
	parmetis_v3_partmeshkway,
	parmetis_v3_partmeshkway_,
	parmetis_v3_partmeshkway__,
	(idxtype *elmdist, idxtype *eptr, idxtype *eind, idxtype *elmwgt, int *wgtflag, 
	 int *numflag, int *ncon, int *ncommonnodes, int *nparts, float *tpwgts, 
	 float *ubvec, int *options, int *edgecut, idxtype *part, MPI_Comm *comm),
	(elmdist, eptr, eind, elmwgt, wgtflag, numflag, ncon, ncommonnodes, nparts, tpwgts, 
	 ubvec, options, edgecut, part, comm)
)
	 
FRENAME(ParMETIS_V3_NodeND,
        PARMETIS_V3_NODEND,
        parmetis_v3_nodend,
        parmetis_v3_nodend_,
        parmetis_v3_nodend__,
        (idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, int *numflag, int *options, 
	 idxtype *order, idxtype *sizes, MPI_Comm *comm),
        (vtxdist, xadj, adjncy, numflag, options, order, sizes, comm)
)

FRENAME(ParMETIS_V3_RefineKway,
        PARMETIS_V3_REFINEKWAY,
        parmetis_v3_refinekway,
        parmetis_v3_refinekway_,
        parmetis_v3_refinekway__,
        (idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, 
	 int *wgtflag, int *numflag, int *ncon, int *nparts, float *tpwgts, float *ubvec, 
	 int *options, int *edgecut, idxtype *part, MPI_Comm *comm),
        (vtxdist, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, ncon, nparts, tpwgts, 
	 ubvec, options, edgecut, part, comm)
)


/*************************************************************************
* Renames for Release 2.0 API
**************************************************************************/
FRENAME(ParMETIS_PartKway,
        PARMETIS_PARTKWAY,
	parmetis_partkway,
	parmetis_partkway_,
	parmetis_partkway__,
        (idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, 
	 int *wgtflag, int *numflag, int *nparts, int *options, int *edgecut, idxtype *part, 
	 MPI_Comm *comm),
        (vtxdist, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, options, edgecut, 
	 part, comm)
)
	 
FRENAME(ParMETIS_PartGeomKway,
        PARMETIS_PARTGEOMKWAY,
        parmetis_partgeomkway,
        parmetis_partgeomkway_,
        parmetis_partgeomkway__,
        (idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, 
	 int *wgtflag, int *numflag, int *ndims, float *xyz, int *nparts, int *options, 
	 int *edgecut, idxtype *part, MPI_Comm *comm),
        (vtxdist, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, ndims, xyz, nparts, options, 
	 edgecut, part, comm)
) 
	 
FRENAME(ParMETIS_PartGeom,
        PARMETIS_PARTGEOM,
	parmetis_partgeom,
	parmetis_partgeom_,
	parmetis_partgeom__,
	(idxtype *vtxdist, int *ndims, float *xyz, idxtype *part, MPI_Comm *comm),
	(vtxdist, ndims, xyz, part, comm)
)

FRENAME(ParMETIS_PartGeomRefine,
        PARMETIS_PARTGEOMREFINE,
	parmetis_partgeomrefine,
	parmetis_partgeomrefine_,
	parmetis_partgeomrefine__,
        (idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, 
	 int *wgtflag, int *numflag, int *ndims, float *xyz, int *options, int *edgecut, 
	 idxtype *part, MPI_Comm *comm),
        (vtxdist, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, ndims, xyz, options, 
	 edgecut, part, comm)
)

FRENAME(ParMETIS_RefineKway,
        PARMETIS_REFINEKWAY,
	parmetis_refinekway,
	parmetis_refinekway_,
	parmetis_refinekway__,
        (idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, 
	 int *wgtflag, int *numflag, int *options, int *edgecut, idxtype *part, MPI_Comm *comm),
        (vtxdist, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, options, edgecut, part, comm)
)

FRENAME(ParMETIS_RepartLDiffusion,
        PARMETIS_REPARTLDIFUSSION,
	parmetis_repartldiffusion,
	parmetis_repartldiffusion_,
	parmetis_repartldiffusion__,
	(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, 
	 int *wgtflag, int *numflag, int *options, int *edgecut, idxtype *part, MPI_Comm *comm),
	(vtxdist, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, options, edgecut, part, comm)
)

FRENAME(ParMETIS_RepartGDiffusion,
        PARMETIS_REPARTGDIFFUSION,
        parmetis_repartgdiffusion,
        parmetis_repartgdiffusion_,
        parmetis_repartgdiffusion__,
	(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, 
	 int *wgtflag, int *numflag, int *options, int *edgecut, idxtype *part, MPI_Comm *comm),
	(vtxdist, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, options, edgecut, part, comm)
)

FRENAME(ParMETIS_RepartRemap,
        PARMETIS_REPARTREMAP,
        parmetis_repartremap,
        parmetis_repartremap_,
        parmetis_repartremap__,
        (idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, 
	 int *wgtflag, int *numflag, int *options, int *edgecut, idxtype *part, MPI_Comm *comm),
        (vtxdist, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, options, edgecut, part, comm)
)

FRENAME(ParMETIS_RepartMLRemap,
        PARMETIS_REPARTMLREMAP,
        parmetis_repartmlremap,
        parmetis_repartmlremap_,
        parmetis_repartmlremap__,
        (idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, idxtype *vwgt, idxtype *adjwgt, 
	 int *wgtflag, int *numflag, int *options, int *edgecut, idxtype *part, MPI_Comm *comm),
        (vtxdist, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, options, edgecut, part, comm)
)

FRENAME(ParMETIS_NodeND,
        PARMETIS_NODEND,
        parmetis_nodend,
        parmetis_nodend_,
        parmetis_nodend__,
	(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, int *numflag, int *options, 
	 idxtype *order, idxtype *sizes, MPI_Comm *comm),
	(vtxdist, xadj, adjncy, numflag, options, order, sizes, comm)
)

FRENAME(ParMETIS_SerialNodeND,
        PARMETIS_SERIALNODEND,
	parmetis_serialnodend,
	parmetis_serialnodend_,
	parmetis_serialnodend__,
	(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, int *numflag, int *options, 
	 idxtype *order, idxtype *sizes, MPI_Comm *comm),
	(vtxdist, xadj, adjncy, numflag, options, order, sizes, comm)
)




/*************************************************************************
* Renames for Release 1.0 API
**************************************************************************/
FRENAME(PARKMETIS,
        PARKMETIS_,
	parkmetis,
	parkmetis_,
	parkmetis__,
	(idxtype *vtxdist, idxtype *xadj, idxtype *vwgt, idxtype *adjncy, idxtype *adjwgt, 
	 idxtype *part, int *options, MPI_Comm comm),
	(vtxdist, xadj, vwgt, adjncy, adjwgt, part, options, comm)
)

FRENAME(PARGKMETIS,
        PARGKMETIS_,
        pargkmetis,
        pargkmetis_,
        pargkmetis__,
        (idxtype *vtxdist, idxtype *xadj, idxtype *vwgt, idxtype *adjncy, idxtype *adjwgt,
         int ndims, float *xyz, idxtype *part, int *options, MPI_Comm comm),
        (vtxdist, xadj, vwgt, adjncy, adjwgt, ndims, xyz, part, options, comm)
)	

FRENAME(PARGRMETIS,
        PARGRMETIS_,
	pargrmetis,
	pargrmetis_,
	pargrmetis__,
        (idxtype *vtxdist, idxtype *xadj, idxtype *vwgt, idxtype *adjncy, idxtype *adjwgt,
         int ndims, float *xyz, idxtype *part, int *options, MPI_Comm comm),
        (vtxdist, xadj, vwgt, adjncy, adjwgt, ndims, xyz, part, options, comm)
)

FRENAME(PARGMETIS,
        PARGMETIS_,
	pargmetis,
	pargmetis_,
	pargmetis__,
	(idxtype *vtxdist, idxtype *xadj, idxtype *adjncy, int ndims, float *xyz,
         idxtype *part, int *options, MPI_Comm comm),
	(vtxdist, xadj, adjncy, ndims, xyz, part, options, comm)
)

FRENAME(PARRMETIS,
        PARRMETIS_,
	parrmetis,
	parrmetis_,
	parrmetis__,
	(idxtype *vtxdist, idxtype *xadj, idxtype *vwgt, idxtype *adjncy, idxtype *adjwgt, 
	 idxtype *part, int *options, MPI_Comm comm),
	(vtxdist, xadj, vwgt, adjncy, adjwgt, part, options, comm)
)

FRENAME(PARUAMETIS,
        PARUAMETIS_,
	paruametis,
	paruametis_,
	paruametis__,
	(idxtype *vtxdist, idxtype *xadj, idxtype *vwgt, idxtype *adjncy, idxtype *adjwgt, 
	 idxtype *part, int *options, MPI_Comm comm),
	(vtxdist, xadj, vwgt, adjncy, adjwgt, part, options, comm)
)

FRENAME(PARDAMETIS,
        PARDAMETIS_,
	pardametis,
	pardametis_,
	pardametis__,
        (idxtype *vtxdist, idxtype *xadj, idxtype *vwgt, idxtype *adjncy, idxtype *adjwgt,
         idxtype *part, int *options, MPI_Comm comm),
        (vtxdist, xadj, vwgt, adjncy, adjwgt, part, options, comm)
)

