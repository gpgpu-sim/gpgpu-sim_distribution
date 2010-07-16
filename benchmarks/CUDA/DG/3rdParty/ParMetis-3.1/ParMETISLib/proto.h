/*
 * Copyright 1997, Regents of the University of Minnesota
 *
 * proto.h
 *
 * This file contains header files
 *
 * Started 10/19/95
 * George
 *
 * $Id: proto.h,v 1.11 2003/07/25 13:52:03 karypis Exp $
 *
 */

/* kmetis.c */
void Moc_Global_Partition(CtrlType *, GraphType *, WorkSpaceType *);

/* mmetis.c */

/* gkmetis.c */

/* match.c */
void Moc_GlobalMatch_Balance(CtrlType *, GraphType *, WorkSpaceType *);

/* coarsen.c */
void Moc_Global_CreateCoarseGraph(CtrlType *, GraphType *, WorkSpaceType *, int);

/* initpart.c */
void Moc_InitPartition_RB(CtrlType *, GraphType *, WorkSpaceType *);
void Moc_KeepPart(GraphType *, WorkSpaceType *, idxtype *, int);

/* kwayrefine.c */
void Moc_ProjectPartition(CtrlType *, GraphType *, WorkSpaceType *);
void Moc_ComputePartitionParams(CtrlType *, GraphType *, WorkSpaceType *);

/* kwayfm.c */
void Moc_KWayFM(CtrlType *, GraphType *, WorkSpaceType *, int);

/* kwaybalance.c */
void Moc_KWayBalance(CtrlType *, GraphType *, WorkSpaceType *, int);

/* remap.c */
void ParallelReMapGraph(CtrlType *, GraphType *, WorkSpaceType *);
void ParallelTotalVReMap(CtrlType *, idxtype *, idxtype *, WorkSpaceType *, int, int);
int SimilarTpwgts(float *, int, int, int);

/* move.c */
GraphType *Moc_MoveGraph(CtrlType *, GraphType *, WorkSpaceType *);
/* move.c */
void CheckMGraph(CtrlType *, GraphType *); 
void ProjectInfoBack(CtrlType *, GraphType *, idxtype *, idxtype *, WorkSpaceType *);
void FindVtxPerm(CtrlType *, GraphType *, idxtype *, WorkSpaceType *);

/* memory.c */
void PreAllocateMemory(CtrlType *, GraphType *, WorkSpaceType *);
void FreeWSpace(WorkSpaceType *);
void FreeCtrl(CtrlType *);
GraphType *CreateGraph(void);
void InitGraph(GraphType *);
void FreeGraph(GraphType *);
void FreeInitialGraphAndRemap(GraphType *, int);


/* ametis.c */
void Adaptive_Partition(CtrlType *, GraphType *, WorkSpaceType *);

/* rmetis.c */


/* lmatch.c */
void Mc_LocalMatch_HEM(CtrlType *, GraphType *, WorkSpaceType *);
void Mc_Local_CreateCoarseGraph(CtrlType *, GraphType *, WorkSpaceType *, int);

/* wave.c */
float WavefrontDiffusion(CtrlType *, GraphType *, idxtype *);

/* balancemylink.c */
int BalanceMyLink(CtrlType *, GraphType *, idxtype *, int, int, float *, float, float *, float *, float);

/* redomylink.c */
void RedoMyLink(CtrlType *, GraphType *, idxtype *, int, int, float *, float *, float *);

/* initbalance.c */
void Balance_Partition(CtrlType *, GraphType *, WorkSpaceType *);
GraphType *Moc_AssembleAdaptiveGraph(CtrlType *, GraphType *, WorkSpaceType *);

/* mdiffusion.c */
int Moc_Diffusion(CtrlType *, GraphType *, idxtype *, idxtype *, idxtype *, WorkSpaceType *, int);
GraphType *ExtractGraph(CtrlType *, GraphType *, idxtype *, idxtype *, idxtype *);

/* diffutil.c */
void SetUpConnectGraph(GraphType *, MatrixType *, idxtype *);
void Mc_ComputeMoveStatistics(CtrlType *, GraphType *, int *, int *, int *);
 int Mc_ComputeSerialTotalV(GraphType *, idxtype *);
void ComputeLoad(GraphType *, int, float *, float *, int);
void ConjGrad2(MatrixType *, float *, float *, float, float *);
void mvMult2(MatrixType *, float *, float *);
void ComputeTransferVector(int, MatrixType *, float *, float *, int);
int ComputeSerialEdgeCut(GraphType *);
int ComputeSerialTotalV(GraphType *, idxtype *);

/* akwayfm.c */
void Moc_KWayAdaptiveRefine(CtrlType *, GraphType *, WorkSpaceType *, int);

/* selectq.c */
void Moc_DynamicSelectQueue(int, int, int, int, idxtype *, float *, int *, int *, int, float, float);
int Moc_HashVwgts(int, float *);
int Moc_HashVRank(int, int *);


/* csrmatch.c */
void CSR_Match_SHEM(MatrixType *, idxtype *, idxtype *, idxtype *, int);

/* serial.c */
void Moc_SerialKWayAdaptRefine(GraphType *, int, idxtype *, float *, int);
void Moc_ComputeSerialPartitionParams(GraphType *, int, EdgeType *);
int AreAllHVwgtsBelow(int, float, float *, float, float *, float *);
void ComputeHKWayLoadImbalance(int, int, float *, float *);
void SerialRemap(GraphType *, int, idxtype *, idxtype *, idxtype *, float *);
int SSMIncKeyCmp(const void *, const void *);
void Moc_Serial_FM_2WayRefine(GraphType *, float *, int);
void Serial_SelectQueue(int, float *, float *, int *, int *, FPQueueType [MAXNCON][2]);
int Serial_BetterBalance(int, float *, float *, float *);
float Serial_Compute2WayHLoadImbalance(int, float *, float *);
void Moc_Serial_Balance2Way(GraphType *, float *, float);
void Moc_Serial_Init2WayBalance(GraphType *, float *);
int Serial_SelectQueueOneWay(int, float *, float *, int, FPQueueType [MAXNCON][2]);
void Moc_Serial_Compute2WayPartitionParams(GraphType *);
int Serial_AreAnyVwgtsBelow(int, float, float *, float, float *, float *);

/* weird.c */
void PartitionSmallGraph(CtrlType *, GraphType *, WorkSpaceType *);
void CheckInputs(int partType, int npes, int dbglvl, int *wgtflag, int *iwgtflag,
                 int *numflag, int *inumflag, int *ncon, int *incon, int *nparts, 
		 int *inparts, float *tpwgts, float **itpwgts, float *ubvec, 
		 float *iubvec, float *ipc2redist, float *iipc2redist, int *options, 
		 int *ioptions, idxtype *part, MPI_Comm *comm);

/* mesh.c */

/* ometis.c */

/* pspases.c */
GraphType *AssembleEntireGraph(CtrlType *, idxtype *, idxtype *, idxtype *);

/* node_refine.c */
void ComputeNodePartitionParams0(CtrlType *, GraphType *, WorkSpaceType *);
void ComputeNodePartitionParams(CtrlType *, GraphType *, WorkSpaceType *);
void KWayNodeRefine0(CtrlType *, GraphType *, WorkSpaceType *, int, float);
void KWayNodeRefine(CtrlType *, GraphType *, WorkSpaceType *, int, float);
void KWayNodeRefine2(CtrlType *, GraphType *, WorkSpaceType *, int, float);
void PrintNodeBalanceInfo(CtrlType *, int, idxtype *, idxtype *, idxtype *, int);

/* initmsection.c */
void InitMultisection(CtrlType *, GraphType *, WorkSpaceType *);
GraphType *AssembleMultisectedGraph(CtrlType *, GraphType *, WorkSpaceType *);

/* order.c */
void MultilevelOrder(CtrlType *, GraphType *, idxtype *, idxtype *, WorkSpaceType *);
void LabelSeparators(CtrlType *, GraphType *, idxtype *, idxtype *, idxtype *, idxtype *, WorkSpaceType *);
void CompactGraph(CtrlType *, GraphType *, idxtype *, WorkSpaceType *);
void LocalOrder(CtrlType *, GraphType *, idxtype *, int, WorkSpaceType *);
void LocalNDOrder(CtrlType *, GraphType *, idxtype *, int, WorkSpaceType *);
void Order_Partition(CtrlType *, GraphType *, WorkSpaceType *);

/* xyzpart.c */
void Coordinate_Partition(CtrlType *, GraphType *, int, float *, int, WorkSpaceType *);
void PartSort(CtrlType *, GraphType *, KeyValueType *, WorkSpaceType *);


/* fpqueue.c */
void FPQueueInit(FPQueueType *, int);
void FPQueueReset(FPQueueType *);
void FPQueueFree(FPQueueType *);
int FPQueueGetSize(FPQueueType *);
int FPQueueInsert(FPQueueType *, int, float);
int FPQueueDelete(FPQueueType *, int);
int FPQueueUpdate(FPQueueType *, int, float, float);
void FPQueueUpdateUp(FPQueueType *, int, float, float);
int FPQueueGetMax(FPQueueType *);
int FPQueueSeeMaxVtx(FPQueueType *);
float FPQueueSeeMaxGain(FPQueueType *);
float FPQueueGetKey(FPQueueType *);
int FPQueueGetQSize(FPQueueType *);
int CheckHeapFloat(FPQueueType *);

/* stat.c */
void Moc_ComputeSerialBalance(CtrlType *, GraphType *, idxtype *, float *);
void Moc_ComputeParallelBalance(CtrlType *, GraphType *, idxtype *, float *);
void Moc_PrintThrottleMatrix(CtrlType *, GraphType *, float *);
void Moc_ComputeRefineStats(CtrlType *, GraphType *, float *);

/* debug.c */
void PrintVector(CtrlType *, int, int, idxtype *, char *);
void PrintVector2(CtrlType *, int, int, idxtype *, char *);
void PrintPairs(CtrlType *, int, KeyValueType *, char *);
void PrintGraph(CtrlType *, GraphType *);
void PrintGraph2(CtrlType *, GraphType *);
void PrintSetUpInfo(CtrlType *ctrl, GraphType *graph);
void PrintTransferedGraphs(CtrlType *, int, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *);
void WriteMetisGraph(int, idxtype *, idxtype *, idxtype *, idxtype *);

/* comm.c */
void CommInterfaceData(CtrlType *, GraphType *, idxtype *, idxtype *, idxtype *);
void CommChangedInterfaceData(CtrlType *, GraphType *, int, idxtype *, idxtype *, KeyValueType *, KeyValueType *, idxtype *);
int GlobalSEMax(CtrlType *, int);
double GlobalSEMaxDouble(CtrlType *, double);
int GlobalSEMin(CtrlType *, int);
int GlobalSESum(CtrlType *, int);
float GlobalSEMaxFloat(CtrlType *, float);
float GlobalSEMinFloat(CtrlType *, float);
float GlobalSESumFloat(CtrlType *, float);

/* util.c */
void errexit(char *,...);
void myprintf(CtrlType *, char *f_str,...);
void rprintf(CtrlType *, char *f_str,...);
#ifndef DMALLOC
int *imalloc(int, char *);
idxtype *idxmalloc(int, char *);
float *fmalloc(int, char *);
int *ismalloc(int, int, char *);
idxtype *idxsmalloc(int, idxtype, char *);
void *GKmalloc(int, char *);
#endif
/*void GKfree(void **,...); */
int *iset(int n, int val, int *x);
idxtype * idxset(int n, idxtype val, idxtype *x);
int idxamax(int n, idxtype *x);
int idxamin(int n, idxtype *x);
int idxasum(int n, idxtype *x);
float snorm2(int, float *);
float sdot(int n, float *, float *);
void saxpy(int, float, float *, float *);
void ikeyvalsort_org(int, KeyValueType *);
int IncKeyValueCmp(const void *, const void *);
void dkeyvalsort(int, KeyValueType *);
int DecKeyValueCmp(const void *, const void *);
int BSearch(int, idxtype *, int);
void RandomPermute(int, idxtype *, int);
void FastRandomPermute(int, idxtype *, int);
int ispow2(int);
int log2Int(int);
void BucketSortKeysDec(int, int, idxtype *, idxtype *);
float *sset(int n, float val, float *x);
int iamax(int, int *);
int idxamax_strd(int, idxtype *, int);
int idxamin_strd(int, idxtype *, int);
int samax_strd(int, float *, int);
int sfamax(int, float *);
int samin_strd(int, float *, int);
float idxavg(int, idxtype *);
float savg(int, float *);
int samax(int, float *);
int sfavg(int n, float *x);
int samax2(int, float *);
int samin(int, float *);
int idxsum(int, idxtype *);
int idxsum_strd(int, idxtype *, int);
void idxadd(int, idxtype *, idxtype *);
float ssum(int, float *);
float ssum_strd(int, float *, int);
void sscale(int, float, float *);
void saneg(int, float *);
float BetterVBalance(int, float *, float *, float *);
int IsHBalanceBetterTT(int, float *, float *, float *, float *);
int IsHBalanceBetterFT(int, float *, float *, float *, float *);
int myvalkeycompare(const void *, const void *);
int imyvalkeycompare(const void *, const void *);
float *fsmalloc(int, float, char *);
void saxpy2(int, float, float *, int, float *, int);
void GetThreeMax(int, float *, int *, int *, int *);

/* qsort_special.c */
void iidxsort(int, idxtype *);
void iintsort(int, int *);
void ikeysort(int, KeyValueType *);
void ikeyvalsort(int, KeyValueType *);

/* grsetup.c */
GraphType *Moc_SetUpGraph(CtrlType *, int, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, int *);
void SetUpCtrl(CtrlType *ctrl, int, int, MPI_Comm);
void ChangeNumbering(idxtype *, idxtype *, idxtype *, idxtype *, int, int, int);
void ChangeNumberingMesh(idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, int, int, int, int);
void ChangeNumberingMesh2(idxtype *elmdist, idxtype *eptr, idxtype *eind,
                          idxtype *xadj, idxtype *adjncy, idxtype *part,
			  int npes, int mype, int from);
void GraphRandomPermute(GraphType *);
void ComputeMoveStatistics(CtrlType *, GraphType *, int *, int *, int *);

/* timer.c */
void InitTimers(CtrlType *);
void PrintTimingInfo(CtrlType *);
void PrintTimer(CtrlType *, timer, char *);

/* setup.c */
void SetUp(CtrlType *, GraphType *, WorkSpaceType *);
int Home_PE(int, int, idxtype *, int);


/*********************/
/* METIS subroutines */
/*********************/
void METIS_WPartGraphKway2(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
void METIS_mCPartGraphRecursive2(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
int MCMlevelRecursiveBisection2(CtrlType *, GraphType *, int, float *, idxtype *, float, int); 
void METIS_PartGraphKway(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, int *, int *, idxtype *);
void METIS_mCPartGraphKway(int *, int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, int *, float *, int *, int *, idxtype *);
void METIS_EdgeComputeSeparator(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, idxtype *); 
void METIS_NodeComputeSeparator(int *, idxtype *, idxtype *, idxtype *, idxtype *, int *, int *, idxtype *); 
void METIS_NodeND(int *, idxtype *, idxtype *, int *, int *, idxtype *, idxtype *);
void METIS_NodeNDP(int, idxtype *, idxtype *, int, int *, idxtype *, idxtype *, idxtype *);



/***********************/
/* TESTing subroutines */
/***********************/

/* pio.c */
void ParallelReadGraph(GraphType *, char *, MPI_Comm);
void Moc_ParallelWriteGraph(CtrlType *, GraphType *, char *, int, int);
void ReadTestGraph(GraphType *, char *, MPI_Comm);
float *ReadTestCoordinates(GraphType *, char *, int, MPI_Comm);
void ReadMetisGraph(char *, int *, idxtype **, idxtype **);
void Moc_SerialReadGraph(GraphType *, char *, int *, MPI_Comm);
void Moc_SerialReadMetisGraph(char *, int *, int *, int *, int *, idxtype **, idxtype **, idxtype **, idxtype **, int *);

/* adaptgraph */
void AdaptGraph(GraphType *, int, MPI_Comm);
void AdaptGraph2(GraphType *, int, MPI_Comm);
void Mc_AdaptGraph(GraphType *, idxtype *, int, int, MPI_Comm);

/* ptest.c */
void TestParMetis(char *, MPI_Comm);

/* NEW_ptest.c */
void TestParMetis_V3(char *, MPI_Comm);
int ComputeRealCut(idxtype *, idxtype *, char *, MPI_Comm);
int ComputeRealCut2(idxtype *, idxtype *, idxtype *, idxtype *, char *, MPI_Comm);
void TestMoveGraph(GraphType *, GraphType *, idxtype *, MPI_Comm);
GraphType *SetUpGraph(CtrlType *, idxtype *, idxtype *, idxtype *, idxtype *, idxtype *, int);

/* mienio.c */
void mienIO(MeshType *, char *, int, int, MPI_Comm);

/* meshio.c */
void ParallelReadMesh(MeshType *, char *, MPI_Comm);

/* parmetis.c */
void ChangeToFortranNumbering(idxtype *, idxtype *, idxtype *, int, int);

