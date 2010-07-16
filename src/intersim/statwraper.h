#ifndef STAT_WRAPER_H
#define STAT_WRAPER_H

void* StatCreate (const char * name, double bin_size, int num_bins) ;
void StatClear(void * st);
void StatAddSample (void * st, int val);
double StatAverage(void * st) ;
double StatMax(void * st) ;
double StatMin(void * st) ;
void StatDisp (void * st);
void StatDumptofile (void * st, FILE f);

#endif
