// a Wraper function for stats class
#include <stdio.h>
#include "intersim2/stats.hpp"

Stats *StatCreate(const char *name, double bin_size, int num_bins) {
  Stats *newstat = new Stats(NULL, name, bin_size, num_bins);
  newstat->Clear();
  return newstat;
}

void StatClear(void *st) { ((Stats *)st)->Clear(); }

void StatAddSample(void *st, int val) { ((Stats *)st)->AddSample(val); }

double StatAverage(void *st) { return ((Stats *)st)->Average(); }

double StatMax(void *st) { return ((Stats *)st)->Max(); }

double StatMin(void *st) { return ((Stats *)st)->Min(); }

void StatDisp(void *st) {
  printf("Stats for ");
  ((Stats *)st)->DisplayHierarchy();
  //   if (((Stats *)st)->NeverUsed()) {
  //      printf (" was never updated!\n");
  //   } else {
  printf("Min %f Max %f Average %f \n", ((Stats *)st)->Min(),
         ((Stats *)st)->Max(), StatAverage(st));
  ((Stats *)st)->Display();
  //   }
}

#if 0 
int main ()
{
   void * mytest = StatCreate("Test",1,5);
   StatAddSample(mytest,4);
   StatAddSample(mytest,4);StatAddSample(mytest,4);
   StatAddSample(mytest,2);
   StatDisp(mytest);
}
#endif
