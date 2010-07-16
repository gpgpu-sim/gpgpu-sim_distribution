cudaEvent_t cstart, cstop;

void cudatic(){
  
  cudaEventCreate(&cstart);
  cudaEventCreate(&cstop);

  cudaEventRecord(cstart, 0); 
}

float cudatoc(){
  
  cudaEventRecord(cstop, 0); 
  cudaEventSynchronize(cstop); 
  float elapsedTime; 
  cudaEventElapsedTime(&elapsedTime, cstart, cstop);
  
  /* return elapsed time in seconds */
  return elapsedTime/1000.0;

}


