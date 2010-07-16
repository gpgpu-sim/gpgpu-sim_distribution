/***********************************************************************************
Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

Copyright (c) 2008 International Institute of Information Technology. 
All rights reserved.
  
Permission to use, copy, modify and distribute this software and its documentation for 
educational purpose is hereby granted without fee, provided that the above copyright 
notice and this permission notice appear in all copies of this software and that you do 
not sell the software.
  
THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
OTHERWISE.

Created by Pawan Harish.
************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>

#define MAX_THREADS_PER_BLOCK 256

int no_of_nodes;
int edge_list_size;
FILE *fp;

//Structure to hold a node information
struct Node
{
int starting;
int no_of_edges;
};

#include <kernel.cu>

void BFSGraph(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	no_of_nodes=0;
	edge_list_size=0;
    BFSGraph( argc, argv);
    //CUT_EXIT(argc, argv);
	return 0;
}



////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{

//    CUT_DEVICE_INIT();

	printf("Reading File\n");
	static char *input_file_name;
	//printf("argc=%d\n", argc);
	if (argc == 2 ) {
		input_file_name = argv[1];
		printf("Input file: %s\n", input_file_name);
	}
	else 
	{
		input_file_name = "SampleGraph.txt";
		printf("No input file specified, defaulting to SampleGraph.txt\n");
	}
	//Read in Graph from a file
	fp = fopen(input_file_name,"r");
	if(!fp)
	{
	printf("Error Reading graph file\n");
	return;
	}
	
	int source = 0;
		
	fscanf(fp,"%d",&no_of_nodes);
	
	int num_of_blocks = 1;
	int num_of_threads_per_block = no_of_nodes;
	
	//Make execution Parameters according to the number of nodes
	//Distribute threads across multiple Blocks if necessary
	if(no_of_nodes>MAX_THREADS_PER_BLOCK)
	{
	num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
	num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
	}
		
	// allocate host memory
    Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
    bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
    bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);
    
    int start, edgeno;   
    // initalize the memory
    for( unsigned int i = 0; i < no_of_nodes; i++) 
    {
		fscanf(fp,"%d %d",&start,&edgeno);
		h_graph_nodes[i].starting = start;
        h_graph_nodes[i].no_of_edges = edgeno;
        h_graph_mask[i]=false;
        h_graph_visited[i]=false;
    }
    
    //read the source node from the file
    fscanf(fp,"%d",&source);
         
    //set the source node as true in the mask
    h_graph_mask[source]=true;
      
    fscanf(fp,"%d",&edge_list_size);
     
    int id,cost;
    int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
    for(int i=0; i < edge_list_size ; i++)
    {
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i] = id;
    }
     
	if(fp)
	fclose(fp);    
	
	printf("Read File\n");

	//Copy the Node list to device memory
    Node* d_graph_nodes;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes) );
    CUDA_SAFE_CALL( cudaMemcpy( d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice) );

	//Copy the Edge List to device Memory
	int* d_graph_edges;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_graph_edges, sizeof(int)*edge_list_size) );
    CUDA_SAFE_CALL( cudaMemcpy( d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice) );
    
    //Copy the Mask to device memory
    bool* d_graph_mask;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_graph_mask, sizeof(bool)*no_of_nodes) );
    CUDA_SAFE_CALL( cudaMemcpy( d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) );
    
    //Copy the Visited nodes array to device memory
    bool* d_graph_visited;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_graph_visited, sizeof(bool)*no_of_nodes) );
    CUDA_SAFE_CALL( cudaMemcpy( d_graph_visited, h_graph_visited, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) );
    
    // allocate mem for the result on host side
	int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
	for(int i=0;i<no_of_nodes;i++)
	h_cost[i]=-1;
	h_cost[source]=0;

	// allocate device memory for result
    int* d_cost;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_cost, sizeof(int)*no_of_nodes));
    CUDA_SAFE_CALL( cudaMemcpy( d_cost, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice) );

    //make a bool to check if the execution is over
    bool *d_over;
    CUDA_SAFE_CALL( cudaMalloc( (void**) &d_over, sizeof(bool)));

	printf("Copied Everything to GPU memory\n");
    
    // setup execution parameters
    dim3  grid( num_of_blocks, 1, 1);
    dim3  threads( num_of_threads_per_block, 1, 1);

	//start the timer
    unsigned int timer = 0;
    float timer_acc = 0.0f;
    CUT_SAFE_CALL( cutCreateTimer( &timer));
    CUT_SAFE_CALL( cutStartTimer( timer));
	int k=0;
	
	bool stop;
	//Call the Kernel untill all the elements of Frontier are not false
    do
    {
	//if no thread changes this value then the loop stops
	stop=false;
	CUDA_SAFE_CALL( cudaMemcpy( d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice) );
    	CUT_SAFE_CALL( cutStartTimer( timer));
	Kernel<<< grid, threads, 0 >>>( d_graph_nodes, d_graph_edges, d_graph_mask, d_graph_visited, d_cost, d_over, no_of_nodes);
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	CUT_SAFE_CALL( cutStopTimer( timer));
	timer_acc += cutGetTimerValue(timer); 
	CUT_SAFE_CALL( cutResetTimer( timer));
	// check if kernel execution generated and error
    CUT_CHECK_ERROR("Kernel execution failed");
	CUDA_SAFE_CALL( cudaMemcpy( &stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost) );
    k++;
	}
    while(stop);
    
    
    printf("Kernel Executed %d times\n",k);

    // copy result from device to host
    CUDA_SAFE_CALL( cudaMemcpy( h_cost, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost) );

	//Stop the Timer
    CUT_SAFE_CALL( cutStopTimer( timer));
    //printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer));
    
    printf( "Processing time: %f (ms)\n", timer_acc);
    CUT_SAFE_CALL( cutDeleteTimer( timer));
    
	
	//Store the result into a file
	FILE *fpo = fopen("result.txt","w");
	for(int i=0;i<no_of_nodes;i++)
	fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	fclose(fpo);
	printf("Result stored in result.txt\n");
	
	
    // cleanup memory
    free( h_graph_nodes);
    free( h_graph_edges);
    free( h_graph_mask);
    free( h_graph_visited);
    free( h_cost);
    CUDA_SAFE_CALL(cudaFree(d_graph_nodes));
    CUDA_SAFE_CALL(cudaFree(d_graph_edges));
    CUDA_SAFE_CALL(cudaFree(d_graph_mask));
    CUDA_SAFE_CALL(cudaFree(d_graph_visited));
    CUDA_SAFE_CALL(cudaFree(d_cost));
}
