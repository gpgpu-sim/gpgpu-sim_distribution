/*
 * Copyright 2008 BOROUJERDI Maxime. Tous droits reserves.
 */

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include "makebmp.h"

/*#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda_gl_interop.h>*/
#include <cutil.h>

typedef unsigned int uint;
typedef unsigned char uchar;

#define numObj 4

#define PI 3.141592654f
#define Angle(a) ((a*PI)/180.0)

//#define DEVICE_EMU
//#define DEBUG_RT_CUDA
#define FIXED_CONST_PARSE
#ifdef DEBUG_RT_CUDA
#define DEBUG_NUM 8
float4 *d_debug_float4;
uint *d_debug_uint;
float4 *h_debug_float4;
uint *h_debug_uint;
#endif
int g_verbose;

#include <rayTracing_kernel.cu>

unsigned width = 64; //640; //512; //16; //32; //512;
unsigned height = 64; //480; //512; //16;//512;
dim3 blockSize(16,8);
dim3 gridSize(width/blockSize.x, height/blockSize.y);

float3 viewRotation;
float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
float invViewMatrix[12];

//static int fpsCount = 0;        // FPS count for averaging
//static int fpsLimit = 1;        // FPS limit for sampling
unsigned int timer;


//GLuint pbo = 0;     // Pixel buffer d'OpenGL


void initPixelBuffer();

class Observateur
{
  private:
    matrice3x4  M;   // U, V, W
    float       df;  // distance focale
    
  public:
    Observateur( );
    Observateur(const float3 &, const float3 &, const float3 &, double );
    
	inline const matrice3x4 & getMatrice( ) const { return M; }
	inline float getDistance( ) const { return df; }
};

Observateur::Observateur()
{
  M.m[0] = make_float4(0.0f,0.0f,1.0f,0.0f);
  M.m[1] = make_float4(0.0f,1.0f,0.0f,0.0f);
  M.m[2] = make_float4(1.0f,0.0f,0.0f,0.0f);
  df = 1.0 / tan(Angle(65)/2.0);
}

Observateur::Observateur(const float3 & p, const float3 & u, const float3 & v, double a )
{
  float3 VP, U, V, W;
  VP = normalize(v);
  U = normalize(u);
  V = normalize(VP - dot(U,VP)*U);
  W = normalize(cross(U,V));
  M.m[0] = make_float4(U.x,U.y,U.z,p.x);
  M.m[1] = make_float4(V.x,V.y,V.z,p.y);
  M.m[2] = make_float4(W.x,W.y,W.z,p.z);
  df = 1.0 / tan(Angle(a)/2.0);
}

float anim = 0.0f, pas = 0.015f;
Observateur obs = Observateur(make_float3(0.0f,0.5f,2.0f),normalize(make_float3(0.0f,0.0f,0.0f)-make_float3(0.0f,0.5f,2.0f)),make_float3(0.0f,1.0f,0.0f),65.0f);;

uint * values = NULL, * d_output, * d_temp, NUM;
uint * c_output;

Node node[numObj], * d_node;

Sphere s, s1, s2;
float phi;

uint  * nObj;
float * prof;
Rayon * ray;
float3 * A, *u;
int t = 1;


void initObjet()
{
	srand(47);
	node->s.r = 1.0f;
	node[0].s.C = make_float3(0.0f,-1.5f,-0.0f); node[0].s.r = 0.5f;
	node[1].s.C = make_float3(-1.0f,0.0f,-1.0f); node[1].s.r = 0.5f;
	node[2].s.C = make_float3(1.0f,-0.f,-1.0f); node[2].s.r = 0.5f;
	node[3].s.C = make_float3(0.0f,-0.f,-2.0f); node[3].s.r = 0.75f;
	for( int i(4); i < numObj; i++ ) {
		float r,v,b;
		float tmp1(5.0f*((r=(float(rand()%255)/255.0f)))-2.5f);
		float tmp2(5.0f*((v=(float(rand()%255)/255.0f)))-2.5f);
		float tmp3(-5.0f*((b=(float(rand()%255)/255.0f))));
		float tmp4((rand()%100)/100.0f);
		node[i].s.C = make_float3(tmp1,tmp2,tmp3); node[i].s.r = tmp4;
		node[i].s.R = r; node[i].s.V = v; node[i].s.B = b; node[i].s.A = 1.0f;
		node[i].fg = 0; node[i].fd = 0;
	}
	node[0].s.R = 0.0f; node[0].s.V = 1.0f; node[0].s.B = 1.0f; node[0].s.A = 1.0f;
	node[1].s.R = 1.0f; node[1].s.V = 0.0f; node[1].s.B = 0.0f; node[1].s.A = 1.0f;
	node[2].s.R = 0.0f; node[2].s.V = 0.0f; node[2].s.B = 1.0f; node[2].s.A = 1.0f;
	node[3].s.R = 0.0f; node[3].s.V = 1.0f; node[3].s.B = 0.0f; node[3].s.A = 1.0f;
	//createNode(&node[0], &node[1], &node[2], 1.0f);
	node[0].fg = 1;	node[0].fd = 2;
	node[1].fg = 0; node[1].fd = 0;
	node[2].fg = 0; node[2].fd = 0;
	node[3].fg = 0; node[3].fd = 0;

   #ifdef DEBUG_RT_CUDA
   h_debug_float4 = (float4*) calloc(DEBUG_NUM, sizeof(float4));
   h_debug_uint = (uint*) calloc(DEBUG_NUM, sizeof(uint));
   CUDA_SAFE_CALL( cudaMalloc( (void**)&d_debug_float4, DEBUG_NUM*sizeof(float4)));
   CUDA_SAFE_CALL( cudaMalloc( (void**)&d_debug_uint, DEBUG_NUM*sizeof(uint)));
   CUDA_SAFE_CALL( cudaMemcpy( d_debug_float4, h_debug_float4, DEBUG_NUM*sizeof(float4), cudaMemcpyHostToDevice) );
   CUDA_SAFE_CALL( cudaMemcpy( d_debug_uint, h_debug_uint, DEBUG_NUM*sizeof(uint), cudaMemcpyHostToDevice) );
   #endif
   c_output = (uint*) calloc(width*height, sizeof(uint));
   CUDA_SAFE_CALL( cudaMalloc( (void**)&d_output, width*height*sizeof(uint)));

    CUDA_SAFE_CALL( cudaMalloc( (void**)&d_node, numObj*sizeof(Node) ));
    CUDA_SAFE_CALL( cudaMemcpy( d_node, node, numObj*sizeof(Node), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(cnode, node, numObj*sizeof(Node)) );
    CUDA_SAFE_CALL( cudaMemcpyToSymbol(MView, (void*)&obs, 3*sizeof(float4)) );	
	CUDA_SAFE_CALL( cudaMalloc( (void**)&d_temp, width * height*sizeof(uint)));
	CUDA_SAFE_CALL( cudaMemset(d_temp, 0, width * height*sizeof(uint)) );
	
	CUDA_SAFE_CALL( cudaMalloc( (void**)&nObj, width * height*sizeof(uint)));
	CUDA_SAFE_CALL( cudaMalloc( (void**)&prof, width * height*sizeof(float)));
	CUDA_SAFE_CALL( cudaMalloc( (void**)&ray, width * height*sizeof(Rayon)));
	
	CUDA_SAFE_CALL( cudaMalloc( (void**)&A, width * height*sizeof(float3)));
	CUDA_SAFE_CALL( cudaMalloc( (void**)&u, width * height*sizeof(float3)));
}

#define PRINT_PIXELS

// Rendu de l'image avec CUDA
void render()
{
    // map PBO to get CUDA device pointer <GY: replace with memcpy?>
    //CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&d_output, pbo));
    //CUDA_SAFE_CALL( cudaMemcpy( d_output, c_output, width*height*sizeof(uint), cudaMemcpyHostToDevice) );
    // call CUDA kernel, writing results to PBO
    CUT_SAFE_CALL(cutStartTimer(timer)); 
    #ifdef DEBUG_RT_CUDA
    render<<<gridSize, blockSize>>>(d_debug_float4, d_debug_uint, d_output, d_node, width, height, anim, obs.getDistance());
    #else
    render<<<gridSize, blockSize>>>(d_output, d_node, width, height, anim, obs.getDistance());
    #endif
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL(cutStopTimer(timer));

    #ifdef DEBUG_RT_CUDA
    CUDA_SAFE_CALL( cudaMemcpy( h_debug_float4, d_debug_float4, DEBUG_NUM*sizeof(float4), cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaMemcpy( h_debug_uint, d_debug_uint, DEBUG_NUM*sizeof(uint), cudaMemcpyDeviceToHost) );

    printf("debug_float4\n");
    for (int i=0; i< DEBUG_NUM; i++) {
       printf("%e %e %e %e\n", h_debug_float4[i].x, h_debug_float4[i].y, h_debug_float4[i].z, h_debug_float4[i].w);
    }
    printf("debug_uint\n");
    for (int i=0; i< DEBUG_NUM; i++) {
       printf("0x%x\n", h_debug_uint[i]);
    }
    #endif

    CUDA_SAFE_CALL( cudaMemcpy( c_output, d_output, width*height*sizeof(uint), cudaMemcpyDeviceToHost) );
    unsigned long long int checksum = 0;
    for (int y=(height-1); y >= 0; y--){
       if (g_verbose) printf("\n");
       for  (int x=0; x< width; x++) {
          if (g_verbose) printf("%010u ", (unsigned) c_output[x+y*width]);
          checksum += c_output[x+y*width];
       }
    }
    printf("\n");
    printf("checksum=%llx\n", checksum);
    CUT_CHECK_ERROR("Erreur kernel");

    //CUDA_SAFE_CALL(cudaGLUnmapBufferObject(pbo)); //<GY: replace with memcpy?>

}

// Affichage du resultat avec OpenGL
void display()
{

    // Affichage du resultat
    //glClear(GL_COLOR_BUFFER_BIT);

    //CUT_SAFE_CALL(cutStartTimer(timer)); 
	 render();
    //CUT_SAFE_CALL(cutStopTimer(timer));
	 printf("Kernel Time: %f \n", cutGetTimerValue(timer));
	/*fpsCount++;
	if (fpsCount == fpsLimit) {
		char fps[256];
		float ifps = 1.f / (cutGetAverageTimerValue(timer) / 1000.f);
		sprintf(fps, "Cuda Ray Tracing: %.1f fps", ifps);  
		glutSetWindowTitle(fps);
		fpsCount = 0; 
		fpsLimit = (int)max(ifps, 1.f);
		CUT_SAFE_CALL(cutResetTimer(timer));
	}*/

	if( anim >= 1.0f ) pas = -0.015f;
	else if( anim <= -1.0f ) pas = 0.015f;
	anim += pas;

    // Dessin de l'image de PBO
    /*glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glutSwapBuffers();
    glutReportErrors();*/
    t--;
    if (!t) {
       return;
    }

}

/*void idle()
{
	glutPostRedisplay();
}

void keyboard(unsigned char , int , int )
{
	//glutPostRedisplay();
}*/

int ox, oy;
int buttonState = 0;

/*void mouse(int , int , int , int )
{
    if (state == GLUT_DOWN)
        buttonState |= 1<<button;
    else if (state == GLUT_UP)
        buttonState = 0;

    ox = x; oy = y;
    glutPostRedisplay();
}

void motion(int , int )
{
    float dx, dy;
    dx = x - ox;
    dy = y - oy;

    if (buttonState == 3) {
        // left+middle = zoom
        viewTranslation.z += dy / 100.0;
    } 
    else if (buttonState & 2) {
        // middle = translate
        viewTranslation.x += dx / 100.0;
        viewTranslation.y -= dy / 100.0;
    }
    else if (buttonState & 1) {
        // left = rotate
        viewRotation.x += dy / 5.0;
        viewRotation.y += dx / 5.0;
    }

    ox = x; oy = y;
    glutPostRedisplay();
}

void reshape(int x, int y)
{
    width = x; height = y;
    initPixelBuffer();

    glViewport(0, 0, x, y);
    //glViewport(-x/2, -y/2, x/2, y/2);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 
}

void cleanup()
{
	CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(pbo));    
	glDeleteBuffersARB(1, &pbo);
    CUT_SAFE_CALL(cutDeleteTimer(timer));  
}*/

int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

void initPixelBuffer()
{
    /*if (pbo) {
        // delete old buffer
        CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(pbo));
        glDeleteBuffersARB(1, &pbo);
    }*/

	NUM = width * height;
	phi = 2.0f/(float)min(width,height);

    // create pixel buffer object for display
   /* glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width*height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	CUDA_SAFE_CALL(cudaGLRegisterBufferObject(pbo));*/

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
}


////////////////////////////////////////////////////////////////////////////////
// Programme principal
////////////////////////////////////////////////////////////////////////////////



int main( int argc, char** argv) 
{
  // initialise card and timer
  int deviceCount;                                                         
  CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceCount(&deviceCount));                
  if (deviceCount == 0) {                                                  
      fprintf(stderr, "There is no device.\n");                            
      exit(EXIT_FAILURE);                                                  
  }                                                                        
  int dev;                                                                 
  for (dev = 0; dev < deviceCount; ++dev) {                                
      cudaDeviceProp deviceProp;                                           
      CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceProperties(&deviceProp, dev));   
      if (deviceProp.major >= 1)                                           
          break;                                                           
  }                                                                        
  if (dev == deviceCount) {                                                
      fprintf(stderr, "There is no device supporting CUDA.\n");            
      exit(EXIT_FAILURE);                                                  
  }                                                                        
  else                                                                     
      CUDA_SAFE_CALL(cudaSetDevice(dev));  
	int i, commandline_error;
	commandline_error = 0;
	g_verbose = 0;
	if (argc >= 3) {
		width = atoi(argv[1]);
        height = atoi(argv[2]);
		for (i=3; i < argc;i++) {
			if (argv[i][0] == '-') {
				switch (argv[i][1]) {
				case 'v': g_verbose = 1;
					break;
				default: commandline_error=1;
				}
			}
			else commandline_error=1;
		}
	} else commandline_error=1;

	if (commandline_error || !width || !height) {
		printf("Usage: ./rayTracing <WIDTH> <HEIGHT> [-v]\n");
		printf("where WIDTH and HEIGHT are the screen dimensions and -v is used to display an abstract representation of the output.\n");
		return 1;
	}
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    CUT_SAFE_CALL(cutResetTimer(timer));  

    initialize_bmp(width,height,32);

    // initialise les functions callback de GLUT
    /*glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("CUDA Ray Tracing");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object")) {
        fprintf(stderr, "Les extensions minimales d'OpenGL sont absentes.");
        exit(-1);
    }
    initPixelBuffer();
	 initObjet();

    atexit(cleanup);

    glutMainLoop();*/
    initObjet();
    initPixelBuffer();
    display();
    create_bmp(c_output);
    CUT_SAFE_CALL(cutDeleteTimer(timer)); 
    return 0;
}
