#ifndef CUDA_SOLVER_CUH
#define CUDA_SOLVER_CUH

#pragma once

#include "Constants.hpp"
#include "support/CudaHelper.cuh"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>

#include "vec2.h"
#include "vector"

#include "support/tinycolormap.hpp"



using namespace std;

class Cuda_solver
{
public:
    Cuda_solver();
    void CudaInit();


    gpuParams* h_params;

    // general Particle properties
    float  yield=0.08f;
    float  stiffness= 0.18f;
    float  nearStiffness= 0.01f;
    float  linearViscocity =0.5f;
    float  quadraticViscocity= 1.f;

    float temp_increase=0.05f;
    float temp_decrease=0.005f;
    float gravity_coeff_max=11.0f;
    float gravity_coeff_min=3.0f;

    Source sources[4] =
    {
        Source(redParticle,     Vec2(0.05f*viewWidth, 0.43f*viewHeight), Vec2( 1.00f, 0.0f), 1.5f),
        Source(blueParticle,    Vec2(0.95f*viewWidth, 0.5f*viewHeight),  Vec2(-1.25f, 0.0f), 1.5f),
        Source(greenParticle,   Vec2(0.15f*viewWidth, 0.15f*viewHeight), Vec2(-1.00f, 0.0f), 1.5f),
        Source(yellowParticle,  Vec2(0.15f*viewWidth, 0.75f*viewHeight), Vec2( 1.00f, 0.0f), 1.5f),
    };

    //world boundaries
    Boundary boundaries[4] =
    {
        Boundary(1, 0, 0),
        Boundary(0, 1, 0),
        Boundary(-1, 0, -viewWidth),
        Boundary(0, -1, -viewHeight)
    };
    bool activeSpout = false;

    //Host arrays
    Particle* h_particles;//[maxParticles];
    Particle* h_artparticles;//[maxParticles];
   // Springs* h_neighbours;//[maxParticles];
    float2* h_prevPos;//[maxParticles];
    ParticleType* h_particleTypes;//[maxParticles];
    Vec4* h_savedParticleColors;//[maxParticles];
    Vec4* h_artParticleColors;//[maxParticles];
    int* h_map;
    int* h_map_size;
    int2* h_mapCoords;//[maxParticles][2];
    float3* h_boundaries;

    int* h_neightb_size;

    //Device arrays
    Particle* d_particles;//[maxParticles];
    Particle* d_artparticles;
   // Springs* d_neighbours;//[maxParticles];
    float2* d_prevPos;//[maxParticles];
    ParticleType* d_particleTypes;//[maxParticles];
    Vec4* d_savedParticleColors;//[maxParticles];
    int* d_map;
    int* d_map_size;

    int* d_neightb_index;
    int* d_neightb_size;
    float* d_neightb_r;
    float* d_neightb_Lij;

    int2* d_mapCoords;//[maxParticles][2];

    float3* d_boundaries;

    float2 gravity;
    int delay = 0;
    int artPixelGroup=0;
    bool showDigits=true;

    int colorMapNumbers=254;
    int* h_colorsMap;
    int* d_colorsMap;

    Vec4* h_colorsMapVec4;
    Vec4* d_colorsMapVec4;

    PosColorLayout* h_particlesDraw;
    GLuint* h_particlesIndexDraw;


    bool Update();
    void generateParticles();

    void UpdateGPUBuffers();
    void UpdateHostBuffers();
    void applyTemp();
    void applyGravity();
    void applyViscosity();
    void advance();
    void adjustSprings();
    void updateMap();
    void storeNeighbors();
    void doubleDensityRelaxation();
    void computeNextVelocity();
    void resolveCollisions();
    void adjustColor();


/*

    bool addParticles(int x, int y);

*/

    void ClearMap();
    void UpdateGPUBuffersSmall();
    bool UpdateActivePixels();
    bool isActivePixel(int index);

    void ArtPixelRender(int height, int column);
    void TexColor(void *devPtr, int m_texWidth, int m_texHeight);
    void ClearTex(void *devPtr, int m_texWidth, int m_texHeight);
    void RegisterGLTextureForCUDA(GLuint *gl_tex, cudaGraphicsResource **cuda_tex, unsigned int size_x, unsigned int size_y);
    void renderParticles(PosColorLayout *vbo, int m_texWidth, int m_texHeight);
    void PerlinTex(void *devPtr, int m_texWidth, int m_texHeight);
protected:

};

static const unsigned int tpb=256;



__global__ void CudaApplyTemp(Particle* particles, Particle* artparticles);
__global__ void CudaApplyGravity(Particle* particles);
__global__ void CudaApplyViscosity(Particle* particles,int* neightb_index, int* neightb_size, float* neightb_r, float* neightb_Lij);
__global__ void CudaAdvance(Particle* particles, float2* prevPos);
__global__ void CudaAdjustSprings(Particle* particles,int* neightb_index, int* neightb_size, float* neightb_r, float* neightb_Lij);

__global__ void CudaUpdateMap(Particle* particles, int* map, int *map_size, int2* mapCoords);

__global__ void CudaStoreNeighbors(Particle* particles, int* map, int *map_size, int2* mapCoords,
                                   int* neightb_index, int* neightb_size, float* neightb_r, float* neightb_Lij);
__global__ void CudaDoubleDensityRelaxation(Particle* particles, int* neightb_index, int* neightb_size, float* neightb_r, float* neightb_Lij);
__global__ void CudaComputeNextVelocity(Particle* particles, float2* prevPos);
__global__ void CudaResolveCollisions(Particle* particles, float3* boundaries);
__global__ void CudaAdjustColor(Particle* particles, Vec4* savedParticleColors, ParticleType* particleTypes);

__global__ void CudaClearMap(int* map, int *map_size);

__global__ void CudaClearTex(uchar4 *ptr, int max_w, int max_h);
__global__ void CudaTexColor(uchar4 *ptr, int max_w, int max_h, Particle* particles, ParticleType* particleTypes, Vec4* savedParticleColors,  Vec4* d_colorMap);
__global__ void CudaRenderParticles(PosColorLayout *vbo, int max_w, int max_h, Particle* particles, ParticleType* particleTypes,Vec4* savedParticleColors,  Vec4* d_colorMap);


__global__ void CudaPerlinTex(uchar4 *ptr, int max_w, int max_h);

#endif // CUDA_SOLVER_CUH
