#include "cuda_solver.cuh"

__constant__  gpuParams params;

__device__ int GetIndex(int x, int y)
{
    return y*params.mapWidth+x;
}

__host__ __device__ int scrIndex(const int x, const  int y, const int scrWidth)
{
    return y*scrWidth+x;
}


__device__ int GetIndexMap(int i,int x, int y)
{
    return  i*params.maxMapCoeff1+y*params.mapWidth+x;
}

__device__ int GetIndexNeightb(int i, int j)
{
    return  i*maxSprings+j;
}


__host__ bool Cuda_solver::isActivePixel(int index)
{
    return false;
}

__host__ __device__ inline unsigned cRGB(unsigned int rc, unsigned int gc, unsigned int bc, unsigned int ac)
    {
        const unsigned char r = ac;
        const unsigned char g = bc;
        const unsigned char b = gc;
        const unsigned char a = rc ;
        return (r << 24) |  (g << 16) |  (b << 8)| a ;
    }


__device__ inline unsigned int mapPosCorrection(int x, int y, unsigned int width, unsigned int height)
{
    unsigned int xf=(x+width)  % width;
    unsigned int yf=(y+height) % height;
    return yf*height+xf;
}

__device__ inline unsigned int mapPosCorrection(int x, int y)
{
    unsigned int xf=(x+params.mapWidth) % params.mapWidth;
    unsigned int yf=(y+params.mapHeight) % params.mapHeight;
    return yf*params.mapHeight+xf;
}

__device__ inline unsigned int mapOnePosCorrection(const int x, const int scr_lenth)
{
    unsigned int xf=(x+scr_lenth) % scr_lenth;
    return xf;
}

__device__ inline unsigned int mapPosCorrectionX(const int x)
{
    unsigned int xf=(x+params.mapWidth) % params.mapWidth;
    return xf;
}

__device__ inline unsigned int mapPosCorrectionY(const int y)
{
    unsigned int yf=(y+params.mapHeight) % params.mapHeight;
    return yf;
}

__device__ float TDist(float x1, float y1, float x2, float y2)
{
    float dx = fabsf(x2 - x1);
    float dy = fabsf(y2 - y1);

    if (dx > params.vWidth2)
        dx = params.vWidth - dx;

    if (dy > params.vHeight2)
        dy = params.vHeight - dy;

    return sqrtf(dx*dx + dy*dy);
}

__device__ float TdeltaX(float x1, float x2)
{
    float dxt=x1-x2;
    float dx = fabsf(dxt);
    if (dx >= params.vWidth2 and x1>=0 and x2>=0)
    {
        if(x1>x2)
            dxt=-(params.vWidth - dx);
        else
            dxt=(params.vWidth - dx);
    }
    return dxt;
}

__device__ float TdeltaY(float y1, float y2)
{
    float dyt=y1-y2;
    float dy = fabsf(dyt);
    if (dy >= params.vWidth2 and y1>=0 and y2>=0)
    {
        if(y1>y2)
            dyt=-(params.vHeight - dy);
        else
            dyt=(params.vHeight - dy);
    }
    return dyt;
}

Cuda_solver::Cuda_solver()
{

         CudaInit();
        // params init

        h_params=new gpuParams;

        h_params->yield=0.08f;
        h_params->stiffness= 0.18f;
        h_params->nearStiffness= 0.01f;
        h_params->linearViscocity =0.5f;
        h_params->quadraticViscocity= 1.0f;

        h_params->alphaSpring=0.3f;
        h_params->kSpring =0.3f;

        h_params->temp_increase=1.0f;
        h_params->temp_decrease=0.0f;
        h_params->gravity_coeff_max=0.0f;
        h_params->gravity_coeff_min=0.0f;

        h_params->totalParticles=0;

        gravity.x=0.0f;
        gravity.y=0.0f;
        h_params->gravity.x=gravity.x;
        h_params->gravity.y=gravity.y;

        h_params->maxMapCoeff1=mapWidth*mapHeight;
        h_params->maxMapCoeff2=mapWidth*mapHeight;

        h_params->mapWidth=mapWidth;
        h_params->mapHeight=mapHeight;

        h_params->restDensity=75.0f;
        h_params->surfaceTension=0.0006f;

        h_params->tick=0;
        h_params->tickLoop=0;

        h_params->scrHeight=12;
        h_params->scrWidth=19;
        h_params->scrRatio=screenWidth / viewWidth;

        h_params->pixelRadius=0.7f;
        h_params->pixelDiametr=2.0f*h_params->pixelRadius;

        h_params->number_artp=h_params->scrHeight*h_params->scrWidth;
        h_params->stepDelay=35;

        h_params->radiusViscocity=0.995f;

        h_params->drawArtPoint=true;
        h_params->drawListSet=0;//DrawList::simpleColor;
        h_params->artPixelTransperency=0.35;
        h_params->magnetRadiusCoeff=2.5f;

        h_params->timeDelayRow=500;
        h_params->timeDelayDigit=1500;

        h_params->vHeight=viewHeight;
        h_params->vWidth=viewWidth;

        h_params->vHeight2=viewHeight/2.0f;
        h_params->vWidth2=viewWidth/2.0f;

        h_particles=(Particle*)malloc(sizeof(Particle)*maxParticles);

        h_artparticles=(Particle*)malloc(sizeof(Particle)*h_params->number_artp);
        //h_neighbours=(Springs*)malloc(sizeof(Springs)*maxParticles);
        h_prevPos=(float2*)malloc(sizeof(float2)*maxParticles);

        h_particleTypes=(ParticleType*)malloc(sizeof(ParticleType)*maxParticles);

        h_savedParticleColors=(Vec4*)malloc(sizeof(Vec4)*maxParticles);

        int sizeBuffer=sizeof(int)*mapWidth*mapHeight*maxParticles;

        h_map=(int*)malloc(sizeof(int)*mapWidth*mapHeight*maxParticles);
        h_map_size=(int*)malloc(sizeof(int)*mapWidth*mapHeight);

        h_mapCoords=(int2*)malloc(sizeof(int2)*maxParticles);
        h_boundaries=(float3*)malloc(sizeof(float3)*4);

        h_neightb_size=(int*)malloc(sizeof(int)*maxParticles);


        checkCudaCall(cudaMalloc(&d_particles,sizeof(Particle)*maxParticles));
        checkCudaCall(cudaMalloc(&d_artparticles,sizeof(Particle)*h_params->number_artp));

        checkCudaCall(cudaMalloc(&d_prevPos,sizeof(float2)*maxParticles));
        checkCudaCall(cudaMalloc(&d_particleTypes,sizeof(ParticleType)*maxParticles));
        checkCudaCall(cudaMalloc(&d_savedParticleColors,sizeof(Vec4)*maxParticles));


        checkCudaCall(cudaMalloc(&d_map,sizeof(int)*mapWidth*mapHeight*maxParticles));
        checkCudaCall(cudaMalloc(&d_map_size,sizeof(int)*mapWidth*mapHeight));

        checkCudaCall(cudaMalloc(&d_mapCoords,sizeof(int2)*maxParticles));
        checkCudaCall(cudaMalloc(&d_boundaries,sizeof(float3)*4));

        gravity=make_float2(0.0f, 0.0f);

        for(int i=0;i<4;i++)
        {
            h_boundaries[i].x=boundaries[i].x;
            h_boundaries[i].y=boundaries[i].y;
            h_boundaries[i].z=boundaries[i].c;

        }


        checkCudaCall(cudaMalloc(&d_neightb_index,sizeof(int)*maxSprings*maxParticles));
        checkCudaCall(cudaMalloc(&d_neightb_size,sizeof(int)*maxParticles));
        checkCudaCall(cudaMalloc(&d_neightb_r,sizeof(float)*maxSprings*maxParticles));
        checkCudaCall(cudaMalloc(&d_neightb_Lij,sizeof(float)*maxSprings*maxParticles));

        memset(h_map, 0, mapWidth*mapHeight*sizeof(int));
        memset(h_neightb_size, 0, maxParticles* sizeof(int));
        checkCudaCall(cudaMemcpyToSymbol(params,h_params,sizeof(gpuParams)));


        h_colorsMap=(int*)malloc(sizeof(int)*colorMapNumbers);
        h_colorsMapVec4=(Vec4*)malloc(sizeof(Vec4)*colorMapNumbers);
        for(int i=0; i<colorMapNumbers;i++)
        {
            float value=(float)i/colorMapNumbers;
            const tinycolormap::Color color = tinycolormap::GetColor(value, tinycolormap::ColormapType::Viridis);
            // h_colorsMap[i]=complementRGB(unsigned(colorMapNumbers*color.b()) << 16) | (unsigned(colorMapNumbers*color.g()) << 8) | unsigned(colorMapNumbers*color.r());
            h_colorsMap[i]=cRGB( unsigned(colorMapNumbers*color.r()),
                                 unsigned(colorMapNumbers*color.g()),
                                 unsigned(colorMapNumbers*color.b()),
                                 254);

            h_colorsMapVec4[i].a=1.0f;
            h_colorsMapVec4[i].r=color.r();
            h_colorsMapVec4[i].g=color.g();
            h_colorsMapVec4[i].b=color.b();

        }
        cudaMalloc(&d_colorsMap,sizeof(unsigned)*colorMapNumbers);
        cudaMalloc(&d_colorsMapVec4,sizeof(Vec4)*colorMapNumbers);
        checkCudaCall(cudaMemcpy(d_colorsMap, h_colorsMap, sizeof(unsigned)*colorMapNumbers, cudaMemcpyHostToDevice));
        checkCudaCall(cudaMemcpy(d_colorsMapVec4, h_colorsMapVec4, sizeof(Vec4)*colorMapNumbers, cudaMemcpyHostToDevice));

        h_particlesDraw=(PosColorLayout*)malloc(sizeof(PosColorLayout)*maxParticles);
        h_particlesIndexDraw=(GLuint*)malloc(sizeof(GLuint)*maxParticles);

        UpdateGPUBuffers();
        ClearMap();
        updateMap();

}



void Cuda_solver::CudaInit()
{
    int devID = gpuGetMaxGflopsDeviceId();

    int devicesCount;
    cudaGetDeviceCount(&devicesCount);
    int deviceIndex =0;
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, deviceIndex);
        cudaDeviceAttr cattr;
        //int* valueAtrr;
        //cudaDeviceGetAttribute(valueAtrr, cattr,1 );

        if (deviceProperties.major >= 2
            && deviceProperties.minor >= 0)
        {
            checkCudaErrors(cudaSetDevice(deviceIndex));

        }
    }


   // checkCudaErrors(cudaSetDevice(devID));
    int major = 0, minor = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
    checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
           devID, _ConvertSMVer2ArchName(major, minor), major, minor);
}


void Cuda_solver::generateParticles(){
    if (h_params->totalParticles == maxParticles)
        return;
    if (delay++ < 2)
        return;

    for (int turn = 0; turn<4; turn++){
        Source& source = sources[turn];
        if (source.count >= maxParticles ) continue;

        for (int i = 0; i <=9 && h_params->totalParticles<maxParticles; i++){
            Particle& p = h_particles[h_params->totalParticles];
            ParticleType& pt = h_particleTypes[h_params->totalParticles];

            PosColorLayout& pdraw=h_particlesDraw[h_params->totalParticles];
            h_particlesIndexDraw[h_params->totalParticles]=h_params->totalParticles;

            h_params->totalParticles++;

            source.count++;

            //for an even distribution of particles
            float offset = float(i) / 1.5f;
            offset *= 0.2f;
            p.posX = source.position.x - offset*source.direction.y;
            p.posY = source.position.y + offset*source.direction.x;

            if(turn==0)
                p.posY +=h_params->temp_increase;
            p.velX = 2.97f*source.speed *source.direction.x;
            p.velY = 2.97f*source.speed *source.direction.y;
            p.m = source.pt.mass;
            p.temp=0;

            pt = source.pt;

            pdraw.m_pos[0]=p.posX;
            pdraw.m_pos[1]=p.posY;
            pdraw.m_pos[2]=0;
        }
    }
    delay = 0;

    checkCudaCall(cudaMemcpyToSymbol(params,h_params,sizeof(gpuParams)));
    UpdateGPUBuffers();
}

void Cuda_solver::UpdateGPUBuffers()
{
    //checkCudaCall(cudaMemcpyToSymbol(params,h_params,sizeof(gpuParams)));

    checkCudaCall(cudaMemcpy(d_boundaries,h_boundaries,sizeof(float3)*4,cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(d_particles,h_particles,sizeof(Particle)*maxParticles,cudaMemcpyHostToDevice));

    checkCudaCall(cudaMemcpy(d_artparticles,h_artparticles,sizeof(Particle)*h_params->number_artp,cudaMemcpyHostToDevice));

    checkCudaCall(cudaMemcpy(d_neightb_size,h_neightb_size,sizeof(int)*maxParticles,cudaMemcpyHostToDevice));
    //checkCudaCall(cudaMemcpy(d_neighbours,h_neighbours,sizeof(Springs)*maxParticles,cudaMemcpyHostToDevice));
   // checkCudaCall(cudaMemcpy(d_prevPos,h_prevPos,sizeof(float2)*maxParticles,cudaMemcpyHostToDevice));

    checkCudaCall(cudaMemcpy(d_particleTypes,h_particleTypes,sizeof(ParticleType)*maxParticles,cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(d_savedParticleColors,h_savedParticleColors,sizeof(int4)*maxParticles,cudaMemcpyHostToDevice));

   // checkCudaCall(cudaMemcpy(d_map,h_map,sizeof(int)*mapWidth*mapHeight,cudaMemcpyHostToDevice));
   // checkCudaCall(cudaMemcpy(d_mapCoords,h_mapCoords,sizeof(int2)*mapWidth*mapHeight,cudaMemcpyHostToDevice));

}


void Cuda_solver::UpdateHostBuffers()
{

    checkCudaCall(cudaMemcpy(h_particles,d_particles,sizeof(Particle)*maxParticles,cudaMemcpyDeviceToHost));
  //  checkCudaCall(cudaMemcpy(h_neighbours,d_neighbours,sizeof(Springs)*maxParticles,cudaMemcpyDeviceToHost));
  //  checkCudaCall(cudaMemcpy(h_prevPos,d_prevPos,sizeof(float2)*maxParticles,cudaMemcpyDeviceToHost));

  //  checkCudaCall(cudaMemcpy(h_particleTypes,d_particleTypes,sizeof(ParticleType)*maxParticles,cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(h_savedParticleColors,d_savedParticleColors,sizeof(int4)*maxParticles,cudaMemcpyDeviceToHost));
  //  checkCudaCall(cudaMemcpy(h_map_size,d_map_size,sizeof(int)*mapWidth*mapHeight,cudaMemcpyDeviceToHost));
  //  checkCudaCall(cudaMemcpy(h_map,d_map,sizeof(int)*mapWidth*mapHeight*maxParticles,cudaMemcpyDeviceToHost));
  //  checkCudaCall(cudaMemcpy(h_mapCoords,d_mapCoords,sizeof(int*)*mapWidth*mapHeight,cudaMemcpyDeviceToHost));

}



//2-Dimensional gravity for player input
void Cuda_solver::applyTemp(){

    if(h_params->totalParticles>1)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid =
                (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaApplyTemp<<<blocksPerGrid, threadsPerBlock>>>(d_particles, d_artparticles);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}

__global__ void CudaApplyTemp(Particle* particles, Particle* artparticles)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        Particle& p = particles[i];
        p.temp=0;
        p.magnetX=0.0f;
        p.magnetY=0.0f;

    }
}

//2-Dimensional gravity for player input
void Cuda_solver::applyGravity()
{
    if(h_params->totalParticles>0)
    {
         int threadsPerBlock = 256;
         int blocksPerGrid =
                 (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaApplyGravity<<<blocksPerGrid, threadsPerBlock>>>(d_particles);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}

__global__ void CudaApplyGravity(Particle* particles)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        Particle& p = particles[i];
        p.velX += params.gravity.x*deltaTime;
        p.velY += params.gravity.y*deltaTime;//+p.temp*deltaTime+p.magnetX*deltaTime;
    }
}

//applies viscosity impulses to particles
void Cuda_solver::applyViscosity()
{
     if(h_params->totalParticles>0)
    {
         int threadsPerBlock = 256;
         int blocksPerGrid =
                 (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaApplyViscosity<<<blocksPerGrid, threadsPerBlock>>>(d_particles,
                                                               d_neightb_index, d_neightb_size, d_neightb_r, d_neightb_Lij);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}

__global__ void CudaApplyViscosity(Particle* particles,
                                   int* neightb_index, int* neightb_size, float* neightb_r, float* neightb_Lij)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        Particle& p = particles[i];

        for (int j = 0; j < neightb_size[i]; j++){
            const Particle& pNear = particles[neightb_index[GetIndexNeightb(i,j)]];

            float diffX = TdeltaX(pNear.posX, p.posX);
            float diffY = TdeltaY(pNear.posY, p.posY);

            float r2 = diffX*diffX + diffY*diffY;
            float r = sqrtf(r2);

            float q = r / particleHeight;

            if (q>1) continue;

            float diffVelX =p.velX-pNear.velX;
            float diffVelY =p.velY-pNear.velY;
            float u = diffVelX*diffX + diffVelY*diffY;

            if (u > 0){
                float a = 1 - q;
                diffX /= r;
                diffY /= r;
                u /= r;

                float I = 0.5f * deltaTime * a * (params.linearViscocity*u + params.quadraticViscocity*u*u);

                particles[i].velX -= I * diffX;
                particles[i].velY -= I * diffY;
            }
        }

    }
}

//Advances particle along its velocity
void Cuda_solver::advance()
{
     if(h_params->totalParticles>0)
    {
         int threadsPerBlock = 256;
         int blocksPerGrid =
                 (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaAdvance<<<blocksPerGrid, threadsPerBlock>>>(d_particles, d_prevPos);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}


__global__ void CudaAdvance(Particle* particles, float2* prevPos)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        Particle& p = particles[i];

        prevPos[i].x = p.posX;
        prevPos[i].y = p.posY;

        p.posX = p.posX+deltaTime * p.velX;
        p.posY = p.posY+deltaTime * p.velY;

    }
}


void Cuda_solver::adjustSprings()
{
     if(h_params->totalParticles>0)
    {
         int threadsPerBlock = 256;
         int blocksPerGrid =
                 (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaAdjustSprings<<<blocksPerGrid, threadsPerBlock>>>(d_particles,
                                                              d_neightb_index, d_neightb_size, d_neightb_r, d_neightb_Lij);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}

__global__ void CudaAdjustSprings(Particle* particles,
                                  int* neightb_index, int* neightb_size, float* neightb_r, float* neightb_Lij)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        Particle& p = particles[i];
        //iterate through that particles neighbors
        for (int j = 0; j < neightb_size[i]; j++){
            const Particle& pNear =  particles[neightb_index[GetIndexNeightb(i,j)]];

            float r = neightb_r[GetIndexNeightb(i,j)];
            float q = r / particleHeight;

            if (q <= 1.0f && q > 0.0000000000001f){
                float d = params.yield*neightb_Lij[GetIndexNeightb(i,j)];

                //calculate spring values
                if (r>particleHeight + d){
                    neightb_Lij[GetIndexNeightb(i,j)]+= deltaTime*params.alphaSpring*(r - particleHeight - d);
                }
                else if (r<particleHeight - d){
                    neightb_Lij[GetIndexNeightb(i,j)]-= deltaTime*params.alphaSpring*(particleHeight - d - r);
                }

                //apply those changes to the particle
                float Lij = neightb_Lij[GetIndexNeightb(i,j)];
                float diffX = TdeltaX(pNear.posX, p.posX);
                float diffY = TdeltaY(pNear.posY, p.posY);
                float displaceX = deltaTime*deltaTime*params.kSpring*(1 - Lij / particleHeight)*(Lij - r)*diffX;
                float displaceY = deltaTime*deltaTime*params.kSpring*(1 - Lij / particleHeight)*(Lij - r)*diffY;

                particles[i].posX-=0.5f*displaceX;
                particles[i].posY-=0.5f*displaceY;

            }
        }


        if (p.posX >= params.vWidth) p.posX-=params.vWidth;
        if (p.posX <0.0f) p.posX+=params.vWidth;
        if (p.posY >= params.vHeight) p.posY-=params.vHeight;
        if (p.posY <0.0f) p.posY+=params.vHeight;
    }
}



//Resets the map of the scene, re-adding every particle based on where it is at this moment
void Cuda_solver::updateMap()
{
    ClearMap();

    if(h_params->totalParticles>0)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid =
                (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;


        CudaUpdateMap<<<blocksPerGrid, threadsPerBlock>>>(d_particles, d_map, d_map_size, d_mapCoords);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}


__global__ void CudaUpdateMap(Particle* particles, int* map, int *map_size, int2* mapCoords)
{

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        Particle& p = particles[i];

        int x = truncf(p.posX / particleHeight);
        int y = truncf(p.posY / particleHeight);

        if (x < 0)
            x = 0;
        else if (x > params.mapWidth - 1)
            x = params.mapWidth - 1;

        if (y < 0)
            y =  0;
        else if (y > params.mapHeight - 1)
            y = params.mapHeight - 1;

        //this handles the linked list between particles on the same square
        int& indexAdd=map_size[GetIndex(x,y)];
        map[GetIndexMap(atomicAdd(&indexAdd,1),x,y)] = i;
        mapCoords[i].x = x;
        mapCoords[i].y = y;

    }
}


//Resets the map of the scene, re-adding every particle based on where it is at this moment
void Cuda_solver::ClearMap()
{

    int threadsPerBlock = 256;
    int blocksPerGrid =
            (h_params->maxMapCoeff1 + threadsPerBlock - 1) / threadsPerBlock;

    CudaClearMap<<<blocksPerGrid, threadsPerBlock>>>(d_map, d_map_size);
    cudaDeviceSynchronize();
    getLastCudaError("Kernel execution failed");

}

__global__ void CudaClearMap(int* map, int *map_size)
{

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.maxMapCoeff1)
    {
        map_size[i]=0;
        if(map_size[i]>0)
           printf("not Clear mapsize %i" , map_size[i] );
    }

}


//saves neighbors for lookup in other functions
void Cuda_solver::storeNeighbors(){
     if(h_params->totalParticles>0)
    {
         int threadsPerBlock = 256;
         int blocksPerGrid =
                 (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaStoreNeighbors<<<blocksPerGrid, threadsPerBlock>>>(d_particles,  d_map, d_map_size, d_mapCoords,
                                                               d_neightb_index, d_neightb_size, d_neightb_r, d_neightb_Lij);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}


__global__ void CudaStoreNeighbors(Particle* particles,  int* map, int *map_size, int2* mapCoords,
                                   int* neightb_index, int* neightb_size, float* neightb_r, float* neightb_Lij)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        Particle& p = particles[i];
        int pX = mapCoords[i].x;
        int pY = mapCoords[i].y;

        neightb_size[i]=0;

        //iterate over the nine squares on the grid around p
        for (int mapX = pX - 1; mapX <= pX + 1; mapX++){
            for (int mapY = pY - 1; mapY <= pY + 1; mapY++){

                int mapXt=mapPosCorrectionX(mapX);
                int mapYt=mapPosCorrectionY(mapY);
                for(int ip=0; ip<map_size[GetIndex(mapXt,mapYt)];ip++)
                {
                    const Particle& pj =  particles[map[GetIndexMap(ip,mapXt,mapYt)]];

                    float diffX = TdeltaX(pj.posX , p.posX);
                    float diffY = TdeltaY(pj.posY , p.posY);
                    float r2 = diffX*diffX + diffY*diffY;
                    float r = sqrtf(r2);
                    float q = r / particleHeight;

                    //save this neighbor
                    if (q <= 1.0f && q > 0.000000000001f){ //0.0000000000001f){

                        const int j=neightb_size[i];// (neightb_size[i]==0)?0:neightb_size[i]-1;
                        if (neightb_size[i] < maxSprings){
                            neightb_index[GetIndexNeightb(i, j)]=map[GetIndexMap(ip,mapXt,mapYt)];
                            neightb_r[GetIndexNeightb(i,j)]=r;
                            neightb_Lij[GetIndexNeightb(i,j)]=particleHeight;
                            neightb_size[i]++;
                        }

                    }
                }
            }
        }

    }
}


//This maps pretty closely to the outline in the paper. Find density and pressure for all particles,
//then apply a displacement based on that. There is an added if statement to handle surface tension for multiple weights of particles
void Cuda_solver::doubleDensityRelaxation(){
    if(h_params->totalParticles>0)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid =
                (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaDoubleDensityRelaxation<<<blocksPerGrid, threadsPerBlock>>>(d_particles,
                                                                        d_neightb_index, d_neightb_size, d_neightb_r, d_neightb_Lij);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}

__global__ void CudaDoubleDensityRelaxation(Particle* particles,
                                            int* neightb_index, int* neightb_size, float* neightb_r, float* neightb_Lij)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        Particle& p = particles[i];

        float density = 0;
        float nearDensity = 0;

        for (int j = 0; j < neightb_size[i]; j++){
            const Particle& pNear = particles[neightb_index[GetIndexNeightb(i,j)]];// *neighbours[i].particles[j];

            float r = neightb_r[GetIndexNeightb(i,j)];
            float q = r / particleHeight;
            if (q>1) continue;
            float a = 1 - q;

            density += a*a * pNear.m * 20;
            nearDensity += a*a*a * pNear.m * 30;
        }
        p.pressure = params.stiffness * (density - p.m*params.restDensity);
        p.nearPressure = params.nearStiffness * nearDensity;
        float dx = 0, dy = 0;

        for (int j = 0; j < neightb_size[i]; j++){
            const Particle& pNear = particles[neightb_index[GetIndexNeightb(i,j)]];

            float diffX = TdeltaX(pNear.posX, p.posX);
            float diffY = TdeltaY(pNear.posY, p.posY);

            float r = neightb_r[GetIndexNeightb(i,j)];
            float q = r / particleHeight;
            if (q>1) continue;
            float a = 1 - q;
            float d = (deltaTime*deltaTime) * ((p.nearPressure + pNear.nearPressure)*a*a*a*53 + (p.pressure + pNear.pressure)*a*a*35) / 2;

            // weight is added to the denominator to reduce the change in dx based on its weight
            dx -=d*diffX/(r*p.m);
            dy -=d*diffY/(r*p.m);

            //surface tension is mapped with one type of particle,
            //this allows multiple weights of particles to behave appropriately
            if (p.m == pNear.m && multipleParticleTypes == true){
                dx +=params.surfaceTension * diffX;
                dy +=params.surfaceTension * diffY;
            }
        }

        p.posX +=dx;
        p.posY +=dy;

    }
}

void Cuda_solver::computeNextVelocity(){
    if(h_params->totalParticles>0)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid =
                (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaComputeNextVelocity<<<blocksPerGrid, threadsPerBlock>>>(d_particles, d_prevPos);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}

__global__ void CudaComputeNextVelocity(Particle* particles, float2* prevPos)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        Particle& p = particles[i];

        p.velX = TdeltaX(p.posX, prevPos[i].x) / deltaTime;
        p.velY = TdeltaY(p.posY, prevPos[i].y) / deltaTime;
    }
}


//Only checks if particles have left window, and push them back if so
void Cuda_solver::resolveCollisions(){
    if(h_params->totalParticles>0)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid =
                (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaResolveCollisions<<<blocksPerGrid, threadsPerBlock>>>(d_particles, d_boundaries);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}


__global__ void CudaResolveCollisions(Particle* particles, float3* boundaries)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        Particle& p = particles[i];


        if (p.posX >= params.vWidth) p.posX-=params.vWidth;
        if (p.posX <0.0f) p.posX+=params.vWidth;
        if (p.posY >= params.vHeight) p.posY-=params.vHeight;
        if (p.posY <0.0f) p.posY+=params.vHeight;

        if (p.velX > 4.5f) p.velX = 4.5f;
        if (p.velY > 4.5f) p.velY = 4.5f;
        if (p.velX < -4.5f) p.velX = -4.5f;
        if (p.velY < -4.5f) p.velY = -4.5f;




/*
        for (int j = 0; j<4; j++){
            const float3& boundary = boundaries[j];
            float distance = boundary.x*p.posX + boundary.y*p.posY - boundary.z;

            if (distance < particleRadius){
                if (distance < 0)
                    distance = 0;
                p.velX += 0.99f*(particleRadius - distance) * boundary.x / deltaTime;
                p.velY += (particleRadius - distance) * boundary.y / deltaTime;
            }

            //The resolve collisions tends to overestimate the needed counter velocity, this limits that

            if (p.velX > 17.5f) p.velX = 17.5f;
            if (p.velY > 17.5f) p.velY = 17.5f;
            if (p.velX < -17.5f) p.velX = -17.5f;
            if (p.velY < -17.5f) p.velY = -17.5f;

        }        
       */
    }
}

__global__ void CudaResolveCollisions_old(Particle* particles, float3* boundaries)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        Particle& p = particles[i];

        for (int j = 0; j<4; j++){
            const float3& boundary = boundaries[j];
            float distance = boundary.x*p.posX + boundary.y*p.posY - boundary.z;

            if (distance < particleRadius){
                if (distance < 0)
                    distance = 0;
                p.velX += 0.99f*(particleRadius - distance) * boundary.x / deltaTime;
                p.velY += (particleRadius - distance) * boundary.y / deltaTime;
            }

        }

    }
}


//Iterates through every particle and multiplies its RGB values based on speed.
//speed^2 is just used to make the difference in speeds more noticeable.
void Cuda_solver::adjustColor()
{
    if(h_params->totalParticles>0)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid =
                (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaAdjustColor<<<blocksPerGrid, threadsPerBlock>>>(d_particles, d_savedParticleColors, d_particleTypes);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}


__global__ void CudaAdjustColor(Particle* particles, Vec4* savedParticleColors, ParticleType* particleTypes)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        const Particle& p = particles[i];
        const ParticleType& pt = particleTypes[i];

        if(params.drawListSet==0)//DrawList::simpleColor)
        {
            float speed2 = 0.0f;

            Vec4& color = savedParticleColors[i];
            color = pt.color;
            color.r *= 0.95f + velocityFactor*speed2;
            color.g *= 0.95f + velocityFactor*speed2;
            color.b *= 0.95f + velocityFactor*speed2;
        }

        if(params.drawListSet==1)//DrawList::tempColor)
        {
            float speed2 = (p.magnetX*p.magnetX + p.velX*p.velX)*0.001f;

            Vec4& color = savedParticleColors[i];
            color = pt.color;
            color.r *= 0.5f + velocityFactor*speed2;
            color.g *= 0.5f + velocityFactor*speed2;
            color.b *= 0.5f + velocityFactor*speed2;
        }

        if(params.drawListSet==2)//DrawList::velocityColor)
        {
            float speed2 = (p.velX*p.velX + p.velY*p.velY)*0.05f;

            Vec4& color = savedParticleColors[i];
            color = pt.color;
            color.r *= 0.5f + velocityFactor*speed2;
            if(color.r>0.70)
                color.r=0.70f;
            color.g *= 0.5f + velocityFactor*speed2;
            if(color.g>0.70)
                color.g=0.70f;
            color.b *= 0.5f + velocityFactor*speed2;
            if(color.b>0.70)
                color.b=0.70f;
        }

    }
}



//Resets the map of the scene, re-adding every particle based on where it is at this moment
void Cuda_solver::ClearTex(void *devPtr, int m_texWidth, int m_texHeight)
{
    dim3 threadsperBlock(32, 32);
    dim3 numBlocks1((m_texWidth + threadsperBlock.x - 1) / threadsperBlock.x,
                             (m_texHeight + threadsperBlock.y - 1) / threadsperBlock.y);

    CudaClearTex<<<numBlocks1, threadsperBlock>>>((uchar4 *) devPtr,m_texWidth, m_texHeight);
    cudaDeviceSynchronize();
    getLastCudaError("Kernel execution failed");
}

__global__ void CudaClearTex(uchar4 *ptr, int max_w, int max_h)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;


   if (x < max_w && y < max_h)
   {
        uchar4 bgra = {0x0,0x0,0x0,0x0};

        bgra.x = 25;
        bgra.y = 25;
        bgra.z = 25;
        bgra.w= 255;

        const unsigned int pixelIndex=y*max_w+x;
        ptr[pixelIndex] = bgra;
   }
}

//Resets the map of the scene, re-adding every particle based on where it is at this moment
void Cuda_solver::PerlinTex(void *devPtr, int m_texWidth, int m_texHeight)
{
    dim3 blockSize {16, 16};
    dim3 gridSize { static_cast<int>(screenWidth) / blockSize.x, static_cast<int>(screenHeight) / blockSize.y };

    CudaPerlinTex<<<gridSize, blockSize>>>((uchar4 *) devPtr,m_texWidth, m_texHeight);
    cudaDeviceSynchronize();
    getLastCudaError("Kernel execution failed");
}

__global__ void CudaPerlinTex(uchar4 *ptr, int max_w, int max_h)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    long idx = x + y * blockDim.x *  gridDim.x;

    float fx = static_cast<float>(x) / (blockDim.x * gridDim.x) * 16.0f;
    float fy = static_cast<float>(y) / (blockDim.y * gridDim.y) * 16.0f;

    ptr[idx].y=15;
    ptr[idx].x=15;
    ptr[idx].z=75;
    ptr[idx].w=255;
}



void Cuda_solver::TexColor(void *devPtr, int m_texWidth, int m_texHeight)
{
    if(h_params->totalParticles>0)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid =
                (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaTexColor<<<blocksPerGrid, threadsPerBlock>>>((uchar4 *) devPtr,m_texWidth, m_texHeight, d_particles, d_particleTypes, d_savedParticleColors,  d_colorsMapVec4);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }
}


__global__ void CudaTexColor(uchar4 *ptr, int max_w, int max_h, Particle* particles, ParticleType* particleTypes, Vec4* savedParticleColors,  Vec4* d_colorMap)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        const Particle& p = particles[i];
        const ParticleType& pt = particleTypes[i];

        float speed2 = sqrtf(p.velX*p.velX + p.velY*p.velY);
        speed2/=7.5f;
        speed2*=253;

        int speedColor=(int)speed2;
        Vec4& color = savedParticleColors[i];
        {
            color=d_colorMap[speedColor];
        }

        uchar4 bgra = {0x0,0x0,0x0,0x0};

        bgra.x = 100;
        bgra.y = 100;
        bgra.z = 245;
        bgra.w=  255;

        int x_coord=(int)(coeffDisplay*p.posX);
        int y_coord=(int)(coeffDisplay*p.posY);

        if(x_coord < max_w && y_coord < max_h and x_coord>=0 and y_coord>=0)
        {
            const unsigned int pixelIndex=(int)(max_h-y_coord)*max_w+x_coord;


            if(params.drawListSet==0)//DrawList::simpleColor)
            {
                float speed2 = 0.0f;

                Vec4& color = savedParticleColors[i];
                color = pt.color;
                color.r *= 1.25f + velocityFactor*speed2;
                color.g *= 1.25f + velocityFactor*speed2;
                color.b *= 1.25f + velocityFactor*speed2;
            }

            if(params.drawListSet==1)//DrawList::tempColor)
            {
                float speed2 = (p.magnetX*p.magnetX + p.velX*p.velX)*0.001f;

                Vec4& color = savedParticleColors[i];
                color = pt.color;
                color.r *= 0.5f + velocityFactor*speed2;
                color.g *= 0.5f + velocityFactor*speed2;
                color.b *= 0.5f + velocityFactor*speed2;
            }

            if(params.drawListSet==2)//DrawList::velocityColor)
            {
                float speed2 = (p.velX*p.velX + p.velY*p.velY)*0.05f;

                Vec4& color = savedParticleColors[i];
                color = pt.color;
                color.r *= 0.5f + velocityFactor*speed2;
                if(color.r>0.70)
                    color.r=0.70f;
                color.g *= 0.5f + velocityFactor*speed2;
                if(color.g>0.70)
                    color.g=0.70f;
                color.b *= 0.5f + velocityFactor*speed2;
                if(color.b>0.70)
                    color.b=0.70f;
            }

            bgra.x = color.r*255;
            bgra.y = color.g*255;
            bgra.z = color.b*255;
            bgra.w=  255;


            ptr[pixelIndex] = bgra;
        }
    }
}

void Cuda_solver::RegisterGLTextureForCUDA(GLuint *gl_tex, cudaGraphicsResource **cuda_tex, unsigned int size_x, unsigned int size_y)
{
       //SDK_CHECK_ERROR_GL();
       checkCudaErrors(cudaGraphicsGLRegisterImage(cuda_tex, *gl_tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

}



void Cuda_solver::renderParticles(PosColorLayout *vbo,int m_texWidth, int m_texHeight)
{

    if(h_params->totalParticles>0)
    {
        int threadsPerBlock = 256;
        int blocksPerGrid =
                (h_params->totalParticles + threadsPerBlock - 1) / threadsPerBlock;

        CudaRenderParticles<<<blocksPerGrid, threadsPerBlock>>>(vbo ,m_texWidth, m_texHeight, d_particles, d_particleTypes, d_savedParticleColors,  d_colorsMapVec4);
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");
    }

}

__global__ void CudaRenderParticles(PosColorLayout *vbo, int max_w, int max_h, Particle* particles, ParticleType* particleTypes, Vec4* savedParticleColors,  Vec4* d_colorMap)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<params.totalParticles)
    {
        const Particle& p = particles[i];
        const ParticleType& pt = particleTypes[i];
        float speed2 = sqrtf(p.velX*p.velX + p.velY*p.velY);
        speed2/=1.5f;
        speed2*=253;

        int speedColor=(int)speed2;
        Vec4& color = savedParticleColors[i];
        {
            color=d_colorMap[speedColor];
        }

        vbo[i].m_pos[0]=coeffDisplay*(p.posX+particleHeight/2.0f)/2.0f;
        vbo[i].m_pos[1]=max_h-coeffDisplay*(p.posY-particleHeight/2.0f)/2.0f;
        vbo[i].m_pos[2]=0.0f;
/*
        if(i==0)
        {
            printf("vbo posx %f",vbo[i].m_pos[0] );
        }
*/
        vbo[i].m_color[0]=55;//color.a;
        vbo[i].m_color[1]=155;
        vbo[i].m_color[2]=155;
        vbo[i].m_color[3]=255;

        if(params.drawListSet==0)//DrawList::simpleColor)
        {
            float speed2 = 0.0f;

            Vec4& color = savedParticleColors[i];
            color = pt.color;
            color.r *= 1.25f + velocityFactor*speed2;
            color.g *= 1.25f + velocityFactor*speed2;
            color.b *= 1.25f + velocityFactor*speed2;
        }

        if(params.drawListSet==1)//DrawList::tempColor)
        {
            float speed2 = (p.velX*p.velX + p.velY*p.velY);
            speed2*=0.5f;
            Vec4& color = savedParticleColors[i];
            color = pt.color;
            color.r *= 0.5f + velocityFactor*speed2;
            if(color.r>0.99)
                color.r=0.99f;
            color.g *= 0.5f + velocityFactor*speed2;
            if(color.g>0.99)
                color.g=0.99f;
            color.b *= 0.5f + velocityFactor*speed2;
            if(color.b>0.99)
                color.b=0.99f;
        }

        if(params.drawListSet==2)//DrawList::velocityColor)
        {
            float speed2 = (p.velX*p.velX + p.velY*p.velY);
            speed2/=2.5f;
            Vec4& color = savedParticleColors[i];

            color = pt.color;
            color.r *= 0.5f + velocityFactor*speed2;
            if(color.r>0.99)
                color.r=0.99f;
            color.g *= 0.5f + velocityFactor*speed2;
            if(color.g>0.99)
                color.g=0.99f;
            color.b *= 0.5f + velocityFactor*speed2;
            if(color.b>0.99)
                color.b=0.99f;
        }

        vbo[i].m_color[0] = color.r*255;
        vbo[i].m_color[1] = color.g*255;
        vbo[i].m_color[2] = color.b*255;
        vbo[i].m_color[3]=  255;


    }
}

//Runs through all of the logic 7 times a frame
bool Cuda_solver::Update()
{

    if(h_params->totalParticles<maxParticles)
    {
        for (int step = 0; step<timeStep; step++)
        {

            checkCudaCall(cudaMemcpyToSymbol(params,h_params,sizeof(gpuParams)));
            generateParticles();
            applyTemp();
            applyGravity();
            applyViscosity();
            advance();
            adjustSprings();
            ClearMap();
            updateMap();
            storeNeighbors();
            doubleDensityRelaxation();
            computeNextVelocity();
            resolveCollisions();
            UpdateHostBuffers();

/*
            for(int i=0; i<h_params->totalParticles;i++)
            {
                h_particlesDraw[i].m_pos[0]=10.0f*h_particles[i].posX;
                h_particlesDraw[i].m_pos[1]=10.0f*h_particles[i].posY;
                h_particlesDraw[i].m_pos[2]=0;
            }
*/
        }
    }
    else
    {
        for (int step = 0; step<timeStep; step++)
        {
            //generateParticles();
            checkCudaCall(cudaMemcpyToSymbol(params,h_params,sizeof(gpuParams)));

            applyTemp();
            applyGravity();
            applyViscosity();
            advance();
            adjustSprings();
            ClearMap();
            updateMap();
            storeNeighbors();
            doubleDensityRelaxation();
            computeNextVelocity();
            resolveCollisions();

            h_params->tick++;
            h_params->tickLoop++;

        }
       //  UpdateHostBuffers();
    }

    return true;
}

bool Cuda_solver::UpdateActivePixels()
{



    return true;
}
