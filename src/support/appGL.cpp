#include "appGL.h"



AppGL::AppGL(AppWindow *appWindow, unsigned maxWidth, unsigned maxHeight, const std::string &shaderPath)
{
    std::cout << "maxWidth,height = " << maxWidth << "," << maxHeight << std::endl;
    mWindow = appWindow;
    glClearColor(1.0, 1.0, 0.5, 0.0);
    // Shared CUDA/GL pixel buffer
    //mSharedPboTex = new AppPboTex(maxWidth, maxHeight);
    //mSharedTex = new AppTexture(screenWidth,screenHeight);
    mPixels = nullptr;

    {
        glm::mat4 projection = glm::ortho(0.0f, 800.0f, 600.0f, 0.0f, -1.0f, 1.0f);

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(glm::vec2(appWindow->width()/2.0f,appWindow->height()/2.0f), 0.0f));
        model = glm::scale(model, glm::vec3(glm::vec2(400,300), 1.0f));

        mCameraToView =projection*model;

    }

    {
        glm::mat4 projection = glm::ortho(0.0f, (float)appWindow->width(), (float)appWindow->height(), 0.0f, -1.0f, 1.0f);

        glm::mat4 model = glm::mat4(1.0f);

        model = glm::translate(model, glm::vec3(glm::vec2(0.0f,0.f), 0.0f));
        model = glm::scale(model, glm::vec3(glm::vec2(1.0f,1.0f), 1.0f));


        mCameraToView2 =projection*model;

    }


    const std::string pathSep = "/";

    std::cout << "source directory = " << shaderPath << std::endl;

    mBasicProgTex = new AppGLProgramTex(
        shaderPath + pathSep + "basic_vert.glsl",
        shaderPath + pathSep + "basic_frag.glsl");

   const uint32_t tempColor1=   (uint8_t(255) << 24) +   //a
                                (uint8_t(155) << 16) +   //r
                                (uint8_t(155) << 8)  +   //g
                                 uint8_t(155);           //b

    float coords[] = {-1.0f,  1.0f, 0.0f,  0.0f, 1.0f,
                       1.0f,  1.0f, 0.0f , 1.0f, 1.0f,
                      -1.0f, -1.0f, 0.0f,  0.0f, 0.0f,
                       1.0f, -1.0f, 0.0f,  1.0f, 0.0f};

    GLuint indexs[] =   {0, 1, 2,
                         1, 3, 2};

    mVertsTex = new AppVertsTex(4,coords,6,indexs);



    mBasicProgPoint = new AppGLProgramPoint(
        shaderPath + pathSep + "basic_vertPoint.glsl",
        shaderPath + pathSep + "basic_fragPoint.glsl");

/*
    PosColorLayout coords2[] = {PosColorLayout{-1.0f,   1.0f,  0.0f,  tempColor1},
                                PosColorLayout{ 1.0f,   1.0f,  0.0f,  tempColor1},
                                PosColorLayout{-1.0f,  -1.0f,  0.0f,  tempColor1},
                                PosColorLayout{ 1.0f,  -1.0f,  0.0f,  tempColor1}};
    */
    PosColorLayout coords2[] = {PosColorLayout{-1.0f,   1.0f,  0.0f,  255, 155,125, 155},
                                PosColorLayout{ 1.0f,   1.0f,  0.0f,  255,  55,135, 155},
                                PosColorLayout{-1.0f,  -1.0f,  0.0f,  255, 135, 55, 155},
                                PosColorLayout{ 1.0f,  -1.0f,  0.0f,  255, 155,155, 135}};

    GLuint indexs2[] =   {0, 1, 2,3};//
                        // 1, 3, 2};

    mVertsPoint = new AppVertsPoint(4,coords2,4,indexs2);

    //mSharedPboPoint.registerBuffer(mVertsPoint->mId, mCudaVbo);




}

AppGL::~AppGL()
{
    if (mPixels != nullptr) {
        delete[](mPixels);
    }
}


void AppGL::handleResize()
{

}

// OpenGL-related code for JMandelbrot.  This is a static class to
// reflect a single GL context.
//
// - Creates a fullscreen quad (2 triangles)
// - The quad has x,y and s,t coordinates & the upper-left corner
//   is always 0,0 for both.
// - When resized,
//   - the x,y for the larger axis ranges from 0-1 and the shorter
//     axis 0-ratio where ratio is < 1.0
//   - the s,t is a ratio of the window size to the shared CUDA/GL
//     texture size.
//   - the shared CUDA/GL texture size should be set to the maximum
//     size you expect. (Monitor width/height)
// - These values are updated inside the vertex buffer.
//
// t y
// 0 0 C--*--D triangle_strip ABCD
//     |\....|
//     |.\...|
//     *..*..*
//     |...\.|
//     |....\|
// 1 1 A--*--B
//     0     1 x position coords
//     0     1 s texture coords
//

void AppGL::render(Cuda_solver& cuda_solver)
{
    glClearColor(0.25f, 0.25f, 0.25f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
/*
    {

        //Cuda Block

        devPtrTex =mSharedPboTex->mapGraphicsResource();
        //cuda_solver.ClearTex(devPtrTex, screenWidth, screenHeight);

        if(oneRunClearTexture)
        {
            cuda_solver.ClearTex(devPtrTex, screenWidth, screenHeight);
            oneRunClearTexture=false;
        }
        cuda_solver.TexColor(devPtrTex, screenWidth, screenHeight);
        mSharedPboTex->unmapGraphicsResource();

        mSharedPboTex->bind();
        mSharedTex->bind();
        mSharedTex->copyFromPbo();

        mBasicProgTex->bind();
        mBasicProgTex->updateCameraToView(mCameraToView);
        mVertsTex->bind();
        mVertsTex->draw();

        mVertsTex->unbind();
        mSharedPboTex->unbind();
        mSharedTex->unbind();
        mBasicProgTex->unbind();
    }

    {
       // cuda_solver.Update();

        if(cuda_solver.h_params->totalParticles<maxParticles)
        {
            mVertsPoint->updatePosition(cuda_solver.h_params->totalParticles, cuda_solver.h_particlesDraw);
            mVertsPoint->updateIndexs(cuda_solver.h_params->totalParticles, cuda_solver.h_particlesIndexDraw);


        }
        else
        {
            if(oneRun)
            {
                mVertsPoint->updatePosition(cuda_solver.h_params->totalParticles, cuda_solver.h_particlesDraw);
                mVertsPoint->updateIndexs(cuda_solver.h_params->totalParticles, cuda_solver.h_particlesIndexDraw);

                cudaErrChk(cudaGraphicsGLRegisterBuffer(&mCudaVbo,
                                                        mVertsPoint->mVbo->m_vboId,
                                                        cudaGraphicsMapFlagsWriteDiscard));

                oneRun=false;
            }

          //  mVertsPoint->bind();
            cudaErrChk(cudaGraphicsMapResources(1, &mCudaVbo,0));
            size_t num_bytes;
            cudaErrChk(cudaGraphicsResourceGetMappedPointer((void**)&devPtrPoint, &num_bytes, mCudaVbo));
            int verNumber=num_bytes/sizeof(PosColorLayout);

            cuda_solver.renderParticles(devPtrPoint, screenWidth,screenHeight);
            //mSharedPboPoint.unmapGraphicsResource(mCudaVbo);
            cudaErrChk(cudaGraphicsUnmapResources(1, &mCudaVbo,0));
         //    mVertsPoint->unbind();

        }



        //mSharedPbo->bind();
        //mSharedTex->bind();
        //mSharedTex->copyFromPbo();

        mBasicProgPoint->bind();
        mBasicProgPoint->updateCameraToView(mCameraToView2);
        mVertsPoint->bind();
        mVertsPoint->draw();

        mVertsPoint->unbind();
        //mSharedPbo->unbind();
       //SharedTex->unbind();
        mBasicProgPoint->unbind();
    }
*/




}
