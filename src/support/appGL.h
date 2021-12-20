#ifndef SMANDELBROTR_APP_GL_H
#define SMANDELBROTR_APP_GL_H

#include "appGLProgram.h"
#include "appPbo.h"
#include "appTexture.h"
#include "appVerts.h"
#include "appWindow.h"

#include <GL/glew.h>
#include "glm/ext.hpp"
#include "glm/glm.hpp"

#include <string>
#include "../Constants.hpp"


#include "../cuda_solver.cuh"


class AppGL {

    AppWindow *mWindow;

    AppVertsTex *mVertsTex;
    AppGLProgramTex *mBasicProgTex;

    AppVertsPoint *mVertsPoint;
    AppGLProgramPoint *mBasicProgPoint;


    AppTexture *mSharedTex;
 public:
    AppPboTex* mSharedPboTex;
    void *devPtrTex;

    AppPboPoint mSharedPboPoint;
    struct cudaGraphicsResource *mCudaVbo;
    void *d_vbo_buffer = NULL;

    PosColorLayout *devPtrPoint;
    bool oneRun=true;
    bool oneRunClearTexture=true;

    glm::mat4 mCameraToView;
    glm::mat4 mCameraToView2;

    unsigned char *mPixels; // storage for data from framebuffer



    AppGL(AppWindow *appWindow, unsigned maxWidth, unsigned maxHeight, const std::string &shaderPath);
    ~AppGL();

   // AppPboTex *sharedPboTex()
   //     { return mSharedPboTex; }
    unsigned textureWidth()
        { return mSharedTex->width(); }
    unsigned textureHeight()
        { return mSharedTex->height(); }

    void handleResize();
    void render(Cuda_solver& cuda_solver);

    unsigned char *readPixels()
    {
        glReadPixels(0, 0, mWindow->width(), mWindow->height(), GL_RGBA, GL_UNSIGNED_BYTE, mPixels);
        return mPixels;
    }
};

#endif
