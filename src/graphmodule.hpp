#ifndef GRAPHMODULE_HPP
#define GRAPHMODULE_HPP

#pragma once

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include "Constants.hpp"

#include "imgui.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_opengl3.h"

#include <SDL2/SDL.h>
#include "cuda_solver.cuh"

#include "support/appGLProgram.h"
#include "support/appTexture.h"
#include "support/appVerts.h"
#include "support/appPbo.h"

class GraphModule
{
public:
    GraphModule();

    bool InitSDL();
    bool InitImGui();

    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    float f = 0.0f;
    int counter = 0;

    SDL_Window* screen = NULL;
    SDL_GLContext gContext;

    SDL_Event event;

    const std::string shaderPath="../shaders";

    glm::mat4 mCameraToView;
    glm::mat4 mCameraToView2;

    AppVertsTex *mVertsTex;
    AppGLProgramTex *mBasicProgTex;

    AppTexture *mSharedTex;
    AppPboTex* mSharedPboTex;

    cudaGraphicsResource* cuda_tex_resource;

    cudaArray *texture_ptr;
    GLuint* cuda_dev_render_buffer;


    AppVertsPoint *mVertsPoint;
    AppGLProgramPoint *mBasicProgPoint;
    struct cudaGraphicsResource *mCudaVbo;
    PosColorLayout *devPtrPoint;

    bool useVBOrender=true;
    bool useTextureRender=true;

    void *devPtrTex;

    bool oneRun=true;
    bool oneRunClearTexture=true;

    GLuint my_image_texture = 0;

    void HandleEvents(SDL_Event e, Cuda_solver& sph_solver);
    void Render(Cuda_solver &sph_solver);
    void GuiRender(Cuda_solver &sph_solver);

    void ClearScreen();

    void CloseRender();

    bool runSimulation=true;
    bool restartSimulation=false;


    void Render2(Cuda_solver &sph_solver);
    bool InitShaders();

    void ClearRenderScreen();
    void Render3(Cuda_solver &cuda_solver);
    bool LoadTextureFromFile(const char *filename, GLuint *out_texture, int *out_width, int *out_height);
    void TextureCudaUpdater3(Cuda_solver &cuda_solver);
    void TextureCudaUpdater(Cuda_solver &cuda_solver);
};

#endif // GRAPHMODULE_HPP
