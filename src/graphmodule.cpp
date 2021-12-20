#include "graphmodule.hpp"


GraphModule::GraphModule()
{
    if(!InitSDL())
        printf("Some problems in inint SDL");

    if(!InitImGui())
        printf("Some problems in inint ImGui");

    InitShaders();

}

bool GraphModule::InitShaders()
{
    {
        glm::mat4 projection = glm::ortho(0.0f, (float)screenWidth, (float)screenHeight, 0.0f, -1.0f, 10.0f);
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(glm::vec2(mainscreenWidth/2.0f,mainscreenHeight/2.0f), 0.0f));
        model = glm::scale(model, glm::vec3(glm::vec2(screenWidth/2.0f,screenHeight/2.0f), 1.0f));
        mCameraToView =projection*model;
    }

    {
        glm::mat4 projection = glm::ortho(0.0f, (float)screenWidth, (float)screenHeight, 0.0f, -1.0f, 1.0f);
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(glm::vec2(25.0f,-25.f), 0.0f));
        model = glm::scale(model, glm::vec3(glm::vec2(1.0f,1.0f), 1.0f));
        mCameraToView2 =projection*model;
    }

    const std::string pathSep = "/";
    std::cout << "source directory = " << shaderPath << std::endl;

    //Cuda texture update block init
    if(useTextureRender)
    {
        mBasicProgTex = new AppGLProgramTex(
            shaderPath + pathSep + "basic_vert.glsl",
            shaderPath + pathSep + "basic_frag.glsl");

        const uint32_t tempColor1=   (uint8_t(255) << 24) +   //a
                                     (uint8_t(155) << 16) +   //r
                                     (uint8_t(155) << 8)  +   //g
                                      uint8_t(155);           //b


        PosTexLayout2 coords[] = {  PosTexLayout2{ -1.0f,  1.0f, 0.0f,  0.0f, 1.0f},
                                    PosTexLayout2{  1.0f,  1.0f, 0.0f , 1.0f, 1.0f},
                                    PosTexLayout2{ -1.0f, -1.0f, 0.0f,  0.0f, 0.0f},
                                    PosTexLayout2{  1.0f, -1.0f, 0.0f,  1.0f, 0.0f}};

        GLuint indexs[] =   {0, 1, 2,
                             1, 3, 2};

        mVertsTex = new AppVertsTex(4,coords,6,indexs);

       // mSharedPboTex=new AppPboTex(screenWidth, screenHeight);
       // mSharedPboTex->registerBuffer();
        mSharedTex=new AppTexture(screenWidth, screenHeight);

        // set up vertex data parameters
        int num_texels = screenWidth* screenHeight;
        int num_values = num_texels * 4;
        size_t size_tex_data = sizeof(int) * num_values;

        // We don't want to use cudaMallocManaged here - since we definitely want
        checkCudaErrors(cudaMalloc(&cuda_dev_render_buffer, size_tex_data));
        checkCudaErrors(cudaGraphicsGLRegisterImage(&cuda_tex_resource, mSharedTex->mId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
    }

    // cuda VBO update block init
    if(useVBOrender)
    {
        mBasicProgPoint = new AppGLProgramPoint(
            shaderPath + pathSep + "basic_vertPoint.glsl",
            shaderPath + pathSep + "basic_fragPoint.glsl");

        PosColorLayout coords[maxParticles];
        GLuint indexs[maxParticles];

        for(int i=0; i<maxParticles; i++)
        {
            coords[i]=PosColorLayout{-1.0f,   1.0f,  0.0f,  255, 155,125, 155};
            indexs[i]=(GLuint)i;
        }

        mVertsPoint = new AppVertsPoint(maxParticles,coords,maxParticles,indexs);
        cudaErrChk(cudaGraphicsGLRegisterBuffer(&mCudaVbo,
                                                mVertsPoint->mVbo->m_vboId,
                                                cudaGraphicsMapFlagsWriteDiscard));
    }

    //load external texture
    /*
    {
        int my_image_width = 0;
        int my_image_height = 0;
        bool ret = LoadTextureFromFile("../shaders/test800_600.png", &my_image_texture, &my_image_width, &my_image_height);
        IM_ASSERT(ret);
    }
    */

    return true;
}


bool GraphModule::InitSDL()
{
    // Setup SDL
    // (Some versions of SDL before <2.0.10 appears to have performance/stalling issues on a minority of Windows systems,
    // depending on whether SDL_INIT_GAMECONTROLLER is enabled or disabled.. updating to latest version of SDL is recommended!)
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
    {
        printf("Error: %s\n", SDL_GetError());
        return false;
    }

    const char* glsl_version = "#version 150";
    // Initialize rendering context
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);


    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                       SDL_GL_CONTEXT_PROFILE_COMPATIBILITY );



    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);

    screen = SDL_CreateWindow("SPH", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, mainscreenWidth, mainscreenHeight, window_flags);
    gContext = SDL_GL_CreateContext(screen);

    SDL_GL_MakeCurrent(screen, gContext);
    SDL_GL_SetSwapInterval(1);

    GLenum err;
    glewExperimental = GL_TRUE; // Please expose OpenGL 3.x+ interfaces
    err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "Failed to init GLEW" << std::endl;
        SDL_GL_DeleteContext(gContext);
        SDL_DestroyWindow(screen);
        SDL_Quit();
        return true;
    }

    //Main loop flag
    bool quit = false;
    SDL_Event e;
    SDL_StartTextInput();

    int major, minor;
    SDL_GL_GetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, &major);
    SDL_GL_GetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, &minor);
    std::cout << "OpenGL version       | " << major << "." << minor << std::endl;
    std::cout << "GLEW version         | " << glewGetString(GLEW_VERSION) << std::endl;
    std::cout << "---------------------+-------" << std::endl;


    return true;
}

bool GraphModule::InitImGui()
{
    // Setup Dear ImGui context
       IMGUI_CHECKVERSION();
       ImGui::CreateContext();
       ImGui::LoadIniSettingsFromDisk("tempImgui.ini");
       ImGuiIO& io = ImGui::GetIO(); (void)io;
       static ImGuiStyle* style = &ImGui::GetStyle();
       style->Alpha = 1.00f; //0.75f

       io.WantCaptureMouse=true;
       //io.WantCaptureKeyboard=false;
       //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
       //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

       // Setup Dear ImGui style
       ImGui::StyleColorsDark();
       //ImGui::StyleColorsClassic();

       // Setup Platform/Renderer backends
       ImGui_ImplSDL2_InitForOpenGL(screen, gContext);
       ImGui_ImplOpenGL3_Init();

       return true;

}

void GraphModule::Render2(Cuda_solver& cuda_solver){

     if(cuda_solver.h_params->totalParticles<maxParticles)
     {
         mVertsPoint->updatePosition(cuda_solver.h_params->totalParticles, cuda_solver.h_particlesDraw);
         mVertsPoint->updateIndexs(cuda_solver.h_params->totalParticles, cuda_solver.h_particlesIndexDraw);


         cudaErrChk(cudaGraphicsMapResources(1, &mCudaVbo,0));
         size_t num_bytes;
         cudaErrChk(cudaGraphicsResourceGetMappedPointer((void**)&devPtrPoint, &num_bytes, mCudaVbo));
         int verNumber=num_bytes/sizeof(PosColorLayout);

         cuda_solver.renderParticles(devPtrPoint, screenWidth,screenHeight);
         //mSharedPboPoint.unmapGraphicsResource(mCudaVbo);
         cudaErrChk(cudaGraphicsUnmapResources(1, &mCudaVbo,0));
     }
     else
     {
         cudaErrChk(cudaGraphicsMapResources(1, &mCudaVbo,0));
         size_t num_bytes;
         cudaErrChk(cudaGraphicsResourceGetMappedPointer((void**)&devPtrPoint, &num_bytes, mCudaVbo));
         int verNumber=num_bytes/sizeof(PosColorLayout);

         cuda_solver.renderParticles(devPtrPoint, screenWidth,screenHeight);
         //mSharedPboPoint.unmapGraphicsResource(mCudaVbo);
         cudaErrChk(cudaGraphicsUnmapResources(1, &mCudaVbo,0));
     }

     mBasicProgPoint->bind();
     mBasicProgPoint->updateCameraToView(mCameraToView2);
     mVertsPoint->bind();
     mVertsPoint->draw();

     mVertsPoint->unbind();
     mBasicProgPoint->unbind();

}


void GraphModule::ClearRenderScreen(){

    glClearColor(0.15f, 0.15f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

void GraphModule::Render(Cuda_solver& cuda_solver){

    mBasicProgTex->bind();
    mBasicProgTex->updateCameraToView(mCameraToView);

    mSharedTex->bind();
    mVertsTex->bind();
    mVertsTex->draw();

    mVertsTex->unbind();
    mSharedPboTex->unbind();
    mSharedTex->unbind();

    mBasicProgTex->unbind();

}

void GraphModule::TextureCudaUpdater(Cuda_solver& cuda_solver){

    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource, 0, 0));

    int size_tex_data_w = sizeof(unsigned int) * (int)screenWidth;
    size_t wOffset=sizeof(int)*0;
    size_t hOffset=0;
    size_t height=(int)screenHeight;

    cuda_solver.ClearTex((unsigned int *)cuda_dev_render_buffer, screenWidth, screenHeight);
    cuda_solver.TexColor((unsigned int *)cuda_dev_render_buffer, screenWidth, screenHeight);


    checkCudaErrors(cudaMemcpy2DToArray(texture_ptr, wOffset, hOffset,cuda_dev_render_buffer, size_tex_data_w, size_tex_data_w,height, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));

}

void GraphModule::TextureCudaUpdater3(Cuda_solver& cuda_solver){

    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource, 0, 0));

    int size_tex_data_w = sizeof(int) * screenWidth;
    size_t wOffset=sizeof(int)*0;
    size_t hOffset=0;
    size_t height=screenHeight;
    cuda_solver.PerlinTex((unsigned int *)cuda_dev_render_buffer, screenWidth, screenHeight);
    //cuda_solver.TexColor((unsigned int *)cuda_dev_render_buffer, screenWidth, screenHeight);

    checkCudaErrors(cudaMemcpy2DToArray(texture_ptr, wOffset, hOffset,cuda_dev_render_buffer, size_tex_data_w, size_tex_data_w,height, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));

}

void GraphModule::Render3(Cuda_solver& cuda_solver){

    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_tex_resource, 0));
    checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_tex_resource, 0, 0));

    int size_tex_data_w = sizeof(int) * screenWidth;
    size_t wOffset=sizeof(int)*0;
    size_t hOffset=0;
    size_t height=screenHeight;
    cuda_solver.PerlinTex((unsigned int *)cuda_dev_render_buffer, screenWidth, screenHeight);
    //cuda_solver.TexColor((unsigned int *)cuda_dev_render_buffer, screenWidth, screenHeight);

    checkCudaErrors(cudaMemcpy2DToArray(texture_ptr, wOffset, hOffset,cuda_dev_render_buffer, size_tex_data_w, size_tex_data_w,height, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_tex_resource, 0));

    mBasicProgTex->bind();
    mBasicProgTex->updateCameraToView(mCameraToView);

    mSharedTex->bind();
    mVertsTex->bind();
    mVertsTex->draw();

    mVertsTex->unbind();
    mSharedPboTex->unbind();
    mSharedTex->unbind();

    mBasicProgTex->unbind();

}

// Simple helper function to load an image into a OpenGL texture with common settings
bool GraphModule::LoadTextureFromFile(const char* filename, GLuint* out_texture, int* out_width, int* out_height)
{
    // Load from file
    int image_width = 0;
    int image_height = 0;
    unsigned char* image_data = stbi_load(filename, &image_width, &image_height, NULL, 4);
    if (image_data == NULL)
        return false;

    // Create a OpenGL texture identifier
    GLuint image_texture;
    glGenTextures(1, &image_texture);
    glBindTexture(GL_TEXTURE_2D, image_texture);

    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

    // Upload pixels into texture
#if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
#endif
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data);
    stbi_image_free(image_data);

    *out_texture = image_texture;
    *out_width = image_width;
    *out_height = image_height;

    return true;
}

void GraphModule::GuiRender(Cuda_solver& sph_solver)
{
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame(screen);
    ImGui::NewFrame();

    {
        ImGui::Begin("SPH particles parameters");
            ImGui::Text("number particles: %i", sph_solver.h_params->totalParticles);
            ImGui::Text("Tick: %i", sph_solver.h_params->tick);
            ImGui::Text("Framerate  : %.1f ms or %.1f Hz", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::Checkbox("Run simulation", &runSimulation);

            if(ImGui::Button("Reset"))
                sph_solver.h_params->totalParticles=0;

            ImGui::SliderFloat("yield", &sph_solver.h_params->yield, 0.0f, 0.3f);
            ImGui::SliderFloat("stiffness", &sph_solver.h_params->stiffness, 0.0f, 0.5f);
            ImGui::SliderFloat("nearStiffness", &sph_solver.h_params->nearStiffness, 0.0f, 0.15f);
            ImGui::SliderFloat("linearViscocity", &sph_solver.h_params->linearViscocity, 0.0f, 27.5f);
            ImGui::SliderFloat("quadraticViscocity", &sph_solver.h_params->quadraticViscocity, 0.0f, 27.5f);

            ImGui::SliderFloat("alphaSpring", &sph_solver.h_params->alphaSpring, 0.05f, 1.15f);
            ImGui::SliderFloat("kSpring", &sph_solver.h_params->kSpring, 0.05f, 1.15f);


            static int item_current = 0;
            const char* items[] = {  "simpleColor",
                                      "tempColor",
                                      "velocityColor" };

            ImGui::SliderFloat("temp increase", &sph_solver.h_params->temp_increase, -10.0f, 10.0f);
            ImGui::SliderFloat("temp decrease", &sph_solver.h_params->temp_decrease, 0.0f, 4.75f);

            ImGui::SliderFloat("gravity x", &sph_solver.h_params->gravity.x, -9.0f, 9.0f);
            ImGui::SliderFloat("gravity y", &sph_solver.h_params->gravity.y, -9.0f, 9.0f);




            ImGui::SliderFloat("left-right", &sph_solver.h_params->gravity_coeff_max, -15.0f, 15.0f);
            ImGui::SliderFloat("up-down", &sph_solver.h_params->gravity_coeff_min, -15.0f, 15.0f);


            ImGui::Combo("Draw type", &item_current, items, IM_ARRAYSIZE(items));
            sph_solver.h_params->drawListSet=item_current;




            ImGui::Checkbox("Draw ArtPoints", &sph_solver.h_params->drawArtPoint);
            ImGui::SliderFloat("Pixel transperency", &sph_solver.h_params->artPixelTransperency, 0.0f, 1.0f);

        ImGui::End();



        ImVec2 uv0 = ImVec2(10.0f/256.0f, 10.0f/256.0f);

        // Normalized coordinates of pixel (110,210) in a 256x256 texture.
        ImVec2 uv1 = ImVec2((10.0f+100.0f)/256.0f, (10.0f+200.0f)/256.0f);

        ImGui::Begin("Texture 2 viewer");
            ImGui::Text("pointer = %p", mSharedTex->mId);
            ImGui::Text("size = %d x %d", mSharedTex->mWidth,  mSharedTex->mHeight);
            ImGui::Image((void*)(intptr_t)mSharedTex->mId, ImVec2(mSharedTex->mWidth, mSharedTex->mHeight));
        ImGui::End();




    }

    // Rendering
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

   // ImGuiIO& io = ImGui::GetIO(); (void)io;
   // glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
   // glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
   // glClear(GL_COLOR_BUFFER_BIT);

}

void GraphModule::ClearScreen()
{

}

void GraphModule::CloseRender()
{
    //close program, return true
    SDL_StopTextInput();
    SDL_DestroyWindow(screen);
    screen = NULL;
    SDL_Quit();

}

void  GraphModule::HandleEvents(SDL_Event e, Cuda_solver& sph_solver){

}
