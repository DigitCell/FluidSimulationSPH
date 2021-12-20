#include "app.h"

#include "imgui.h"
#include "backends/imgui_impl_sdl.h"
#include "backends/imgui_impl_opengl3.h"

#ifdef WIN32
// don't interfere with std::min,max
#define NOMINMAX
// https://seabird.handmade.network/blogs/p/2460-be_aware_of_high_dpi
#pragma comment(lib, "Shcore.lib")

#include <ShellScalingAPI.h>
#include <comdef.h>
#include <windows.h>
#endif

App::App()
{
    mAppWindow = nullptr;
    mAppGL = nullptr;
    mSDLWindow = nullptr;
    mSDLGLContext = nullptr;

    mSwitchFullscreen = false;
    mIsFullscreen = false;
    mZoomOutMode = false;
    mSaveImage = false;
    mMouseDown = false;
    mReverseZoomMode = false;
    mShowGUI = true;

    //cuda_solver=nullptr;

    cuda_solver=new Cuda_solver();

    mPrevWindowWidth = mPrevWindowHeight = -1;
    mPrevWindowX = mPrevWindowY = -1;

    mMonitorWidth = mMonitorHeight = -1;

    mMouseStartX = mMouseStartY =
        mMouseX = mMouseY = mCenterStartX = mCenterStartY = -1;
}

void App::run(const int cudaDevice, const std::string &shaderPath)
{
    std::cout << "SDL2 CUDA OpenGL Mandelbrotr" << std::endl;
    std::cout << "Versions-------------+-------" << std::endl;
    SDL_version compiled;
    SDL_version linked;

    SDL_VERSION(&compiled);
    SDL_GetVersion(&linked);
    std::cout << "SDL compiled version | " << int(compiled.major) << "." << int(compiled.minor) << "." << int(compiled.patch) << std::endl;
    std::cout << "SDL linked version   | " << int(linked.major) << "." << int(linked.minor) << "." << int(linked.patch) << std::endl;

    int v;
    cudaRuntimeGetVersion(&v);
    int major = v / 1000;
    int minor = (v - 1000 * major) / 10;
    std::cout << "CUDA runtime version | " << major << "." << minor << std::endl;
    cudaRuntimeGetVersion(&v);
    major = v / 1000;
    minor = (v - 1000 * major) / 10;
    std::cout << "CUDA driver version  | " << major << "." << minor << std::endl;

    std::cout << "GLM version          | " << GLM_VERSION << std::endl;

    if (!init(cudaDevice, shaderPath)) {
        loop();
    }
    cleanup();
}

void App::cleanup()
{
    std::cout << "Exiting..." << std::endl;
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    SDL_DestroyWindow(mSDLWindow);
}

bool App::init(const int cudaDevice, const std::string &shaderPath)
{
    if (initWindow()) {
        std::cerr << "Failed to create main window" << std::endl;
        return true;
    }
    mAppGL = new AppGL(mAppWindow, mAppWindow->width(),mAppWindow->height(), shaderPath);
 //   mAppGL->mSharedPboTex->registerBuffer();

    return false;
}

// initialize SDL2 window
// return true on error
bool App::initWindow()
{
#ifdef WIN32
    SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);
#endif

    // Initialize SDL Video
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "Failed to initialize SDL video" << std::endl;
        return true;
    }

    SDL_DisplayMode DM;
    SDL_GetCurrentDisplayMode(0, &DM);
    mMonitorWidth = DM.w;
    mMonitorHeight = DM.h;

    //mAppWindow = new AppWindow(startDim, startDim);
    mAppWindow = new AppWindow(screenWidth, screenHeight);

    // Create main window
    mSDLWindow = SDL_CreateWindow(
        "SMandelbrotr",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        mAppWindow->width(),mAppWindow->height(),
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    if (mSDLWindow == NULL) {
        std::cerr << "Failed to create main window" << std::endl;
        SDL_Quit();
        return true;
    }

    // Initialize rendering context
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);


    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 8);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                       SDL_GL_CONTEXT_PROFILE_COMPATIBILITY );

    mSDLGLContext = SDL_GL_CreateContext(mSDLWindow);
    if (mSDLGLContext == NULL) {
        std::cerr << "Failed to create GL context" << std::endl;
        SDL_DestroyWindow(mSDLWindow);
        SDL_Quit();
        return true;
    }

    SDL_GL_SetSwapInterval(1); // Use VSYNC

    // Initialize GL Extension Wrangler (GLEW)

    GLenum err;
    glewExperimental = GL_TRUE; // Please expose OpenGL 3.x+ interfaces
    err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "Failed to init GLEW" << std::endl;
        SDL_GL_DeleteContext(mSDLGLContext);
        SDL_DestroyWindow(mSDLWindow);
        SDL_Quit();
        return true;
    }

    // initialize imgui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    // setup Dear ImGui style
    ImGui::StyleColorsDark();

    ImGui::LoadIniSettingsFromDisk("tempImgui.ini");
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    static ImGuiStyle* style = &ImGui::GetStyle();
    style->Alpha = 0.5f;

    //io.WantCaptureMouse=true;

    // setup platform/renderer bindings
    ImGui_ImplSDL2_InitForOpenGL(mSDLWindow, mSDLGLContext);
    ImGui_ImplOpenGL3_Init("#version 330");

    // colors are set in RGBA, but as float
    //ImVec4 background = ImVec4(35/255.0f, 35/255.0f, 35/255.0f, 1.00f);


#ifndef NDEBUG
    int major, minor;
    SDL_GL_GetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, &major);
    SDL_GL_GetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, &minor);
    std::cout << "OpenGL version       | " << major << "." << minor << std::endl;
    std::cout << "GLEW version         | " << glewGetString(GLEW_VERSION) << std::endl;
    std::cout << "---------------------+-------" << std::endl;
#endif

    return false;
}


void App::loop()
{
    std::cout << "Running main loop" << std::endl;
    bool running = true;
    static Uint32 lastFrameEventTime = 0;
    const Uint32 debounceTime = 100; // 100ms

    while (running) {
        Uint32 curTime = SDL_GetTicks();
        SDL_Event event;
        while (SDL_PollEvent(&event)) {

            ImGui_ImplSDL2_ProcessEvent(&event);
            ImGuiIO& io = ImGui::GetIO();
            if (io.WantCaptureKeyboard || io.WantCaptureMouse) {
                break;
            }
            HandleEvents(event, running);
        }

        update();

        cuda_solver->Update();
       // mAppGL->render(cuda_solver);
        GuiRender();

        SDL_GL_SwapWindow(mSDLWindow);
    }
}

void App::GuiRender()
{
    {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame(mSDLWindow);
        ImGui::NewFrame();
        if(mShowGUI) {
            ImGui::Begin("Information", NULL, ImGuiWindowFlags_AlwaysAutoResize);
            ImGui::Text("Number particles %i",cuda_solver->h_params->totalParticles);
            ImGui::Text("Framerate  : %.1f ms or %.1f Hz", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }
}

// Thanks! https://gist.github.com/wduminy/5859474
SDL_Surface *flip_surface(SDL_Surface *sfc)
{
    SDL_Surface *result = SDL_CreateRGBSurface(sfc->flags, sfc->w, sfc->h,
                                               sfc->format->BytesPerPixel * 8, sfc->format->Rmask, sfc->format->Gmask,
                                               sfc->format->Bmask, sfc->format->Amask);
    const auto pitch = sfc->pitch;
    const auto pxlength = pitch * (sfc->h - 1); // FIXED BUG
    auto pixels = static_cast<unsigned char *>(sfc->pixels) + pxlength;
    auto rpixels = static_cast<unsigned char *>(result->pixels);
    for (auto line = 0; line < sfc->h; ++line) {
        memcpy(rpixels, pixels, pitch);
        pixels -= pitch;
        rpixels += pitch;
    }
    return result;
}

void App::update()
{
    mAppGL->handleResize();

    // handle fullscreen
    /*
    if (mSwitchFullscreen) {

        std::cout << "switch fullscreen ";

        mSwitchFullscreen = false;
        if (mIsFullscreen) { // switch to windowed
            std::cout << "to windowed" << std::endl;

            mIsFullscreen = false;
            SDL_SetWindowFullscreen(mSDLWindow, 0);
            SDL_RestoreWindow(mSDLWindow); // Seemingly required for Jetson
            SDL_SetWindowSize(mSDLWindow, mPrevWindowWidth, mPrevWindowHeight);
            SDL_SetWindowPosition(mSDLWindow, mPrevWindowX, mPrevWindowY);
        }
        else { // switch to fullscreen

            std::cout << std::endl;
            mIsFullscreen = true;
            mPrevWindowWidth = mAppWindow->width();
            mPrevWindowHeight = mAppWindow->height();
            SDL_GetWindowPosition(mSDLWindow, &mPrevWindowX, &mPrevWindowY);
            SDL_SetWindowSize(mSDLWindow, mMonitorWidth, mMonitorHeight);
            SDL_SetWindowFullscreen(mSDLWindow, SDL_WINDOW_FULLSCREEN_DESKTOP); // "fake" fullscreen
        }
    }
*/

    if (mMouseDown) {
        double dx = mMouseX - mMouseStartX;
        double dy = mMouseY - mMouseStartY;
        //#ifdef DEBUG
        //        std::cerr << "dx,dy = " << dx << ", " << dy << std::endl;
        //#endif

        //mAppMandelbrot->centerX(mCenterStartX - centerDx);
        //mAppMandelbrot->centerY(mCenterStartY - centerDy);
    }

    // saveImage
    if (mSaveImage) {
        mSaveImage = false;
        std::cout << "Saving save.bmp" << std::endl;
        SDL_Surface *surface = SDL_CreateRGBSurfaceFrom(
            (void *)mAppGL->readPixels(),
            mAppWindow->width(), mAppWindow->height(), 32, 4 * mAppWindow->width(),
            0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000);
        SDL_Surface *flipped_surface = flip_surface(surface);
        SDL_SaveBMP(flipped_surface, "save.bmp");
        SDL_FreeSurface(flipped_surface);
        SDL_FreeSurface(surface);
    }
}

void App::resize(unsigned width, unsigned height)
{
    if (width > 0 && height > 0 &&
        (mAppWindow->width() != width || mAppWindow->height() != height)) {
     //   mAppWindow->width(width);
     //   mAppWindow->height(height);
    }
}

void App::HandleEvents(SDL_Event event, bool& running)
{

    if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE) {
        running = false;

    }
    else if (event.type == SDL_KEYDOWN) {
        switch (event.key.keysym.sym) {
        case SDLK_ESCAPE:
            running = false;
            ImGui::SaveIniSettingsToDisk("tempImgui.ini");
            break;
        case SDLK_TAB:
            mShowGUI = !mShowGUI;

        case SDLK_RETURN:
            mZoomOutMode = true;
            break;

        case SDLK_w:
            mSaveImage = true;
            break;
        case SDLK_LSHIFT:
            mReverseZoomMode = true;
            break;
        }
    }
    else if (event.type == SDL_KEYUP) {
        switch (event.key.keysym.sym) {
        case SDLK_LSHIFT:
            mReverseZoomMode = false;
            break;
        }
    }
    else if (event.type == SDL_WINDOWEVENT) {
        if ((event.window.event == SDL_WINDOWEVENT_RESIZED) ||
            (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)) {
            resize(event.window.data1, event.window.data2);
        }
    }
    else if (event.type == SDL_MOUSEBUTTONDOWN) {
        if (event.button.button == SDL_BUTTON_LEFT) {
            mMouseStartX = mMouseX = event.button.x;
            mMouseStartY = mMouseY = event.button.y;
            //mCenterStartX = mAppMandelbrot->centerX();
            //mCenterStartY = mAppMandelbrot->centerY();
            mMouseDown = true;
        }
    }
    else if (event.type == SDL_MOUSEMOTION) {
        mMouseX = event.motion.x;
        mMouseY = event.motion.y;
    }
    else if (event.type == SDL_MOUSEBUTTONUP) {
        if (event.button.button == SDL_BUTTON_LEFT) {
            mMouseDown = false;
        }
    }
    else if (event.type == SDL_MOUSEWHEEL) {
        const double zoomFactor = 1.1;
        // my Lenovo laptop's trackpad does not work with mousewheel--strange!
        // Alright then, use LSHIFT key to control direction if necessary.
        bool zoomIn = mReverseZoomMode ? event.wheel.y >= 0 : event.wheel.y < 0;

    }
}
