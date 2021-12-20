#include "mainloop.hpp"

MainLoop::MainLoop()
{

}

bool MainLoop::RunLoop()
{

    std::cout << "Running main loop" << std::endl;
    bool running = true;
    static Uint32 lastFrameEventTime = 0;
    const Uint32 debounceTime = 100; // 100ms

    bool done = false;
    while (!done)
    {
            Uint32 curTime = SDL_GetTicks();
            while (SDL_PollEvent(&graphModule.event))
            {
                ImGui_ImplSDL2_ProcessEvent(&graphModule.event);
                ImGuiIO& io = ImGui::GetIO();
                if (graphModule.event.type == SDL_QUIT)
                {
                    done = true;
                    ImGui::SaveIniSettingsToDisk("tempImgui.ini");
                }
                if (graphModule.event.type == SDL_WINDOWEVENT && graphModule.event.window.event == SDL_WINDOWEVENT_CLOSE && graphModule.event.window.windowID == SDL_GetWindowID(graphModule.screen))
                    done = true;
                else
                {
                    graphModule.HandleEvents(graphModule.event, cuda_solver);
                }
            }

       if(graphModule.runSimulation)
           cuda_solver.Update();

        graphModule.ClearRenderScreen();



        if(graphModule.useTextureRender){
           graphModule.TextureCudaUpdater(cuda_solver);
          // graphModule.Render(cuda_solver);
        }// update buffer
        if(graphModule.useVBOrender)
            graphModule.Render2(cuda_solver);

        graphModule.GuiRender(cuda_solver);
        SDL_GL_SwapWindow(graphModule.screen); //update window
    }

    graphModule.CloseRender();

    return true;
}
