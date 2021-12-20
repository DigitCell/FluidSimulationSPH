cmake_minimum_required(VERSION 3.14)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61)
  message(CMAKE_CUDA_ARCHITECTURES)
endif()





set(GLM_DIR    ${CMAKE_SOURCE_DIR}/external/glm-0.9.9.8)
set(IMGUI_ROOT ${CMAKE_SOURCE_DIR}/external/imgui-1.85)

# SDL2, glew, glm, ImGui
add_subdirectory("external")

include(CheckLanguage)
check_language(CUDA)

project(smandelbrotr
    VERSION     0.12
    DESCRIPTION "SDL2 CUDA OpenGL Fluid Simulation"
    LANGUAGES   CUDA CXX
)

# IMGUI
set(IMGUI_SOURCES ${IMGUI_ROOT}/imgui.cpp ${IMGUI_ROOT}/imgui_draw.cpp ${IMGUI_ROOT}/imgui_widgets.cpp ${IMGUI_ROOT}/imgui_tables.cpp)
set(IMGUI_IMPL_SOURCES ${IMGUI_ROOT}/backends/imgui_impl_sdl.cpp ${IMGUI_ROOT}/backends/imgui_impl_opengl3.cpp)


SET(TARGET_H
   # src/support/app.h
   # src/support/appGL.h
    src/support/appVbo.h
    src/support/appPbo.h
    src/support/appVerts.h
    src/support/appTexture.h
   # src/support/appWindow.h
    src/support/appGLProgram.h

    src/support/tinycolormap.hpp
    src/structs.h
    src/Constants.hpp

    src/graphmodule.hpp
    src/mainloop.hpp
    src/Constants.hpp

)

SET(TARGET_SRC

   # src/support/app.cpp
   # src/support/appGL.cpp

    src/support/appTexture.cpp

    src/graphmodule.cpp
    src/mainloop.cpp
    src/main.cpp
)

SET(TARGET_CUDA_H

    src/support/cudaErrorCheck.h
    src/cuda_solver.cuh
    src/support/CudaHelper.cuh
    src/support/cuda_noise.cuh

)

SET(TARGET_CUDA_SRC

    src/cuda_solver.cu

)

add_executable(smandelbrotr ${TARGET_CUDA_H} ${TARGET_CUDA_SRC}
                            ${TARGET_SRC} ${TARGET_H}
                            ${IMGUI_SOURCES} ${IMGUI_IMPL_SOURCES})

target_include_directories(smandelbrotr PUBLIC "${IMGUI_ROOT}")
target_include_directories(smandelbrotr PUBLIC "${IMGUI_ROOT}/backends")
                            
# CUDA wants C++11
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(Cuda_common_include "/usr/local/cuda/samples/common/inc")
include_directories(${Cuda_common_include})

set(homeDir "/home/geotyper")
set(core_math  ${homeDir}/CPP_lib/Core_math)
include_directories(${core_math})

set(stb_lib  ${homeDir}/CPP_lib/stb)
include_directories(${stb_lib})

set_target_properties(smandelbrotr PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(smandelbrotr PUBLIC cxx_std_11)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_include_directories(smandelbrotr PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_include_directories(smandelbrotr PUBLIC "${GLM_DIR}")

# Windows prereq: get dev libraries from https://www.libsdl.org/download-2.0.php
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
# workaround for CMP0004 issue with the SDL2 libraries on Linux
string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)
target_link_libraries(smandelbrotr ${SDL2_LIBRARIES})

# OpenGL
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED COMPONENTS OpenGL)
if(NOT TARGET OpenGL::OpenGL)
  target_link_libraries(smandelbrotr OpenGL::GL)
  target_include_directories(smandelbrotr PUBLIC OpenGL::GL)
else()
  target_link_libraries(smandelbrotr OpenGL::OpenGL)
  target_include_directories(smandelbrotr PUBLIC OpenGL::OpenGL)
endif()

# GL Extension Wrangler (GLEW) get dev libraries from http://glew.sourceforge.net/
find_package(GLEW REQUIRED)
target_link_libraries(smandelbrotr "${GLEW_LIBRARIES}")
target_include_directories(smandelbrotr PUBLIC "${GLEW_INCLUDE_DIRS}")
target_include_directories(smandelbrotr PRIVATE ${INCLUDES} ${core_math} )

