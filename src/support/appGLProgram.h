#ifndef SMANDELBROTR_APP_GL_PROGRAM_H
#define SMANDELBROTR_APP_GL_PROGRAM_H

#include "glm/ext.hpp"
#include "glm/glm.hpp"
#include <GL/glew.h>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>



class AppGLProgram {

   public:
    GLuint mId;
    GLuint mUniCameraToView;
    GLuint texLocation;
    std::string readFile(const std::string &fileName)
    {
        std::ifstream ifs(fileName.c_str(), std::ios::in | std::ios::binary | std::ios::ate);

        if (ifs.is_open()) {
            std::ifstream::pos_type fileSize = ifs.tellg();
            ifs.seekg(0, std::ios::beg);

            std::vector<char> bytes(fileSize);
            ifs.read(bytes.data(), fileSize);

            return std::string(bytes.data(), fileSize);
        }
        else {
            std::cerr << "Failed to open file: " << fileName << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    GLuint createShader(const std::string &resource, GLuint type)
    {
        GLuint shader = glCreateShader(type);
        std::string source = readFile(resource);
        const char *source_c_str = source.c_str();
        glShaderSource(shader, 1, &source_c_str, NULL);
        glCompileShader(shader);
        GLint compiled;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
        GLsizei lengthOfLog;
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, &lengthOfLog, infoLog);
        if (lengthOfLog > 0) {
            std::cout << infoLog << std::endl;
        }
        if (compiled == 0) {
            std::cerr << "Could not compile shader: " << resource << std::endl;
            exit(99);
        }
        return shader;
    }

    AppGLProgram(const std::string &vertProgramPath, const std::string &fragProgramPath)
    {
        GLuint vshader = createShader(vertProgramPath, GL_VERTEX_SHADER);
        GLuint fshader = createShader(fragProgramPath, GL_FRAGMENT_SHADER);

        mId = glCreateProgram();
        glAttachShader(mId, vshader);
        glAttachShader(mId, fshader);
        glLinkProgram(mId);
        int linked;
        glGetProgramiv(mId, GL_LINK_STATUS, &linked);
        GLsizei lengthOfLog;
        char programLog[512];
        glGetProgramInfoLog(mId, 512, &lengthOfLog, programLog);
        if (lengthOfLog > 0) {
            std::cout << programLog << std::endl;
        }
        if (linked == 0) {
            std::cerr << "Could not link program" << std::endl;
            exit(99);
        }

    };

    // bind this program so OpenGL will use it.
    void bind() {
        glUseProgram(mId);
    }
    // unbind this program so OpenGL will not use it.
    void unbind() {
        glUseProgram(0);
    }
    // update the camera-to-view matrix uniform.

    void updateCameraToView(glm::mat4 cameraToView)
    {
        mUniCameraToView = glGetUniformLocation(mId, "cameraToView");
        glUniformMatrix4fv(mUniCameraToView, 1, GL_FALSE, glm::value_ptr(cameraToView));
    }
};


class AppGLProgramTex: public AppGLProgram {
    GLuint mAttrPosition;
    GLuint mAttrTexCoords;

  public:

    AppGLProgramTex(const std::string &vertProgramPath, const std::string &fragProgramPath):
        AppGLProgram(vertProgramPath, fragProgramPath) { init();};

    void init()
    {
        glUseProgram(mId);
        texLocation = glGetUniformLocation(mId, "uTexture");
        glUniform1i(texLocation, 0);
        glUseProgram(0);
    }

    void UpdateTexture(GLuint texture)
    {
        glUseProgram(mId);
        glUniform1i(texLocation, texture);
        glUseProgram(0);
    }
    GLuint attrPosition() { return mAttrPosition; }
    GLuint attrTexCoords() { return mAttrTexCoords; }
};



class AppGLProgramPoint: public AppGLProgram {
    GLuint mAttrPosition;
    GLuint mAttrColor;

  public:

    AppGLProgramPoint(const std::string &vertProgramPath, const std::string &fragProgramPath):
        AppGLProgram(vertProgramPath, fragProgramPath) { init();};

    void init()
    {
        glUseProgram(mId);
       // mAttrPosition =     glGetAttribLocation(mId,  "position");
       // mAttrColor =        glGetAttribLocation(mId,  "color");
      //  mUniCameraToView =  glGetUniformLocation(mId, "cameraToView");
        glUseProgram(0);
    }
    GLuint attrPosition() { return mAttrPosition; }
    GLuint attrColor() { return mAttrColor; }
};



#endif
