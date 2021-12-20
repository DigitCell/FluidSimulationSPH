#ifndef SMANDELBROTR_APP_TEXTURE_H
#define SMANDELBROTR_APP_TEXTURE_H

#pragma once
#include <GL/glew.h>
#include "stb_image.h"
#include "iostream"

using namespace std;

class AppTexture {
  public:
    GLuint mId;
    unsigned mWidth, mHeight;

    AppTexture(unsigned width, unsigned height);

    // copy from the pixel buffer object to this texture. Since the
    // TexSubImage pixels parameter (final one) is 0, Data is coming
    // from a PBO, not host memory
    void copyFromPbo()
    {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, GL_BGRA, GL_UNSIGNED_BYTE, 0);
    }

    unsigned width() { return mWidth; }
    unsigned height() { return mHeight; }

    // bind this texture so OpenGL will use it
    void bind()
    {
        glBindTexture(GL_TEXTURE_2D, mId);
    }

    // unbind this texture so OpenGL will stop using it
    void unbind()
    {
        glBindTexture(GL_TEXTURE_2D, 0);
    }


    void createGLTexturefromFile(GLuint *gl_tex, unsigned int size_x, unsigned int size_y)
    {

        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &mId);
        glBindTexture(GL_TEXTURE_2D, mId);
        // set basic texture parameters

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        float borderColor[] = { 0.1f, 1.0f, 0.0f, 1.0f };
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);


        int width, height, nrChannels;
        unsigned char *data = stbi_load("../shaders/test800_600.png", &width, &height, &nrChannels, 0);
        if (data)
        {
            glTexImage2D(GL_TEXTURE_2D, 0,  GL_RGBA, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
           // glGenerateMipmap(GL_TEXTURE_2D);
           // glBindTexture(GL_TEXTURE_2D, 0);

        }
        else
        {
            std::cout << "Failed to load texture" << std::endl;
        }
        stbi_image_free(data);
        /*
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, gl_tex); // generate 1 texture
        glBindTexture(GL_TEXTURE_2D, *gl_tex); // set it as current target
        // set basic texture parameters

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Specify 2D texture
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size_x, size_y, 0,  GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        */
}

};

#endif
