#include "appTexture.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

AppTexture::AppTexture(unsigned width, unsigned height) : mWidth(width), mHeight(height)
{
    /*
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &mId);
    glBindTexture(GL_TEXTURE_2D, mId);
    // Allocate the texture memory. This will be filled in by the
    // PBO during rendering
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);
    // Set filter mode
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
    */

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &mId);
    glBindTexture(GL_TEXTURE_2D, mId);
    // set basic texture parameters
/*
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    */

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);

    glBindTexture(GL_TEXTURE_2D, 0);
/*
   // float borderColor[] = { 0.1f, 1.0f, 0.0f, 1.0f };
  //  glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);


    int widths, heights, nrChannels;
    unsigned char *data = stbi_load("../shaders/test800_600.png", &widths, &heights, &nrChannels, 0);
    if (data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0,  GL_RGBA, widths, heights, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
       // glGenerateMipmap(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);

        mWidth=widths;
        mHeight=heights;

    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);

*/

   // glBindTexture(GL_TEXTURE_2D, 0);
}
