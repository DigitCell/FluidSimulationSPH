#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#pragma once


#include <GL/glew.h>
#include "structs.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "glm/ext.hpp"
#include "glm/glm.hpp"


struct PosColorLayout
{
    float m_pos[3];
    uint8_t m_color[4];
};

struct PosTexLayout
{
    float m_pos[3];
    float m_texcoord[3];
};

struct PosTexLayout2
{
    float m_pos[3];
    float m_texcoord[2];
};

struct PosColorVertex
{
    float m_pos[3];
    float m_normal[3];
    uint32_t m_abgr;
};

//screen properties
const int SCREEN_FPS = 30;
#define screenWidth  1200
#define screenHeight 800

#define mainscreenWidth  1600
#define mainscreenHeight 1600

#define viewWidth 40.0f
#define viewHeight (screenHeight*viewWidth/screenWidth)
#define coeffDisplay 30.0f

// simulation parameters in constant memory


#define maxParticles 35050//6000

//Particle size
#define particleRadius 0.05f
#define particleHeight (6*particleRadius)

//Particle rendering properties
#define drawRatio 3.0f
#define velocityFactor 1.00f

#define frameRate 39.0f
#define timeStep 3
#define Pi 3.14159265f
#define deltaTime ((1.0f/frameRate) / timeStep)

//#define restDensity 75.0f
//#define surfaceTension 0.0006f
#define multipleParticleTypes true
/*
SDL_Window* screen = NULL;
SDL_GLContext gContext;

//world boundaries
Boundary boundaries[4] =
{
    Boundary(1, 0, 0),
    Boundary(0, 1, 0),
    Boundary(-1, 0, -viewWidth),
    Boundary(0, -1, -viewHeight)
};
*/
//types of materials, rgb orders lightest to heaviest, and the sources that will use them
#define redParticle ParticleType(   Vec4(0.5f, 0.71f, 0.1f, 1.0f), 1.25f)
#define greenParticle ParticleType( Vec4(0.3f, 0.95f, 0.71f, 1.0f), 1.24f)
#define blueParticle ParticleType(  Vec4(0.1f, 0.7f, 0.75f, 1.0f), 1.27f)
#define yellowParticle ParticleType(Vec4(0.7f, 0.51f, 0.1f, 1.0f), 1.26f)


//the map of the scene
const int mapWidth =  (int)(viewWidth / particleHeight);
const int mapHeight = (int)(viewHeight / particleHeight);




#endif // CONSTANTS_HPP
