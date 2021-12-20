#ifndef SMANDELBROTR_APP_VBO_H
#define SMANDELBROTR_APP_VBO_H

//#include <GL/glew.h>

#include "../Constants.hpp"


#define _GL_CHECK(_check, _call) \
                BX_MACRO_BLOCK_BEGIN \
                    /*BX_TRACE(#_call);*/ \
                    _call; \
                    GLenum gl_err = glGetError(); \
                    _check(0 == gl_err, #_call "; GL error 0x%x: %s", gl_err, glEnumName(gl_err) ); \
                    BX_UNUSED(gl_err); \
                BX_MACRO_BLOCK_END

#define IGNORE_GL_ERROR_CHECK(...) BX_NOOP()

#if BGFX_CONFIG_DEBUG
#	define GL_CHECK(_call)   _GL_CHECK(BX_ASSERT, _call)
#	define GL_CHECK_I(_call) _GL_CHECK(IGNORE_GL_ERROR_CHECK, _call)
#else
#	define GL_CHECK(_call)   _call
#	define GL_CHECK_I(_call) _call
#endif // BGFX_CONFIG_DEBUG



class AppVboTex {
    public:

    GLuint m_vboId;
    GLuint m_iboId;

    int sizeofIndex;

    AppVboTex(int numVertex, PosTexLayout2 *vertex_data, int numIndexs, GLuint *index_data)
    {
        sizeofIndex=numIndexs;
        glGenBuffers(1, &m_vboId);
        glBindBuffer(GL_ARRAY_BUFFER, m_vboId);
        GL_CHECK(glBufferData(GL_ARRAY_BUFFER, numVertex*sizeof(PosTexLayout2), vertex_data, GL_DYNAMIC_DRAW));

        glGenBuffers(1, &m_iboId);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_iboId);
        GL_CHECK(glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeofIndex*sizeof(GLuint), index_data, GL_DYNAMIC_DRAW));

    }
    void updateVbo(int numVertex, float *vertex_data)
    {
        // data better be the same length as mNumAttrs!
        glBindBuffer(GL_ARRAY_BUFFER, m_vboId);
        glBufferSubData(GL_ARRAY_BUFFER, 0, numVertex*sizeof(PosTexLayout2), vertex_data);
    }
};


class AppVboPoint {
    public:

    GLuint m_vboId;
    GLuint m_iboId;

    AppVboPoint(int numVertex, PosColorLayout *vertex_data, int numIndexs, GLuint *index_data)
    {
        glGenBuffers(1, &m_vboId);
        glBindBuffer(GL_ARRAY_BUFFER, m_vboId);
        GL_CHECK(glBufferData(GL_ARRAY_BUFFER, numVertex*sizeof(PosColorLayout), vertex_data, GL_DYNAMIC_DRAW));

        glGenBuffers(1, &m_iboId);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_iboId);
        GL_CHECK(glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndexs*sizeof(GLuint), index_data, GL_DYNAMIC_DRAW));

    }
    void updateVbo(int numVertex, PosColorLayout *vertex_data)
    {
        // data better be the same length as mNumAttrs!
        glBindBuffer(GL_ARRAY_BUFFER, m_vboId);
      //  GL_CHECK(glBufferData(GL_ARRAY_BUFFER, numVertex*sizeof(PosColorLayout), vertex_data, GL_DYNAMIC_DRAW));
      //  GL_CHECK(glBufferData(GL_ARRAY_BUFFER, numVertex*sizeof(PosColorLayout),0, GL_DYNAMIC_DRAW));
        glBufferSubData(GL_ARRAY_BUFFER, 0, numVertex*sizeof(PosColorLayout), vertex_data);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }


    void updateIbo(int numIndexs, GLuint *index_data)
    {
        // data better be the same length as mNumAttrs!
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_iboId);
     //   GL_CHECK(glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndexs*sizeof(GLuint), index_data, GL_DYNAMIC_DRAW));
      //  GL_CHECK(glBufferData(GL_ELEMENT_ARRAY_BUFFER, numIndexs*sizeof(GLuint), 0, GL_DYNAMIC_DRAW));
        GL_CHECK(glBufferSubData(GL_ELEMENT_ARRAY_BUFFER,0, numIndexs*sizeof(GLuint), index_data));
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
};


#endif
