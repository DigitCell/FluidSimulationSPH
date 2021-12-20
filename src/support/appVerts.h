#ifndef SMANDELBROTR_APP_VERTS_H
#define SMANDELBROTR_APP_VERTS_H

#include "appVbo.h"

#include <GL/glew.h>

class AppVertsTex {
    GLuint mId;
    AppVboTex *mVbo;
    int mNumVerts;
    int mNumIndexs;

  public:
    AppVertsTex(int numVertex, PosTexLayout2 *vertex_data, int numIndexs, GLuint *index_data)
    {
        mNumVerts=numVertex;
        mNumIndexs=numIndexs;

        glGenVertexArrays(1, &mId);
        glBindVertexArray(mId);

        mVbo =  new AppVboTex(numVertex, vertex_data, mNumIndexs, index_data);

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(PosTexLayout2), reinterpret_cast<void*>( offsetof(PosTexLayout2,m_pos)));
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(PosTexLayout2), reinterpret_cast<void*>( offsetof(PosTexLayout2, m_texcoord)));

        glBindVertexArray(0);

    }
    void draw()
    {
       glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void*)0);

       // glPointSize(10.0f);
       // glEnable(GL_POINT_SMOOTH);
       // glDrawElements(GL_POINTS, 6, GL_UNSIGNED_INT, (void*)0);
    }
    void updatePosition(float *newCoords)
    {
      //  mPositionVbo->update(newCoords);
    }
    void updateTexCoords(float *newCoords)
    {
      //  mTexCoordsVbo->update(newCoords);
    }
    // bind this VertexArray so OpenGL can use it
    void bind()
    {
        glBindVertexArray(mId);
    }
    // unbind this VertexArray so OpenGL will stop using it
    void unbind()
    {
         glBindVertexArray(0);
    }
};

class AppVertsPoint {

public:
    GLuint mId;
    AppVboPoint *mVbo;
    int mNumVerts;
    int mNumIndexs;


    AppVertsPoint(int numVertex, PosColorLayout *vertex_data, int numIndexs, GLuint *index_data)
    {
        mNumVerts=numVertex;
        mNumIndexs=numIndexs;

        glGenVertexArrays(1, &mId);
        glBindVertexArray(mId);

        mVbo =  new AppVboPoint(numVertex, vertex_data, mNumIndexs, index_data);

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(PosColorLayout),
                              reinterpret_cast<void*>( offsetof(PosColorLayout,m_pos)));
        glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_FALSE, sizeof(PosColorLayout),
                              reinterpret_cast<void*>( offsetof(PosColorLayout, m_color)));

        glBindVertexArray(0);

    }
    void draw()
    {
       //  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void*)0);
        glPointSize(5.1f);
        glEnable(GL_POINT_SMOOTH);
        glDrawElements(GL_POINTS, mNumIndexs, GL_UNSIGNED_INT, (void*)0);
    }
    void updatePosition(int numVertex, PosColorLayout *newCoords)
    {
       mNumVerts=numVertex;
       mVbo->updateVbo(numVertex, newCoords);
    }
    void updateIndexs(int numIndexs, GLuint *index_data)
    {
        mNumIndexs=numIndexs;
        mVbo->updateIbo(numIndexs, index_data);

    }
    // bind this VertexArray so OpenGL can use it
    void bind()
    {
        glBindVertexArray(mId);
    }
    // unbind this VertexArray so OpenGL will stop using it
    void unbind()
    {
         glBindVertexArray(0);
    }
};


#endif
