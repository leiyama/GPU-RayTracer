#ifndef SCENE_HPP
#define SCENE_HPP

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include "Eigen/Dense"

//#include "command_line.hpp"
#include "model.hpp"
#include "Utilities.hpp"

using namespace std;
using namespace Eigen;

struct PointLight {
    float position[4];
    float color[3];
    float k; // Attenuation coefficient

    PointLight() = default;
    PointLight(float *position, float *color, float k);
};

static const bool default_needs_update = false;

class Scene {
    public:
        static Scene *singleton;
        static Scene *getSingleton();

        vector<Object *> root_objs;

        unordered_map<Primitive*, unsigned int> prm_tessellation_start;
        vector<PointLight> lights;

        vector<Vector3f> vertices;
        vector<Vector3f> normals;

        Scene();

        void createLights();
        int getLightCount();

        // THIS IS THE FUNC WHICH NEED CHANGE WITHOUT command_line CLASS!!!!
        void update();
        void update(Renderable* ren);

    private:
        bool needs_update;

        void generateVertex(Primitive *prm, float u, float v);
        void tessellatePrimitive(Primitive *prm);
        void tessellateObject(Object *obj);
};

#endif