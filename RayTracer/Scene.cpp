#include "Scene.hpp"

using namespace std;

Scene *Scene::singleton;

/* Constructs a light in our scene. */
PointLight::PointLight(float *position, float *color, float k) {
    for (int i = 0; i < 3; i++) {
        this->position[i] = position[i];
        this->color[i] = color[i];
    }
    this->position[3] = position[3];
    this->k = k;
}

/* Initializes the scene's data structures. */
Scene::Scene() : needs_update(default_needs_update) {
    this->prm_tessellation_start = unordered_map<Primitive*, unsigned int>();
    // Scene::objects = vector<Object>();
    this->vertices = vector<Vector3f>();
    this->normals = vector<Vector3f>();
    this->lights = vector<PointLight>();
    this->root_objs = vector<Object *>();
    this->createLights();
}

/* Returns/sets up the singleton instance of the class. */
Scene *Scene::getSingleton() {
    if (!Scene::singleton) {
        Scene::singleton = new Scene();
    }

    return Scene::singleton;
}

/* Creates the scene's lights. */
void Scene::createLights() {
    float position[4] = {3.0, 4.0, 5.0, 1.0};
    float color[3] = {1.0, 1.0, 1.0};
    float k = 0.002;

    // Set up a single light
    this->lights.emplace_back(position, color, k); // different from push_back, emplace_back doesn't create new class.
}

/*
 * Adds a vertex and normal to the respective buffers, calculated from a 
 * parametric (u, v) point on a superquadric's surface.
 */
void Scene::generateVertex(Primitive *prm, float u, float v) {
    this->vertices.push_back(prm->getVertex(u, v));
    this->normals.push_back(prm->getNormal(this->vertices.back()));
}

/* Tesselates a primitive, adding the vertices and normals to their buffers. */
void Scene::tessellatePrimitive(Primitive *prm) {
    if (this->prm_tessellation_start.find(prm) != this->prm_tessellation_start.end()) {
        return;
    }

    this->prm_tessellation_start.insert({prm, vertices.size()});

    // Less typing, computation at runtime
    int ures = prm->getPatchX();
    int vres = prm->getPatchY();
    float u, v;
    float half_pi = M_PI / 2;
    float du = 2 * M_PI / ures, dv = M_PI / vres;

    // Create GL_TRIANGLE_STRIPs by moving circumferentially around each of its
    // vres - 2 non-polar latitude ranges
    v = dv - half_pi;
    for (int j = 1; j < vres - 1; j++) {
        // U sweeps counterclockwise from -pi to pi, so the first edge should
        // point down in order for the right-hand rule to make the normal
        // point out of the primitive
        u = -M_PI;
        for (int i = 0; i < ures; i++) {
            this->generateVertex(prm, u, v + dv);
            this->generateVertex(prm, u, v);
            u += du;
        }
        // Connect back to the beginning
        this->generateVertex(prm, -M_PI, v + dv);
        this->generateVertex(prm, -M_PI, v);

        v += dv;
    }

    // Draw the primitive's bottom by filling in its southernmost latitude range
    // with a GL_TRIANGLE_FAN centered on the south pole
    u = M_PI;
    v = dv - half_pi;
    this->generateVertex(prm, u, -half_pi);
    for (int i = 0; i < ures; i++) {
        // U sweeps clockwise to make the normals point out
        this->generateVertex(prm, u, v);
        u -= du;
    }
    // Connect back to the beginning
    u = -M_PI;
    this->generateVertex(prm, u, v);

    // Now we tessellate its top by doing the same at the north pole
    v *= -1;
    this->generateVertex(prm, u, half_pi);
    for (int i = 0; i < ures; i++) {
        // U sweeps counterclockwise to make the normals point out
        this->generateVertex(prm, u, v);
        u += du;
    }
    // Connect back to the beginning
    u = M_PI;
    this->generateVertex(prm, u, v);
}

void Scene::tessellateObject(Object *obj) {
    for (auto& child_it : obj->getChildren()) {
        Renderable* ren = Renderable::get(child_it.second.name);
        switch (ren->getType()) {
            case OBJ: {
                Object* obj = dynamic_cast<Object*>(ren);
                this->tessellateObject(obj);
                break;
            }
            case PRM: {
                Primitive* prm = dynamic_cast<Primitive*>(ren);
                this->tessellatePrimitive(prm);
                break;
            }
            default:
                fprintf(stderr, "Scene::tessellateObject ERROR invalid Renderable type %s\n",
                    toCstr(ren->getType()));
                exit(1);
        }
    }
}

/*
 * TODO: THIS FUNC CONTAINS COMMAND_LINE CLASS WHICH NEED TO BE DELETED
 * UPDATE: if the scene remains the same all the time, there is no need to update the scene.
 * 2016/5/29
 *
 * Regenerates the scene's vertex and normal buffers based on currently selected
 * Renderable
 */

 void Scene::update(Renderable* ren) {
    assert(ren->getType() == PRM);
    this->tessellatePrimitive(dynamic_cast<Primitive*>(ren));
 }
 
void Scene::update() {
    // this->root_objs.clear();
    // this->prm_tessellation_start.clear();
    // this->vertices.clear();
    // this->normals.clear();

    // //const Line* cur_state = CommandLine::getState();

    // Renderable* ren = NULL;
    // char sphere;
    // char* name_char = &sphere;
    // Name *name = Name::getSingleton(name_char);
    // ren = Renderable::get(*name);

    // assert(ren->getType() == PRM);
    // this->tessellatePrimitive(dynamic_cast<Primitive*>(ren));

    // if (cur_state) {
    //     switch (cur_state->toCommandID()) {
    //         case Commands::primitive_get_cmd_id: {
    //             Renderable* ren = Renderable::get(cur_state->tokens[1]);
    //             assert(ren->getType() == PRM);
    //             this->tessellatePrimitive(dynamic_cast<Primitive*>(ren));
    //             break;
    //         }
    //         case Commands::object_get_cmd_id: {
    //             Renderable* ren = Renderable::get(cur_state->tokens[1]);
    //             assert(ren->getType() == OBJ);
    //             this->root_objs.push_back(dynamic_cast<Object*>(ren));
    //             break;
    //         }
    //         default:
    //            fprintf(stderr, "ERROR Commands:info invalid state CommandID %d from current state\n",
    //                cur_state->toCommandID());
    //             exit(1);
    //     }
    // }

    // for (Object* obj : root_objs) {
    //     this->tessellateObject(obj);
    // }
}