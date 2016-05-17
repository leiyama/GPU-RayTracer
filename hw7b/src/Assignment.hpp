#ifndef ASSIGNMENT_HPP
#define ASSIGNMENT_HPP

#include <vector>
#include "model.hpp"
#include "Utilities.hpp"
#include "Scene.hpp"
#include "UI.hpp"

#include "PNGMaker.hpp"

class Camera;
class Scene;

class Intersect
{
public:
    float t;
    Vector3f point;
    Vector3f normal;
    Primitive * prm;
};

using namespace std;

class Assignment
{
public:
    Assignment() = default;
    
    static void raytrace(Camera camera, Scene scene);
    static void traverse(Renderable *ren, Matrix4f curr_mat, int depth);
    static Intersect intersect_with_Prm(Primitive* prm, Matrix4f curr_mat, Vector3f a, Vector3f b);
    static Intersect getInteract(vector<pair<Primitive*, Matrix4f>> pairs, Vector3f a, Vector3f b);

    static Vector3f lighting( Vector3f & pos, Vector3f & norm, Primitive * prm, PointLight &l, Vector3f &position, bool hit);

    static float sq_io(Vector3f &a,  Vector3f &b, float t, float e, float n);
    static float sq_iod(Vector3f &a,  Vector3f &b, float t, float e, float n);
    static float NewtonMethod(float initial_t, Vector3f a, Vector3f b, float e, float n);

};
Matrix4f cumulate_mat(vector<Transformation> transformations);
Vector3f apply_transform_to_vector(Vector3f v, Matrix4f t);

Matrix4f get_matrix(Transformation transf);
Matrix4f get_translation_matrix(float x, float y, float z);
Matrix4f get_scaling_matrix(float x, float y, float z);
Matrix4f get_rotation_matrix(float x, float y, float z, float angle);

#endif