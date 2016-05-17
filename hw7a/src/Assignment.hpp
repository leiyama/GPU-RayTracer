
#ifndef ASSIGNMENT_HPP
#define ASSIGNMENT_HPP

#include "model.hpp"

class Camera;

class Intersect
{
public:
    float t;
    Vector3f point;
    Vector3f normal;
};

class Assignment {
public:
    Assignment() = default;
    
    static void drawIOTest();
    static void drawSphere(int type, float i, float j, float k);
    static float randSalt(float base, float top);
    
    static void traverse(Renderable *ren, Matrix4f curr_mat, int depth);
    static bool isInsidePrm(float i, float j, float k);
    static bool isInsidePrm(float i, float j, float k, Primitive* prm, Matrix4f curr_mat);
    
    static void drawIntersectTest(Camera* camera);
    
    static Intersect intersect_with_Prm(Camera* camera, Primitive* prm, Matrix4f curr_mat, Vector3f a_world, Vector3f b_world);
    static float getInitialGuesses(Vector3f cam_dir, Vector3f cam_pos);
};

Vector3f apply_transform_to_vector(Vector3f v, Matrix4f t);

Matrix4f get_matrix(Transformation transf);
Matrix4f get_translation_matrix(float x, float y, float z);
Matrix4f get_scaling_matrix(float x, float y, float z);
Matrix4f get_rotation_matrix(float x, float y, float z, float angle);

#endif