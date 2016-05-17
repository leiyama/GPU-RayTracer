#include "Assignment.hpp"

#include "Utilities.hpp"

#include "Scene.hpp"
#include "UI.hpp"

#include "Eigen/Dense"
#include <limits>
#include "float.h"
#include <math.h>

using namespace std;
using namespace Eigen;

vector<pair<Primitive*, Matrix4f>> scene_list;

const float bright_green[] = {0.0, 1.0, 0.0};
const float black[] = {0.0, 0.0, 0.0};
const float red[] = {1.0, 0.0, 0.0};
const float blue[] = {0.0, 0.0, 1.0};


/*
 Part1 of assignment7a:
 IO test
 */
void Assignment::drawIOTest()
{
    
    // Retreive the current Renderable
    const Line* curr_state = CommandLine::getState();
    Renderable* ren = NULL;
    
    if (curr_state)
        ren = Renderable::get(curr_state->tokens[1]);
    else
        return;
    
    scene_list.clear();
    Matrix4f curr_mat = Matrix4f::Identity();
    traverse(ren, curr_mat, 0);
    
    /*For each point p of the form (0.5i, 0.5j, 0.5k) where i,j,k in [-10, 10]
     
     check if p is inside one of the superquadrics of the scene (using the
     inside-outside function).
     
     Draw a red sphere of radius 0.1 at that location
     if it's inside a superquadric and a blue sphere if it's not.
     */
    for (int i = -10; i <= 10; i++) {
        for (int j = -10; j <= 10; j++) {
            for (int k = -10; k <= 10; k++) {
                float x = i * 0.5;
                float y = j * 0.5;
                float z = k * 0.5;
                
                // Check if point is inside one of the scene's primitives
                if (isInsidePrm(x, y, z))
                    drawSphere(1, x, y, z); // red
                else
                    drawSphere(2, x, y, z); // translucent red
            }
        }
    }
    
}

void Assignment::drawSphere(int type, float i, float j, float k)
{
    glMaterialfv(GL_FRONT, GL_DIFFUSE, black);
    glMaterialfv(GL_FRONT, GL_SPECULAR, black);
    glMaterialf(GL_FRONT, GL_SHININESS, 1);
    glMaterialfv(GL_FRONT, GL_AMBIENT, red);
    
    if (type == 1) {
        glMaterialfv(GL_FRONT, GL_AMBIENT, red);
    }else {
        glMaterialfv(GL_FRONT, GL_AMBIENT, blue);
    }
    
    i += randSalt(-0.05, 0.05);
    j += randSalt(-0.05, 0.05);
    k += randSalt(-0.05, 0.05);
    
    glBegin(GL_POINTS);
    glVertex3f(i, j, k);
    glPointSize(0.01);
    glEnd();
    
}

float Assignment::randSalt(float base, float top)
{
    int rand_int = rand();
    float rand = (float)rand_int / (float)RAND_MAX;
    float range = top - base;
    return base + range * rand;
}

Matrix4f cumulate_mat(vector<Transformation> transformations)
{
    Matrix4f prod = Matrix4f::Identity();
    Matrix4f current_matrix;
    for (int i = (int) transformations.size() - 1; i >= 0; i--)
    {
        current_matrix = get_matrix(transformations[i]);
        prod = prod * current_matrix;
    }
    return prod;
}

void Assignment::traverse(Renderable *ren, Matrix4f curr_mat, int depth)
{
    // avoid infinite recursive
    // set MAX DEPTH BUFFER to 1000
    depth++;
    if (depth > 1000) {
        return;
    }
    
    if (ren->getType() == OBJ) {
        
        Object* obj = dynamic_cast<Object*>(ren);
        const unordered_map<Name, Child, NameHasher>& child_map = obj->getChildren();
        
        curr_mat = cumulate_mat(obj->getOverallTransformation()).inverse() * curr_mat;
        
        Matrix4f child_mat = Matrix4f::Identity();
        
        for (auto it = child_map.begin(); it != child_map.end(); ++it)
        {
            Child child = it->second;
            
            Renderable *ren = Renderable::get(child.name);
            
            child_mat = cumulate_mat(child.transformations).inverse() * curr_mat;
            
            traverse(ren, child_mat, depth);
        }
    }else {
        Primitive* prm = dynamic_cast<Primitive*> (ren);
        scene_list.push_back(make_pair(prm, curr_mat));
    }
}

bool Assignment::isInsidePrm(float i, float j, float k)
{
    for (pair<Primitive*, Matrix4f> p : scene_list)
    {
        if (isInsidePrm(i, j, k, p.first, p.second)) {
            return true;
        }
    }
    return false;
}

bool Assignment::isInsidePrm(float i, float j, float k, Primitive *prm, Matrix4f curr_mat)
{
    Vector4f vec(i, j, k, 1);
    Vector4f transformed = curr_mat * vec;
    
    float e = prm->getExp0();
    float n = prm->getExp1();
    
    float w = transformed(3);
    // change x, y, z from homogeneous system to world system
    float x = transformed(0) / w;
    float y = transformed(1) / w;
    float z = transformed(2) / w;
    
    Vector3f coeff = prm->getCoeff();
    
    x = x / coeff(0);
    y = y / coeff(1);
    z = z / coeff(2);
    
    float inside_outside = pow((pow((x * x), 1.0 / e) + pow(y * y, 1.0 / e)), e / n) + pow((z * z), 1.0 / n) - 1.0;
    
    if (inside_outside < 0)
        return true;
    
    return false;
}

Matrix4f get_matrix(Transformation transf)
{
    
    TransformationType type = transf.type;
    float x = transf.trans(0);
    float y = transf.trans(1);
    float z = transf.trans(2);
    float angle = transf.trans(3);
    
    // get matrix among different type:
    switch (type) {
        case TRANS: {
            // Translation matrix
            return get_translation_matrix(x, y, z);
            break;
        }
        case SCALE: {
            // Scaling matrix
            return get_scaling_matrix(x, y, z);
            break;
        }
        case ROTATE: {
            // Rotation matrix
            return get_rotation_matrix(x, y, z, angle);
            break;
        }
        default:
            cout << "Invalid Transformations" <<endl;
            exit(EXIT_FAILURE);
    }
}

Matrix4f get_translation_matrix(float x, float y, float z)
{
    Matrix4f m;
    m << 1, 0, 0, x,
    0, 1, 0, y,
    0, 0, 1, z,
    0, 0, 0, 1;
    return m;
}

Matrix4f get_scaling_matrix(float x, float y, float z)
{
    Matrix4f m;
    m << x, 0, 0, 0,
    0, y, 0, 0,
    0, 0, z, 0,
    0, 0, 0, 1;
    return m;
}

Matrix4f get_rotation_matrix(float x, float y, float z, float angle)
{
    Matrix4f m;
    // Make input vector a unit vector
    float norm = sqrt(x * x + y * y + z * z);
    x /= norm;
    y /= norm;
    z /= norm;
    
    m << (x * x) + (1 - (x * x)) * cos(angle), (x * y) * (1 - cos(angle)) - z * sin(angle),
    (x * z) * (1 - cos(angle)) + y * sin(angle), 0,
    (y * x) * (1 - cos(angle)) + z * sin(angle), (y * y) + (1 - y * y) * cos(angle),
    (y * z) * (1 - cos(angle)) - x * sin(angle), 0,
    (z * x) * (1 - cos(angle)) - y * sin(angle), (z * y) * (1 - cos(angle)) + x * sin(angle),
    (z * z) + (1 - (z * z)) * cos(angle), 0,
    0, 0, 0, 1;
    return m;
}

/*
 Part2 of assignment7a:
 Raytracing method
 */

Vector3f apply_transform_to_vector(Vector3f v, Matrix4f t)
{
    Vector4f v_h(v(0), v(1), v(2), 1);
    Vector4f v_t = t * v_h;
    Vector3f v_norm;
    
    v_norm(0) = v_t(0) / v_t(3);
    v_norm(1) = v_t(1) / v_t(3);
    v_norm(2) = v_t(2) / v_t(3);
    
    return v_norm;
}


void Assignment::drawIntersectTest(Camera* camera)
{
    
    const Line* curr_state = CommandLine::getState();
    Renderable* ren = NULL;
    
    if (curr_state)
        ren = Renderable::get(curr_state->tokens[1]);
    else
        return;
    
    // Populate scene_list
    scene_list.clear();
    Matrix4f curr_mat = Matrix4f::Identity();
    traverse(ren, curr_mat, 0);
    
    // Vector to record the intersection point.
    vector<pair<Vector3f, Vector3f>> pairs;
    
    Vector3f a;
    a << 0.0, 0.0, -1.0;
    Vector3f b = camera->getPosition();
    
    // get the camera rotation matrix;
    Vector3f rotation_axis = camera->getAxis();
    float rotation_angle = camera->getAngle();
    Matrix4f cam_rotation = get_rotation_matrix(rotation_axis(0), rotation_axis(1),
                                                rotation_axis(2), rotation_angle);
    
    Vector3f a_world, b_world;
    
    a_world = apply_transform_to_vector(a, cam_rotation * (UI::arcball_object_mat).inverse());
    b_world = apply_transform_to_vector(b, (UI::arcball_object_mat).inverse());
    
    // loop through all primitives to find intersection points;
    for (pair<Primitive*, Matrix4f> p : scene_list) {
        
        Intersect intersected = intersect_with_Prm(camera, p.first, p.second, a_world, b_world);
        
        if (!isnan(intersected.t)) {
            pairs.push_back(make_pair(intersected.point, intersected.normal));
        }else {
            
        }
    }
    
    Vector3f intersection, normal;
    
    if (pairs.size() > 0) {
        
        // find out the closest point near camera;
        int min_index = 0;
        float min_distance = numeric_limits<float> ::max();
        
        for (int i = 0; i < (int)pairs.size(); i++) {
            Vector3f difference = b_world - pairs[i].first;
           
            float distance = difference.norm();
            if (distance < min_distance) {
                min_distance = distance;
                min_index = i;
            }
        }
        
        intersection = pairs[min_index].first;
        normal = pairs[min_index].second;
        normal.normalize();
    }
    
    Vector3f p1 = intersection;
    Vector3f p2 = intersection + normal;

    glMaterialfv(GL_FRONT, GL_AMBIENT, bright_green);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, black);
    glMaterialfv(GL_FRONT, GL_SPECULAR, black);
    glMaterialf(GL_FRONT, GL_SHININESS, 1);
    
    glBegin(GL_POINTS);
    glVertex3f(p1(0), p1(1), p1(2));
    glPointSize(0.1f);
    glEnd();
    
    glMaterialfv(GL_FRONT, GL_AMBIENT, bright_green);
    glMaterialfv(GL_FRONT, GL_DIFFUSE, black);
    glMaterialfv(GL_FRONT, GL_SPECULAR, black);
    glMaterialf(GL_FRONT, GL_SHININESS, 1);
    
    // Draw the line
    glBegin(GL_LINES);
    glLineWidth(2.5);
    glVertex3f(p1(0), p1(1), p1(2));
    glVertex3f(p2(0), p2(1), p2(2));
    glEnd();
}

Intersect Assignment::intersect_with_Prm(Camera* camera, Primitive* prm,
                                         Matrix4f curr_mat, Vector3f a_world, Vector3f b_world)
{
    Intersect inter;
    Intersect not_inter;
    
    Vector3f p_obj, n_obj;
    
    not_inter.t = nanf("");
    
    float e = prm->getExp0();
    float n = prm->getExp1();
    
    Vector3f coeff = prm->getCoeff();
    
    Vector3f a_transformed, b_transformed, o_transformed;
    Vector3f a_divcoeff, b_divcoeff;
    Vector3f o;
    o << 0.0, 0.0, 0.0;
    
    a_transformed = apply_transform_to_vector(a_world, curr_mat) - apply_transform_to_vector(o, curr_mat);
    
    a_divcoeff(0) = a_transformed(0) / coeff(0);
    a_divcoeff(1) = a_transformed(1) / coeff(0);
    a_divcoeff(2) = a_transformed(2) / coeff(0);
    
    b_transformed = apply_transform_to_vector(b_world, curr_mat);
    
    b_divcoeff(0) = b_transformed(0) / coeff(0);
    b_divcoeff(1) = b_transformed(1) / coeff(0);
    b_divcoeff(2) = b_transformed(2) / coeff(0);
    
    // get the initial guesses of the solution
    float initial_t = getInitialGuesses(a_divcoeff, b_divcoeff);
    
    if (!isnan(initial_t)) {
        
        float t = initial_t;
        
        double sq_io = 100;
        while (abs(sq_io) > 0.01) {
            
            //Newton's Method
            
            Vector3f ray = a_divcoeff * t + b_divcoeff;
            Vector3f gradient_sq_io;
            
            double x = ray(0);
            double y = ray(1);
            double z = ray(2);
            
            sq_io = pow((pow((x * x), 1.0 / e) + pow(y * y, 1.0 / e)), e / n) + pow((z * z), 1.0 / n) - 1.0;
            
            gradient_sq_io(0) = (2.0 * x * pow(x * x, 1.0 / e - 1.0) *
                                 pow(pow(x * x, 1.0 / e) + pow(y * y, 1.0 / e), e / n - 1.0)) / n;
            gradient_sq_io(1) = (2.0 * y * pow(y * y, 1.0 / e - 1.0) *
                                 pow(pow(x * x, 1.0 / e) + pow(y * y, 1.0 / e), e / n - 1.0)) / n;
            gradient_sq_io(2) = (2 * z * pow(z * z, 1.0 / n - 1.0)) / n;
            
            double deri_g = a_divcoeff.dot(gradient_sq_io);
            
            if (deri_g >= 0) {
                break;
            }
            t = t - sq_io / deri_g;
        }
        if (abs(sq_io) < 0.01)
        {
            inter.t = t;
            inter.point = a_world * t + b_world;
            p_obj = a_transformed * t + b_transformed;
            n_obj = prm->getNormal(p_obj);
            inter.normal = apply_transform_to_vector(n_obj, curr_mat.inverse())
                              - apply_transform_to_vector(o, curr_mat.inverse());
            return inter;
        }else {
            return not_inter;
        }
    }else
        return not_inter;
}

float Assignment::getInitialGuesses(Vector3f cam_dir, Vector3f cam_pos)
{
    float t = 0.0;
    
    float a = cam_dir.dot(cam_dir);
    float b = 2 * cam_dir.dot(cam_pos);
    float c = cam_pos.dot(cam_pos) - 3;
    
    float discriminant = b * b - 4 * a * c;
    
    if (discriminant < 0) {
        return nanf("");
    }else {
        float t_pos = ( -b + discriminant ) / ( 2 * a );
        float t_neg = ( -b - discriminant ) / ( 2 * a );
        
        if (t_neg < 0 && t_pos < 0) {
            return nanf("");
        }else {
            // starting point is inside the sphere;
            // here we record both t_pos and t_neg, and iterate both using Newton's Method.
            t = min(abs(t_neg), abs(t_pos));
        }
    }
    return t;
}

