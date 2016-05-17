#include <limits>
#include "float.h"
#include <math.h>

#include "Assignment.hpp"

#define XRES 250
#define YRES 250

using namespace std;

vector<pair<Primitive*, Matrix4f>> scene_list;
vector<Intersect> intersect_list;

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

Matrix4f cumulate_mat(vector<Transformation> transformations)
{
    Matrix4f prod = Matrix4f::Identity();
    Matrix4f current_matrix;
    for (int i = (int) transformations.size() - 1; i >= 0; i--)
    {
        current_matrix = get_matrix(transformations[i]);
        prod = current_matrix.inverse() * prod;
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
        curr_mat = cumulate_mat(obj->getOverallTransformation()) * curr_mat;
        
        Matrix4f child_mat = Matrix4f::Identity();
        
        for (auto it = child_map.begin(); it != child_map.end(); ++it)
        {
            Child child = it->second;
            
            Renderable *ren = Renderable::get(child.name);
            
            child_mat = cumulate_mat(child.transformations) * curr_mat;
            
            traverse(ren, child_mat, depth);
        }
    }else {
        Primitive* prm = dynamic_cast<Primitive*> (ren);
        scene_list.push_back(make_pair(prm, curr_mat));
    }
}

/* Ray traces the scene. */
void Assignment::raytrace(Camera camera, Scene scene)
{
    // LEAVE THIS UNLESS YOU WANT TO WRITE YOUR OWN OUTPUT FUNCTION
    PNGMaker png = PNGMaker(XRES, YRES);
    
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
    
//    for (int i = 0; i < scene_list.size(); i++) {
//        cout << "prm: " << scene_list[i].first <<endl;
//    }
    // get the height and width of the screen
    float n = camera.getNear();
    float theta = degToRad(camera.getFov());
    float h = 2 * n * tan(theta / 2.0);
    float w = camera.getAspect() * h;
    
    // get initial camera direction (e1) and e2, e3
    Vector3f e1, e2, e3;
    e1 << 0.0, 0.0, -1.0;
    e2 << 1.0, 0.0, 0.0;
    e3 << 0.0, 1.0, 0.0;
    
    Vector3f cam_pos = camera.getPosition();
    
    // get the camera rotation matrix
    Vector3f rotation_axis = camera.getAxis();
    float rotation_angle = camera.getAngle();
    Matrix4f cam_rotation = get_rotation_matrix(rotation_axis(0), rotation_axis(1),
                                                rotation_axis(2), rotation_angle);
    
    // apply camera rotation and Arcball rotation
    e1 = apply_transform_to_vector(e1, (cam_rotation * UI::singleton->arcball_object_mat).inverse());
    e2 = apply_transform_to_vector(e2, (cam_rotation * UI::singleton->arcball_object_mat).inverse());
    e3 = apply_transform_to_vector(e3, (cam_rotation * UI::singleton->arcball_object_mat).inverse());
    
    cam_pos = apply_transform_to_vector(cam_pos, (UI::singleton->arcball_object_mat).inverse());
    
    // REPLACE THIS WITH YOUR CODE
    for (int i = 0; i < XRES; i++) {
        for (int j = 0; j < YRES; j++) {
            
            intersect_list.clear();
            float xi = (i - XRES / 2.0) * w / XRES;
            float yi = (j - YRES / 2.0) * h / YRES;

            Vector3f a, b;
            a = n * e1 + xi * e2 + yi * e3;
            b = cam_pos;
            
            Intersect inter_final = getInteract(scene_list, a, b);
            
            if (inter_final.t < INFINITY) {
                
                Vector3f obj_pos = a * inter_final.t + b;
                //vector<Transformation> vec2;
                
                Vector3f color, one;
                color << 0.0, 0.0, 0.0;
                one << 1.0, 1.0, 1.0;
                
                for (int i = 0; i < (int)scene.lights.size(); i++) {
                    
                    Vector3f light_pos;
                    
                    light_pos(0) = scene.lights[i].position[0]/scene.lights[i].position[3];
                    light_pos(1) = scene.lights[i].position[1]/scene.lights[i].position[3];
                    light_pos(2) = scene.lights[i].position[2]/scene.lights[i].position[3];

                    Vector3f light_a = obj_pos - light_pos;
                    
                    Intersect p_light = getInteract(scene_list, light_a, light_pos);
                    
                    if (fabsf(p_light.t - 1) < 1e-3) {
                        color += lighting(obj_pos, inter_final.normal, inter_final.prm, scene.lights[i], b, true);
                    }
                    else {
                        color += lighting(obj_pos, inter_final.normal, inter_final.prm, scene.lights[i], b, false);
                    }
                }
                color = one.cwiseMin(color);
                png.setPixel(i, j, color(0), color(1), color(2));

            }else {
                png.setPixel(i, j, 0.0, 0.0, 0.0);
            }
        }
    }
    
    // LEAVE THIS UNLESS YOU WANT TO WRITE YOUR OWN OUTPUT FUNCTION
    if (png.saveImage())
        printf("Error: couldn't save PNG image\n");
}

Intersect Assignment::getInteract(vector<pair<Primitive*, Matrix4f>> pairs, Vector3f a, Vector3f b){
    for (pair<Primitive*, Matrix4f> p : scene_list) {
        
        Intersect intersected = intersect_with_Prm(p.first, p.second, a, b);
        
        if (intersected.t < INFINITY) {
            intersect_list.push_back(intersected);
        }
    }
    Intersect nearest;
    Intersect none;
    none.t = INFINITY;
    Vector3f color;

    if (intersect_list.size() > 0) {
        
        // find out the closest point near camera;
        int min_index = 0;
        float min_distance = numeric_limits<float> ::max();
        
        for (int i = 0; i < (int)intersect_list.size(); i++) {
            Vector3f point = a * intersect_list[i].t + b;
            Vector3f difference = b - point;
            float distance = difference.norm();
            
            if (distance < min_distance) {
                min_distance = distance;
                min_index = i;
            }
        }
        nearest.t = intersect_list[min_index].t;
        Vector3f point = a * nearest.t + b;
        //cout << "point: " << point(0) << " " << point(1) << " "<<point(2) << endl;
        nearest.normal = intersect_list[min_index].normal;
        nearest.prm = intersect_list[min_index].prm;
        return nearest;
    }else {
        return none;
    }
}

float Assignment::sq_io(Vector3f &a,  Vector3f &b, float t, float e, float n) {
    Vector3f ray = a * t + b;
    float x = ray(0);
    float y = ray(1);
    float z = ray(2);
    return pow(pow(x * x, 1/e) + pow(y * y, 1/e), e/n) + pow(z * z, 1/n) - 1;
}

float Assignment::sq_iod(Vector3f &a,  Vector3f &b, float t, float e, float n) {
    Vector3f ray = a * t + b;
    float x = ray(0);
    float y = ray(1);
    float z = ray(2);
    float dx = 2 * x * pow(x * x, 1/e-1) * pow(pow(x * x, 1/e) + pow(y * y, 1/e), e/n-1)/n;
    float dy = 2 * y * pow(y * y, 1/e-1) * pow(pow(x * x, 1/e) + pow(y * y, 1/e), e/n-1)/n;
    float dz = 2 * z * pow(z * z, 1/n-1)/n;
    return a(0) * dx + a(1) * dy + a(2) * dz;
}


float Assignment::NewtonMethod(float initial_t, Vector3f a, Vector3f b, float e, float n)
{
    float tOld = initial_t;
    float tNew = initial_t;
    int sign = 0;
    while (fabsf(Assignment::sq_io(a, b, tNew, e, n)) > 1e-3) {
        tOld = tNew;
        float sq = Assignment::sq_io(a, b, tOld, e, n);
        float sqd = Assignment::sq_iod(a, b, tOld, e, n);
        if (sign == 0) sign = (sqd > 0) ? 1:-1;
        if (sqd * sign < 0 || fabsf(sqd) < 1e-3) {
            return INFINITY;
        }
        sign = (sqd > 0) ? 1:-1;
        tNew = tOld - sq/sqd;
    }
    return tNew;
}

Intersect Assignment::intersect_with_Prm(Primitive* prm, Matrix4f curr_mat, Vector3f a, Vector3f b)
{
    Intersect inter;
    
    inter.t = INFINITY;
    inter.normal << 1, 0, 0;
    inter.prm = nullptr;
    
    float e = prm->getExp0();
    float n = prm->getExp1();

    Vector3f coeff = prm->getCoeff();
    
    Vector3f o;
    o << 0.0, 0.0, 0.0;
    Vector3f a_transformed, b_transformed, o_transformed;
    Vector3f a_divcoeff, b_divcoeff;
    
    a_transformed = apply_transform_to_vector(a, curr_mat) - apply_transform_to_vector(o, curr_mat);
    
    a_divcoeff(0) = a_transformed(0) / coeff(0);
    a_divcoeff(1) = a_transformed(1) / coeff(1);
    a_divcoeff(2) = a_transformed(2) / coeff(2);
    
    b_transformed = apply_transform_to_vector(b, curr_mat);
    
    b_divcoeff(0) = b_transformed(0) / coeff(0);
    b_divcoeff(1) = b_transformed(1) / coeff(1);
    b_divcoeff(2) = b_transformed(2) / coeff(2);
    
    float A = a_divcoeff.dot(a_divcoeff);
    float B = 2 * a_divcoeff.dot(b_divcoeff);
    float C = b_divcoeff.dot(b_divcoeff) - 3;
    
    if (B < 0) {
        A = -A;
        B = -B;
        C = -C;
    }
    
    float delta = B * B - 4 * A * C;
    
    if (delta < 0) {
        return inter;
    }else {
        float modifier = - B - sqrt(delta);
        float t1 = (-B + sqrt(delta)) / (2 * A);
        float t2 = (-B + sqrt(delta)) / (2 * A);
        if (t1 >= 0 && t2 >= 0) {
            float initial_t = min(t1, t2) * modifier / modifier;
            inter.t = NewtonMethod(initial_t, a_divcoeff, b_divcoeff, e, n);
        }else if (t1 < 0 && t2 < 0){
            return inter;
        }else {
            float t1_after = NewtonMethod(t1 * modifier/modifier, a_divcoeff, b_divcoeff, e, n);
            float t2_after = NewtonMethod(t2 * modifier/modifier, a_divcoeff, b_divcoeff, e, n);
            if (t1_after >= 0 && t2_after >= 0) {
                inter.t = min(t1_after, t2_after);
            }else {
                return inter;
            }
        }
        if (inter.t == INFINITY) return inter;
        
        inter.point = a * inter.t + b;
        
        Vector3f p_obj, n_obj;
        
        p_obj = a_transformed * inter.t + b_transformed;
        n_obj = prm->getNormal(p_obj);
        inter.normal = apply_transform_to_vector(n_obj, curr_mat.inverse())
        - apply_transform_to_vector(o, curr_mat.inverse());
        
        inter.prm = prm;
        return inter;
    }
}

Vector3f Assignment::lighting( Vector3f & pos, Vector3f & norm, Primitive * prm, PointLight & l, Vector3f & position, bool hit)
{
    const RGBf& color = prm->getColor();
    const float ambient = prm->getAmbient();
    const float diffuse = prm->getDiffuse();
    const float specular = prm->getSpecular();
    
    Vector3f one, vecColor, vecAmbient, vecDiffuse, vecSpecular;
    one << 1, 1, 1;
    vecColor << color.r, color.g, color.b;
    vecAmbient = vecColor * ambient;
    vecDiffuse = vecColor * diffuse;
    vecSpecular = vecColor * specular;
    
    if (hit == false) return vecAmbient;
    
    Vector3f light_position, light_color;
    light_position << l.position[0], l.position[1], l.position[2];
    light_color << l.color[0], l.color[1], l.color[2];
    
    float k = l.k;
    
    Vector3f ldirection = light_position-pos;
    float distance = ldirection.norm();
    float iver_attenuation = 1 + k * distance * distance;

    Vector3f lc = light_color/iver_attenuation;
    
    // direction
    Vector3f ld = ldirection/distance;
    
    // diffuse
    float cosine = norm.dot(ld)>0 ? norm.dot(ld):0;
    Vector3f ldiff = lc * cosine;
    
    // specular
    Vector3f edirection = position - pos;
    Vector3f ed = edirection/edirection.norm();
    Vector3f hdirection = ed+ld;
    Vector3f hd = hdirection/hdirection.norm();
    float phi = norm.dot(hd)>0 ? norm.dot(hd):0;
    Vector3f lspec = lc * pow(phi,prm->getReflected());
    
    Vector3f cPotential = vecAmbient+ldiff.cwiseProduct(vecDiffuse)+lspec.cwiseProduct(vecSpecular);
    Vector3f color_final = one.cwiseMin(cPotential);
    return color_final;
}


