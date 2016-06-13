#include <vector>
#include <string>
#include <fstream>

#include "superquadric.cuh"
#include "point.cuh"
#include "matrix.cuh"
#include "camera.h"

const char * DELIMITER = " ";

// Set of parser functions to help streamline the ray tracing process
Camera * parseObjects(const char *filename,
                      std::vector<Superquadric> &scene,
                      std::vector<pointLight> &lights)
{
    const int MAX_CHARS_PER_LINE = 200;
    const int MAX_TOKENS_PER_LINE = 25;
    bool object(false), camera(false), light(false);

    Camera *c;
    int num_cameras = 0;

    std::ifstream file;
    file.open(filename);

    if (!file.good())
    {
        std::cerr << "Error! File invalid!\n";
        exit(0);
    }

    while(!file.eof())
    {
        char buf[MAX_CHARS_PER_LINE];
        file.getline(buf, MAX_CHARS_PER_LINE);

        int n = 0;
        const char* token[MAX_TOKENS_PER_LINE] = {};
        token[0] = strtok(buf, DELIMITER);

        object = light = camera = false;

        if (token[0])
        {
            if (!std::strcmp(token[0], "object"))
                object = true;
            else if (!std::strcmp(token[0], "lights"))
                light  = true;
            else if (!std::strcmp(token[0], "camera"))
                camera = true;
            else
                continue;
            for (n = 1; n < MAX_TOKENS_PER_LINE; n++)
            {
                token[n] = strtok(0, DELIMITER);
                if (!token[n]) break;
            }
        }
        else
            continue;

        //////////////////////////////////
        // Deconstruct line into parts. //
        //////////////////////////////////

        if (object)
        {
            // Translation elements
            float trax, tray, traz;
            trax = atof(token[1]);
            tray = atof(token[2]);
            traz = atof(token[3]);
            Point * tra = new Point(trax, tray, traz);

            // Scaling elements
            float scax, scay, scaz;
            scax = atof(token[4]);
            scay = atof(token[5]);
            scaz = atof(token[6]);
            Point *sca = new Point(scax, scay, scaz);

            // Rotation elements
            float rotx, roty, rotz, theta;
            rotx = atof(token[7]);
            roty = atof(token[8]);
            rotz = atof(token[9]);
            theta = atof(token[10]) * 3.1415926 / 180;
            Point *rot = new Point(rotx, roty, rotz);

            // Eccentricity values
            float E, N;
            E = atof(token[11]);
            N = atof(token[12]);

            // Color diffusion properties
            float difR, difG, difB;
            difR = atoi(token[13]);
            difG = atoi(token[14]);
            difB = atoi(token[15]);
            Point * dif = new Point(difR, difG, difB);

            // Ambient color properties
            float ambR, ambG, ambB;
            ambR = atoi(token[16]);
            ambG = atoi(token[17]);
            ambB = atoi(token[18]);
            Point * amb = new Point(ambR, ambG, ambB);

            // Specular color properties
            float speR, speG, speB;
            speR = atoi(token[19]);
            speG = atoi(token[20]);
            speB = atoi(token[21]);
            Point * spe = new Point(speR, speG, speB);

            // Other light properties
            float  shi,  sne,  opa;
            shi = atof(token[22]);
            sne = atof(token[23]);
            opa = atof(token[24]);

            Superquadric * s = new Superquadric(*tra, *sca, *rot, theta, E, N,
                                                *dif, *amb, *spe, shi, sne, opa);

            scene.push_back(*s);

            delete s;
        }
        else if (light)
        {
            float x, y, z;
            x = atof(token[1]);
            y = atof(token[2]);
            z = atof(token[3]);

            int R, G, B;
            R = atoi(token[4]);
            G = atoi(token[5]);
            B = atoi(token[6]);

            float att_k = atof(token[7]);

             pointLight * l = new pointLight(x, y, z, R, G, B, att_k);

             lights.push_back(*l);

             delete l;
        }
        else if (camera && num_cameras == 0)
        {
            num_cameras = 1;

            float lookx, looky, lookz;
            lookx = atof(token[1]);
            looky = atof(token[2]);
            lookz = atof(token[3]);

            Point * LookFrom = new Point(lookx, looky, lookz);

            float atx, aty, atz;
            atx = atof(token[4]);
            aty = atof(token[5]);
            atz = atof(token[6]);

            Point * LookAt = new Point(atx, aty, atz);

            float upx, upy, upz;
            upx = atof(token[7]);
            upy = atof(token[8]);
            upz = atof(token[9]);

            Point * Up = new Point(upx, upy, upz);

            float Fd, Fx;
            int   Nx, Ny;
            Fd = atof(token[10]);
            Fx = atof(token[11]);
            Nx = atoi(token[12]);
            Ny = atoi(token[13]);

            c = new Camera(*LookFrom, *LookAt, *Up, Fd, Fx, Nx, Ny);

            pointLight *l = new pointLight(lookx, looky, lookz, 255, 255, 255, 0.005);
        }

    }
    if (num_cameras == 0)
        c = new Camera();
    return c;
}
