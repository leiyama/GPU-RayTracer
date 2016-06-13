#ifndef  SUPERQUADRIC_H
#define  SUPERQUADRIC_H

#include "matrix.cuh"
#include "point.cuh"

#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust/device_vector.h>

#include <cstdio>
#include <math.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <float.h>

class Superquadric {
    private:
        traMat t;   // Position members
        rotMat r;   // Orientation
        scaMat s;   // Scaling
        float e, n; // Eccentricity values

        // Light properties
        Point diffuse;
        Point ambient;
        Point specular;
        float shine;
        float snell;
        float opacity;
    
        // Index to prevent self comparison
        int obj_num;

    public:
        // Constructors
        Superquadric();
        Superquadric(float, float);
        Superquadric(Point, Point, Point, float, float, float);
        Superquadric(Point, Point, Point, float, float, float,
                     Point, Point, Point, float, float, float);

	// Accessor
        __host__ __device__ float getEcc();
        // Point Transformation functions
        __host__ __device__ Point applyTransforms(Point);
        __host__ __device__ Point applyDirTransforms(Point);
        __host__ __device__ Point revertTransforms(Point);
        __host__ __device__ Point revertDirTransforms(Point);

        // Superquadric functions
        __host__ __device__ float   isq(Point);
        __host__ __device__ float   isq_prime(Point, Ray);
        __host__ __device__ Point isq_g(Point);
        __host__ __device__ Point getNormal(Point);

        // Basic raytracing functions
        __host__ __device__ float   get_initial_guess(Ray);
        __host__ __device__ float   get_intersection(Ray);
        void    rayTrace(Ray&, Point lookFrom,
                         std::vector<pointLight>,
                         std::vector<Superquadric>);

        // Light modeling functions 
        __host__ Point lighting(Point p, Point n, Point lookFrom,
                         std::vector<pointLight> lights,
                         std::vector<Superquadric> scene);

        __device__ Point lighting(Point p, Point n, Point lookFrom,
                         pointLight * lightStart,
                         Superquadric * sceneStart,
                         int lightSize,
                         int sceneSize);

        bool    checkShadow(Point, pointLight, std::vector<Superquadric>);

        // Handle indexing
        __host__ __device__ void setNum(int i) {this->obj_num = i;}
};



#endif
