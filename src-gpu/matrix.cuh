#ifndef MATRIX_H
#define MATRIX_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <cstdlib>

#include "point.cuh"

// Base matrix class
class Matrix
{
    protected:
        Point xyz;

    public:
        void set(Point);
};

// Rotation matrix
class rotMat : public Matrix
{
    protected:
        // Rotation component
        float theta;           

    public:
        // Default constructor
        rotMat();

        rotMat(float, float, float, float);

        // Normal  constructor
        rotMat(Point, float); 

        void setTheta(float t) {this->theta = t;}

        __host__ __device__ Point    apply(Point);
        __host__ __device__ Point  unapply(Point);

};

// Scaling matrix
class scaMat : public Matrix
{
    public:
        scaMat();
        scaMat(float, float, float);
        scaMat(Point);

        __host__ __device__ Point    apply(Point);
        __host__ __device__ Point  unapply(Point);
};

// Translation matrix
class traMat : public Matrix
{
    public:
        traMat();
        traMat(float, float, float);
        traMat(Point);

        __host__ __device__ Point    apply(Point);
        __host__ __device__ Point  unapply(Point);
};
#endif
