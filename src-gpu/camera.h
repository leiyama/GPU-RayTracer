#ifndef CAMERA_H
#define CAMERA_H

#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <fstream>

#include "point.cuh"
#include "superquadric.cuh"
#include "matrix.cuh"

class Camera
{
    private:
        std::vector<Ray> rayScreen;
        Point LookFrom, LookAt, Up;
        Point e1, e2, e3;
        float Fd, Fx, Fy;
        int Nx, Ny;

        void init();
    public:
        // Default constructor
        Camera();

        // Camera constructor
        Camera(Point, Point, Point, float, float, float, float);
        Camera(Camera*);

        // Accessor
        std::vector<Ray> getRayScreen();
        Point * getFrom() {return new Point(LookFrom.X(), LookFrom.Y(), LookFrom.Z());}
	    Point getLookFrom();
	    void setRayScreen(std::vector<Ray> screen);

        void runRayTracer(std::vector<Superquadric>, std::vector<pointLight>);
        void scenePrep(std::vector<Superquadric>);
        void printImage();
	void gpuPrintImage();
};

#endif
