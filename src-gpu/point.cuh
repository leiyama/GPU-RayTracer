#ifndef POINT_H
#define POINT_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <cmath>
#include <float.h>

class Point 
{
    protected:
        float x, y, z;

    public:
        // Constructors
        __host__ __device__ Point();
        __host__ __device__ Point(float, float, float);

        // Accessors
        __host__ __device__ float X() {return this->x;} 
        __host__ __device__ float Y() {return this->y;} 
        __host__ __device__ float Z() {return this->z;} 
        
        // Mutators
        __host__ __device__ void setX(float X) {this->x = X;}      
        __host__ __device__ void setY(float Y) {this->y = Y;}
        __host__ __device__ void setZ(float Z) {this->z = Z;}
        __host__ __device__ void set(float X, float Y, float Z) 
        {
            this->x = X;
            this->y = Y;
            this->z = Z;
        }
        __host__ __device__ void set(Point p)
        {
            this->x = p.x;
            this->y = p.y;
            this->z = p.z;
        }

        // Other functions
        __host__ __device__ Point norm();
        __host__ __device__ float   dot(Point);
        __host__ __device__ Point cross(Point);
        __host__ __device__ float   dist(Point);
        __host__ __device__ Point cwiseMin(Point);

        // Operator overloads
        __host__ __device__ Point operator+ (Point);
        __host__ __device__ Point operator- (Point);
        __host__ __device__ Point operator/ (Point);
        __host__ __device__ Point operator* (Point);
        __host__ __device__ Point operator+=(Point);
        __host__ __device__ Point operator-=(Point);
        __host__ __device__ Point operator/=(Point);
        __host__ __device__ Point operator*=(Point);
        __host__ __device__ Point operator= (Point);
        __host__ __device__ bool    operator==(Point);

        __host__ __device__ Point operator+ (float);
        __host__ __device__ Point operator- (float);
        __host__ __device__ Point operator/ (float);
        __host__ __device__ Point operator* (float);
        __host__ __device__ Point operator+=(float);
        __host__ __device__ Point operator-=(float);
        __host__ __device__ Point operator/=(float);
        __host__ __device__ Point operator*=(float);
        
        __host__ friend std::ostream& operator<< (std::ostream&, Point);
};

// 3D ray class that inherits from point.
class Ray : public Point
{
    protected:
        float posx, posy, posz;
        int R, G, B; // Returned color value
        float t;     // time to closest object
    public:
        __host__ __device__ Ray();
        __host__ __device__ Ray(float, float, float, float, float, float);
        __host__ __device__ Ray(Point , Point);

        // Mutation functions
        __host__ __device__ void setColor(int, int, int);
        __host__ __device__ void setStart(Point );
        __host__ __device__ void setDir(Point );
        __host__ __device__ void setTime(float T) {this->t = T;}

        // Accessor functions
        __host__ __device__ float getR() {return this->R;}
        __host__ __device__ float getG() {return this->G;}
        __host__ __device__ float getB() {return this->B;}
        __host__ __device__ float getTime() {return this->t;}
        __host__ __device__ Point getStart() {return Point(this->posx, this->posy, this->posz);}
        __host__ __device__ Point getDir()   {return Point(this->x, this->y, this->z);}
        __host__ __device__ Point propagate(float);
};

// Point light source in 3D coordinates
class pointLight : public Point
{
    protected:
        int R, G, B;
        float attenuation_k;

    public:
        __host__ __device__ pointLight();
        __host__ __device__ pointLight(float, float, float, int, int, int, float);
        __host__ __device__ pointLight(Point, int, int, int, float);

        __host__ __device__ void    setColor(int, int, int);
        __host__ __device__ Point getColor();
        __host__ __device__ void    setAtt_k(float);
        __host__ __device__ float   getAtt_k();
        __host__ __device__ void    setPos(Point p);
        __host__ __device__ void    setPos(float, float, float);
        __host__ __device__ Point getPos();

};
#endif
