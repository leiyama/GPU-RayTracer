#include "point.cuh"

// Default constructor
__host__ __device__
Point::Point()
{
    this->x = 0;
    this->y = 0;
    this->z = 0;
}

// Normal constructor
__host__ __device__
Point::Point(float x, float y, float z)
{
    this->x = x;
    this->y = y;
    this->z = z;
}

// Returns the norm of a point treated like a vector
__host__ __device__
Point Point::norm()
{
    Point result;
    result = *this;
    
    float mag = sqrt(this->x * this->x +
                     this->y * this->y +
                     this->z * this->z);
    result /= mag;
    return result;
}

// Dot product of two points treated like vectors
__host__ __device__
float Point::dot(Point p)
{
    return this->x * p.x + this->y * p.y + this->z * p.z;
}

// Cross product of two points treated like vectors
__host__ __device__
Point Point::cross(Point p)
{
    Point result =  Point(this->y * p.z - this->z * p.y,
                               this->z * p.x - this->x * p.z,
                               this->x * p.y - this->y * p.x);

    return result;
}

// Calculates the Euclidean distance between 2 points
__host__ __device__
float Point::dist(Point p)
{
    float dx = p.x - this->x;
    float dy = p.y - this->y;
    float dz = p.z - this->z;

    return sqrt(dx * dx + dy * dy + dz * dz);
}

__host__ __device__
Point Point::cwiseMin(Point p)
{
    float minx, miny, minz;
    minx = (this->x < p.X()) ? this->x : p.X();
    miny = (this->y < p.Y()) ? this->y : p.Y();
    minz = (this->z < p.Z()) ? this->z : p.Z();

    return Point (minx, miny, minz);
}

// OPERATOR OVERLOADS
__host__ __device__
Point Point::operator+(Point p)
{
    Point result = Point(this->x + p.x,
                               this->y + p.y,
                               this->z + p.z);
    return result;
}

__host__ __device__
Point Point::operator-(Point p)
{
    Point result = Point(this->x - p.x,
                               this->y - p.y,
                               this->z - p.z);
    return result;
}

__host__ __device__
Point Point::operator*(Point p)
{
    Point result = Point(this->x * p.x,
                               this->y * p.y,
                               this->z * p.z);
    return result;
}

__host__ __device__
Point Point::operator/(Point p)
{
    Point result = Point(this->x / p.x,
                               this->y / p.y,
                               this->z / p.z);
    return result;
}

__host__ __device__
Point Point::operator+=(Point p)
{
    this->x += p.x;
    this->y += p.y;
    this->z += p.z;

    return *this;
}

__host__ __device__
Point Point::operator-=(Point p)
{
    this->x -= p.x;
    this->y -= p.y;
    this->z -= p.z;

    return *this;
}

__host__ __device__
Point Point::operator*=(Point p)
{
    this->x *= p.x;
    this->y *= p.y;
    this->z *= p.z;

    return *this;
}

__host__ __device__
Point Point::operator/=(Point p)
{
    this->x /= p.x;
    this->y /= p.y;
    this->z /= p.z;

    return *this;
}

__host__ __device__
Point Point::operator+(float f)
{
    Point result = Point(this->x + f,
                               this->y + f,
                               this->z + f);
    return result;
}

__host__ __device__
Point Point::operator-(float f)
{
    Point result = Point(this->x - f,
                               this->y - f,
                               this->z - f);
    return result;
}

__host__ __device__
Point Point::operator*(float f)
{
    Point result = Point(this->x * f,
                               this->y * f,
                               this->z * f);
    return result;
}

__host__ __device__
Point Point::operator/(float f)
{
    Point result = Point(this->x / f,
                               this->y / f,
                               this->z / f);
    return result;
}

__host__ __device__
Point Point::operator+=(float f)
{
    this->x += f;
    this->y += f;
    this->z += f;

    return *this;
}

__host__ __device__
Point Point::operator-=(float f)
{
    this->x -= f;
    this->y -= f;
    this->z -= f;
    return *this;
}

__host__ __device__
Point Point::operator*=(float f)
{
    this->x *= f;
    this->y *= f;
    this->z *= f;
    return *this;
}

__host__ __device__
Point Point::operator/=(float f)
{
    this->x /= f;
    this->y /= f;
    this->z /= f;
    return *this;
}

__host__ __device__
Point Point::operator=(Point p)
{
    this->x = p.x;
    this->y = p.y;
    this->z = p.z;
    return *this;
}

__host__ __device__
bool Point::operator==(Point p)
{
    if (this->x == p.x && this->y == p.y && this->z == p.z)
    {
        return true;
    }

    return false;
}


// Stream output operator
__host__
std::ostream& operator<<(std::ostream &out, Point p)
{
    out << p.x << "\t" << p.y << "\t" << p.z << "\n";
    return out;
}

/******************************************************************************/
/*                           Methods for Ray Class                            */
/******************************************************************************/
__host__ __device__
Ray::Ray()
{
    // Default direction
    this->x = 1;
    this->y = 1;
    this->z = 1;

    // Located at the origin
    this->posx = 0;
    this->posy = 0;
    this->posz = 0;
    
    // Default color black
    this->setColor(0, 0, 0);
    this->t = FLT_MAX;
}

__host__ __device__
Ray::Ray(float X, float Y, float Z, float dX, float dY, float dZ)
{
    this->x = dX;
    this->y = dY;
    this->z = dZ;
    this->posx = X;
    this->posy = Y;
    this->posz = Z;

    // Default color black
    this->setColor(0, 0, 0);
    this->t = FLT_MAX;
}

__host__ __device__
Ray::Ray(Point dp, Point p)
{
    this->x = dp.X();
    this->y = dp.Y();
    this->z = dp.Z();
    this->posx = p.X();
    this->posy = p.Y();
    this->posz = p.Z();
    
    // Default color black./
    this->setColor(0, 0, 0);
    this->t = FLT_MAX;
}

__host__ __device__
void Ray::setColor(int r, int g, int b)
{
    this->R = r;
    this->G = g;
    this->B = b;
}

__host__ __device__
void Ray::setDir(Point p)
{
    this->x = p.X();
    this->y = p.Y();
    this->z = p.Z();
}

__host__ __device__
void Ray::setStart(Point p)
{
    this->posx = p.X();
    this->posy = p.Y();
    this->posz = p.Z();
}

__host__ __device__
Point Ray::propagate(float time)
{
    return Point(this->x * time + this->posx,
                     this->y * time + this->posy,
                     this->z * time + this->posz);
}


///////////////////////////////////////////////////////////////////////////////
// Point Light operations
///////////////////////////////////////////////////////////////////////////////
__host__ __device__
pointLight::pointLight()
{
    this->setPos(5, 5, 5);
    this->setColor(0, 140, 125);
    this->setAtt_k(0.0005);
}

__host__ __device__
pointLight::pointLight(float X, float Y, float Z,
                       int r, int g, int b, float att_k)
{
    this->setPos(X, Y, Z);
    this->setColor(r, g, b);
    this->setAtt_k(att_k);
}

__host__ __device__
pointLight::pointLight(Point p, int r, int g, int b, float att_k)
{
    this->setPos(p);
    this->setColor(r, g, b);
    this->setAtt_k(att_k);
}

__host__ __device__
void pointLight::setColor(int r, int g, int b)
{
    this->R = r;
    this->G = g;
    this->B = b;
}

__host__ __device__
Point pointLight::getColor()
{
    return Point(this->R, this->G, this->B);
}

__host__ __device__
void pointLight::setAtt_k(float att_k)
{
    this->attenuation_k = att_k;
}

__host__ __device__
float pointLight::getAtt_k()
{
    return this->attenuation_k;
}

__host__ __device__
void pointLight::setPos(Point p)
{
    this->setPos(p.X(), p.Y(), p.Z());
}

__host__ __device__
void pointLight::setPos(float X, float Y, float Z)
{
    this->x = X;
    this->y = Y;
    this->z = Z;
}

__host__ __device__
Point pointLight::getPos()
{
    return Point(this->x, this->y, this->z);
}
