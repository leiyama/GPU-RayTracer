#include <vector>
#include "superquadric.cuh"
#include "point.cuh"
#include "camera.h"

Camera * parseObjects(const char *filename,
                      std::vector<Superquadric> &scene,
                      std::vector<pointLight> &lights);
