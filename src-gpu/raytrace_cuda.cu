#include "raytrace_cuda.cuh"

// This kernel will parallelize the scene preparation
__global__
void cudaScenePrep(Superquadric * start, Superquadric * dev_out_scene, unsigned int size) {
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    // Thread Resiliency
    while (index < size) {
        (*(start + index)).setNum(index);
        dev_out_scene[index] = *(start + index);
        index += blockDim.x * gridDim.x;
    }
    // Syncing threads so that they all finish.
    __syncthreads();
}


// This will just call the kernel...
void cudaCallScenePrep(thrust::device_vector<Superquadric> scene, Superquadric * dev_out_scene,
                       unsigned int size,
                       int blocks, int threadsPerBlock) {

    Superquadric * start = thrust::raw_pointer_cast(&scene[0]);
    cudaScenePrep<<<blocks, threadsPerBlock>>>(start, dev_out_scene, size);
}


// This kernel will be called in the "runRayTrace" thing from camera.
// This will be parallelized based on the screen.
__global__
void cudaRayTrace(Superquadric * object,
                  Superquadric * sceneStart, 
                  pointLight * lightStart,
                  Ray * start,
                  unsigned int size, unsigned int lightSize, unsigned int sceneSize, Point * lookFrom) {
    // Thread resiliency measuresi.
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    while (index < size) {
        
        Ray targetRay = *(start + index);
        Point origin = targetRay.getStart();
        Point dir = targetRay.getDir();

        Point temp_lookFrom;
        temp_lookFrom.set(lookFrom->X(), lookFrom->Y(), lookFrom->Z());

        // Transform frame of reference so that this object is at origin.
        Point new_origin = object->applyTransforms(origin);
        Point new_dir = (object->applyDirTransforms(dir)).norm();


        // Create new ray to do intersection test.
        Ray transR;
        transR.setStart(new_origin);
        transR.setDir(new_dir);

        // Check for intersection
        float intersects = object->get_intersection(transR);
        // If there is an intersection
        if (intersects != FLT_MAX && intersects < targetRay.getTime()) {
            // Calculate the intersection point
            Point pTran = transR.propagate(intersects);
            Point pTrue = object->revertTransforms(pTran); 
            // Get the normal at the intersection point
            Point n = object->revertDirTransforms((object->getNormal(pTran)).norm());
            // Point *showNorm = *pTran + *(*n / 10);
            Point color = object->lighting(pTrue, n, temp_lookFrom, lightStart, sceneStart,
                                            lightSize, sceneSize);

            (*(start + index)).setColor(color.X(), color.Y(), color.Z());                       
            (*(start + index)).setTime(intersects);
           }
        index += blockDim.x * gridDim.x;
    } 
    // Syncing threads so that they all finish...
    __syncthreads();
}

void cudaCallRayTrace(Superquadric * object,
                      thrust::device_vector<Superquadric> scene, 
                      thrust::device_vector<pointLight> lights,
                      Ray * RayScreen,
                      unsigned int size, Point * lookFrom, int blocks,
                      int threadsPerBlock) {
    pointLight * lightStart = thrust::raw_pointer_cast(&lights[0]);
    Superquadric * sceneStart = thrust::raw_pointer_cast(&scene[0]);

    unsigned int lightSize = lights.size();
    unsigned int sceneSize = scene.size();

    cudaRayTrace<<<blocks, threadsPerBlock>>> (object, sceneStart, lightStart,
        RayScreen, size, lightSize, sceneSize, lookFrom);
}

