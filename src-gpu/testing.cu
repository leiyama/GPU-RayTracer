#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <iostream>

#include "raytrace_cuda.cuh"
#include "superquadric.cuh"
#include "point.cuh"
#include "matrix.cuh"
#include "camera.h"
#include "parser.h"

// Timing setup Code
cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {              \
    cudaEventCreate(&start);         \
    cudaEventCreate(&stop);          \
    cudaEventRecord(start);          \
}

#define STOP_RECORD_TIMER(name) {    \
    cudaEventRecord(stop);           \
    cudaEventSynchronize(stop);      \
    cudaEventElapsedTime(&name, start, stop); \
    cudaEventDestroy(start);         \
    cudaEventDestroy(stop);          \
}


int main(int argc, char ** argv)
{

    // Now, for GPU implementation
    if (argc != 4) {
        std::cout << "For GPU usage: ./raytracer scene.txt <numBlocks> <threadsPerBlock>" << std::endl;
        return 0;
    }
    // First, get number of blocks
    int blocks = atoi(argv[2]);

    // Then, threadsPerBlock
    int threadsPerBlock = atoi(argv[3]);
    

    // Preparing for CPU stuff
    std::cout << "Preparing for CPU Raytracing..." << std::endl;
    
    std::vector<Superquadric> scene;
    std::vector<pointLight> lights;
    Camera *c = parseObjects(argv[1], scene, lights);
    Point * LookFrom = c->getFrom();
    //std::cout << "Raytracing..." << std::endl;
    float time_elapsed;

    std::cout << "Preparing for GPU Raytracing..." << std::endl;


    // Create a new camera with the same things as above.
    Camera * d_c = new Camera(c);
    
    // Create two device_vectors from the std::vectors above.
    thrust::device_vector<Superquadric> d_scene(scene.begin(), scene.end());
    thrust::device_vector<pointLight> d_lights(lights.begin(), lights.end());


    // Create a device_vector based on the screen from the camera.
    std::vector<Ray> camScreen = d_c->getRayScreen();
    thrust::device_vector<Ray> d_screen(camScreen.begin(), camScreen.end());

    // Get size values for the thread resiliency...
    unsigned int d_scene_size = d_scene.size();
    unsigned int d_lights_size = d_lights.size();
    unsigned int d_screen_size = d_screen.size();

    // Allocate space for the out_scene.
    Superquadric * dev_out_scene;
    cudaMalloc(&dev_out_scene, sizeof(Superquadric) * d_scene_size);
    
    // Prepare the scene...
    cudaCallScenePrep(d_scene, dev_out_scene, d_scene_size, blocks, threadsPerBlock);
    
    std::cout << "Scene Done Being Prepared" << std::endl;
    // Running the Ray Tracer..

    std::cout << "Raytracing..." << std::endl;
    // Allocate space for the point on the GPU
    Point * d_lookFrom;
    cudaMalloc(&d_lookFrom, sizeof(Point));
    cudaMemcpy(d_lookFrom, LookFrom, sizeof(Point), cudaMemcpyHostToDevice);

    Ray * RayScreen;
    cudaMalloc(&RayScreen, sizeof(Ray) * d_screen_size);
    Ray * dev_vector_start = thrust::raw_pointer_cast(&d_screen[0]);
    cudaMemcpy(RayScreen, dev_vector_start, sizeof(Ray) * d_screen_size, cudaMemcpyDeviceToDevice);

    START_TIMER();
    for(int i = 0; i < d_scene_size; i++) {
        cudaCallRayTrace(dev_out_scene + i, d_scene, d_lights, RayScreen, d_screen_size, 
                         d_lookFrom, blocks, threadsPerBlock);
    }
    STOP_RECORD_TIMER(time_elapsed);
	std::cout << "Done with raytrace..." << std::endl;

    // The screen is done. Set the camera's ray vector to be equal to the 
    // screen thrust::vector.
    Ray * host_Screen;
    host_Screen = (Ray*) malloc(sizeof(Ray) * d_screen_size);
    cudaMemcpy(host_Screen, RayScreen, sizeof(Ray) * d_screen_size, cudaMemcpyDeviceToHost);

    std::vector<Ray> out_screen(host_Screen, host_Screen + d_screen_size);
    d_c->setRayScreen(out_screen);


    std::cout << "Printing..." << std::endl;
    
    d_c->gpuPrintImage();

    std::cout << "GPU RayTracing done! Time is " << time_elapsed << std::endl;

    // Free all the things.
    delete c;
    delete d_c;
    free(host_Screen);
    cudaFree(dev_out_scene);
    cudaFree(d_lookFrom);
    cudaFree(RayScreen);

    // Thrust vectors automatically freed upon returning.
    return 0;
}
