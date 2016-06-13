#include "Renderer.hpp"
#include "model.hpp"

#include <cstdio>

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: ./modeler xres yres\n");
        return 1;
    }

    // Initialize GLUT and its window
    glutInit(&argc, argv);

    int xres = atoi(argv[1]);
    int yres = atoi(argv[2]);

    glutInitWindowSize(xres, yres);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutCreateWindow("Phong Renderer");

    char* name = "sphere";
    Renderable::create(PRM, name);
    
    Renderer *renderer = Renderer::getSingleton(xres, yres);
    renderer->init();
    renderer->start();

    return 0;
}