###############################################################################
# Makefile for modeler
# written by: Parker Won
###############################################################################

CC = g++
FLAGS = -Wall -g -std=c++11
LDFLAGS = -pthread -L/usr/X11R6/lib -L/usr/local/lib 
LDLIBS = -lm -lGLEW -lGL -lGLU -lglut -lpng
INCLUDE = -I../lib -I/usr/include -I/usr/X11R6/include -I/usr/include/GL -I/usr/include/libpng
SOURCES = main.cpp model.o Renderer.o Scene.o UI.o Utilities.o Shader.o Assignment.o PNGMaker.o
EXENAME = modeler

all: $(EXENAME)

$(EXENAME): $(SOURCES)
	$(CC) $(FLAGS) -o $(EXENAME) $(INCLUDE) $(LDFLAGS) $(SOURCES) $(LDLIBS)

%.o: %.cpp %.hpp
	$(CC) $(FLAGS) -o $@ $(LIBDIR) $(INCLUDE) $(LDFLAGS) -c $< $(LDLIBS)

clean:
	rm -rf *.o $(EXENAME)

.PHONY: all clean
