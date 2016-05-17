*** PLEASE READ BEFORE STARTING THE ASSIGNMENT ***
NOTE: If you have not read the assignment notes yet, I highly recommend that you
do that BEFORE you read this. All of the math discussed here are all explained
in the assignment notes. Also, this assignment (at least the first portion) is
overwhelmingly about understanding the math and the tools we provide you so make
sure you take your time to read the assignment notes and this manual. The actual
coding you need to do for the first part of the assignment should be quite
simple.


    In this assignment, you will be using some tools that we provide for you to
build/load a scene and render the scene. These tools have been built and
blackboxed so do not worry about the implementation details and instead focus on
learning how to use the tools. If you are curious as to how everything fits
together behind the scenes, feel free to read through the files but it may be
quite time consuming so I do not recommend doing so.


/********************************* Terminology ********************************/

Primitive   we call superquadrics primitives in our program
Object      we group superquadrics as Objects. Objects have children Objects and
            Primitives. The end goal of this assignment is to select an Object
            in the command line and raytrace it.
Renderable  we call Primitives and Objects Renderables as a whole. as the name
            suggests, essentially anything we're going to render is called a
            Renderable


/******************************* The CommandLine ******************************/

    One of the main tools we provide you the command line modeling tool. When
you first setup, compile and run the code we provide for you (it should compile
and run without you touching anything) you will see an OpenGL display open along
with a command line prompt within your command line like so:

username@your_computer:<path to assignment>/modeler/src$ ./modeler 800 800
CS 171 Modeling Language
type help for the instruction manual
> 

    If the command line is active, the OpenGL display will be non-interactive.
You can interact with the scene by using the command 'interact'. Once the OpenGL
display is interactive you'll be able to arcball rotate the scene, zoom in and
out (use w key to zoom in and the s key to zoom out), etc using your keyboard
and mouse. If you wish to return to the command line simply press 'q' or the esc
key (Hint: you're going to have to get used to using alt+tab to switch between
the command line and the display). One thing you'll notice are the red, green
and blue lines on the screen. Those are the axes which are setup like so:

    red = x
    green = y
    blue = z
    (RGB <=> XYZ)

    bright line = positive, dim line = negative

    Once you're familiar with switching between the command line and the display
type in 'help' in the command line. This will show you all of the commands you
can use in the command line. Note that some of the commands have alias(es). A
lot of the commands can be quite cumbersome to type in so get familiar with all
the aliases (e.g. typing in 'prm' is much easier to do than typing in
'Primitive'). Play around with the 'help' command to get familiar with how to
use it but don't worry about learning all the commands just yet. That will be
explained below.

    Now that you know the basics, let's get into the specifics. In order to use
this command line well, you just have to keep one thing in mind.

    THE DISPLAY IS SHOWING YOU THE CURRENTLY SELECTED RENDERABLE.
    THE DISPLAY IS SHOWING YOU THE CURRENTLY SELECTED RENDERABLE.
    THE DISPLAY IS SHOWING YOU THE CURRENTLY SELECTED RENDERABLE.
    THE DISPLAY IS SHOWING YOU THE CURRENTLY SELECTED RENDERABLE.
    THE DISPLAY IS SHOWING YOU THE CURRENTLY SELECTED RENDERABLE.

    This is the idea behind the basic intuition on how this command line and the
display was designed so it's very important that you understand this (hence the
repetition and the capitalization lol). Now, try creating and selecting a
primitive by typing in:

    prm test_prm

    You should see a white ball appear in the display. This is the default
primitive. Basically what you just did was request a selection of a primitive
called test_prm. Since such a primitive does not exist yet, the program
generated one and selected it. Type in 'info' and you'll see this:

> info
printing scene info
    currently active Primitive(s):
        test_prm
    currently active Object(s):
DONE
currently selected: PRM test_prm

    Because you didn't specify what Renderable you wanted the info of, the
command line simply printed out information on everything stored in the program.
Note that now there is 1 active primitive called test_prm and the currently
selected Renderable is a PRM (primitive) called test_prm.

    Now, if you type in 'info selected' or 'info test_prm' you'll see this:

> info selected
currently selected: PRM test_prm
    coeff: 1.000000 1.000000 1.000000
    exponents: 1.000000 1.000000
    patch: 15 15
    color: 1.000000 1.000000 1.000000
    ambient: 0.100000
    reflected: 0.300000
    refracted: 0.500000
    gloss: 0.300000
    diffuse: 0.800000
    specular: 0.100000

or 

> info test_prm
test_prm:
    coeff: 1.000000 1.000000 1.000000
    exponents: 1.000000 1.000000
    patch: 15 15
    color: 1.000000 1.000000 1.000000
    ambient: 0.100000
    reflected: 0.300000
    refracted: 0.500000
    gloss: 0.300000
    diffuse: 0.800000
    specular: 0.100000

    You're now looking at the information on test_prm. You can change all of
these values using the following commands (they are quite self explanatory if
you've read the assignment notes on superquadrics). Play around with it a bit.

    setCoeff       (aliases: coeff Coeff)
    setExponent    (aliases: Exponent exp exponent)
    setPatch       (aliases: patch Patch)
    setColor       (aliases: surface color)
    setAmbient     (aliases: Ambient ambient)
    setReflected   (aliases: Reflected reflected)
    setRefracted   (aliases: Refracted refracted)
    setGloss       (aliases: gloss)
    setDiffuse     (aliases: diffuse)
    setSpecular    (aliases: specular)

    Now, before we go onto Object related commands, play around more and create
some more primitives. Having more primitives will help you understand the next
steps better. In my example, I played around till typing in 'info all' or 'info'
prints out:

> info
printing scene info
    currently active Primitive(s):
        test_prm3
        test_prm2
        test_prm
    currently active Object(s):
DONE
currently selected: PRM test_prm3

    and typing in 'info test_prm test_prm2 test_prm3' prints out:

> info test_prm test_prm2 test_prm3
test_prm:
    coeff: 2.000000 1.000000 1.000000
    exponents: 1.000000 1.000000
    patch: 100 100
    color: 1.000000 0.200000 0.200000
    ambient: 0.100000
    reflected: 0.300000
    refracted: 0.500000
    gloss: 0.300000
    diffuse: 0.800000
    specular: 0.500000

test_prm2:
    coeff: 1.000000 1.000000 1.000000
    exponents: 1.000000 1.000000
    patch: 15 15
    color: 1.000000 1.000000 1.000000
    ambient: 0.100000
    reflected: 0.300000
    refracted: 0.500000
    gloss: 0.300000
    diffuse: 0.800000
    specular: 0.100000

test_prm3:
    coeff: 1.000000 1.000000 1.000000
    exponents: 0.100000 0.100000
    patch: 100 100
    color: 1.000000 1.000000 1.000000
    ambient: 0.100000
    reflected: 0.300000
    refracted: 0.500000
    gloss: 0.300000
    diffuse: 0.800000
    specular: 0.100000

    You don't have to have this exact setup. I'm just trying to give you an
example. Have some fun! See what shapes you can make.

    Now, let's go onto Objects. Type in 'obj test_obj'. Your display will go
blank except for the axes. Don't be alarmed. This is because the command line
selected your test_obj but your test_obj does not have anything inside of it.
You can confirm this by typing in 'info selected':

> info selected
currently selected: OBJ test_obj
    overall transformation(s):
    child object(s):
    child primitive(s):
    current cursor set at: [NONE]

    Now, let's make it a bit more interesting by adding in a primitive. Type in
'addPrimitive test_prm' or 'ap test_prm' for short. Your display will now show
your test_prm. Now typing in 'info selected' will print:

> info selected
currently selected: OBJ test_obj
    overall transformation(s):
    child object(s):
    child primitive(s):
        PRM test_prm
    current cursor set at: test_prm
    transformation(s) on cursor renderable:

    Now, it shows that test_prm is a child primitive and the cursor is set to
test_prm. The cursor is a tool you're going to use within an object to apply
transformations on a child. To see what this means, type in 'translate 1 0 0'.
You will see that test_prm in the display got translated by 1 in the x
direction. If you type 'info selected' again, you will now see:

> info selected
currently selected: OBJ test_obj
    overall transformation(s):
    child object(s):
    child primitive(s):
        PRM test_prm
    current cursor set at: test_prm
    transformation(s) on cursor renderable:
        TRANS 1.000000 0.000000 0.000000

    Now add another primitive and apply transformations on that primitive. In my
example, I added test_prm2 and applied a scaling by 2 2 2 and a translation by
0 3 0. 'info selected' prints:

> info selected
currently selected: OBJ test_obj
    overall transformation(s):
    child object(s):
    child primitive(s):
        PRM test_prm2
        PRM test_prm
    current cursor set at: test_prm2
    transformation(s) on cursor renderable:
        SCALE 2.000000 2.000000 2.000000
        TRANS 0.000000 3.000000 0.000000

    If I want to access or modify the transformations applied on test_prm child,
I can input 'cursor test_prm'. After this change in cursor, 'info selected' now
prints:

> info selected
currently selected: OBJ test_obj
    overall transformation(s):
    child object(s):
    child primitive(s):
        PRM test_prm2
        PRM test_prm
    current cursor set at: test_prm
    transformation(s) on cursor renderable:
        TRANS 1.000000 0.000000 0.000000

    This cursor operation along with the addPrimitive (or ap) and the addObject
(or ao) commands is how you're going to combine primtives and objects inside an
object. Play around with these combining feature of the command line. As you
add children objects and primitives, you will be able to see the changes in the
display. This is the primary method with which you can build scenes using this
tool.

    Note that you can give children aliases. For example, if I do
'ap test_prm test_prm_alias', the 'info selected' command will print:

> info selected
currently selected: OBJ test_obj
    overall transformation(s):
    child object(s):
    child primitive(s):
        PRM test_prm aliased as test_prm_alias
        PRM test_prm2
        PRM test_prm
    current cursor set at: test_prm_alias
    transformation(s) on cursor renderable:

    This is how you can reuse prmitives and objects. There are other cool things
you can do with these commands such as cause the program to recursively draw a
single object. We put in a recursion limit to the drawing so you'll be able to
draw fractal patterns by adding objects to themselves. HOWEVER, be careful when
using this feature. The recursion limit is only applied to the drawing of the
display so if you're recursing through the children yourself in your assignment,
you're going to have to apply this limit yourself. NOT DOING SO WILL CAUSE AN
INFINITE LOOP THAT WILL CAUSE A STACK OVERFLOW.

    Now that I've shown you how to build scenes, I'll show you how to save and
load your work. Typing in 'save' will save your progress into a file called
quicksave.scn in the data directory. If you type in 'load', your quicksave.scn
file will be loaded. You can also specify which file to save your progress in.
Type in 'save data/my_scene.scn' and the program will save your progress to a
file called my_scene.scn in the data directory. The load command works in the
same way.

    We've already provided for you a robot_arm.scn file. Load this file by
typing in 'load data/robot_arm.scn'. IMPORTANT NOTE: loading a new scene will
wipe all of your current progress. PLEASE save your progress if you want to keep
it.

    Don't forget to use the 'interact' command to interact with the display.


/****************************** Source Code Tools *****************************/

    Now, you're going to be using some of the blackboxed functionality we've
already written for you to recurse through Objects and their children in the
assignment. Therefore, it is important that you understand how to call and use
these functions. All of this code can be found in the source code and are
included in the assignment.cpp file.

    Below, I'm going to explain three classes and the functions in those classes
that you will be using. The three classes are:

    - struct Transformation
    - class Renderable
    - class Primitive : public Renderable
    - class Object : public Renderable

IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!
IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!
NOTE: If you do not understand how any of the following in C++ works PLEASE
PLEASE PLEASE PLEASE PLEASE look them up in Google and learn them before you
continue:
    
    - polymorphism
    - dynamic_cast
    - static functions
    - C++ std::vector
    - C++ std::unordered_map
    - tree structure and tree recursion

Seriously, if you don't understand the above three things, you will have no idea
how to use any of the tools correctly.



/************************** The Transformation Struct *************************/

enum TransformationType {TRANS, SCALE, ROTATE};
struct Transformation {
    TransformationType type;
    Vector4f trans;
    // translation defined by a 3f displacement vector with trailing 1
    // scaling defined by a 3f scaling vector with trailing 1
    // rotation defined by a quaternion (x, y, z, theta) where the theta is in
    // radians
    Transformation(TransformationType type, const Vector4f& trans);
    Transformation(
        TransformationType type,
        const float x,
        const float y,
        const float z,
        const float w);
};

    This struct is pretty self explanatory. Just note that the rotation stores
theta in radians. You input theta as degrees in the command line but it is
converted to radians before being stored.



/**************************** The Renderable Class ****************************/

enum RenderableType {PRM, OBJ, MSH};
class Renderable {
// private and protected variables and methods YOU DO NOT NEED THESE ...

public:
    // static instance controller functions YOU DO NOT NEED THESE...

    // two static instance controller functions that you will have to use
    static Renderable* get(const Name& name);
    static bool exists(const Name& name);

    // pure virtual function used to determining type of a Renderable
    virtual const RenderableType getType() const = 0;
};

    This is the stripped down version of the class. You ONLY need the methods I
left in this version for this assignment. All other functions work behind the
scenes. In order to understand what this class does, you have to understand how
the program keeps track of all Renderables. Any Renderable you use will be
associated with a name. In the tutorial for the CommandLine above, you will have
noticed that every time you create or select a Primitive or an Object, you have
to specify a name. This Renderable class keeps track of all those names and
links them up to a Primitive or an Object. The functions shown above can be used
with a name to check the Renderable's existance or get the Renderable itself.
Those functions can be called like so:

Renderable* ren = Renderable::get("test_prm");
bool ren_exists = Renderable::exists("test_prm");

IMPORTANT
NOTE: the Renderable::get() function will return NULL if the Renderable does not
exist. If you get a NULL pointer dereference error, this is probably where you
made your mistake
IMPORTANT

    The getType() function is a pure virtual function that is overriden in the
Primitive and the Object classes to return their corresponding types (i.e.
calling getType() from a Primitive will return PRM, and calling getType() from a
Object will return OBJ). This function along with dynamic_cast will come in
handy for retrieving a Renderable pointer and converting it into the pointer of
the appropriate type.



/***************************** The Primitive Class ****************************/

class Primitive : public Renderable {
private:
    // stuff you don't need

public:
    // extraneous functions you do not need...

    const RenderableType getType() const {
        return PRM;
    }

    // accessors for private variables
    const Vector3f& getCoeff() const;
    const float getExp0() const;
    const float getExp1() const;
    const unsigned int getPatchX() const;
    const unsigned int getPatchY() const;
    const RGBf& getColor() const;
    float getAmbient() const;
    float getReflected() const;
    float getRefracted() const;
    float getGloss() const;
    float getDiffuse() const;
    float getSpecular() const;

    const Vector3f getNormal(const Vector3f& vertex);
};

    Fairly straight forward.



/****************************** The Object Class ******************************/

struct Child {
    Name name;
    vector<Transformation> transformations;

    Child();
    Child(const Name& name);
};

class Object : public Renderable {
private:
    // stuff you don't need

public:
    // extraneous functions you do not need...

    const RenderableType getType() const {
        return OBJ;
    }

    // accessors and modifiers below

    bool aliasExists(const Name& name);

    // overall transformation
    const vector<Transformation>& getOverallTransformation() const;

    // children objects and primitives
    const unordered_map<Name, Child, NameHasher>& getChildren() const;
};

    The only important thing to note here is that you have to be careful not to
confuse aliases and names. I repeat.

    DO NOT CONFUSE ALIASES AND NAMES.
    DO NOT CONFUSE ALIASES AND NAMES.
    DO NOT CONFUSE ALIASES AND NAMES.
    DO NOT CONFUSE ALIASES AND NAMES.
    DO NOT CONFUSE ALIASES AND NAMES.

    Aliases are only significant within the Object. Names that you want to use
in order to traverse the tree of Renderables is stored in the Child struct. The
Name key used in the unordered_map returned by the getChildren() function is the
alias.

    In the assignment you will be recursing through a tree of Renderables from
the currently selected Renderable (i.e. the one being displayed). The way you
retrieve the current Renderable is to use the following code:

const Line* cur_state = CommandLine::getState();
Renderable* ren = NULL;

if (cur_state) {
    ren = Renderable::get(cur_state->tokens[1]);
}

After this bit of code, the currently selected Renderable will be stored in the
pointer ren. This code is very much tied to how the backend of the command line.
If you would like to learn more about how it works, email me at jwon@caltech.edu