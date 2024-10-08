#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>

int windowWidth = 2560;
int windowHeight = 1440;

GLuint pbo;
float globalZoom = 1.0f;
float offsetx = 0.0f;
float offsety = 0.0f;

void initPBO() {
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, windowWidth * windowHeight * 4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

cudaGraphicsResource* cudaPBOResource;

void registerPBOWithCUDA() {
    cudaGraphicsGLRegisterBuffer(&cudaPBOResource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

__global__ void computeFractalKernel(uchar4* pixels, int width, int height, float zoom, float offsetX, float offsetY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    float jx = 1.5f * (x - width / 2) / (0.5f * zoom * width) + offsetX;
    float jy = (y - height / 2) / (0.5f * zoom * height) + offsetY;

    float zx = 0.0f;
    float zy = 0.0f;
    int iter = 0;
    const int maxIter = 2000;

    while (zx * zx + zy * zy < 4.0f && iter < maxIter) {
        float temp = zx * zx - zy * zy + jx;
        zy = 2.0f * zx * zy + jy;
        zx = temp;
        iter++;
    }

    uchar4 color;
    if (iter == maxIter) {
        color = make_uchar4(0, 0, 0, 255);
    } else {
        color = make_uchar4(iter % 256, iter % 256, iter % 256, 255);
    }

    int idx = y * width + x;
    pixels[idx] = color;
}

void launchFractalKernel(uchar4* d_pixels) {
    dim3 blockSize(16, 16);
    dim3 gridSize((windowWidth + blockSize.x - 1) / blockSize.x, (windowHeight + blockSize.y - 1) / blockSize.y);

    computeFractalKernel<<<gridSize, blockSize>>>(d_pixels, windowWidth, windowHeight, 1.0f, 0.0f, 0.0f);

    cudaDeviceSynchronize();
}

void handleMouseWheel(int wheel, int direction, int x, int y) {
    float mouseNormX = 2.0f * (x / (float)windowWidth) - 1.0f;
    float mouseNormY = 1.0f - 2.0f * (y / (float)windowHeight);

    float oldFractalX = (mouseNormX / globalZoom) + offsetx;
    float oldFractalY = (mouseNormY / globalZoom) + offsety;

    if (direction > 0) {
        globalZoom *= 1.25f;
    } else if (direction < 0) {
        globalZoom *= 0.8f;
    }

    float newFractalX = (mouseNormX / globalZoom) + offsetx;
    float newFractalY = (mouseNormY / globalZoom) + offsety;

    offsetx += (oldFractalX - newFractalX);
    offsety += (oldFractalY - newFractalY);

    if (globalZoom < 1.0f) {
        globalZoom = 1.0f;
    }
}

void display() {
    cudaGraphicsMapResources(1, &cudaPBOResource, 0);
    uchar4* d_pixels;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_pixels, &num_bytes, cudaPBOResource);

    launchFractalKernel(d_pixels);

    cudaGraphicsUnmapResources(1, &cudaPBOResource, 0);

    glClear(GL_COLOR_BUFFER_BIT);
    glRasterPos2i(-1, -1);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glDrawPixels(windowWidth, windowHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glutSwapBuffers();
    glutPostRedisplay();
}

void cleanup() {
    if (cudaPBOResource) {
        cudaGraphicsUnregisterResource(cudaPBOResource);
        cudaPBOResource = nullptr;
    }
    if (pbo) {
        glDeleteBuffers(1, &pbo);
        pbo = 0;
    }
    cudaDeviceReset();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(windowWidth, windowHeight);
    glutCreateWindow("CUDA Fractal");
    glutMouseWheelFunc(handleMouseWheel);

    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "Error initializing GLEW: " << glewGetErrorString(err) << std::endl;
        return -1;
    }

    initPBO();

    registerPBOWithCUDA();

    glutDisplayFunc(display);

    atexit(cleanup);

    glutMainLoop();

    return 0;
}
