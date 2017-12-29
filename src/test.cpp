#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mvnc.h>
#include <cmath>
#include <iostream>

/*OpenCV includes*/
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#include "fp16.h"
// #include "half.hpp"

#define GRAPH_PATH "resources/graph"
#define TEST_IMAGE "resources/test2.jpg"

// 16 bits.  will use this to store half precision floats since C++ has no
// built in support for it.
typedef unsigned short half;
const int networkDim = 416;
const float anchors[] = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828};
const int coffs[] = {0, 4, 8, 12, 16};
const int scoffs[] = {20, 21, 22, 23, 24};
const int pcoffs[] = {25, 26, 27, 28, 29};

void *LoadGraphFile(const char *path, unsigned int *length);
half *LoadImage(const char *path, int reqsize);

using namespace cv;

float sigmoid(float x)
{
    float exp_value;
    float return_value;

    /*** Exponential calculation ***/
    exp_value = exp((double)-x);

    /*** Final sigmoid value ***/
    return_value = 1 / (1 + exp_value);

    return return_value;
}

int main(int argc, char **argv)
{
    mvncStatus retCode;
    void *deviceHandle;
    void *graphHandle;
    char devName[100];
    unsigned int graphFileLen;
    void *graphFileBuf;
    Mat img = imread(TEST_IMAGE);

    //buscando stick
    retCode = mvncGetDeviceName(0, devName, 100);
    if (retCode != MVNC_OK)
    {
        printf("Error. No encontrado ningun NCS. \n");
        printf("\tmvncStatus: %d\n", retCode);
        exit(-1);
    }

    //intentando abrir dispositivo usando nombre encontrado
    retCode = mvncOpenDevice(devName, &deviceHandle);
    if (retCode != MVNC_OK)
    {
        printf("Error. No se ha podido acceder al dispositivo NCS. \n");
        printf("\tmvncStatus: %d\n", retCode);
        exit(-1);
    }

    //desipositivo listo para usarse
    printf("Dispositoivo NCS preparado\n");

    graphFileBuf = LoadGraphFile(GRAPH_PATH, &graphFileLen);

    // allocate the graph
    retCode = mvncAllocateGraph(deviceHandle, &graphHandle, graphFileBuf, graphFileLen);
    free(graphFileBuf);
    if (retCode != MVNC_OK)
    { // error allocating graph
        printf("No se ha podido cargar el grafo del fichero: %s\n", GRAPH_PATH);
        printf("\tError: %d\n", retCode);
        exit(-1);
    }

    // successfully allocated graph.  Now graphHandle is ready to go.
    // use graphHandle for other API calls and call mvncDeallocateGraph
    // when done with it.
    printf("Successfully allocated graph for %s\n", GRAPH_PATH);

    half *image = LoadImage(TEST_IMAGE, networkDim);
    unsigned int lenImage = 3 * networkDim * networkDim * sizeof(half);

    retCode = mvncLoadTensor(graphHandle, image, lenImage, NULL);
    free(image);
    if (retCode != MVNC_OK)
    { // error loading tensor
        printf("No se ha podido cargar la imagen de prueba: %s\n", GRAPH_PATH);
        printf("\tError: %d\n", retCode);
        exit(-1);
    }

    void *resultData16;
    void *userParam;
    unsigned int lenResultData;
    retCode = mvncGetResult(graphHandle, &resultData16, &lenResultData, &userParam);
    if (retCode != MVNC_OK)
    {
        printf("Error - No se ha podido obtener el resultado de la imagen\n");
        printf("\tError: %d\n", retCode);
        exit(-1);
    }

    // convert half precision floats to full floats
    int numResults = lenResultData / sizeof(half);
    float *resultData32;
    resultData32 = (float *)malloc(numResults * sizeof(float));

    fp16tofloat(resultData32, (unsigned char *)resultData16, numResults);

    printf("Hey %d\n", numResults);
    printf("Obtenida inferencia de la imagen\n");

    float maxResult = 0.0, classProb = 0.0;
    int maxIndex = -1;
    for (int cx = 0; cx < 13; cx++)
    {
        for (int cy = 0; cy < 13; cy++)
        {
            int index = 13 * cx + cy;
            printf("Cell %d:\n", index);
            for (int j = 0; j < 5; j++)
            {
                float x = 0.0f, y = 0.0f, w = 0.0f, h = 0.0f, score = 0.0f, prob = 1.0f;
                x = (cx + sigmoid(resultData32[30 * index + coffs[j]])) * 32;
                y = (cy + sigmoid(resultData32[30 * index + coffs[j] + 1])) * 32;
                w = exp(resultData32[30 * index + coffs[j] + 2]) * anchors[2 * j] * 32;
                h = exp(resultData32[30 * index + coffs[j] + 3]) * anchors[2 * j + 1] * 32;
                score = sigmoid(resultData32[30 * index + scoffs[j]]);
                // in this case, always 1 as there is only one possilbe class
                // prob = vector of softmax function

                if (score > 0.7f)
                {
                    Point pt1, pt2;
                    pt1.x = (int)round(x - w / 2);
                    pt1.y = (int)round(y - h / 2);
                    pt2.x = (int)round(x + w / 2);
                    pt2.y = (int)round(y + h / 2);
                    rectangle(img, pt1, pt2, Scalar(255, 0, 0), 2);
                    printf("\tBox[x:%f, y:%f, w: %f, h:%f] score: %f, class: %f \n", x, y, w, h, score, prob);
                }
                if (maxResult < score)
                {
                    maxResult = score;
                    maxIndex = index;
                    classProb = prob;
                }
            }
        }
    }
    imshow("test", img);
    cv::waitKey();
    printf("Index of top result is: %d\n", maxIndex);
    printf("Probability of top result is: %f\n", maxResult);

    printf("Imagen cargada en dispositivo con exito\n");

    retCode = mvncDeallocateGraph(graphHandle);
    graphHandle = NULL;
    if (retCode != MVNC_OK)
    {
        printf("Error. No se ha podido eliminar el grafo en memoria. \n");
        printf("\tCodigo: %d\n", retCode);
        exit(-1);
    }

    retCode = mvncCloseDevice(deviceHandle);
    deviceHandle = NULL;
    if (retCode != MVNC_OK)
    {
        printf("Error. No se ha podido cerrar el dispositivo NCS. \n");
        printf("\tmvncStatus: %d\n", retCode);
        exit(-1);
    }

    printf("Dispositivo NCS cerrado correctamente.\n");
}

//Movidius NCAPPZoo github: https://github.com/movidius/ncappzoo
void *LoadGraphFile(const char *path, unsigned int *length)
{
    FILE *fp;
    char *buf;

    fp = fopen(path, "rb");
    if (fp == NULL)
        return 0;
    fseek(fp, 0, SEEK_END);
    *length = ftell(fp);
    rewind(fp);
    if (!(buf = (char *)malloc(*length)))
    {
        fclose(fp);
        return 0;
    }
    if (fread(buf, 1, *length, fp) != *length)
    {
        fclose(fp);
        free(buf);
        return 0;
    }
    fclose(fp);
    return buf;
}

//Movidius NCAPPZoo github: https://github.com/movidius/ncappzoo
half *LoadImage(const char *path, int reqSize)
{
    int width, height, cp, i;
    unsigned char *img, *imgresized;
    float *imgfp32;
    half *imgfp16;

    img = stbi_load(path, &width, &height, &cp, 3);
    if (!img)
    {
        printf("Error - the image file %s could not be loaded\n", path);
        return NULL;
    }
    imgresized = (unsigned char *)malloc(3 * reqSize * reqSize);
    if (!imgresized)
    {
        free(img);
        perror("malloc");
        return NULL;
    }
    stbir_resize_uint8(img, width, height, 0, imgresized, reqSize, reqSize, 0, 3);
    free(img);
    imgfp32 = (float *)malloc(sizeof(*imgfp32) * reqSize * reqSize * 3);
    if (!imgfp32)
    {
        free(imgresized);
        perror("malloc");
        return NULL;
    }
    for (i = 0; i < reqSize * reqSize * 3; i++)
        imgfp32[i] = imgresized[i];
    free(imgresized);
    imgfp16 = (half *)malloc(sizeof(*imgfp16) * reqSize * reqSize * 3);
    if (!imgfp16)
    {
        free(imgfp32);
        perror("malloc");
        return NULL;
    }
    for (i = 0; i < reqSize * reqSize; i++)
    {
        float blue, green, red;
        blue = imgfp32[3 * i + 2];
        green = imgfp32[3 * i + 1];
        red = imgfp32[3 * i + 0];

        imgfp32[3 * i + 0] = blue / 255.0f;
        imgfp32[3 * i + 1] = green / 255.0f;
        imgfp32[3 * i + 2] = red / 255.0f;

        // uncomment to see what values are getting passed to mvncLoadTensor() before conversion to half float
        //printf("Blue: %f, Green: %f,  Red: %f \n", imgfp32[3*i+0], imgfp32[3*i+1], imgfp32[3*i+2]);
    }
    floattofp16((unsigned char *)imgfp16, imgfp32, 3 * reqSize * reqSize);
    free(imgfp32);
    // for(i = 0; i < reqSize*reqSize; i++) {
    //     half blue, green, red;

    //     blue = imgfp16[3*i+2];
    //     green = imgfp16[3*i+1];
    //     red = imgfp16[3*i+0];
    //     printf("Blue: %d, Green: %d,  Red: %d \n", blue, green, red);
    // }
    return imgfp16;
}