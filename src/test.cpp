#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#define TEST_IMAGE "resources/image.jpg"

// 16 bits.  will use this to store half precision floats since C++ has no
// built in support for it.
typedef unsigned short half;
const int networkDim = 300;
const char *labels[] = {"background", "marcador"};
const float anchors[] = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828};
const int coffs[] = {0, 4, 8, 12, 16};
const int scoffs[] = {20, 21, 22, 23, 24};
const int pcoffs[] = {25, 26, 27, 28, 29};

void *LoadGraphFile(const char *path, unsigned int *length);
half *LoadImage(const char *path, int reqsize);
void normalize_image(std::vector<float> image, float **dst);

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

    Mat cvImg, aux;
    Size size(300, 300);
    resize(img, aux, size);
    aux.convertTo(cvImg, CV_32FC3, 1 / 255.0);

    std::vector<float> array;
    if (cvImg.isContinuous())
    {
        array.assign((float *)cvImg.datastart, (float *)cvImg.dataend);
    }
    else
    {
        for (int i = 0; i < cvImg.rows; ++i)
        {
            array.insert(array.end(), cvImg.ptr<float>(i), cvImg.ptr<float>(i) + cvImg.cols);
        }
    }

    //NORMALIZE
    printf("%d\n", (int)array.size());
    float *cvImg32 = (float *)malloc(((int)array.size()) * sizeof(float));
    for (int i = 0; i < (int)array.size() / 3; i++)
    {
        cvImg32[3 * i] = (array[3 * i] * 2.0f - 1.0f);
        cvImg32[3 * i + 1] = (array[3 * i + 1] * 2.0f - 1.0f);
        cvImg32[3 * i + 2] = (array[3 * i + 2] * 2.0f - 1.0f);
    }

    half *image = (half *)malloc(((int)array.size()) * sizeof(half));
    unsigned int lenImage = ((int)array.size());
    floattofp16((unsigned char *)(void *)image, cvImg32, lenImage);
    // half *image = LoadImage(TEST_IMAGE, networkDim);

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
    float len = resultData32[0];

    for (int i = 0; i < len; i++)
    {
        int index = 7 + i * 7;

        if (resultData32[index] != INFINITY && resultData32[index + 1] != INFINITY && resultData32[index + 1] != INFINITY && resultData32[index + 2] != INFINITY && resultData32[index + 3] != INFINITY && resultData32[index + 4] != INFINITY && resultData32[index + 5] != INFINITY && resultData32[index + 6] != INFINITY && !isnanf(resultData32[index]) && !isnanf(resultData32[index + 1]) && !isnanf(resultData32[index + 1]) && !isnanf(resultData32[index + 2]) && !isnanf(resultData32[index + 3]) && !isnanf(resultData32[index + 4]) && !isnanf(resultData32[index + 5]) && !isnanf(resultData32[index + 6]))
        {
            float x1, x2, y1, y2, score;
            int center[2];
            x1 = resultData32[index + 3] * img.size().width;
            x2 = resultData32[index + 5] * img.size().width;
            y1 = resultData32[index + 4] * img.size().height;
            y2 = resultData32[index + 6] * img.size().height;

            /* Discard if is out of bounds*/
            if (x1 < 0 || y1 < 0 || x2 < 0 || y2 < 0)
            {
                continue;
            }

            if (x1 > img.size().width || y1 > img.size().height || x2 > img.size().width || y2 > img.size().height)
            {
                continue;
            }
            /*******************************/

            /* Discard if dimensions make no sense */
            float rel = (y2 - y1) / (x2 - x1);
            if (rel > 3.0f || rel < 0.33f)
            {
                continue;
            }
            /***************************************/

            center[0] = (int)round((x1 + x2) / 2);
            center[1] = (int)round((y1 + y2) / 2);
            score = resultData32[index + 2] * 100.0f;
            const char *label = labels[(int)round(resultData32[index + 1])];
            Point pt1, pt2;
            pt1.x = (int)x1;
            pt1.y = (int)y1;
            pt2.x = (int)x2;
            pt2.y = (int)y2;
            if (score > 0.6f)
            {
                char prob[20];
                sprintf(prob, "%f", score);
                rectangle(img, pt1, pt2, Scalar(255, 0, 0), 2);
                Scalar col(125, 175, 75);
                Size labelsize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, NULL);
                // rectangle(img, Point(pt1.x - 1, pt1.y - labelsize.height - 1), Point(pt1.x + labelsize.width + 1, pt1.y + labelsize.height + 1), col, -1);
                putText(img, prob, Point(pt1.x, pt1.y + labelsize.height), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
            }
            printf("\tBox%d[x1:%f, y1:%f, x2: %f, y2:%f] score: %f label: %s\n", i, x1, y1, x2, y2, score, label);
        }
    }
    // printf("Obtenida inferencia de la imagen\n");

    // float maxResult = 0.0, classProb = 0.0;
    // int maxIndex = -1;
    // for (int cx = 0; cx < 13; cx++)
    // {
    //     for (int cy = 0; cy < 13; cy++)
    //     {
    //         int index = 13 * cx + cy;
    //         printf("Cell %d:\n", index);
    //         for (int j = 0; j < 5; j++)
    //         {
    //             float x = 0.0f, y = 0.0f, w = 0.0f, h = 0.0f, score = 0.0f, prob = 1.0f;
    //             x = (sigmoid(resultData32[30 * index + coffs[j]]) + cx) * 32;
    //             y = (sigmoid(resultData32[30 * index + coffs[j] + 1]) + cy) * 32;
    //             w = exp(resultData32[30 * index + coffs[j] + 2]) * anchors[2 * j] * 32;
    //             h = exp(resultData32[30 * index + coffs[j] + 3]) * anchors[2 * j + 1] * 32;
    //             score = sigmoid(resultData32[30 * index + scoffs[j]]);
    //             // in this case, always 1 as there is only one possilbe class
    //             prob = exp(resultData32[30 * index + pcoffs[j]]);

    //             if (y > 140 && y < 300 && x > 140 && x < 300 && w > 70 && w < 300 && h > 70 && h < 300)
    //             {
    //                 Point pt1, pt2;
    //                 pt1.x = (int)round((x - w / 2));
    //                 pt1.y = (int)round((y - h / 2));
    //                 pt2.x = (int)round((x + w / 2));
    //                 pt2.y = (int)round((y + h / 2));
    //                 rectangle(img, pt1, pt2, Scalar(255, 0, 0), 2);
    //                 printf("\tBox[x:%f, y:%f, w: %f, h:%f] score: %f, class: %f \n", x, y, w, h, score, prob);
    //             }
    //             if (maxResult < score * prob)
    //             {
    //                 maxResult = score * prob;
    //                 maxIndex = index;
    //                 classProb = score;
    //             }
    //         }
    //     }
    // }
    // imshow("test", img);
    // cv::waitKey();
    imwrite("result.jpg", img);
    // printf("Index of top result is: %d\n", maxIndex);
    // printf("Probability of top result is: %f\n", maxResult);

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
    //normalize
    // normalize_image(imgfp32);
    floattofp16((unsigned char *)imgfp16, imgfp32, 3 * reqSize * reqSize);
    free(imgfp32);
    return imgfp16;
}
//normalize image
void normalize_image(std::vector<float> image, float **dst)
{
    // float maxRed = 0.0f, minRed = 255.0f, maxGreen = 0.0f, minGreen = 255.0f, maxBlue = 0.0f, minBlue = 255.0f;
    // for (int i = 0; i < 173056; i++)
    // {
    //     //red
    //     if (image[3 * i] > maxRed)
    //         maxRed = image[3 * i];
    //     if (image[3 * i] < minRed)
    //         minRed = image[3 * i];
    //     //green
    //     if (image[3 * i + 1] > maxGreen)
    //         maxGreen = image[3 * i + 1];
    //     if (image[3 * i + 1] < minGreen)
    //         minGreen = image[3 * i + 1];
    //     //blue
    //     if (image[3 * i + 2] > maxBlue)
    //         maxBlue = image[3 * i + 2];
    //     if (image[3 * i + 2] < minBlue)
    //         minBlue = image[3 * i + 2];
    // }
    // for (int i = 0; i < 173056; i++)
    // {
    //     image[3 * i] = (image[3 * i] - minRed) / (maxRed - minRed);
    //     image[3 * i + 1] = (image[3 * i + 1] - minGreen) / (maxGreen - minGreen);
    //     image[3 * i + 2] = (image[3 * i + 2] - minBlue) / (maxBlue - minBlue);
    // }
    float *toRet = (float *)malloc(((int)image.size()) * sizeof(float));
    for (int i = 0; i < sizeof(toRet) / 3; i++)
    {
        toRet[3 * i] = (image[3 * i] * 255.0f - 127.5) * 0.007843;
        toRet[3 * i + 1] = (image[3 * i + 1] * 255.0f - 127.5) * 0.007843;
        toRet[3 * i + 2] = (image[3 * i + 2] * 255.0f - 127.5) * 0.007843;
    }
    *dst = toRet;
}