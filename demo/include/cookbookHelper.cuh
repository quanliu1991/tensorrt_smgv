#ifndef COOKBOOKHELPER_CUH
#define COOKBOOKHELPER_CUH

#include <NvInfer.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace nvinfer1;

#define DEBUG

#ifdef DEBUG
    #define WHERE_AM_I()                          \
        do                                        \
        {                                         \
            printf("%14p[%s]\n", this, __func__); \
        } while (0);
#else
    #define WHERE_AM_I()
#endif // ifdef DEBUG

#define CEIL_DIVIDE(X, Y) (((X) + (Y)-1) / (Y))

// get the string of a TensorRT data format
__inline__ std::string formatToString(TensorFormat format)
{
    switch (format)
    {
        case TensorFormat::kLINEAR:
            return std::string("LINE ");
        case TensorFormat::kCHW2:
            return std::string("CHW2 ");
        case TensorFormat::kHWC8:
            return std::string("HWC8 ");
        case TensorFormat::kCHW4:
            return std::string("CHW4 ");
        case TensorFormat::kCHW16:
            return std::string("CHW16");
        case TensorFormat::kCHW32:
            return std::string("CHW32");
        case TensorFormat::kHWC:
            return std::string("HWC  ");
        case TensorFormat::kDLA_LINEAR:
            return std::string("DLINE");
        case TensorFormat::kDLA_HWC4:
            return std::string("DHWC4");
        case TensorFormat::kHWC16:
            return std::string("HWC16");
        default: 
            return std::string("None ");
    }
}
// get the string of a TensorRT data type
__inline__ std::string dataTypeToString(DataType dataType)
{
    switch (dataType)
    {
    case DataType::kFLOAT:
        return std::string("FP32 ");
    case DataType::kHALF:
        return std::string("FP16 ");
    case DataType::kINT8:
        return std::string("INT8 ");
    case DataType::kINT32:
        return std::string("INT32");
    case DataType::kBOOL:
        return std::string("BOOL ");
    default:
        return std::string("Unknown");
    }
}


#endif