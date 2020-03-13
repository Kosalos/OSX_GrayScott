#pragma once

#ifdef __METAL_VERSION__
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
#define NSInteger metal::int32_t
#else
#import <Foundation/Foundation.h>
#endif

#include <simd/simd.h>

typedef struct {
    float value;
    vector_float4 color;
} Cell;

typedef struct {
    int numCells,zoom;
    int cxSize,cySize;  // cell grid size
    int txSize,tySize;  // texture grid size
    
    float feedRate;
    float killRate;
    float diffusionRateA;
    float diffusionRateB;
    float scale;    
} Control;
