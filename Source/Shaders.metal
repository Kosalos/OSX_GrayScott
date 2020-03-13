#include <metal_stdlib>
#import "Shader.h"
using namespace metal;

kernel void drawShader
(
 constant Control &ctrl  [[ buffer(0) ]],
 device Cell *cells      [[ buffer(1) ]],
 constant float3 *cTable [[ buffer(2) ]],   // color lookup table[256]
 texture2d<float, access::write> dest [[texture(0)]],
 uint2 pt [[thread_position_in_grid]]
 ) {
    int i = pt.x + pt.y * ctrl.cxSize;
    if(i >= ctrl.numCells) return;
    
    int tx = pt.x * ctrl.zoom;
    int ty = pt.y * ctrl.zoom;
    if(tx >= ctrl.txSize - ctrl.zoom) return;
    if(ty >= ctrl.tySize - ctrl.zoom) return;
    
    int colorIndex = int(cells[i].value * 255);
    float4 color = float4(cTable[colorIndex],1);
    
    uint2 pixel;
    for(int x = 0; x < ctrl.zoom; ++x) {
        pixel.x = tx + x;
        for(int y = 0; y < ctrl.zoom; ++y) {
            pixel.y = ty + y;
            dest.write(color,pixel);
        }
    }
}

//MARK: -

kernel void evolveShader
(
 constant Control &ctrl [[ buffer(0) ]],
 device Cell *A         [[ buffer(1) ]],
 device Cell *B         [[ buffer(2) ]],
 device Cell *newA      [[ buffer(3) ]],
 device Cell *newB      [[ buffer(4) ]],
 uint2 pt [[thread_position_in_grid]]
 ) {
    int x = int(pt.x);
    int y = int(pt.y);
    int w = ctrl.cxSize;
    int h = ctrl.cySize;
    if(x >= w || y >= h) return;

    int i = x + y * w;
    
    // compute u and v coordinates of pixel in range [0, 1]
    const float u = (float)x / float(w - 1);
    const float v = 1 - (float)y / float(h - 1);

    // compute kill and feed rates
    const float k = ctrl.killRate + (u - 0.5) * 0;
    const float f = ctrl.feedRate + (v - 0.5) * 0;
    
    // find neighboring pixels, wrapping around edges
    const int xp = x == 0 ? w - 1 : x - 1;
    const int xn = x == w - 1 ? 0 : x + 1;
    const int yp = y == 0 ? h - 1 : y - 1;
    const int yn = y == h - 1 ? 0 : y + 1;
    
    // get the values for A and B at this pixel
    const float a = A[i].value;
    const float b = B[i].value;

    #define centerWeight -1.0
    #define adjacentWeight 0.2
    #define diagonalWeight 0.05

    // compute A diffusion
    float dda = 0;
    dda += a * centerWeight;
    dda += A[yp * w + xp].value * diagonalWeight;
    dda += A[yp * w + xn].value * diagonalWeight;
    dda += A[yn * w + xp].value * diagonalWeight;
    dda += A[yn * w + xn].value * diagonalWeight;
    dda += A[yp * w + x].value * adjacentWeight;
    dda += A[yn * w + x].value * adjacentWeight;
    dda += A[y * w + xp].value * adjacentWeight;
    dda += A[y * w + xn].value * adjacentWeight;
    
    // compute B diffusion
    float ddb = 0;
    ddb += b * centerWeight;
    ddb += B[yp * w + xp].value * diagonalWeight;
    ddb += B[yp * w + xn].value * diagonalWeight;
    ddb += B[yn * w + xp].value * diagonalWeight;
    ddb += B[yn * w + xn].value * diagonalWeight;
    ddb += B[yp * w + x].value * adjacentWeight;
    ddb += B[yn * w + x].value * adjacentWeight;
    ddb += B[y * w + xp].value * adjacentWeight;
    ddb += B[y * w + xn].value * adjacentWeight;
    
    // apply reaction diffusion formula
    const float da = ctrl.diffusionRateA * dda - a * b * b + f * (1 - a);
    const float db = ctrl.diffusionRateB * ddb + a * b * b - (f + k) * b;
    
    // write new A and B values
    newA[i].value = a + da * ctrl.scale;
    newB[i].value = b + db * ctrl.scale;
}
   
