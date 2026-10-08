#pragma once
#include "varlen/operands.cuh"
namespace DISM_VARIANT::dism_varlen {
using ChunkGlobal=kt::gl<float,-1,-1,-1,-1>;
struct ChunkArgs {
    ChunkGlobal a,b;
    float* output;
    int heads;
};

}
