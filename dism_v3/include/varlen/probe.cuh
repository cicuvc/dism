#pragma once
#include "summary/primitives.cuh"
#include "varlen/layout.cuh"
namespace DISM_VARIANT::dism_varlen {
template<int Rows> struct ProbeRecord {
    using Tile=KPermutationSharedBuffer<kt::bf16,Rows,64>;
    using Global=kt::gl<kt::bf16,-1,-1,-1,64,Tile>;
    Global input;
    kt::bf16* output;
    int heads;
};

}
