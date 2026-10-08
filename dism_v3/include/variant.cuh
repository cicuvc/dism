#pragma once

// Each translation unit selects one explicit configuration. The namespace also
// isolates legacy non-template aliases/helpers across configurations (ODR).
template<int Readout, int Key, int Value> struct KernelConfig {
    static_assert(Readout == 16 || Readout == 32);
    static_assert(Key == 32 || Key == 64 || Key == 128);
    static_assert(Value == 32 || Value == 64 || Value == 128);
    static constexpr int R = Readout, D = Key, DV = Value;
};
#ifndef DISM_READOUT_DIM
#define DISM_READOUT_DIM 32
#endif
#ifndef DISM_KC_KEY_DIM
#define DISM_KC_KEY_DIM 64
#endif
#ifndef DISM_KC_HEAD_DIM
#define DISM_KC_HEAD_DIM 64
#endif
#ifndef DISM_VARIANT
#define DISM_VARIANT dism_r32_d64_v64
#endif
#ifndef DISM_ENABLE_FP32
#define DISM_ENABLE_FP32 0
#endif
namespace DISM_VARIANT {
using ActiveConfig = KernelConfig<DISM_READOUT_DIM, DISM_KC_KEY_DIM, DISM_KC_HEAD_DIM>;
}
