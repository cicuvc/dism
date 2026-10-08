/**
 * @file
 * @brief An aggregate header of group memory operations on tiles.
 */

// #include "shared_to_register.cuh"
// #include "global_to_register.cuh"
// #include "global_to_shared.cuh"
#if (defined(KITTENS_FEATURE_MULTIMEM))
#include "pgl.cuh"
#endif
#ifdef KITTENS_FEATURE_UMMA
#include "tensor_to_register.cuh"
#endif
