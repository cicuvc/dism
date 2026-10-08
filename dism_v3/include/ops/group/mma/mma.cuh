/**
 * @file
 * @brief An aggregate header for all group-scope MMA operations.
 */

// Blackwell has its own tensor-scope MMA operations.
#if defined(KITTENS_FEATURE_UMMA)
#include "tensor/tensor.cuh"
#endif