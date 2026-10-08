/**
 * @file
 * @brief An aggregate header file for all the device types defined by
 * ThunderKittens.
 */

#pragma once

#if defined(KITTENS_FEATURE_MULTIMEM)
#include "ipc.cuh"
#include "pgl.cuh"
#include "vmm.cuh"
#endif
