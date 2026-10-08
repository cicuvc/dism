// A modded Thunderkittens

#pragma once

#ifdef KITTENS_HOPPER
#define KITTENS_FEATURE_TMA
#define KITTENS_FEATURE_FP8
#define KITTENS_FEATURE_WGMMA
#define KITTENS_FEATURE_MULTIMEM
#define KITTENS_FEATURE_REG_INCDEC
#define KITTENS_FEATURE_STMATRIX
#define KITTENS_FEATURE_MBARRIER
#endif

#ifdef KITTENS_BLACKWELL
#define KITTENS_FEATURE_TMA
#define KITTENS_FEATURE_FP8
#define KITTENS_FEATURE_FP4
#define KITTENS_FEATURE_UMMA
#define KITTENS_FEATURE_UE8
#define KITTENS_FEATURE_MULTIMEM
#define KITTENS_FEATURE_REG_INCDEC
#define KITTENS_FEATURE_STMATRIX
#define KITTENS_FEATURE_MBARRIER
#define KITTENS_FEATURE_X2FP
#define KITTENS_FEATURE_SELECTIVE_MBAR_TRIGGER
#endif

#ifdef KITTENS_RTX_BLACKWELL
#define KITTENS_FEATURE_TMA
#define KITTENS_FEATURE_FP8
#define KITTENS_FEATURE_REG_INCDEC
#define KITTENS_FEATURE_STMATRIX
#define KITTENS_FEATURE_MBARRIER
#endif

#include "common/common.cuh"
#include "ops/ops.cuh"
#include "pyutils/util.cuh"
#include "types/types.cuh"
// #include "pyutils/pyutils.cuh" // for simple binding without including torch