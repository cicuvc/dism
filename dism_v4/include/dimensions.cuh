#pragma once

#ifndef DISM_KC_KEY_DIM
#define DISM_KC_KEY_DIM 64
#endif
#ifndef DISM_KC_HEAD_DIM
#define DISM_KC_HEAD_DIM 64
#endif
#ifndef DISM_READOUT_DIM
#define DISM_READOUT_DIM 32
#endif
static_assert(DISM_KC_KEY_DIM==32 || DISM_KC_KEY_DIM==64 || DISM_KC_KEY_DIM==128);
static_assert(DISM_KC_HEAD_DIM==32 || DISM_KC_HEAD_DIM==64 || DISM_KC_HEAD_DIM==128);
static_assert(DISM_READOUT_DIM==16 || DISM_READOUT_DIM==32);
