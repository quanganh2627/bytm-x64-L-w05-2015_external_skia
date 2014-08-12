/*
 * Copyright 2009 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkUtils_opts_SSE2_DEFINED
#define SkUtils_opts_SSE2_DEFINED

#include "SkTypes.h"

void sk_memcpy32_SSE2(uint32_t *dst, const uint32_t *src, int count);
void SkMemset16_x86(uint16_t *dst, uint16_t value, int count);
void SkMemset32_x86(uint32_t *dst, uint32_t value, int count);
void SkSetPixels16_x86(uint16_t *dst, uint16_t value, int count, int totalCount);
void SkSetPixels32_x86(uint32_t *dst, uint32_t value, int count, int totalCount);

#endif
