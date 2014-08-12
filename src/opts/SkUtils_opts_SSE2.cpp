/*
 * Copyright 2009 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <emmintrin.h>
#include "SkUtils_opts_SSE2.h"
#include "cutils/memory.h"
#include "SkCacheConfig_x86.h"


static void SkNonTemporalMemset16_SSE2(uint16_t *dst, uint16_t value, int count)
{
    while (((size_t)dst) & 0x0F) {
        *dst++ = value;
        --count;
    }

    __m128i *d = reinterpret_cast<__m128i*>(dst);
    __m128i value_wide = _mm_set1_epi16(value);
    count -= 31;
    do {
        _mm_stream_si128(d    , value_wide);
        _mm_stream_si128(d + 1, value_wide);
        _mm_stream_si128(d + 2, value_wide);
        _mm_stream_si128(d + 3, value_wide);
        d += 4;
        count -= 32;
    } while (count > 0);

    count += 24;
    while (count > 0) {
        _mm_stream_si128(d, value_wide);
        d++;
        count -= 8;
    }

    dst = reinterpret_cast<uint16_t*>(d);
    count += 7;
    while (count > 0) {
        *dst++ = value;
        --count;
    }
}

static void SkNonTemporalMemset32_SSE2(uint32_t *dst, uint32_t value, int count)
{
    while (((size_t)dst) & 0x0F) {
        *dst++ = value;
        --count;
    }

    __m128i *d = reinterpret_cast<__m128i*>(dst);
    __m128i value_wide = _mm_set1_epi32(value);
    count -= 15;
    do {
        _mm_stream_si128(d    , value_wide);
        _mm_stream_si128(d + 1, value_wide);
        _mm_stream_si128(d + 2, value_wide);
        _mm_stream_si128(d + 3, value_wide);
        d += 4;
        count -= 16;
    } while (count > 0);

    count += 12;
    while (count > 0) {
        _mm_stream_si128(d, value_wide);
        d++;
        count -= 4;
    }

    dst = reinterpret_cast<uint32_t*>(d);
    count += 3;
    while (count > 0) {
        *dst++ = value;
        --count;
    }
}

void SkMemset16_x86(uint16_t *dst, uint16_t value, int count)
{
    SkASSERT(dst != NULL && count >= 0);

    // dst must be 2-byte aligned.
    SkASSERT((((size_t) dst) & 0x01) == 0);

    /* Check the size of the operation. If it's more than half the L2 cache
     * we use a local function with non temporal stores, otherwise we use
     * the assembly optimized version in libcutils.
     */
    if (count > (SHARED_CACHE_SIZE_HALF >> 1))
    {
        SkNonTemporalMemset16_SSE2(dst, value, count);
    }
    else
    {
        android_memset16(dst, value, count << 1);
    }
}

void SkMemset32_x86(uint32_t *dst, uint32_t value, int count)
{
    SkASSERT(dst != NULL && count >= 0);

    // dst must be 4-byte aligned.
    SkASSERT((((size_t) dst) & 0x03) == 0);

    /* Check the size of the operation. If it's more than half the L2 cache
     * we use a local function with non temporal stores, otherwise we use
     * the assembly optimized version in libcutils.
     */
    if (count > (SHARED_CACHE_SIZE_HALF >> 2))
    {
        SkNonTemporalMemset32_SSE2(dst, value, count);
    }
    else
    {
        android_memset32(dst, value, count << 2);
    }
}

void SkSetPixels16_x86(uint16_t *dst, uint16_t value, int count, int totalCount)
{
    SkASSERT(dst != NULL && count >= 0);

    // dst must be 2-byte aligned.
    SkASSERT((((size_t) dst) & 0x01) == 0);

    /* Check the total size of the operation. If it's more than half the
     * L2 cache we use a local function with non temporal stores, otherwise
     * we use the assembly optimized version in libcutils.
     */
    if ((totalCount > (SHARED_CACHE_SIZE_HALF >> 1)) && (count >= 39))
    {
        SkNonTemporalMemset16_SSE2(dst, value, count);
    }
    else
    {
        android_memset16(dst, value, count << 1);
    }
}

void SkSetPixels32_x86(uint32_t *dst, uint32_t value, int count, int totalCount)
{
    SkASSERT(dst != NULL && count >= 0);

    // dst must be 4-byte aligned.
    SkASSERT((((size_t) dst) & 0x03) == 0);

    /* Check the total size of the operation. If it's more than half the
     * L2 cache we use a local function with non temporal stores, otherwise
     * we use the assembly optimized version in libcutils.
     */
    if ((totalCount > (SHARED_CACHE_SIZE_HALF >> 2)) && (count >= 19))
    {
        SkNonTemporalMemset32_SSE2(dst, value, count);
    }
    else
    {
        android_memset32(dst, value, count << 2);
    }
}

void sk_memcpy32_SSE2(uint32_t *dst, const uint32_t *src, int count)
{
    if (count >= 16) {
        while (((size_t)dst) & 0x0F) {
            *dst++ = *src++;
            --count;
        }
        __m128i *dst128 = reinterpret_cast<__m128i*>(dst);
        const __m128i *src128 = reinterpret_cast<const __m128i*>(src);
        while (count >= 16) {
            __m128i a =  _mm_loadu_si128(src128++);
            __m128i b =  _mm_loadu_si128(src128++);
            __m128i c =  _mm_loadu_si128(src128++);
            __m128i d =  _mm_loadu_si128(src128++);

            _mm_store_si128(dst128++, a);
            _mm_store_si128(dst128++, b);
            _mm_store_si128(dst128++, c);
            _mm_store_si128(dst128++, d);
            count -= 16;
        }
        dst = reinterpret_cast<uint32_t*>(dst128);
        src = reinterpret_cast<const uint32_t*>(src128);
    }
    while (count > 0) {
        *dst++ = *src++;
        --count;
    }
}
