
/*
 * Copyright 2009 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */


#include <emmintrin.h>
#include "SkBitmapProcState_opts_SSE2.h"
#include "SkPaint.h"
#include "SkUtils.h"
#include "SkBitmapProcState_filter.h"

__attribute__((aligned(16)))
uint32_t vf[4] = {0xF,0xf,0xf,0xf};
uint32_t vmask2[4] = {0xff00ff00,0xff00ff00,0xff00ff00,0xff00ff00};
unsigned short v256[8] = {256,256,256,256,256,256,256,256};
uint32_t vmask[4] = {0x00ff00ff,0x00ff00ff,0x00ff00ff,0x00ff00ff};

void S32_opaque_D32_filter_DX_SSE2(const SkBitmapProcState& s,
                                   const uint32_t* xy,
                                   int count, uint32_t* colors) {
    SkASSERT(count > 0 && colors != NULL);
    SkASSERT(s.fFilterLevel != SkPaint::kNone_FilterLevel);
    SkASSERT(s.fBitmap->config() == SkBitmap::kARGB_8888_Config);
    SkASSERT(s.fAlphaScale == 256);

    const char* srcAddr = static_cast<const char*>(s.fBitmap->getPixels());
    size_t rb = s.fBitmap->rowBytes();
    uint32_t XY = *xy++;
    unsigned y0 = XY >> 14;
    const uint32_t* row0 = reinterpret_cast<const uint32_t*>(srcAddr + (y0 >> 4) * rb);
    const uint32_t* row1 = reinterpret_cast<const uint32_t*>(srcAddr + (XY & 0x3FFF) * rb);
    unsigned subY = y0 & 0xF;

    // ( 0,  0,  0,  0,  0,  0,  0, 16)
    __m128i sixteen = _mm_cvtsi32_si128(16);

    // ( 0,  0,  0,  0, 16, 16, 16, 16)
    sixteen = _mm_shufflelo_epi16(sixteen, 0);

    // ( 0,  0,  0,  0,  0,  0,  0,  y)
    __m128i allY = _mm_cvtsi32_si128(subY);

    // ( 0,  0,  0,  0,  y,  y,  y,  y)
    allY = _mm_shufflelo_epi16(allY, 0);

    // ( 0,  0,  0,  0, 16-y, 16-y, 16-y, 16-y)
    __m128i negY = _mm_sub_epi16(sixteen, allY);

    // (16-y, 16-y, 16-y, 16-y, y, y, y, y)
    allY = _mm_unpacklo_epi64(allY, negY);

    // (16, 16, 16, 16, 16, 16, 16, 16 )
    sixteen = _mm_shuffle_epi32(sixteen, 0);

    // ( 0,  0,  0,  0,  0,  0,  0,  0)
    __m128i zero = _mm_setzero_si128();
    do {
        uint32_t XX = *xy++;    // x0:14 | 4 | x1:14
        unsigned x0 = XX >> 18;
        unsigned x1 = XX & 0x3FFF;

        // (0, 0, 0, 0, 0, 0, 0, x)
        __m128i allX = _mm_cvtsi32_si128((XX >> 14) & 0x0F);

        // (0, 0, 0, 0, x, x, x, x)
        allX = _mm_shufflelo_epi16(allX, 0);

        // (x, x, x, x, x, x, x, x)
        allX = _mm_shuffle_epi32(allX, 0);

        // (16-x, 16-x, 16-x, 16-x, 16-x, 16-x, 16-x)
        __m128i negX = _mm_sub_epi16(sixteen, allX);

        // Load 4 samples (pixels).
        __m128i a00 = _mm_cvtsi32_si128(row0[x0]);
        __m128i a01 = _mm_cvtsi32_si128(row0[x1]);
        __m128i a10 = _mm_cvtsi32_si128(row1[x0]);
        __m128i a11 = _mm_cvtsi32_si128(row1[x1]);

        // (0, 0, a00, a10)
        __m128i a00a10 = _mm_unpacklo_epi32(a10, a00);

        // Expand to 16 bits per component.
        a00a10 = _mm_unpacklo_epi8(a00a10, zero);

        // ((a00 * (16-y)), (a10 * y)).
        a00a10 = _mm_mullo_epi16(a00a10, allY);

        // (a00 * (16-y) * (16-x), a10 * y * (16-x)).
        a00a10 = _mm_mullo_epi16(a00a10, negX);

        // (0, 0, a01, a10)
        __m128i a01a11 = _mm_unpacklo_epi32(a11, a01);

        // Expand to 16 bits per component.
        a01a11 = _mm_unpacklo_epi8(a01a11, zero);

        // (a01 * (16-y)), (a11 * y)
        a01a11 = _mm_mullo_epi16(a01a11, allY);

        // (a01 * (16-y) * x), (a11 * y * x)
        a01a11 = _mm_mullo_epi16(a01a11, allX);

        // (a00*w00 + a01*w01, a10*w10 + a11*w11)
        __m128i sum = _mm_add_epi16(a00a10, a01a11);

        // (DC, a00*w00 + a01*w01)
        __m128i shifted = _mm_shuffle_epi32(sum, 0xEE);

        // (DC, a00*w00 + a01*w01 + a10*w10 + a11*w11)
        sum = _mm_add_epi16(sum, shifted);

        // Divide each 16 bit component by 256.
        sum = _mm_srli_epi16(sum, 8);

        // Pack lower 4 16 bit values of sum into lower 4 bytes.
        sum = _mm_packus_epi16(sum, zero);

        // Extract low int and store.
        *colors++ = _mm_cvtsi128_si32(sum);
    } while (--count > 0);
}

void S32_alpha_D32_filter_DX_SSE2(const SkBitmapProcState& s,
                                  const uint32_t* xy,
                                  int count, uint32_t* colors) {
    SkASSERT(count > 0 && colors != NULL);
    SkASSERT(s.fFilterLevel != SkPaint::kNone_FilterLevel);
    SkASSERT(s.fBitmap->config() == SkBitmap::kARGB_8888_Config);
    SkASSERT(s.fAlphaScale < 256);

    const char* srcAddr = static_cast<const char*>(s.fBitmap->getPixels());
    size_t rb = s.fBitmap->rowBytes();
    uint32_t XY = *xy++;
    unsigned y0 = XY >> 14;
    const uint32_t* row0 = reinterpret_cast<const uint32_t*>(srcAddr + (y0 >> 4) * rb);
    const uint32_t* row1 = reinterpret_cast<const uint32_t*>(srcAddr + (XY & 0x3FFF) * rb);
    unsigned subY = y0 & 0xF;

    // ( 0,  0,  0,  0,  0,  0,  0, 16)
    __m128i sixteen = _mm_cvtsi32_si128(16);

    // ( 0,  0,  0,  0, 16, 16, 16, 16)
    sixteen = _mm_shufflelo_epi16(sixteen, 0);

    // ( 0,  0,  0,  0,  0,  0,  0,  y)
    __m128i allY = _mm_cvtsi32_si128(subY);

    // ( 0,  0,  0,  0,  y,  y,  y,  y)
    allY = _mm_shufflelo_epi16(allY, 0);

    // ( 0,  0,  0,  0, 16-y, 16-y, 16-y, 16-y)
    __m128i negY = _mm_sub_epi16(sixteen, allY);

    // (16-y, 16-y, 16-y, 16-y, y, y, y, y)
    allY = _mm_unpacklo_epi64(allY, negY);

    // (16, 16, 16, 16, 16, 16, 16, 16 )
    sixteen = _mm_shuffle_epi32(sixteen, 0);

    // ( 0,  0,  0,  0,  0,  0,  0,  0)
    __m128i zero = _mm_setzero_si128();

    // ( alpha, alpha, alpha, alpha, alpha, alpha, alpha, alpha )
    __m128i alpha = _mm_set1_epi16(s.fAlphaScale);

    do {
        uint32_t XX = *xy++;    // x0:14 | 4 | x1:14
        unsigned x0 = XX >> 18;
        unsigned x1 = XX & 0x3FFF;

        // (0, 0, 0, 0, 0, 0, 0, x)
        __m128i allX = _mm_cvtsi32_si128((XX >> 14) & 0x0F);

        // (0, 0, 0, 0, x, x, x, x)
        allX = _mm_shufflelo_epi16(allX, 0);

        // (x, x, x, x, x, x, x, x)
        allX = _mm_shuffle_epi32(allX, 0);

        // (16-x, 16-x, 16-x, 16-x, 16-x, 16-x, 16-x)
        __m128i negX = _mm_sub_epi16(sixteen, allX);

        // Load 4 samples (pixels).
        __m128i a00 = _mm_cvtsi32_si128(row0[x0]);
        __m128i a01 = _mm_cvtsi32_si128(row0[x1]);
        __m128i a10 = _mm_cvtsi32_si128(row1[x0]);
        __m128i a11 = _mm_cvtsi32_si128(row1[x1]);

        // (0, 0, a00, a10)
        __m128i a00a10 = _mm_unpacklo_epi32(a10, a00);

        // Expand to 16 bits per component.
        a00a10 = _mm_unpacklo_epi8(a00a10, zero);

        // ((a00 * (16-y)), (a10 * y)).
        a00a10 = _mm_mullo_epi16(a00a10, allY);

        // (a00 * (16-y) * (16-x), a10 * y * (16-x)).
        a00a10 = _mm_mullo_epi16(a00a10, negX);

        // (0, 0, a01, a10)
        __m128i a01a11 = _mm_unpacklo_epi32(a11, a01);

        // Expand to 16 bits per component.
        a01a11 = _mm_unpacklo_epi8(a01a11, zero);

        // (a01 * (16-y)), (a11 * y)
        a01a11 = _mm_mullo_epi16(a01a11, allY);

        // (a01 * (16-y) * x), (a11 * y * x)
        a01a11 = _mm_mullo_epi16(a01a11, allX);

        // (a00*w00 + a01*w01, a10*w10 + a11*w11)
        __m128i sum = _mm_add_epi16(a00a10, a01a11);

        // (DC, a00*w00 + a01*w01)
        __m128i shifted = _mm_shuffle_epi32(sum, 0xEE);

        // (DC, a00*w00 + a01*w01 + a10*w10 + a11*w11)
        sum = _mm_add_epi16(sum, shifted);

        // Divide each 16 bit component by 256.
        sum = _mm_srli_epi16(sum, 8);

        // Multiply by alpha.
        sum = _mm_mullo_epi16(sum, alpha);

        // Divide each 16 bit component by 256.
        sum = _mm_srli_epi16(sum, 8);

        // Pack lower 4 16 bit values of sum into lower 4 bytes.
        sum = _mm_packus_epi16(sum, zero);

        // Extract low int and store.
        *colors++ = _mm_cvtsi128_si32(sum);
    } while (--count > 0);
}

static inline uint32_t ClampX_ClampY_pack_filter(SkFixed f, unsigned max,
                                                 SkFixed one) {
    unsigned i = SkClampMax(f >> 16, max);
    i = (i << 4) | ((f >> 12) & 0xF);
    return (i << 14) | SkClampMax((f + one) >> 16, max);
}

/*  SSE version of ClampX_ClampY_filter_scale()
 *  portable version is in core/SkBitmapProcState_matrix.h
 */
void ClampX_ClampY_filter_scale_SSE2(const SkBitmapProcState& s, uint32_t xy[],
                                     int count, int x, int y) {
    SkASSERT((s.fInvType & ~(SkMatrix::kTranslate_Mask |
                             SkMatrix::kScale_Mask)) == 0);
    SkASSERT(s.fInvKy == 0);

    const unsigned maxX = s.fBitmap->width() - 1;
    const SkFixed one = s.fFilterOneX;
    const SkFixed dx = s.fInvSx;
    SkFixed fx;

    SkPoint pt;
    s.fInvProc(s.fInvMatrix, SkIntToScalar(x) + SK_ScalarHalf,
                             SkIntToScalar(y) + SK_ScalarHalf, &pt);
    const SkFixed fy = SkScalarToFixed(pt.fY) - (s.fFilterOneY >> 1);
    const unsigned maxY = s.fBitmap->height() - 1;
    // compute our two Y values up front
    *xy++ = ClampX_ClampY_pack_filter(fy, maxY, s.fFilterOneY);
    // now initialize fx
    fx = SkScalarToFixed(pt.fX) - (one >> 1);

    // test if we don't need to apply the tile proc
    if (dx > 0 && (unsigned)(fx >> 16) <= maxX &&
        (unsigned)((fx + dx * (count - 1)) >> 16) < maxX) {
        if (count >= 4) {
            // SSE version of decal_filter_scale
            while ((size_t(xy) & 0x0F) != 0) {
                SkASSERT((fx >> (16 + 14)) == 0);
                *xy++ = (fx >> 12 << 14) | ((fx >> 16) + 1);
                fx += dx;
                count--;
            }

            __m128i wide_1    = _mm_set1_epi32(1);
            __m128i wide_dx4  = _mm_set1_epi32(dx * 4);
            __m128i wide_fx   = _mm_set_epi32(fx + dx * 3, fx + dx * 2,
                                              fx + dx, fx);

            while (count >= 4) {
                __m128i wide_out;

                wide_out = _mm_slli_epi32(_mm_srai_epi32(wide_fx, 12), 14);
                wide_out = _mm_or_si128(wide_out, _mm_add_epi32(
                                        _mm_srai_epi32(wide_fx, 16), wide_1));

                _mm_store_si128(reinterpret_cast<__m128i*>(xy), wide_out);

                xy += 4;
                fx += dx * 4;
                wide_fx  = _mm_add_epi32(wide_fx, wide_dx4);
                count -= 4;
            } // while count >= 4
        } // if count >= 4

        while (count-- > 0) {
            SkASSERT((fx >> (16 + 14)) == 0);
            *xy++ = (fx >> 12 << 14) | ((fx >> 16) + 1);
            fx += dx;
        }
    } else {
        // SSE2 only support 16bit interger max & min, so only process the case
        // maxX less than the max 16bit interger. Actually maxX is the bitmap's
        // height, there should be rare bitmap whose height will be greater
        // than max 16bit interger in the real world.
        if ((count >= 4) && (maxX <= 0xFFFF)) {
            while (((size_t)xy & 0x0F) != 0) {
                *xy++ = ClampX_ClampY_pack_filter(fx, maxX, one);
                fx += dx;
                count--;
            }

            __m128i wide_fx   = _mm_set_epi32(fx + dx * 3, fx + dx * 2,
                                              fx + dx, fx);
            __m128i wide_dx4  = _mm_set1_epi32(dx * 4);
            __m128i wide_one  = _mm_set1_epi32(one);
            __m128i wide_maxX = _mm_set1_epi32(maxX);
            __m128i wide_mask = _mm_set1_epi32(0xF);

             while (count >= 4) {
                __m128i wide_i;
                __m128i wide_lo;
                __m128i wide_fx1;

                // i = SkClampMax(f>>16,maxX)
                wide_i = _mm_max_epi16(_mm_srli_epi32(wide_fx, 16),
                                       _mm_setzero_si128());
                wide_i = _mm_min_epi16(wide_i, wide_maxX);

                // i<<4 | TILEX_LOW_BITS(fx)
                wide_lo = _mm_srli_epi32(wide_fx, 12);
                wide_lo = _mm_and_si128(wide_lo, wide_mask);
                wide_i  = _mm_slli_epi32(wide_i, 4);
                wide_i  = _mm_or_si128(wide_i, wide_lo);

                // i<<14
                wide_i = _mm_slli_epi32(wide_i, 14);

                // SkClampMax(((f+one))>>16,max)
                wide_fx1 = _mm_add_epi32(wide_fx, wide_one);
                wide_fx1 = _mm_max_epi16(_mm_srli_epi32(wide_fx1, 16),
                                                        _mm_setzero_si128());
                wide_fx1 = _mm_min_epi16(wide_fx1, wide_maxX);

                // final combination
                wide_i = _mm_or_si128(wide_i, wide_fx1);
                _mm_store_si128(reinterpret_cast<__m128i*>(xy), wide_i);

                wide_fx = _mm_add_epi32(wide_fx, wide_dx4);
                fx += dx * 4;
                xy += 4;
                count -= 4;
            } // while count >= 4
        } // if count >= 4

        while (count-- > 0) {
            *xy++ = ClampX_ClampY_pack_filter(fx, maxX, one);
            fx += dx;
        }
    }
}

/*  SSE version of ClampX_ClampY_nofilter_scale()
 *  portable version is in core/SkBitmapProcState_matrix.h
 */
void ClampX_ClampY_nofilter_scale_SSE2(const SkBitmapProcState& s,
                                    uint32_t xy[], int count, int x, int y) {
    SkASSERT((s.fInvType & ~(SkMatrix::kTranslate_Mask |
                             SkMatrix::kScale_Mask)) == 0);

    // we store y, x, x, x, x, x
    const unsigned maxX = s.fBitmap->width() - 1;
    SkFixed fx;
    SkPoint pt;
    s.fInvProc(s.fInvMatrix, SkIntToScalar(x) + SK_ScalarHalf,
                             SkIntToScalar(y) + SK_ScalarHalf, &pt);
    fx = SkScalarToFixed(pt.fY);
    const unsigned maxY = s.fBitmap->height() - 1;
    *xy++ = SkClampMax(fx >> 16, maxY);
    fx = SkScalarToFixed(pt.fX);

    if (0 == maxX) {
        // all of the following X values must be 0
        memset(xy, 0, count * sizeof(uint16_t));
        return;
    }

    const SkFixed dx = s.fInvSx;

    // test if we don't need to apply the tile proc
    if ((unsigned)(fx >> 16) <= maxX &&
        (unsigned)((fx + dx * (count - 1)) >> 16) <= maxX) {
        // SSE version of decal_nofilter_scale
        if (count >= 8) {
            while (((size_t)xy & 0x0F) != 0) {
                *xy++ = pack_two_shorts(fx >> 16, (fx + dx) >> 16);
                fx += 2 * dx;
                count -= 2;
            }

            __m128i wide_dx4 = _mm_set1_epi32(dx * 4);
            __m128i wide_dx8 = _mm_add_epi32(wide_dx4, wide_dx4);

            __m128i wide_low = _mm_set_epi32(fx + dx * 3, fx + dx * 2,
                                             fx + dx, fx);
            __m128i wide_high = _mm_add_epi32(wide_low, wide_dx4);

            while (count >= 8) {
                __m128i wide_out_low = _mm_srli_epi32(wide_low, 16);
                __m128i wide_out_high = _mm_srli_epi32(wide_high, 16);

                __m128i wide_result = _mm_packs_epi32(wide_out_low,
                                                      wide_out_high);
                _mm_store_si128(reinterpret_cast<__m128i*>(xy), wide_result);

                wide_low = _mm_add_epi32(wide_low, wide_dx8);
                wide_high = _mm_add_epi32(wide_high, wide_dx8);

                xy += 4;
                fx += dx * 8;
                count -= 8;
            }
        } // if count >= 8

        uint16_t* xx = reinterpret_cast<uint16_t*>(xy);
        while (count-- > 0) {
            *xx++ = SkToU16(fx >> 16);
            fx += dx;
        }
    } else {
        // SSE2 only support 16bit interger max & min, so only process the case
        // maxX less than the max 16bit interger. Actually maxX is the bitmap's
        // height, there should be rare bitmap whose height will be greater
        // than max 16bit interger in the real world.
        if ((count >= 8) && (maxX <= 0xFFFF)) {
            while (((size_t)xy & 0x0F) != 0) {
                *xy++ = pack_two_shorts(SkClampMax((fx + dx) >> 16, maxX),
                                        SkClampMax(fx >> 16, maxX));
                fx += 2 * dx;
                count -= 2;
            }

            __m128i wide_dx4 = _mm_set1_epi32(dx * 4);
            __m128i wide_dx8 = _mm_add_epi32(wide_dx4, wide_dx4);

            __m128i wide_low = _mm_set_epi32(fx + dx * 3, fx + dx * 2,
                                             fx + dx, fx);
            __m128i wide_high = _mm_add_epi32(wide_low, wide_dx4);
            __m128i wide_maxX = _mm_set1_epi32(maxX);

            while (count >= 8) {
                __m128i wide_out_low = _mm_srli_epi32(wide_low, 16);
                __m128i wide_out_high = _mm_srli_epi32(wide_high, 16);

                wide_out_low  = _mm_max_epi16(wide_out_low,
                                              _mm_setzero_si128());
                wide_out_low  = _mm_min_epi16(wide_out_low, wide_maxX);
                wide_out_high = _mm_max_epi16(wide_out_high,
                                              _mm_setzero_si128());
                wide_out_high = _mm_min_epi16(wide_out_high, wide_maxX);

                __m128i wide_result = _mm_packs_epi32(wide_out_low,
                                                      wide_out_high);
                _mm_store_si128(reinterpret_cast<__m128i*>(xy), wide_result);

                wide_low  = _mm_add_epi32(wide_low, wide_dx8);
                wide_high = _mm_add_epi32(wide_high, wide_dx8);

                xy += 4;
                fx += dx * 8;
                count -= 8;
            }
        } // if count >= 8

        uint16_t* xx = reinterpret_cast<uint16_t*>(xy);
        while (count-- > 0) {
            *xx++ = SkClampMax(fx >> 16, maxX);
            fx += dx;
        }
    }
}

/*  SSE version of ClampX_ClampY_filter_affine()
 *  portable version is in core/SkBitmapProcState_matrix.h
 *  the address of xy should be 16bytes aligned, otherwise it will
 *  core dump because of _mm_store_si128()
 */
void ClampX_ClampY_filter_affine_SSE2(const SkBitmapProcState& s,
                                      uint32_t xy[], int count, int x, int y) {

    SkASSERT(((size_t)xy & 0x0F) == 0);
    SkPoint srcPt;
    s.fInvProc(s.fInvMatrix,
               SkIntToScalar(x) + SK_ScalarHalf,
               SkIntToScalar(y) + SK_ScalarHalf, &srcPt);

    SkFixed oneX = s.fFilterOneX;
    SkFixed oneY = s.fFilterOneY;
    SkFixed fx = SkScalarToFixed(srcPt.fX) - (oneX >> 1);
    SkFixed fy = SkScalarToFixed(srcPt.fY) - (oneY >> 1);
    SkFixed dx = s.fInvSx;
    SkFixed dy = s.fInvKy;
    unsigned maxX = s.fBitmap->width() - 1;
    unsigned maxY = s.fBitmap->height() - 1;

    if (count >= 2 && (maxX <= 0xFFFF)) {
        SkFixed dx2 = dx + dx;
        SkFixed dy2 = dy + dy;

        __m128i wide_f = _mm_set_epi32(fx + dx, fy + dy, fx, fy);
        __m128i wide_d2  = _mm_set_epi32(dx2, dy2, dx2, dy2);
        __m128i wide_one  = _mm_set_epi32(oneX, oneY, oneX, oneY);
        __m128i wide_max = _mm_set_epi32(maxX, maxY, maxX, maxY);
        __m128i wide_mask = _mm_set1_epi32(0xF);

        while (count >= 2) {
            // i = SkClampMax(f>>16,maxX)
            __m128i wide_i = _mm_max_epi16(_mm_srli_epi32(wide_f, 16),
                                           _mm_setzero_si128());
            wide_i = _mm_min_epi16(wide_i, wide_max);

            // i<<4 | TILEX_LOW_BITS(f)
            __m128i wide_lo = _mm_srli_epi32(wide_f, 12);
            wide_lo = _mm_and_si128(wide_lo, wide_mask);
            wide_i  = _mm_slli_epi32(wide_i, 4);
            wide_i  = _mm_or_si128(wide_i, wide_lo);

            // i<<14
            wide_i = _mm_slli_epi32(wide_i, 14);

            // SkClampMax(((f+one))>>16,max)
            __m128i wide_f1 = _mm_add_epi32(wide_f, wide_one);
            wide_f1 = _mm_max_epi16(_mm_srli_epi32(wide_f1, 16),
                                                   _mm_setzero_si128());
            wide_f1 = _mm_min_epi16(wide_f1, wide_max);

            // final combination
            wide_i = _mm_or_si128(wide_i, wide_f1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(xy), wide_i);

            wide_f = _mm_add_epi32(wide_f, wide_d2);

            fx += dx2;
            fy += dy2;
            xy += 4;
            count -= 2;
        } // while count >= 2
    } // if count >= 2

    while (count-- > 0) {
        *xy++ = ClampX_ClampY_pack_filter(fy, maxY, oneY);
        fy += dy;
        *xy++ = ClampX_ClampY_pack_filter(fx, maxX, oneX);
        fx += dx;
    }
}

/*  SSE version of ClampX_ClampY_nofilter_affine()
 *  portable version is in core/SkBitmapProcState_matrix.h
 */
void ClampX_ClampY_nofilter_affine_SSE2(const SkBitmapProcState& s,
                                      uint32_t xy[], int count, int x, int y) {
    SkASSERT(s.fInvType & SkMatrix::kAffine_Mask);
    SkASSERT((s.fInvType & ~(SkMatrix::kTranslate_Mask |
                             SkMatrix::kScale_Mask |
                             SkMatrix::kAffine_Mask)) == 0);

    SkPoint srcPt;
    s.fInvProc(s.fInvMatrix,
               SkIntToScalar(x) + SK_ScalarHalf,
               SkIntToScalar(y) + SK_ScalarHalf, &srcPt);

    SkFixed fx = SkScalarToFixed(srcPt.fX);
    SkFixed fy = SkScalarToFixed(srcPt.fY);
    SkFixed dx = s.fInvSx;
    SkFixed dy = s.fInvKy;
    int maxX = s.fBitmap->width() - 1;
    int maxY = s.fBitmap->height() - 1;

    if (count >= 4 && (maxX <= 0xFFFF)) {
        while (((size_t)xy & 0x0F) != 0) {
            *xy++ = (SkClampMax(fy >> 16, maxY) << 16) |
                                  SkClampMax(fx >> 16, maxX);
            fx += dx;
            fy += dy;
            count--;
        }

        SkFixed dx4 = dx * 4;
        SkFixed dy4 = dy * 4;

        __m128i wide_fx   = _mm_set_epi32(fx + dx * 3, fx + dx * 2,
                                          fx + dx, fx);
        __m128i wide_fy   = _mm_set_epi32(fy + dy * 3, fy + dy * 2,
                                          fy + dy, fy);
        __m128i wide_dx4  = _mm_set1_epi32(dx4);
        __m128i wide_dy4  = _mm_set1_epi32(dy4);

        __m128i wide_maxX = _mm_set1_epi32(maxX);
        __m128i wide_maxY = _mm_set1_epi32(maxY);

        while (count >= 4) {
            // SkClampMax(fx>>16,maxX)
            __m128i wide_lo = _mm_max_epi16(_mm_srli_epi32(wide_fx, 16),
                                            _mm_setzero_si128());
            wide_lo = _mm_min_epi16(wide_lo, wide_maxX);

            // SkClampMax(fy>>16,maxY)
            __m128i wide_hi = _mm_max_epi16(_mm_srli_epi32(wide_fy, 16),
                                            _mm_setzero_si128());
            wide_hi = _mm_min_epi16(wide_hi, wide_maxY);

            // final combination
            __m128i wide_i = _mm_or_si128(_mm_slli_epi32(wide_hi, 16),
                                          wide_lo);
            _mm_store_si128(reinterpret_cast<__m128i*>(xy), wide_i);

            wide_fx = _mm_add_epi32(wide_fx, wide_dx4);
            wide_fy = _mm_add_epi32(wide_fy, wide_dy4);

            fx += dx4;
            fy += dy4;
            xy += 4;
            count -= 4;
        } // while count >= 4
    } // if count >= 4

    while (count-- > 0) {
        *xy++ = (SkClampMax(fy >> 16, maxY) << 16) |
                              SkClampMax(fx >> 16, maxX);
        fx += dx;
        fy += dy;
    }
}

/*  SSE version of S32_D16_filter_DX_SSE2
 *  Definition is in section of "D16 functions for SRC == 8888" in SkBitmapProcState.cpp
 *  It combines S32_opaque_D32_filter_DX_SSE2 and SkPixel32ToPixel16
 */
void S32_D16_filter_DX_SSE2(const SkBitmapProcState& s,
                                   const uint32_t* xy,
                                   int count, uint16_t* colors) {
    SkASSERT(count > 0 && colors != NULL);
    SkASSERT(s.fFilterLevel != SkPaint::kNone_FilterLevel);
    SkASSERT(s.fBitmap->config() == SkBitmap::kARGB_8888_Config);
    SkASSERT(s.fBitmap->isOpaque());

    SkPMColor dstColor;
    const char* srcAddr = static_cast<const char*>(s.fBitmap->getPixels());
    size_t rb = s.fBitmap->rowBytes();
    uint32_t XY = *xy++;
    unsigned y0 = XY >> 14;
    const uint32_t* row0 = reinterpret_cast<const uint32_t*>(srcAddr + (y0 >> 4) * rb);
    const uint32_t* row1 = reinterpret_cast<const uint32_t*>(srcAddr + (XY & 0x3FFF) * rb);
    unsigned subY = y0 & 0xF;

    // ( 0,  0,  0,  0,  0,  0,  0, 16)
    __m128i sixteen = _mm_cvtsi32_si128(16);

    // ( 0,  0,  0,  0, 16, 16, 16, 16)
    sixteen = _mm_shufflelo_epi16(sixteen, 0);

    // ( 0,  0,  0,  0,  0,  0,  0,  y)
    __m128i allY = _mm_cvtsi32_si128(subY);

    // ( 0,  0,  0,  0,  y,  y,  y,  y)
    allY = _mm_shufflelo_epi16(allY, 0);

    // ( 0,  0,  0,  0, 16-y, 16-y, 16-y, 16-y)
    __m128i negY = _mm_sub_epi16(sixteen, allY);

    // (16-y, 16-y, 16-y, 16-y, y, y, y, y)
    allY = _mm_unpacklo_epi64(allY, negY);

    // (16, 16, 16, 16, 16, 16, 16, 16 )
    sixteen = _mm_shuffle_epi32(sixteen, 0);

    // ( 0,  0,  0,  0,  0,  0,  0,  0)
    __m128i zero = _mm_setzero_si128();

    do {
        uint32_t XX = *xy++;    // x0:14 | 4 | x1:14
        unsigned x0 = XX >> 18;
        unsigned x1 = XX & 0x3FFF;

        // (0, 0, 0, 0, 0, 0, 0, x)
        __m128i allX = _mm_cvtsi32_si128((XX >> 14) & 0x0F);

        // (0, 0, 0, 0, x, x, x, x)
        allX = _mm_shufflelo_epi16(allX, 0);

        // (x, x, x, x, x, x, x, x)
        allX = _mm_shuffle_epi32(allX, 0);

        // (16-x, 16-x, 16-x, 16-x, 16-x, 16-x, 16-x)
        __m128i negX = _mm_sub_epi16(sixteen, allX);

        // Load 4 samples (pixels).
        __m128i a00 = _mm_cvtsi32_si128(row0[x0]);
        __m128i a01 = _mm_cvtsi32_si128(row0[x1]);
        __m128i a10 = _mm_cvtsi32_si128(row1[x0]);
        __m128i a11 = _mm_cvtsi32_si128(row1[x1]);

        // (0, 0, a00, a10)
        __m128i a00a10 = _mm_unpacklo_epi32(a10, a00);

        // Expand to 16 bits per component.
        a00a10 = _mm_unpacklo_epi8(a00a10, zero);

        // ((a00 * (16-y)), (a10 * y)).
        a00a10 = _mm_mullo_epi16(a00a10, allY);

        // (a00 * (16-y) * (16-x), a10 * y * (16-x)).
        a00a10 = _mm_mullo_epi16(a00a10, negX);

        // (0, 0, a01, a10)
        __m128i a01a11 = _mm_unpacklo_epi32(a11, a01);

        // Expand to 16 bits per component.
        a01a11 = _mm_unpacklo_epi8(a01a11, zero);

        // (a01 * (16-y)), (a11 * y)
        a01a11 = _mm_mullo_epi16(a01a11, allY);

        // (a01 * (16-y) * x), (a11 * y * x)
        a01a11 = _mm_mullo_epi16(a01a11, allX);

        // (a00*w00 + a01*w01, a10*w10 + a11*w11)
        __m128i sum = _mm_add_epi16(a00a10, a01a11);

        // (DC, a00*w00 + a01*w01)
        __m128i shifted = _mm_shuffle_epi32(sum, 0xEE);

        // (DC, a00*w00 + a01*w01 + a10*w10 + a11*w11)
        sum = _mm_add_epi16(sum, shifted);

        // Divide each 16 bit component by 256.
        sum = _mm_srli_epi16(sum, 8);

        // Pack lower 4 16 bit values of sum into lower 4 bytes.
        sum = _mm_packus_epi16(sum, zero);

        // Extract low int and store.
        dstColor = _mm_cvtsi128_si32(sum);

        // *colors++ = SkPixel32ToPixel16(dstColor);
        // below is much faster than the above. It's tested for Android benchmark--Softweg
        __m128i _m_temp1 = _mm_set1_epi32(dstColor);
        __m128i _m_temp2 = _mm_srli_epi32(_m_temp1, 3);

        unsigned int r32 = _mm_cvtsi128_si32(_m_temp2);
        unsigned r = (r32 & ((1<<5) -1)) << 11;

        _m_temp2 = _mm_srli_epi32(_m_temp2, 7);
        unsigned int g32 = _mm_cvtsi128_si32(_m_temp2);
        unsigned g = (g32 & ((1<<6) -1)) << 5;

        _m_temp2 = _mm_srli_epi32(_m_temp2, 9);
        unsigned int b32 = _mm_cvtsi128_si32(_m_temp2);
        unsigned b = (b32 & ((1<<5) -1));

        *colors++ = r | g | b;

    } while (--count > 0);
}

extern "C" void S32_opaque_D32_nofilter_DX_SSE2_asm(const uint32_t* xy,
                                                    int count,
                                                    const SkPMColor* srcAddr,
                                                    uint32_t* colors);

void S32_opaque_D32_nofilter_DX_SSE2(const SkBitmapProcState& s,
                                     const uint32_t* xy,
                                     int count, uint32_t* colors) {
    SkASSERT(count > 0 && colors != NULL);
    SkASSERT(s.fInvType <= (SkMatrix::kTranslate_Mask | SkMatrix::kScale_Mask));
    SkASSERT(s.fDoFilter == false);
    SkASSERT(s.fBitmap->config() == SkBitmap::kARGB_8888_Config);
    SkASSERT(s.fAlphaScale == 256);

    const SkPMColor* SK_RESTRICT srcAddr =
        (const SkPMColor*)s.fBitmap->getPixels();

    // buffer is y32, x16, x16, x16, x16, x16
    // bump srcAddr to the proper row, since we're told Y never changes
    SkASSERT((unsigned)xy[0] < (unsigned)s.fBitmap->height());
    srcAddr = (const SkPMColor*)((const char*)srcAddr +
                                                xy[0] * s.fBitmap->rowBytes());
    xy += 1;

    SkPMColor src;

    if (1 == s.fBitmap->width()) {
        src = srcAddr[0];
        uint32_t dstValue = src;
        sk_memset32(colors, dstValue, count);
    } else {
        int i;

        S32_opaque_D32_nofilter_DX_SSE2_asm(xy, count, srcAddr, colors);

        xy     += 2 * (count >> 2);
        colors += 4 * (count >> 2);
        const uint16_t* SK_RESTRICT xx = (const uint16_t*)(xy);
        for (i = (count & 3); i > 0; --i) {
            SkASSERT(*xx < (unsigned)s.fBitmap->width());
            src = srcAddr[*xx++]; *colors++ = src;
        }
    }
}

void S32_opaque_D32_filter_DXDY_SSE2(const SkBitmapProcState& s,
                                  const uint32_t* xy,
                                  int count, uint32_t* colors) {

    SkASSERT(count > 0 && colors != NULL);
    SkASSERT(s.fDoFilter);
    SkASSERT(s.fBitmap->config() == SkBitmap::kARGB_8888_Config);
    SkASSERT(s.fAlphaScale == 256);
    uint32_t data;
    unsigned y0, y1, x0, x1, subX, subY;
    const SkPMColor *row0, *row1;

    const char* srcAddr = static_cast<const char*>(s.fBitmap->getPixels());
    unsigned rb = s.fBitmap->rowBytes();
    if (count >= 4) {
        while (((size_t)xy & 0x0F) != 0)
        {
            data = *xy++;
            y0 = data >> 14;
            y1 = data & 0x3FFF;
            subY = y0 & 0xF;
            y0 >>= 4;

            data = *xy++;
            x0 = data >> 14;
            x1 = data & 0x3FFF;
            subX = x0 & 0xF;
            x0 >>= 4;

            row0 = (const SkPMColor*)(srcAddr + y0 * rb);
            row1 = (const SkPMColor*)(srcAddr + y1 * rb);

            Filter_32_opaque(subX, subY,
                       (row0[x0]),
                       (row0[x1]),
                       (row1[x0]),
                       (row1[x1]),
                       colors);
            colors += 1;
            --count;
        }
        __m128i vf = _mm_set1_epi32(0xF);
        __m128i vmask = _mm_set1_epi32(gMask_00FF00FF);
        __m128i vmask2 = _mm_set1_epi32(0xff00ff00);
        __m128i v256 = _mm_set1_epi16(256);
        __m128i *d = reinterpret_cast<__m128i*>(colors);
        while (count >= 4) {
            __m128i vy_d = _mm_load_si128((__m128i*)xy);
            __m128i vx_d = _mm_load_si128((__m128i*)(xy+4));
            __m128i vy = (__m128i)_mm_shuffle_ps((__m128)vy_d,(__m128)vx_d,0x88);
            __m128i vx = (__m128i)_mm_shuffle_ps((__m128)vy_d,(__m128)vx_d,0xdd);

            uint32_t XY = *xy++;
            const uint32_t* row0 = (const uint32_t*)(srcAddr + (XY >> 18) * rb);
            const uint32_t* row1 = (const uint32_t*)(srcAddr + (XY & 0x3FFF) * rb);

            uint32_t XX = *xy++;    // x0:14 | 4 | x1:14
            unsigned x0 = XX >> 18;
            unsigned x1 = XX & 0x3FFF;

            __m128i a00 = _mm_cvtsi32_si128(row0[x0]);
            __m128i a01 = _mm_cvtsi32_si128(row0[x1]);
            __m128i a10 = _mm_cvtsi32_si128(row1[x0]);
            __m128i a11 = _mm_cvtsi32_si128(row1[x1]);

            XY = *xy++;
            row0 = (const uint32_t*)(srcAddr + (XY >> 18) * rb);
            row1 = (const uint32_t*)(srcAddr + (XY & 0x3FFF) * rb);

            XX = *xy++;    // x0:14 | 4 | x1:14
            x0 = XX >> 18;
            x1 = XX & 0x3FFF;
            a00 = _mm_unpacklo_epi32(a00,_mm_cvtsi32_si128(row0[x0]));
            a01 = _mm_unpacklo_epi32(a01,_mm_cvtsi32_si128(row0[x1]));
            a10 = _mm_unpacklo_epi32(a10,_mm_cvtsi32_si128(row1[x0]));
            a11 = _mm_unpacklo_epi32(a11,_mm_cvtsi32_si128(row1[x1]));

            XY = *xy++;
            row0 = (const uint32_t*)(srcAddr + (XY >> 18) * rb);
            row1 = (const uint32_t*)(srcAddr + (XY & 0x3FFF) * rb);

            XX = *xy++;    // x0:14 | 4 | x1:14
            x0 = XX >> 18;
            x1 = XX & 0x3FFF;
            __m128i a00_d = _mm_cvtsi32_si128(row0[x0]);
            __m128i a01_d = _mm_cvtsi32_si128(row0[x1]);
            __m128i a10_d = _mm_cvtsi32_si128(row1[x0]);
            __m128i a11_d = _mm_cvtsi32_si128(row1[x1]);

            XY = *xy++;
            row0 = (const uint32_t*)(srcAddr + (XY >> 18) * rb);
            row1 = (const uint32_t*)(srcAddr + (XY & 0x3FFF) * rb);

            XX = *xy++;    // x0:14 | 4 | x1:14
            x0 = XX >> 18;
            x1 = XX & 0x3FFF;
            a00_d = _mm_unpacklo_epi32(a00_d,_mm_cvtsi32_si128(row0[x0]));
            a01_d = _mm_unpacklo_epi32(a01_d,_mm_cvtsi32_si128(row0[x1]));
            a10_d = _mm_unpacklo_epi32(a10_d,_mm_cvtsi32_si128(row1[x0]));
            a11_d = _mm_unpacklo_epi32(a11_d,_mm_cvtsi32_si128(row1[x1]));

            vy = _mm_srli_epi32(vy,14);
            vy = _mm_and_si128(vy,vf);

            vx = _mm_srli_epi32(vx,14);
            vx = _mm_and_si128(vx,vf);

            a00 = _mm_unpacklo_epi64(a00,a00_d);
            a01 = _mm_unpacklo_epi64(a01,a01_d);
            a10 = _mm_unpacklo_epi64(a10,a10_d);
            a11 = _mm_unpacklo_epi64(a11,a11_d);

            vy = _mm_shufflelo_epi16(vy,0xa0);
            vy = _mm_shufflehi_epi16(vy,0xa0);
            vx = _mm_shufflelo_epi16(vx,0xa0);
            vx = _mm_shufflehi_epi16(vx,0xa0);

            // unsigned xy = x * y;
            __m128i vxy = _mm_mullo_epi16(vx,vy);
            __m128i v16y = _mm_slli_epi16(vy,4);
            __m128i v16x = _mm_slli_epi16(vx,4);
            // unsigned scale = 256 - 16*y - 16*x + xy;
            __m128i vscale = _mm_add_epi16(v256,vxy);
            vscale = _mm_sub_epi16(vscale,v16y);
            vscale = _mm_sub_epi16(vscale,v16x);

            // uint32_t lo = (a00 & mask) * scale;
            __m128i vlo = _mm_and_si128(a00,vmask);
            vlo = _mm_mullo_epi16(vlo, vscale);

            // uint32_t hi = ((a00 >> 8) & mask) * scale;
            __m128i vhi = _mm_srli_epi32(a00,8);
            vhi = _mm_and_si128(vhi,vmask);
            vhi = _mm_mullo_epi16(vhi, vscale);
            // scale = 16*x-xy;
            vscale = _mm_sub_epi16(v16x,vxy);

            // lo += (a01 & mask) * scale;
            __m128i vlo2 = _mm_and_si128(a01,vmask);
            vlo2 = _mm_mullo_epi16(vlo2, vscale);
            vlo = _mm_add_epi16(vlo,vlo2);

            // hi += ((a01 >> 8) & mask) * scale;
            __m128i vhi2 = _mm_srli_epi32(a01,8);
            vhi2 = _mm_and_si128(vhi2,vmask);
            vhi2 = _mm_mullo_epi16(vhi2, vscale);
            vhi = _mm_add_epi16(vhi,vhi2);

            // scale = 16*y - xy;
            vscale = _mm_sub_epi16(v16y,vxy);

            // lo += (a10 & mask) * scale;
            vlo2 = _mm_and_si128(a10,vmask);
            vlo2 = _mm_mullo_epi16(vlo2, vscale);
            vlo = _mm_add_epi16(vlo,vlo2);

            // hi += ((a10 >> 8) & mask) * scale;
            vhi2 = _mm_srli_epi32(a10,8);
            vhi2 = _mm_and_si128(vhi2,vmask);
            vhi2 = _mm_mullo_epi16(vhi2, vscale);
            vhi = _mm_add_epi16(vhi,vhi2);

            // lo += (a11 & mask) * xy;
            vlo2 = _mm_and_si128(a11,vmask);
            vlo2 = _mm_mullo_epi16(vlo2, vxy);
            vlo = _mm_add_epi16(vlo,vlo2);

            // hi += ((a11 >> 8) & mask) * xy;
            vhi2 = _mm_srli_epi32(a11,8);
            vhi2 = _mm_and_si128(vhi2,vmask);
            vhi2 = _mm_mullo_epi16(vhi2, vxy);
            vhi = _mm_add_epi16(vhi,vhi2);

            // *dstColor = ((lo >> 8) & mask) | (hi & ~mask);
            vlo = _mm_srli_epi32(vlo,8);
            vlo = _mm_and_si128(vlo,vmask);
            vhi = _mm_and_si128(vhi,vmask2);

            _mm_storeu_si128(d,_mm_or_si128(vlo,vhi));
            d++;
            count -= 4;
        }
    colors = reinterpret_cast<SkPMColor*>(d);
    }
    while (count > 0)
    {
        data = *xy++;
        y0 = data >> 14;
        y1 = data & 0x3FFF;
        subY = y0 & 0xF;
        y0 >>= 4;

        data = *xy++;
        x0 = data >> 14;
        x1 = data & 0x3FFF;
        subX = x0 & 0xF;
        x0 >>= 4;

        row0 = (const SkPMColor*)(srcAddr + y0 * rb);
        row1 = (const SkPMColor*)(srcAddr + y1 * rb);

        Filter_32_opaque(subX, subY,
                   (row0[x0]),
                   (row0[x1]),
                   (row1[x0]),
                   (row1[x1]),
                   colors);
        colors += 1;
        count --;
   }

}

void S32_opaque_D32_filter_DXDY_SSE2_asm(const SkBitmapProcState& s,
        const uint32_t* xy, int count, uint32_t* colors) {
    const char* srcAddr = static_cast<const char*>(s.fBitmap->getPixels());
    unsigned rb = s.fBitmap->rowBytes();
    uint32_t data;
    unsigned y0, y1, x0, x1, subX, subY;
    const SkPMColor *row0, *row1;
    if (count >= 4) {
        while (((size_t)xy & 0x0F) != 0)
        {
            data = *xy++;
            y0 = data >> 14;
            y1 = data & 0x3FFF;
            subY = y0 & 0xF;
            y0 >>= 4;

            data = *xy++;
            x0 = data >> 14;
            x1 = data & 0x3FFF;
            subX = x0 & 0xF;
            x0 >>= 4;

            row0 = (const SkPMColor*)(srcAddr + y0 * rb);
            row1 = (const SkPMColor*)(srcAddr + y1 * rb);

            Filter_32_opaque(subX, subY,
                       (row0[x0]),
                       (row0[x1]),
                       (row1[x0]),
                       (row1[x1]),
                       colors);
            colors += 1;
            --count;
        }
        if (count >= 4)
        {
            __attribute__((aligned(16)))
            __m128i tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmpa;
            // load 4 pixes in each run [D, C, B, A]
            __asm__(
            "1:\n"
            // load pixel A
            "mov    (%%edx),%%edi\n"  // *xy
            "mov    %%edi,%%ecx\n"
            "shr    $0x12,%%edi\n"    // *xy > 0x12
            "and    $0x3fff,%%ecx\n"  // *xy & 0x3FFF
            "imul   %[rb],%%edi\n"    // rb * y0
            "imul   %[rb],%%ecx\n"    // rb * y1

            "mov    0x4(%%edx),%%eax\n" // *(xy + 4)
            "mov    %%eax,%%esi\n"
            "shr    $0x12,%%esi\n"      // x0
            "add    %[srcAddr],%%ecx\n" // row1.0 = srcAddr + rb * y1
            "and    $0x3fff,%%eax\n"    // x1
            "add    %[srcAddr],%%edi\n" // row0.0 = srcAddr + rb * y0

            "movd   (%%ecx,%%esi,4),%%xmm6\n"    // A: a10
            "movd   (%%ecx,%%eax,4),%%xmm7\n"    // A: a11
            // load pixel B
            "mov    0x8(%%edx),%%ecx\n"          // *(xy+8)
            "movd   (%%edi,%%esi,4),%%xmm4\n"    // A:a00
            "movd   (%%edi,%%eax,4),%%xmm5\n"    // A:a01
            "mov    %%ecx,%%esi\n"
            "and    $0x3fff,%%ecx\n"      // B:y1
            "shr    $0x12,%%esi\n"        // B:y0
            "imul   %[rb],%%ecx\n"        // rb * y1
            "imul   %[rb],%%esi\n"        // rb * y0
            "mov    0xc(%%edx),%%edi\n"
            "mov    %%edi,%%eax\n"
            "shr    $0x12,%%eax\n"        // B:x0
            "add    %[srcAddr],%%ecx\n"   // B:row1.1
            "and    $0x3fff,%%edi\n"      // B:x1
            "add    %[srcAddr],%%esi\n"   // B:row0.1
            "movd   (%%ecx,%%eax,4),%%xmm2\n"    // B:a10
            "movd   (%%ecx,%%edi,4),%%xmm1\n"    // B:a11
            // load pixel C
            "mov    0x10(%%edx),%%ecx\n"
            "movd   (%%esi,%%eax,4),%%xmm0\n"    // B:a00
            "movd   (%%esi,%%edi,4),%%xmm3\n"    // B:a01

            "mov    %%ecx,%%eax\n"
            "shr    $0x12,%%eax\n"               // C:y0
            "and    $0x3fff,%%ecx\n"             // C:y1
            "imul   %[rb],%%eax\n"
            "imul   %[rb],%%ecx\n"
            // [0, Ba00, 0, Aa00]
            "punpcklqdq %%xmm0,%%xmm4\n"
            // [0, Ba01, 0, Aa01]
            "punpcklqdq %%xmm3,%%xmm5\n"
            "mov    0x14(%%edx),%%esi\n"
            "mov    %%esi,%%edi\n"
            "add    %[srcAddr],%%eax\n"          // C: row0
            "add    %[srcAddr],%%ecx\n"          // C: row1
            // [0, Ba11, 0, Aa11]
            "punpcklqdq %%xmm1,%%xmm7\n"
            // [0, Ba10, 0, Aa10]
            "punpcklqdq %%xmm2,%%xmm6\n"
            "and    $0x3fff,%%esi\n"
            "shr    $0x12,%%edi\n"
            // [0, 0, 0, Ca01]
            "movd   0x0(%%eax,%%esi,4),%%xmm3\n"
            // [0, 0, 0, Ca11]
            "movd   (%%ecx,%%esi,4),%%xmm2\n"
            // load pixel D
            "mov    0x18(%%edx),%%esi\n"
            // [0, 0, 0, Ca00]
            "movd   0x0(%%eax,%%edi,4),%%xmm1\n"
            "mov    %%esi,%%eax\n"
            "shr    $0x12,%%esi\n"            // D:y0
            "and    $0x3fff,%%eax\n"          // D:y1
            // [0, 0, 0, Ca10]
            "movd   (%%ecx,%%edi,4),%%xmm0\n"

            "imul   %[rb],%%eax\n"            // D:rb * y0
            "imul   %[rb],%%esi\n"            // D:rb * y1

            "movdqa %%xmm2,%[tmp6]\n"         // save D:a11
            "movdqa %%xmm0,%[tmp5]\n"         // save D:a10


            "mov    0x1c(%%edx),%%ecx\n"
            "mov    %%ecx,%%edi\n"
            "add    %[srcAddr],%%esi\n"       // D:row0
            "and    $0x3fff,%%ecx\n"          // D:x1
            "shr    $0x12,%%edi\n"            // D:x0
            "add    %[srcAddr],%%eax\n"       // D:row1

            "movd   (%%esi,%%ecx,4),%%xmm2\n"    // D:a01
            "movd   (%%esi,%%edi,4),%%xmm0\n"    // D:a00

            // [0, Da01, 0, Ca01]
            "punpcklqdq %%xmm2,%%xmm3\n"
            // [0, Da00, 0, Ca00]
            "punpcklqdq %%xmm0,%%xmm1\n"

            "mov 0x0(%%eax,%%edi,4), %%esi\n"       // D:a10
            "mov %%esi, %[tmp8]\n"                  // save D:a10

            "movd   0x0(%%eax,%%ecx,4),%%xmm2\n"    // D:a11
            // [Da00, Ca00, Ba00, Aa00]
            "shufps $0x88,%%xmm1,%%xmm4\n"
            "movdqa %[tmp6],%%xmm1\n"        // C:a11
            // [0, Da11, 0, Ca11]
            "punpcklqdq %%xmm2,%%xmm1\n"
            // [Da11, Ca11, Ba11, Aa11]
            "shufps $0x88,%%xmm1,%%xmm7\n"   // a11.3210

            "movdqa  (%%edx),%%xmm1\n"        // vy_d

            "movaps %%xmm1,%%xmm2\n"         // vy_d
            "shufps $0xdd,0x10(%%edx),%%xmm1\n"    // vx
            "shufps $0x88,0x10(%%edx),%%xmm2\n"    // vy

            "psrld  $0xe,%%xmm1\n"        // vx>>14
            // [Da01, Ca01, Ba01, Aa01]
            "shufps $0x88,%%xmm3,%%xmm5\n"
            "psrld  $0xe,%%xmm2\n"         // vy>>14

            "add    $0x20,%%edx\n"        // xy+=8
            "movdqa %[tmp5],%%xmm3\n"     // C:a10
            "movhpd %[tmp8],%%xmm3\n"     // CD:a10

            "pand   vf,%%xmm1\n"           // vx & vf
            "pand   vf,%%xmm2\n"           // vy & vf
            // [Da10, Ca10, Ba10, Aa10]
            "shufps $0x88,%%xmm3,%%xmm6\n"

            "pshuflw $0xa0,%%xmm1,%%xmm3\n"
            "pshuflw $0xa0,%%xmm2,%%xmm0\n"
            "pshufhw $0xa0,%%xmm3,%%xmm1\n"
            "pshufhw $0xa0,%%xmm0,%%xmm2\n"

            "movdqa %%xmm1,%%xmm3\n"
            "pmullw %%xmm2,%%xmm3\n"         // vxy

            "psllw  $0x4,%%xmm2\n"           // v16y
            "movdqa v256,%%xmm0\n"
            "psllw  $0x4,%%xmm1\n"           // v16x

            "paddw  %%xmm3,%%xmm0\n"         // v256+vxy
            "psubw  %%xmm2,%%xmm0\n"         // v256+vxy-v16y
            "psubw  %%xmm3,%%xmm2\n"         // v16y-vxy    vscale 2

            "movaps %%xmm4,%[tmp9]\n"        // a00
            "psubw  %%xmm1,%%xmm0\n"         // v256+vxy-v16y-v16x   vscale0

            "movaps %%xmm5,%[tmpa]\n"        // a01

            "psubw  %%xmm3,%%xmm1\n"         // v16x-vxy   vscale 1
            "pand   vmask,%%xmm4\n"          // a00 & vmask
            "pand   vmask,%%xmm5\n"          // a01 & vmask

            "pmullw %%xmm0,%%xmm4\n"         // a00.l*vscale0\n
            "pmullw %%xmm1,%%xmm5\n"         // a01.l*vscale1\n

            "paddw  %%xmm5,%%xmm4\n"         // a00.l+a01.l
            "movaps %%xmm6,%%xmm5\n"         // a10
            "pand   vmask,%%xmm5\n"          // a10.l
            "psrld  $0x8,%%xmm6\n"           // a10.h
            "pmullw %%xmm2,%%xmm5\n"         // a10.l*vscale2
            "paddw  %%xmm5,%%xmm4\n"         // a00.l+a01.l+a10.l
            "movaps %%xmm7,%%xmm5\n"         // a11
            "pand   vmask,%%xmm5\n"          // a11.l
            "psrld  $0x8,%%xmm7\n"           // a11.h
            "pmullw %%xmm3,%%xmm5\n"         // a11.l*vxy
            "paddw  %%xmm5,%%xmm4\n"         // a00.l+a01.l+a10.l+a11.l
            "movaps %[tmp9],%%xmm5\n"        // a00
            "psrld  $0x8,%%xmm4\n"           // ax.l>>8
            "psrld  $0x8,%%xmm5\n"           // a00.h
            "pand   vmask,%%xmm5\n"          // a00.h
            "pmullw %%xmm0,%%xmm5\n"         // a00.h*vscale0
            "movaps %[tmpa],%%xmm0\n"        // a01
            "psrld  $0x8,%%xmm0\n"           // a01.h
            "mov    %[count], %%edi\n"
            "pand   vmask,%%xmm0\n"          // a01.h
            "pand   vmask,%%xmm6\n"          // a10.h
            "pmullw %%xmm1,%%xmm0\n"         // a01.h * vscale1
            "add    $0xfffffffc,%%edi\n"
            "pmullw %%xmm2,%%xmm6\n"         // a10.h * vscale2
            "pand   vmask,%%xmm7\n"          // a11.h
            "paddw  %%xmm0,%%xmm5\n"         // a00.h+a01.h
            "pmullw %%xmm3,%%xmm7\n"         // a11.h * vscale2
            "paddw  %%xmm6,%%xmm5\n"         // a00.h+a01.h+a10.h
            "mov    %[colors],%%ecx\n"       // colors
            "paddw  %%xmm7,%%xmm5\n"         // a00.h+a01.h+a10.h+a11.h
            "pand   vmask,%%xmm4\n"          // ax.l & vmask
            "pand   vmask2,%%xmm5\n"         // ax.h & vmask2
            "addl   $0x10,%[colors]\n"       // colors+=4
            "por    %%xmm5,%%xmm4\n"         // ax.l|ax.h
            "movdqu %%xmm4,(%%ecx)\n"        // store colors
            "cmp    $0x4,%%edi\n"
            "mov    %%edi,%[count]\n"
            "jge    1b\n"
            :"+d" (xy)
            : [srcAddr] "m" (srcAddr), [colors] "m" (colors), [rb] "m" (rb),
            [count] "m" (count),
            [tmp1] "m" (tmp1), [tmp2] "m" (tmp2), [tmp3] "m" (tmp3), [tmp4] "m" (tmp4),
            [tmp5] "m" (tmp5), [tmp6] "m" (tmp6), [tmp7] "m" (tmp7), [tmp8] "m" (tmp4),
            [tmp9] "m" (tmp9), [tmpa] "m" (tmpa)
            :"memory","ecx","esi","edi", "eax"
        );
        } // count >= 4
    }
    while (count > 0)
    {
        data = *xy++;
         y0 = data >> 14;
        y1 = data & 0x3FFF;
        subY = y0 & 0xF;
        y0 >>= 4;

        data = *xy++;
        x0 = data >> 14;
        x1 = data & 0x3FFF;
        subX = x0 & 0xF;
        x0 >>= 4;

        row0 = (const SkPMColor*)(srcAddr + y0 * rb);
        row1 = (const SkPMColor*)(srcAddr + y1 * rb);

        Filter_32_opaque(subX, subY,
                   (row0[x0]),
                   (row0[x1]),
                   (row1[x0]),
                   (row1[x1]),
                   colors);
        colors += 1;
        count --;
    }
}
void S32_alpha_D32_filter_DXDY_SSE2(const SkBitmapProcState& s,
                                  const uint32_t* xy,
                                  int count, uint32_t* colors) {

    SkASSERT(count > 0 && colors != NULL);
    SkASSERT(s.fDoFilter);
    SkASSERT(s.fBitmap->config() == SkBitmap::kARGB_8888_Config);
    SkASSERT(s.fAlphaScale < 256);
    uint32_t data;
    unsigned y0, y1, x0, x1, subX, subY;
    const SkPMColor *row0, *row1;
    const char* srcAddr = static_cast<const char*>(s.fBitmap->getPixels());
    __m128i alphaScale = _mm_set1_epi16(s.fAlphaScale);
    unsigned rb = s.fBitmap->rowBytes();
    if (count >= 4) {
        while (((size_t)xy & 0x0F) != 0)
        {
            data = *xy++;
            y0 = data >> 14;
            y1 = data & 0x3FFF;
            subY = y0 & 0xF;
            y0 >>= 4;

            data = *xy++;
            x0 = data >> 14;
            x1 = data & 0x3FFF;
            subX = x0 & 0xF;
            x0 >>= 4;

            row0 = (const SkPMColor*)(srcAddr + y0 * rb);
            row1 = (const SkPMColor*)(srcAddr + y1 * rb);

            Filter_32_alpha(subX, subY,
                       (row0[x0]),
                       (row0[x1]),
                       (row1[x0]),
                       (row1[x1]),
                       colors,
                       s.fAlphaScale);
            colors += 1;
            --count;
        }
    __m128i vf = _mm_set1_epi32(0xF);
    __m128i vmask = _mm_set1_epi32(gMask_00FF00FF);
    __m128i vmask2 = _mm_set1_epi32(0xff00ff00);
    __m128i v256 = _mm_set1_epi16(256);
    __m128i *d = reinterpret_cast<__m128i*>(colors);
    while (count>=4) {
        // [Bx1x0, By1y0, Ax1x0, Ay1y0]; load 4 pixes [D, C, B, A] in one run
        __m128i vy_d = _mm_load_si128((__m128i*)xy);
        // [Dx1x0, Dy1y0, Cx1x0, Cy1y0];
        __m128i vx_d = _mm_load_si128((__m128i*)(xy+4));
        // [Dy1y0, Cy1y0, By1y0, Ay1y0]
        __m128i vy = (__m128i)_mm_shuffle_ps((__m128)vy_d,(__m128)vx_d,0x88);
        // [Dx1x0, Cx1x0, Bx1x0, Ax1x0]
        __m128i vx = (__m128i)_mm_shuffle_ps((__m128)vy_d,(__m128)vx_d,0xdd);

        uint32_t XY = *xy++;
        const uint32_t* row0 = (const uint32_t*)(srcAddr + (XY >> 18) * rb);
        const uint32_t* row1 = (const uint32_t*)(srcAddr + (XY & 0x3FFF) * rb);

        uint32_t XX = *xy++;    // x0:14 | 4 | x1:14
        unsigned x0 = XX >> 18;
        unsigned x1 = XX & 0x3FFF;
        // [0, 0, 0, Ay0x0]
        __m128i a00 = _mm_cvtsi32_si128(row0[x0]);
        // [0, 0, 0, Ay0x1]
        __m128i a01 = _mm_cvtsi32_si128(row0[x1]);
        // [0, 0, 0, Ay1x0]
        __m128i a10 = _mm_cvtsi32_si128(row1[x0]);
        // [0, 0, 0, Ay1x1]
        __m128i a11 = _mm_cvtsi32_si128(row1[x1]);

        XY = *xy++;
        row0 = (const uint32_t*)(srcAddr + (XY >> 18) * rb);
        row1 = (const uint32_t*)(srcAddr + (XY & 0x3FFF) * rb);

        XX = *xy++;    // x0:14 | 4 | x1:14
        x0 = XX >> 18;
        x1 = XX & 0x3FFF;
        // [0, 0, By0x0, Ay0x0]
        a00 = _mm_unpacklo_epi32(a00,_mm_cvtsi32_si128(row0[x0]));
        // [0, 0, By0x1, Ay0x1]
        a01 = _mm_unpacklo_epi32(a01,_mm_cvtsi32_si128(row0[x1]));
        // [0, 0, By1x0, Ay1x0]
        a10 = _mm_unpacklo_epi32(a10,_mm_cvtsi32_si128(row1[x0]));
        // [0, 0, By1x1, Ay1x1]
        a11 = _mm_unpacklo_epi32(a11,_mm_cvtsi32_si128(row1[x1]));

        XY = *xy++;
        row0 = (const uint32_t*)(srcAddr + (XY >> 18) * rb);
        row1 = (const uint32_t*)(srcAddr + (XY & 0x3FFF) * rb);

        XX = *xy++;    // x0:14 | 4 | x1:14
        x0 = XX >> 18;
        x1 = XX & 0x3FFF;
        // [0, 0, 0, Cy0x0]
        __m128i a00_d = _mm_cvtsi32_si128(row0[x0]);
        // [0, 0, 0, Cy0x1]
        __m128i a01_d = _mm_cvtsi32_si128(row0[x1]);
        // [0, 0, 0, Cy1x0]
        __m128i a10_d = _mm_cvtsi32_si128(row1[x0]);
        // [0, 0, 0, Cy1x1]
        __m128i a11_d = _mm_cvtsi32_si128(row1[x1]);

        XY = *xy++;
        row0 = (const uint32_t*)(srcAddr + (XY >> 18) * rb);
        row1 = (const uint32_t*)(srcAddr + (XY & 0x3FFF) * rb);

        XX = *xy++;    // x0:14 | 4 | x1:14
        x0 = XX >> 18;
        x1 = XX & 0x3FFF;
        // [0, 0, Dy0x0, Cy0x0]
        a00_d = _mm_unpacklo_epi32(a00_d,_mm_cvtsi32_si128(row0[x0]));
        // [0, 0, Dy0x1, Cy0x1]
        a01_d = _mm_unpacklo_epi32(a01_d,_mm_cvtsi32_si128(row0[x1]));
        // [0, 0, Dy1x0, Cy1x0]
        a10_d = _mm_unpacklo_epi32(a10_d,_mm_cvtsi32_si128(row1[x0]));
        // [0, 0, Dy1x1, Cy1x1]
        a11_d = _mm_unpacklo_epi32(a11_d,_mm_cvtsi32_si128(row1[x1]));

        // [DsubX, CsubY, BsubY, AsubY]
        vy = _mm_srli_epi32(vy,14);
        vy = _mm_and_si128(vy,vf);

        // [DsubX, CsubX, BsubX, AsubX]
        vx = _mm_srli_epi32(vx,14);
        vx = _mm_and_si128(vx,vf);

        // [Dy0x0, Cy0x0, By0x0, Ay0x0]
        a00 = _mm_unpacklo_epi64(a00,a00_d);
        // [Dy0x1, Cy0x1, By0x1, Ay0x1]
        a01 = _mm_unpacklo_epi64(a01,a01_d);
        // [Dy1x0, Cy1x0, By1x0, Ay1x0]
        a10 = _mm_unpacklo_epi64(a10,a10_d);
        // [Dy1x1, Cy1x1, By1x1, Ay1x1]
        a11 = _mm_unpacklo_epi64(a11,a11_d);

        // [0, DsubX, 0, CsubY, BsubY, BsubY, AsubY, AsubY]
        vy = _mm_shufflelo_epi16(vy,0xa0);
        // [CsubY, DsubY, CsubY, CsubY, BsubY, BsubY, AsubY, AsubY]
        vy = _mm_shufflehi_epi16(vy,0xa0);
        vx = _mm_shufflelo_epi16(vx,0xa0);
        // [CsubX, DsubX, CsubX, CsubX, BsubX, BsubX, AsubX, AsubX]
        vx = _mm_shufflehi_epi16(vx,0xa0);

        // unsigned xy = x * y;
        __m128i vxy = _mm_mullo_epi16(vx,vy);
        __m128i v16y = _mm_slli_epi16(vy,4);
        __m128i v16x = _mm_slli_epi16(vx,4);
        // unsigned scale = 256 - 16*y - 16*x + xy;
        __m128i vscale = _mm_add_epi16(v256,vxy);
        vscale = _mm_sub_epi16(vscale,v16y);
        vscale = _mm_sub_epi16(vscale,v16x);

        // uint32_t lo = (a00 & mask) * scale;
        __m128i vlo = _mm_and_si128(a00,vmask);
        vlo = _mm_mullo_epi16(vlo, vscale);

        // uint32_t hi = ((a00 >> 8) & mask) * scale;
        __m128i vhi = _mm_srli_epi32(a00,8);
        vhi = _mm_and_si128(vhi,vmask);
        vhi = _mm_mullo_epi16(vhi, vscale);
        // scale = 16*x-xy;
        vscale = _mm_sub_epi16(v16x,vxy);

        // lo += (a01 & mask) * scale;
        __m128i vlo2 = _mm_and_si128(a01,vmask);
        vlo2 = _mm_mullo_epi16(vlo2, vscale);
        vlo = _mm_add_epi16(vlo,vlo2);

        // hi += ((a01 >> 8) & mask) * scale;
        __m128i vhi2 = _mm_srli_epi32(a01,8);
        vhi2 = _mm_and_si128(vhi2,vmask);
        vhi2 = _mm_mullo_epi16(vhi2, vscale);
        vhi = _mm_add_epi16(vhi,vhi2);

        // scale = 16*y - xy;
        vscale = _mm_sub_epi16(v16y,vxy);

        // lo += (a10 & mask) * scale;
        vlo2 = _mm_and_si128(a10,vmask);
        vlo2 = _mm_mullo_epi16(vlo2, vscale);
        vlo = _mm_add_epi16(vlo,vlo2);

        // hi += ((a10 >> 8) & mask) * scale;
        vhi2 = _mm_srli_epi32(a10,8);
        vhi2 = _mm_and_si128(vhi2,vmask);
        vhi2 = _mm_mullo_epi16(vhi2, vscale);
        vhi = _mm_add_epi16(vhi,vhi2);

        // lo += (a11 & mask) * xy;
        vlo2 = _mm_and_si128(a11,vmask);
        vlo2 = _mm_mullo_epi16(vlo2, vxy);
        vlo = _mm_add_epi16(vlo,vlo2);

        // hi += ((a11 >> 8) & mask) * xy;
        vhi2 = _mm_srli_epi32(a11,8);
        vhi2 = _mm_and_si128(vhi2,vmask);
        vhi2 = _mm_mullo_epi16(vhi2, vxy);
        vhi = _mm_add_epi16(vhi,vhi2);

        // lo = (((lo >> 8) & mask) * alphaScale) >> 8 & mask;
        vlo = _mm_srli_epi32(vlo,8);
        vlo = _mm_mullo_epi16(_mm_and_si128(vlo,vmask), alphaScale);
        vlo = _mm_srli_epi32(vlo,8);
        vlo = _mm_and_si128(vlo,vmask);

        // hi = (((hi >> 8) & mask) * alphaScale) >> 8 &(~mask);
        vhi = _mm_srli_epi32(vhi,8);
        vhi = _mm_mullo_epi16(_mm_and_si128(vhi,vmask), alphaScale);
        vhi = _mm_and_si128(vhi, vmask2);

        _mm_storeu_si128(d,_mm_or_si128(vlo,vhi));
        d++;
        count -= 4;
        }
        colors = reinterpret_cast<SkPMColor*>(d);
    }
    while (count > 0)
    {
        data = *xy++;
        y0 = data >> 14;
        y1 = data & 0x3FFF;
        subY = y0 & 0xF;
        y0 >>= 4;

        data = *xy++;
        x0 = data >> 14;
        x1 = data & 0x3FFF;
        subX = x0 & 0xF;
        x0 >>= 4;

        row0 = (const SkPMColor*)(srcAddr + y0 * rb);
        row1 = (const SkPMColor*)(srcAddr + y1 * rb);

        Filter_32_alpha(subX, subY,
                       (row0[x0]),
                       (row0[x1]),
                       (row1[x0]),
                       (row1[x1]),
                       colors,
                       s.fAlphaScale);
        colors += 1;
        count --;
    }

}

void S32_alpha_D32_filter_DXDY_SSE2_asm(const SkBitmapProcState& s,
                                  const uint32_t* xy,
                                  int count, uint32_t* colors) {
    SkASSERT(count > 0 && colors != NULL);
    SkASSERT(s.fDoFilter);
    SkASSERT(s.fBitmap->config() == SkBitmap::kARGB_8888_Config);
    SkASSERT(s.fAlphaScale < 256);
    uint32_t data;
    unsigned y0, y1, x0, x1, subX, subY;
    const SkPMColor *row0, *row1;
    const char* srcAddr = static_cast<const char*>(s.fBitmap->getPixels());
    unsigned rb = s.fBitmap->rowBytes();
    unsigned alphaScale = s.fAlphaScale;
    if (count >= 4) {
        while (((size_t)xy & 0x0F) != 0)
        {
            data = *xy++;
            y0 = data >> 14;
            y1 = data & 0x3FFF;
            subY = y0 & 0xF;
            y0 >>= 4;

            data = *xy++;
            x0 = data >> 14;
            x1 = data & 0x3FFF;
            subX = x0 & 0xF;
            x0 >>= 4;

            row0 = (const SkPMColor*)(srcAddr + y0 * rb);
            row1 = (const SkPMColor*)(srcAddr + y1 * rb);

            Filter_32_alpha(subX, subY,
                       (row0[x0]),
                       (row0[x1]),
                       (row1[x0]),
                       (row1[x1]),
                      colors,
                       s.fAlphaScale);
            colors += 1;
            --count;
        }

        // BE CAREFUL, count >= 4
        if (count >= 4)
        {
            __attribute__((aligned(16)))
            __m128i tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmpa, tmpb;
           // unsigned int tmpeax;
            __asm__(
           // "mov   %%eax, %[tmpeax]\n"
           "movd   %[alphaScale], %%xmm0\n"
           "pshuflw $0,%%xmm0,%%xmm0\n"
           "punpcklqdq %%xmm0,%%xmm0\n"    // a00._1_0
           "movaps %%xmm0,%[tmpb]\n"
           "1:\n"
           "mov    (%%edx),%%edi\n"
           "mov    %%edi,%%ecx\n"
           "shr    $0x12,%%edi\n"
           "and    $0x3fff,%%ecx\n"
           "imul   %[rb],%%edi\n"
           "imul   %[rb],%%ecx\n"

           "mov    0x4(%%edx),%%eax\n"
           "mov    %%eax,%%esi\n"
           "shr    $0x12,%%esi\n"
           "add    %[srcAddr],%%ecx\n"    // row1.0
           "and    $0x3fff,%%eax\n"
           "add    %[srcAddr],%%edi\n"    // row0.0

           "movd   (%%ecx,%%esi,4),%%xmm6\n"    // a10.0
           "movd   (%%ecx,%%eax,4),%%xmm7\n"    // a11.0
           "mov    0x8(%%edx),%%ecx\n"
           "movd   (%%edi,%%esi,4),%%xmm4\n"    // a00.0
           "movd   (%%edi,%%eax,4),%%xmm5\n"    // a01.0
           "mov    %%ecx,%%esi\n"
           "and    $0x3fff,%%ecx\n"
           "shr    $0x12,%%esi\n"
           "imul   %[rb],%%ecx\n"
           "imul   %[rb],%%esi\n"

           // "movaps %%xmm6,%[tmp1]\n"    // a10.0
           // "movaps %%xmm7,%[tmp2]\n"    // a11.0

           "mov    0xc(%%edx),%%edi\n"
           "mov    %%edi,%%eax\n"
           "shr    $0x12,%%eax\n"
           "add    %[srcAddr],%%ecx\n"    // row1.1
           "and    $0x3fff,%%edi\n"
           "add    %[srcAddr],%%esi\n"    // row0.1
           "movd   (%%ecx,%%eax,4),%%xmm2\n"    // a10.1
           "movd   (%%ecx,%%edi,4),%%xmm1\n"    // a11.1
           "mov    0x10(%%edx),%%ecx\n"
           "movd   (%%esi,%%eax,4),%%xmm0\n"    // a00.1
           "movd   (%%esi,%%edi,4),%%xmm3\n"    // a01.1

           "mov    %%ecx,%%eax\n"
           "shr    $0x12,%%eax\n"
           "and    $0x3fff,%%ecx\n"
           "imul   %[rb],%%eax\n"
           "imul   %[rb],%%ecx\n"

           "punpcklqdq %%xmm0,%%xmm4\n"    // a00._1_0
           "punpcklqdq %%xmm3,%%xmm5\n"    // a01._1_0

           // "movdqa %%xmm2,%[tmp3]\n"    // a10.1
           // "movdqa %%xmm1,%[tmp4]\n"    // a11.1

           "mov    0x14(%%edx),%%esi\n"
           "mov    %%esi,%%edi\n"
           "add    %[srcAddr],%%eax\n"        // row0.2
           "add    %[srcAddr],%%ecx\n"        // row1.2
           "punpcklqdq %%xmm1,%%xmm7\n"    // a01._1_0
           "punpcklqdq %%xmm2,%%xmm6\n"    // a10._1_0
           "and    $0x3fff,%%esi\n"
           "shr    $0x12,%%edi\n"

           // "movaps %%xmm6,%[tmp1]\n"    // a10.0


           // "movaps %[tmp2],%%xmm7\n"        // a11.0
           // "movhpd %[tmp4],%%xmm7\n"        // a11._1_0

           "movd   0x0(%%eax,%%esi,4),%%xmm3\n"    // a01.2
           "movd   (%%ecx,%%esi,4),%%xmm2\n"    // a11.2
           "mov    0x18(%%edx),%%esi\n"
           "movd   0x0(%%eax,%%edi,4),%%xmm1\n"    // a00.2
           "mov    %%esi,%%eax\n"
           "shr    $0x12,%%esi\n"
           "and    $0x3fff,%%eax\n"
           "movd   (%%ecx,%%edi,4),%%xmm0\n"    // a10.2

           "imul   %[rb],%%eax\n"
           "imul   %[rb],%%esi\n"

           "movdqa %%xmm2,%[tmp6]\n"        // a11.2
           "movdqa %%xmm0,%[tmp5]\n"        // a10.2


           "mov    0x1c(%%edx),%%ecx\n"
           "mov    %%ecx,%%edi\n"
           "add    %[srcAddr],%%esi\n"            // row0.3
           "and    $0x3fff,%%ecx\n"        // x1.3
           "shr    $0x12,%%edi\n"            // x0.3
           "add    %[srcAddr],%%eax\n"    // row1.3

           "movd   (%%esi,%%ecx,4),%%xmm2\n"    // a01.3
           "movd   (%%esi,%%edi,4),%%xmm0\n"    // a00.3

           // "movdqa %%xmm2,%[tmp7]\n"        // a01.3 save
           "punpcklqdq %%xmm2,%%xmm3\n"        // a01._3_2
           "punpcklqdq %%xmm0,%%xmm1\n"        // a00._3_2

           // "movd   0x0(%%eax,%%edi,4),%%xmm2\n"    // a10.3
           // "movdqa %%xmm2,%[tmp8]\n"        // 10.3
           "mov 0x0(%%eax,%%edi,4), %%esi\n"  // a10.3
           "mov %%esi, %[tmp8]\n"             // a10.3

           "movd   0x0(%%eax,%%ecx,4),%%xmm2\n"    // a11.3

           "shufps $0x88,%%xmm1,%%xmm4\n"        // a00.3210
           "movdqa %[tmp6],%%xmm1\n"        // a11.2

           "punpcklqdq %%xmm2,%%xmm1\n"        // a11._3_2

           "shufps $0x88,%%xmm1,%%xmm7\n"        // a11.3210

           "movdqa (%%edx),%%xmm1\n"        // vy_d
           // "movhpd %[tmp7],%%xmm3\n"        // a01._3_2

           "movaps %%xmm1,%%xmm2\n"        // vy_d
           "shufps $0xdd,0x10(%%edx),%%xmm1\n"    // vx
           "shufps $0x88,0x10(%%edx),%%xmm2\n"    // vy

           "psrld  $0xe,%%xmm1\n"        // vx>>14
           "shufps $0x88,%%xmm3,%%xmm5\n"    // a01.3210
           "psrld  $0xe,%%xmm2\n"        // vy>>14

           // "movaps %[tmp1],%%xmm6\n"        // a10._1_0
           "add    $0x20,%%edx\n"        // xy+=8
           "movdqa %[tmp5],%%xmm3\n"        // a10.2
           // "movhpd %[tmp3],%%xmm6\n"        // a10._1_0
           "movhpd %[tmp8],%%xmm3\n"        // a10._3_2

           "pand   vf,%%xmm1\n"        // vx & vf
           "pand   vf,%%xmm2\n"        // vy & vf

           "shufps $0x88,%%xmm3,%%xmm6\n"    // a10.3210

           "pshuflw $0xa0,%%xmm1,%%xmm3\n"
           "pshuflw $0xa0,%%xmm2,%%xmm0\n"
           "pshufhw $0xa0,%%xmm3,%%xmm1\n"
           "pshufhw $0xa0,%%xmm0,%%xmm2\n"

           "movdqa %%xmm1,%%xmm3\n"
           "pmullw %%xmm2,%%xmm3\n"        // vxy

           "psllw  $0x4,%%xmm2\n"        // v16y
           "movdqa v256,%%xmm0\n"
           "psllw  $0x4,%%xmm1\n"        // v16x

           "paddw  %%xmm3,%%xmm0\n"        // v256+vxy
           "psubw  %%xmm2,%%xmm0\n"        // v256+vxy-v16y
           "psubw  %%xmm3,%%xmm2\n"        // v16y-vxy    vscale 2

           "movaps %%xmm4,%[tmp9]\n"        // a00
           "psubw  %%xmm1,%%xmm0\n"        // v256+vxy-v16y-v16x   vscale0

           "movaps %%xmm5,%[tmpa]\n"        // a01

           "psubw  %%xmm3,%%xmm1\n"        // v16x-vxy   vscale 1
           "pand   vmask,%%xmm4\n"        // a00 & vmask
           "pmullw %%xmm0,%%xmm4\n"        // a00.l*vscale0\n
           "pand   vmask,%%xmm5\n"        // a01 & vmask
           "pmullw %%xmm1,%%xmm5\n"        // a01.l*vscale1\n

           "paddw  %%xmm5,%%xmm4\n"        // a00.l+a01.l
           "movaps %%xmm6,%%xmm5\n"        // a10
           "pand   vmask,%%xmm5\n"        // a10.l
           "psrld  $0x8,%%xmm6\n"        // a10.h
           "pmullw %%xmm2,%%xmm5\n"        // a10.l*vscale2
           "paddw  %%xmm5,%%xmm4\n"        // a00.l+a01.l+a10.l
           "movaps %%xmm7,%%xmm5\n"        // a11
           "pand   vmask,%%xmm5\n"        // a11.l
           "psrld  $0x8,%%xmm7\n"        // a11.h
           "pmullw %%xmm3,%%xmm5\n"        // a11.l*vxy
           "paddw  %%xmm5,%%xmm4\n"        // a00.l+a01.l+a10.l+a11.l
           "movaps %[tmp9],%%xmm5\n"        // a00
           "psrld  $0x8,%%xmm4\n"        // ax.l>>8
           "psrld  $0x8,%%xmm5\n"        // a00.h
           "pand   vmask,%%xmm4\n"        // ax.l & vmask
           "pand   vmask,%%xmm5\n"        // a00.h
           "pmullw %%xmm0,%%xmm5\n"        // a00.h*vscale0
           "movaps %[tmpa],%%xmm0\n"        // a01
           "psrld  $0x8,%%xmm0\n"        // a01.h
           "mov    %[count], %%edi\n"
           "pand   vmask,%%xmm0\n"        // a01.h
           "pmullw %%xmm1,%%xmm0\n"        // a01.h * vscale1
           "pand   vmask,%%xmm6\n"        // a10.h
           "add    $0xfffffffc,%%edi\n"
           "pmullw %%xmm2,%%xmm6\n"        // a10.h * vscale2
           "pand   vmask,%%xmm7\n"        // a11.h
           "paddw  %%xmm0,%%xmm5\n"        // a00.h+a01.h
           "pmullw %%xmm3,%%xmm7\n"        // a11.h * vscale2
           // "movaps %[tmpb], %%xmm3\n"
           "paddw  %%xmm6,%%xmm5\n"        // a00.h+a01.h+a10.h
           "pmullw %[tmpb], %%xmm4\n"
           "mov    %[colors],%%ecx\n"        // colors
           "paddw  %%xmm7,%%xmm5\n"        // a00.h+a01.h+a10.h+a11.h
           "psrld  $0x8,%%xmm5\n"        // ax.l>>8
           "pand   vmask,%%xmm5\n"        // ax.l & vmask
           "pmullw %[tmpb], %%xmm5\n"
           "psrld  $0x8,%%xmm4\n"        // ax.l>>8
           "pand   vmask,%%xmm4\n"        // ax.l & vmask
           "pand   vmask2,%%xmm5\n"        // ax.h & vmask2
           "addl   $0x10,%[colors]\n"        // colors+=4
           "por    %%xmm5,%%xmm4\n"        // ax.l|ax.h
           "movdqu %%xmm4,(%%ecx)\n"        // store colors
           "cmp    $0x4,%%edi\n"
           "mov    %%edi,%[count]\n"
           "jge    1b\n"
           :"+d" (xy)
           :[srcAddr] "m" (srcAddr), [colors] "m" (colors), [rb] "m" (rb),
           [count] "m" (count),[alphaScale] "m" (alphaScale),
           [tmp1] "m" (tmp1), [tmp2] "m" (tmp2), [tmp3] "m" (tmp3), [tmp4] "m" (tmp4),
           [tmp5] "m" (tmp5), [tmp6] "m" (tmp6), [tmp7] "m" (tmp7), [tmp8] "m" (tmp4),
           [tmp9] "m" (tmp9), [tmpa] "m" (tmpa), [tmpb] "m" (tmpb)
           :"memory","ecx","esi","edi", "eax"
           );
        } // count >= 4
    }
    while (count > 0)
    {
        data = *xy++;
        y0 = data >> 14;
        y1 = data & 0x3FFF;
        subY = y0 & 0xF;
        y0 >>= 4;

        data = *xy++;
        x0 = data >> 14;
        x1 = data & 0x3FFF;
        subX = x0 & 0xF;
        x0 >>= 4;

        row0 = (const SkPMColor*)(srcAddr + y0 * rb);
        row1 = (const SkPMColor*)(srcAddr + y1 * rb);

        Filter_32_alpha(subX, subY,
                       (row0[x0]),
                       (row0[x1]),
                       (row1[x0]),
                       (row1[x1]),
                       colors,
                       s.fAlphaScale);
        colors += 1;
        count --;
    }
}
