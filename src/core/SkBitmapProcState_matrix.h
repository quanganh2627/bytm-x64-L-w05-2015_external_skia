
/*
 * Copyright 2011 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "SkMath.h"
#include "SkMathPriv.h"

#define SCALE_NOFILTER_NAME     MAKENAME(_nofilter_scale)
#define SCALE_FILTER_NAME       MAKENAME(_filter_scale)
#define AFFINE_NOFILTER_NAME    MAKENAME(_nofilter_affine)
#define AFFINE_FILTER_NAME      MAKENAME(_filter_affine)
#define PERSP_NOFILTER_NAME     MAKENAME(_nofilter_persp)
#define PERSP_FILTER_NAME       MAKENAME(_filter_persp)

#define PACK_FILTER_X_NAME  MAKENAME(_pack_filter_x)
#define PACK_FILTER_Y_NAME  MAKENAME(_pack_filter_y)

#ifndef PREAMBLE
    #define PREAMBLE(state)
    #define PREAMBLE_PARAM_X
    #define PREAMBLE_PARAM_Y
    #define PREAMBLE_ARG_X
    #define PREAMBLE_ARG_Y
#endif

// declare functions externally to suppress warnings.
void SCALE_NOFILTER_NAME(const SkBitmapProcState& s,
                                uint32_t xy[], int count, int x, int y);
void AFFINE_NOFILTER_NAME(const SkBitmapProcState& s,
                                 uint32_t xy[], int count, int x, int y);
void PERSP_NOFILTER_NAME(const SkBitmapProcState& s,
                                uint32_t* SK_RESTRICT xy,
                                int count, int x, int y);
void SCALE_FILTER_NAME(const SkBitmapProcState& s,
                              uint32_t xy[], int count, int x, int y);
void AFFINE_FILTER_NAME(const SkBitmapProcState& s,
                               uint32_t xy[], int count, int x, int y);
void PERSP_FILTER_NAME(const SkBitmapProcState& s,
                              uint32_t* SK_RESTRICT xy, int count,
                              int x, int y);

void SCALE_NOFILTER_NAME(const SkBitmapProcState& s,
                                uint32_t xy[], int count, int x, int y) {
    SkASSERT((s.fInvType & ~(SkMatrix::kTranslate_Mask |
                             SkMatrix::kScale_Mask)) == 0);

    PREAMBLE(s);
    // we store y, x, x, x, x, x

    const unsigned maxX = s.fBitmap->width() - 1;
    SkFractionalInt fx;
    {
        SkPoint pt;
        s.fInvProc(s.fInvMatrix, SkIntToScalar(x) + SK_ScalarHalf,
                                  SkIntToScalar(y) + SK_ScalarHalf, &pt);
        fx = SkScalarToFractionalInt(pt.fY);
        const unsigned maxY = s.fBitmap->height() - 1;
        *xy++ = TILEY_PROCF(SkFractionalIntToFixed(fx), maxY);
        fx = SkScalarToFractionalInt(pt.fX);
    }

    if (0 == maxX) {
        // all of the following X values must be 0
        memset(xy, 0, count * sizeof(uint16_t));
        return;
    }

    const SkFractionalInt dx = s.fInvSxFractionalInt;

#ifdef CHECK_FOR_DECAL
    if (can_truncate_to_fixed_for_decal(fx, dx, count, maxX)) {
        decal_nofilter_scale(xy, SkFractionalIntToFixed(fx),
                             SkFractionalIntToFixed(dx), count);
    } else
#endif
    {
        int i;
        for (i = (count >> 2); i > 0; --i) {
            unsigned a, b;
            a = TILEX_PROCF(SkFractionalIntToFixed(fx), maxX); fx += dx;
            b = TILEX_PROCF(SkFractionalIntToFixed(fx), maxX); fx += dx;
#ifdef SK_CPU_BENDIAN
            *xy++ = (a << 16) | b;
#else
            *xy++ = (b << 16) | a;
#endif
            a = TILEX_PROCF(SkFractionalIntToFixed(fx), maxX); fx += dx;
            b = TILEX_PROCF(SkFractionalIntToFixed(fx), maxX); fx += dx;
#ifdef SK_CPU_BENDIAN
            *xy++ = (a << 16) | b;
#else
            *xy++ = (b << 16) | a;
#endif
        }
        uint16_t* xx = (uint16_t*)xy;
        for (i = (count & 3); i > 0; --i) {
            *xx++ = TILEX_PROCF(SkFractionalIntToFixed(fx), maxX); fx += dx;
        }
    }
}

// note: we could special-case on a matrix which is skewed in X but not Y.
// this would require a more general setup thatn SCALE does, but could use
// SCALE's inner loop that only looks at dx

void AFFINE_NOFILTER_NAME(const SkBitmapProcState& s,
                                 uint32_t xy[], int count, int x, int y) {
    SkASSERT(s.fInvType & SkMatrix::kAffine_Mask);
    SkASSERT((s.fInvType & ~(SkMatrix::kTranslate_Mask |
                             SkMatrix::kScale_Mask |
                             SkMatrix::kAffine_Mask)) == 0);

    PREAMBLE(s);
    SkPoint srcPt;
    s.fInvProc(s.fInvMatrix,
               SkIntToScalar(x) + SK_ScalarHalf,
               SkIntToScalar(y) + SK_ScalarHalf, &srcPt);

    SkFractionalInt fx = SkScalarToFractionalInt(srcPt.fX);
    SkFractionalInt fy = SkScalarToFractionalInt(srcPt.fY);
    SkFractionalInt dx = s.fInvSxFractionalInt;
    SkFractionalInt dy = s.fInvKyFractionalInt;
    int maxX = s.fBitmap->width() - 1;
    int maxY = s.fBitmap->height() - 1;

    for (int i = count; i > 0; --i) {
        *xy++ = (TILEY_PROCF(SkFractionalIntToFixed(fy), maxY) << 16) |
                 TILEX_PROCF(SkFractionalIntToFixed(fx), maxX);
        fx += dx; fy += dy;
    }
}

void PERSP_NOFILTER_NAME(const SkBitmapProcState& s,
                                uint32_t* SK_RESTRICT xy,
                                int count, int x, int y) {
    SkASSERT(s.fInvType & SkMatrix::kPerspective_Mask);

    PREAMBLE(s);
    int maxX = s.fBitmap->width() - 1;
    int maxY = s.fBitmap->height() - 1;

    SkPerspIter   iter(s.fInvMatrix,
                       SkIntToScalar(x) + SK_ScalarHalf,
                       SkIntToScalar(y) + SK_ScalarHalf, count);

    while ((count = iter.next()) != 0) {
        const SkFixed* SK_RESTRICT srcXY = iter.getXY();
        while (--count >= 0) {
            *xy++ = (TILEY_PROCF(srcXY[1], maxY) << 16) |
                     TILEX_PROCF(srcXY[0], maxX);
            srcXY += 2;
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

static inline uint32_t PACK_FILTER_Y_NAME(SkFixed f, unsigned max,
                                          SkFixed one PREAMBLE_PARAM_Y) {
    unsigned i = TILEY_PROCF(f, max);
    i = (i << 4) | TILEY_LOW_BITS(f, max);
    return (i << 14) | (TILEY_PROCF((f + one), max));
}

static inline uint32_t PACK_FILTER_X_NAME(SkFixed f, unsigned max,
                                          SkFixed one PREAMBLE_PARAM_X) {
    unsigned i = TILEX_PROCF(f, max);
    i = (i << 4) | TILEX_LOW_BITS(f, max);
    return (i << 14) | (TILEX_PROCF((f + one), max));
}

void SCALE_FILTER_NAME(const SkBitmapProcState& s,
                              uint32_t xy[], int count, int x, int y) {
    SkASSERT((s.fInvType & ~(SkMatrix::kTranslate_Mask |
                             SkMatrix::kScale_Mask)) == 0);
    SkASSERT(s.fInvKy == 0);

    PREAMBLE(s);

    const unsigned maxX = s.fBitmap->width() - 1;
    const SkFixed one = s.fFilterOneX;
    const SkFractionalInt dx = s.fInvSxFractionalInt;
    SkFractionalInt fx;

    {
        SkPoint pt;
        s.fInvProc(s.fInvMatrix, SkIntToScalar(x) + SK_ScalarHalf,
                                  SkIntToScalar(y) + SK_ScalarHalf, &pt);
        const SkFixed fy = SkScalarToFixed(pt.fY) - (s.fFilterOneY >> 1);
        const unsigned maxY = s.fBitmap->height() - 1;
        // compute our two Y values up front
        *xy++ = PACK_FILTER_Y_NAME(fy, maxY, s.fFilterOneY PREAMBLE_ARG_Y);
        // now initialize fx
        fx = SkScalarToFractionalInt(pt.fX) - (SkFixedToFractionalInt(one) >> 1);
    }

#ifdef CHECK_FOR_DECAL
    if (can_truncate_to_fixed_for_decal(fx, dx, count, maxX)) {
        decal_filter_scale(xy, SkFractionalIntToFixed(fx),
                           SkFractionalIntToFixed(dx), count);
    } else
#endif
    {
#if SK_CPU_SSE_LEVEL >= SK_CPU_SSE_LEVEL_SSE2
#ifdef ClampX_ClampY_filter_scale_SSE2_with_shader
    if (count >= 4) {
        float maxX_float = (float)maxX;
        __m128i _m_fx, _m_dx, _m_one, _m_i, _m_F, _m_temp1, _m_temp2, _m_temp3;
        __m128 _m_tmp1, _m_tmp2, _m_maxX;
        _m_maxX = _mm_set_ps(maxX_float, maxX_float, maxX_float, maxX_float);
        _m_dx  = _mm_set1_epi32(dx * 4);
        _m_F = _mm_set1_epi32(0xF);
        _m_one = _mm_set1_epi32(one);
        _m_fx  = _mm_set_epi32(fx + dx * 3, fx + dx *2, fx + dx, fx);
        __m128 zero = _mm_setzero_ps();
        fx += ((count >>2) <<2) * dx;
        do {
            _m_temp1 = _mm_srai_epi32(_m_fx, 16);
            _m_tmp1 = _mm_cvtepi32_ps(_m_temp1);
            _m_tmp2 = _mm_max_ps(_m_tmp1, zero);
            _m_tmp1 = _mm_min_ps(_m_tmp2, _m_maxX);
            _m_i = _mm_cvtps_epi32(_m_tmp1);
            _m_temp1 = _mm_slli_epi32(_m_i, 4);
            _m_temp2 = _mm_srai_epi32(_m_fx, 12);
            _m_i = _mm_and_si128(_m_temp2, _m_F);
            _m_temp2 = _mm_or_si128(_m_temp1, _m_i);
            _m_temp1 = _mm_slli_epi32(_m_temp2, 14);
            _m_temp2 = _mm_add_epi32(_m_fx, _m_one);
            _m_temp3 = _mm_srai_epi32(_m_temp2, 16);
            _m_tmp2 = _mm_cvtepi32_ps(_m_temp3);
            _m_tmp1 = _mm_max_ps(_m_tmp2, zero);
            _m_tmp2 = _mm_min_ps(_m_tmp1, _m_maxX);
            _m_i = _mm_cvtps_epi32(_m_tmp2);
            _m_temp2 = _mm_or_si128(_m_temp1, _m_i);
            _mm_storeu_si128((__m128i *)xy, _m_temp2);
            _m_fx    = _mm_add_epi32(_m_fx, _m_dx);
            xy = xy + 4;
            count = count - 4;
        } while (count >= 4);
    }
    if (count == 0)
        return;
#endif
#endif
        do {
            SkFixed fixedFx = SkFractionalIntToFixed(fx);
            *xy++ = PACK_FILTER_X_NAME(fixedFx, maxX, one PREAMBLE_ARG_X);
            fx += dx;
        } while (--count != 0);
    }
}

void AFFINE_FILTER_NAME(const SkBitmapProcState& s,
                               uint32_t xy[], int count, int x, int y) {
    SkASSERT(s.fInvType & SkMatrix::kAffine_Mask);
    SkASSERT((s.fInvType & ~(SkMatrix::kTranslate_Mask |
                             SkMatrix::kScale_Mask |
                             SkMatrix::kAffine_Mask)) == 0);

    PREAMBLE(s);
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

    do {
        *xy++ = PACK_FILTER_Y_NAME(fy, maxY, oneY PREAMBLE_ARG_Y);
        fy += dy;
        *xy++ = PACK_FILTER_X_NAME(fx, maxX, oneX PREAMBLE_ARG_X);
        fx += dx;
    } while (--count != 0);
}

void PERSP_FILTER_NAME(const SkBitmapProcState& s,
                              uint32_t* SK_RESTRICT xy, int count,
                              int x, int y) {
    SkASSERT(s.fInvType & SkMatrix::kPerspective_Mask);

    PREAMBLE(s);
    unsigned maxX = s.fBitmap->width() - 1;
    unsigned maxY = s.fBitmap->height() - 1;
    SkFixed oneX = s.fFilterOneX;
    SkFixed oneY = s.fFilterOneY;

    SkPerspIter   iter(s.fInvMatrix,
                       SkIntToScalar(x) + SK_ScalarHalf,
                       SkIntToScalar(y) + SK_ScalarHalf, count);

    while ((count = iter.next()) != 0) {
        const SkFixed* SK_RESTRICT srcXY = iter.getXY();
        do {
            *xy++ = PACK_FILTER_Y_NAME(srcXY[1] - (oneY >> 1), maxY,
                                       oneY PREAMBLE_ARG_Y);
            *xy++ = PACK_FILTER_X_NAME(srcXY[0] - (oneX >> 1), maxX,
                                       oneX PREAMBLE_ARG_X);
            srcXY += 2;
        } while (--count != 0);
    }
}

static SkBitmapProcState::MatrixProc MAKENAME(_Procs)[] = {
    SCALE_NOFILTER_NAME,
    SCALE_FILTER_NAME,
    AFFINE_NOFILTER_NAME,
    AFFINE_FILTER_NAME,
    PERSP_NOFILTER_NAME,
    PERSP_FILTER_NAME
};

#undef MAKENAME
#undef TILEX_PROCF
#undef TILEY_PROCF
#ifdef CHECK_FOR_DECAL
    #undef CHECK_FOR_DECAL
#endif

#undef SCALE_NOFILTER_NAME
#undef SCALE_FILTER_NAME
#undef AFFINE_NOFILTER_NAME
#undef AFFINE_FILTER_NAME
#undef PERSP_NOFILTER_NAME
#undef PERSP_FILTER_NAME

#undef PREAMBLE
#undef PREAMBLE_PARAM_X
#undef PREAMBLE_PARAM_Y
#undef PREAMBLE_ARG_X
#undef PREAMBLE_ARG_Y

#undef TILEX_LOW_BITS
#undef TILEY_LOW_BITS
