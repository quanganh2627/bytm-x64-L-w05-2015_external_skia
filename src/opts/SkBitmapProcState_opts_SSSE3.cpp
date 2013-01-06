/*
 * Copyright 2012 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include <tmmintrin.h>  // SSSE3
#include "SkBitmapProcState_opts_SSSE3.h"
#include "SkUtils.h"

// adding anonymous namespace seemed to force gcc to inline directly the
// instantiation, instead of creating the functions
// S32_generic_D32_filter_DX_SSSE3<true> and
// S32_generic_D32_filter_DX_SSSE3<false> which were then called by the
// external functions.
namespace {
// In this file, variations for alpha and non alpha versions are implemented
// with a template, as it makes the code more compact and a bit easier to
// maintain, while making the compiler generate the same exact code as with
// two functions that only differ by a few lines.


// Prepare all necessary constants for a round of processing for two pixel
// pairs.
// @param xy is the location where the xy parameters for four pixels should be
//           read from. It is identical in concept with argument two of
//           S32_{opaque}_D32_filter_DX methods.
// @param mask_3FFF vector of 32 bit constants containing 3FFF,
//                  suitable to mask the bottom 14 bits of a XY value.
// @param mask_000F vector of 32 bit constants containing 000F,
//                  suitable to mask the bottom 4 bits of a XY value.
// @param sixteen_8bit vector of 8 bit components containing the value 16.
// @param mask_dist_select vector of 8 bit components containing the shuffling
//                         parameters to reorder x[0-3] parameters.
// @param all_x_result vector of 8 bit components that will contain the
//              (4x(x3), 4x(x2), 4x(x1), 4x(x0)) upon return.
// @param sixteen_minus_x vector of 8 bit components, containing
//              (4x(16 - x3), 4x(16 - x2), 4x(16 - x1), 4x(16 - x0))
inline void PrepareConstantsTwoPixelPairs(const uint32_t* xy,
                                          const __m128i& mask_3FFF,
                                          const __m128i& mask_000F,
                                          const __m128i& sixteen_8bit,
                                          const __m128i& mask_dist_select,
                                          __m128i* all_x_result,
                                          __m128i* sixteen_minus_x,
                                          int* x0,
                                          int* x1) {
    const __m128i xx = _mm_loadu_si128(reinterpret_cast<const __m128i *>(xy));

    // 4 delta X
    // (x03, x02, x01, x00)
    const __m128i x0_wide = _mm_srli_epi32(xx, 18);
    // (x13, x12, x11, x10)
    const __m128i x1_wide = _mm_and_si128(xx, mask_3FFF);

    _mm_storeu_si128(reinterpret_cast<__m128i *>(x0), x0_wide);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(x1), x1_wide);

    __m128i all_x = _mm_and_si128(_mm_srli_epi32(xx, 14), mask_000F);

    // (4x(x3), 4x(x2), 4x(x1), 4x(x0))
    all_x = _mm_shuffle_epi8(all_x, mask_dist_select);

    *all_x_result = all_x;
    // (4x(16-x3), 4x(16-x2), 4x(16-x1), 4x(16-x0))
    *sixteen_minus_x = _mm_sub_epi8(sixteen_8bit, all_x);
}

// Helper function used when processing one pixel pair.
// @param pixel0..3 are the four input pixels
// @param scale_x vector of 8 bit components to multiply the pixel[0:3]. This
//                will contain (4x(x1, 16-x1), 4x(x0, 16-x0))
//                or (4x(x3, 16-x3), 4x(x2, 16-x2))
// @return a vector of 16 bit components containing:
// (Aa2 * (16 - x1) + Aa3 * x1, ... , Ra0 * (16 - x0) + Ra1 * x0)
inline __m128i ProcessPixelPairHelper(uint32_t pixel0,
                                      uint32_t pixel1,
                                      uint32_t pixel2,
                                      uint32_t pixel3,
                                      const __m128i& scale_x) {
    __m128i a0, a1, a2, a3;
    // Load 2 pairs of pixels
    a0 = _mm_cvtsi32_si128(pixel0);
    a1 = _mm_cvtsi32_si128(pixel1);

    // Interleave pixels.
    // (0, 0, 0, 0, 0, 0, 0, 0, Aa1, Aa0, Ba1, Ba0, Ga1, Ga0, Ra1, Ra0)
    a0 = _mm_unpacklo_epi8(a0, a1);

    a2 = _mm_cvtsi32_si128(pixel2);
    a3 = _mm_cvtsi32_si128(pixel3);
    // (0, 0, 0, 0, 0, 0, 0, 0, Aa3, Aa2, Ba3, Ba2, Ga3, Ga2, Ra3, Ra2)
    a2 = _mm_unpacklo_epi8(a2, a3);

    // two pairs of pixel pairs, interleaved.
    // (Aa3, Aa2, Ba3, Ba2, Ga3, Ga2, Ra3, Ra2,
    //  Aa1, Aa0, Ba1, Ba0, Ga1, Ga0, Ra1, Ra0)
    a0 = _mm_unpacklo_epi64(a0, a2);

    // multiply and sum to 16 bit components.
    // (Aa2 * (16 - x1) + Aa3 * x1, ... , Ra0 * (16 - x0) + Ra1 * x0)
    // At that point, we use up a bit less than 12 bits for each 16 bit
    // component:
    // All components are less than 255. So,
    // C0 * (16 - x) + C1 * x <= 255 * (16 - x) + 255 * x = 255 * 16.
    return _mm_maddubs_epi16(a0, scale_x);
}

// Scale back the results after multiplications to the [0:255] range, and scale
// by alpha when has_alpha is true.
// Depending on whether one set or two sets of multiplications had been applied,
// the results have to be shifted by four places (dividing by 16), or shifted
// by eight places (dividing by 256), since each multiplication is by a quantity
// in the range [0:16].
template<bool has_alpha, int scale>
inline __m128i ScaleFourPixels(__m128i* pixels,
                               const __m128i& alpha) {
    // Divide each 16 bit component by 16 (or 256 depending on scale).
    *pixels = _mm_srli_epi16(*pixels, scale);

    if (has_alpha) {
        // Multiply by alpha.
        *pixels = _mm_mullo_epi16(*pixels, alpha);

        // Divide each 16 bit component by 256.
        *pixels = _mm_srli_epi16(*pixels, 8);
    }
    return *pixels;
}

// Wrapper to calculate two output pixels from four input pixels. The
// arguments are the same as ProcessPixelPairHelper. Technically, there are
// eight input pixels, but since sub_y == 0, the factors applied to half of the
// pixels is zero (sub_y), and are therefore omitted here to save on some
// processing.
// @param alpha when has_alpha is true, scale all resulting components by this
//              value.
// @return a vector of 16 bit components containing:
// ((Aa2 * (16 - x1) + Aa3 * x1) * alpha, ...,
// (Ra0 * (16 - x0) + Ra1 * x0) * alpha) (when has_alpha is true)
// otherwise
// (Aa2 * (16 - x1) + Aa3 * x1, ... , Ra0 * (16 - x0) + Ra1 * x0)
// In both cases, the results are renormalized (divided by 16) to match the
// expected formats when storing back the results into memory.
template<bool has_alpha>
inline __m128i ProcessPixelPairZeroSubY(uint32_t pixel0,
                                        uint32_t pixel1,
                                        uint32_t pixel2,
                                        uint32_t pixel3,
                                        const __m128i& scale_x,
                                        const __m128i& alpha) {
    __m128i sum = ProcessPixelPairHelper(pixel0, pixel1, pixel2, pixel3,
                                         scale_x);
    return ScaleFourPixels<has_alpha, 4>(&sum, alpha);
}

// Same as ProcessPixelPairZeroSubY, expect processing one output pixel at a
// time instead of two. As in the above function, only two pixels are needed
// to generate a single pixel since sub_y == 0.
// @return same as ProcessPixelPairZeroSubY, except that only the bottom 4
// 16 bit components are set.
template<bool has_alpha>
inline __m128i ProcessOnePixelZeroSubY(uint32_t pixel0,
                                       uint32_t pixel1,
                                       __m128i scale_x,
                                       __m128i alpha) {
    __m128i a0 = _mm_cvtsi32_si128(pixel0);
    __m128i a1 = _mm_cvtsi32_si128(pixel1);

    // Interleave
    a0 = _mm_unpacklo_epi8(a0, a1);

    // (a0 * (16-x) + a1 * x)
    __m128i sum = _mm_maddubs_epi16(a0, scale_x);

    return ScaleFourPixels<has_alpha, 4>(&sum, alpha);
}

// Methods when sub_y != 0


// Same as ProcessPixelPairHelper, except that the values are scaled by y.
// @param y vector of 16 bit components containing 'y' values. There are two
//        cases in practice, where y will contain the sub_y constant, or will
//        contain the 16 - sub_y constant.
// @return vector of 16 bit components containing:
// (y * (Aa2 * (16 - x1) + Aa3 * x1), ... , y * (Ra0 * (16 - x0) + Ra1 * x0))
inline __m128i ProcessPixelPair(uint32_t pixel0,
                                uint32_t pixel1,
                                uint32_t pixel2,
                                uint32_t pixel3,
                                const __m128i& scale_x,
                                const __m128i& y) {
    __m128i sum = ProcessPixelPairHelper(pixel0, pixel1, pixel2, pixel3,
                                         scale_x);

    // first row times 16-y or y depending on whether 'y' represents one or
    // the other.
    // Values will be up to 255 * 16 * 16 = 65280.
    // (y * (Aa2 * (16 - x1) + Aa3 * x1), ... ,
    //  y * (Ra0 * (16 - x0) + Ra1 * x0))
    sum = _mm_mullo_epi16(sum, y);

    return sum;
}

// Process two pixel pairs out of eight input pixels.
// In other methods, the distinct pixels are passed one by one, but in this
// case, the rows, and index offsets to the pixels into the row are passed
// to generate the 8 pixels.
// @param row0..1 top and bottom row where to find input pixels.
// @param x0..1 offsets into the row for all eight input pixels.
// @param all_y vector of 16 bit components containing the constant sub_y
// @param neg_y vector of 16 bit components containing the constant 16 - sub_y
// @param alpha vector of 16 bit components containing the alpha value to scale
//        the results by, when has_alpha is true.
// @return
// (alpha * ((16-y) * (Aa2  * (16-x1) + Aa3  * x1) +
//             y    * (Aa2' * (16-x1) + Aa3' * x1)),
// ...
//  alpha * ((16-y) * (Ra0  * (16-x0) + Ra1 * x0) +
//             y    * (Ra0' * (16-x0) + Ra1' * x0))
// With the factor alpha removed when has_alpha is false.
// The values are scaled back to 16 bit components, but with only the bottom
// 8 bits being set.
template<bool has_alpha>
inline __m128i ProcessTwoPixelPairs(const uint32_t* row0,
                                    const uint32_t* row1,
                                    const int* x0,
                                    const int* x1,
                                    const __m128i& scale_x,
                                    const __m128i& all_y,
                                    const __m128i& neg_y,
                                    const __m128i& alpha) {
    __m128i sum0 = ProcessPixelPair(
        row0[x0[0]], row0[x1[0]], row0[x0[1]], row0[x1[1]],
        scale_x, neg_y);
    __m128i sum1 = ProcessPixelPair(
        row1[x0[0]], row1[x1[0]], row1[x0[1]], row1[x1[1]],
        scale_x, all_y);

    // 2 samples fully summed.
    // ((16-y) * (Aa2 * (16-x1) + Aa3 * x1) +
    //  y * (Aa2' * (16-x1) + Aa3' * x1),
    // ...
    //  (16-y) * (Ra0 * (16 - x0) + Ra1 * x0)) +
    //  y * (Ra0' * (16-x0) + Ra1' * x0))
    // Each component, again can be at most 256 * 255 = 65280, so no overflow.
    sum0 = _mm_add_epi16(sum0, sum1);

    return ScaleFourPixels<has_alpha, 8>(&sum0, alpha);
}


// Same as ProcessPixelPair, except that performing the math one output pixel
// at a time. This means that only the bottom four 16 bit components are set.
inline __m128i ProcessOnePixel(uint32_t pixel0, uint32_t pixel1,
                               const __m128i& scale_x, const __m128i& y) {
    __m128i a0 = _mm_cvtsi32_si128(pixel0);
    __m128i a1 = _mm_cvtsi32_si128(pixel1);

    // Interleave
    // (0, 0, 0, 0, 0, 0, 0, 0, Aa1, Aa0, Ba1, Ba0, Ga1, Ga0, Ra1, Ra0)
    a0 = _mm_unpacklo_epi8(a0, a1);

    // (a0 * (16-x) + a1 * x)
    a0 = _mm_maddubs_epi16(a0, scale_x);

    // scale row by y
    return _mm_mullo_epi16(a0, y);
}

// Notes about the various tricks that are used in this implementation:
// - specialization for sub_y == 0.
// Statistically, 1/16th of the samples will have sub_y == 0. When this
// happens, the math goes from:
// (16 - x)*(16 - y)*a00 + x*(16 - y)*a01 + (16 - x)*y*a10 + x*y*a11
// to:
// (16 - x)*a00 + 16*x*a01
// much simpler. The simplification makes for an easy boost in performance.
// - calculating 4 output pixels at a time.
//  This allows loading the coefficients x0 and x1 and shuffling them to the
// optimum location only once per loop, instead of twice per loop.
// This also allows us to store the four pixels with a single store.
// - Use of 2 special SSSE3 instructions (comparatively to the SSE2 instruction
// version):
// _mm_shuffle_epi8 : this allows us to spread the coefficients x[0-3] loaded
// in 32 bit values to 8 bit values repeated four times.
// _mm_maddubs_epi16 : this allows us to perform multiplications and additions
// in one swoop of 8bit values storing the results in 16 bit values. This
// instruction is actually crucial for the speed of the implementation since
// as one can see in the SSE2 implementation, all inputs have to be used as
// 16 bits because the results are 16 bits. This basically allows us to process
// twice as many pixel components per iteration.
//
// As a result, this method behaves faster than the traditional SSE2. The actual
// boost varies greatly on the underlying architecture.
template<bool has_alpha>
void S32_generic_D32_filter_DX_SSSE3(const SkBitmapProcState& s,
                                     const uint32_t* xy,
                                     int count, uint32_t* colors) {
    SkASSERT(count > 0 && colors != NULL);
    SkASSERT(s.fDoFilter);
    SkASSERT(s.fBitmap->config() == SkBitmap::kARGB_8888_Config);
    if (has_alpha) {
        SkASSERT(s.fAlphaScale < 256);
    } else {
        SkASSERT(s.fAlphaScale == 256);
    }

    const uint8_t* src_addr =
            static_cast<const uint8_t*>(s.fBitmap->getPixels());
    const unsigned rb = s.fBitmap->rowBytes();
    const uint32_t XY = *xy++;
    const unsigned y0 = XY >> 14;
    const uint32_t* row0 =
            reinterpret_cast<const uint32_t*>(src_addr + (y0 >> 4) * rb);
    const uint32_t* row1 =
            reinterpret_cast<const uint32_t*>(src_addr + (XY & 0x3FFF) * rb);
    const unsigned sub_y = y0 & 0xF;

    // vector constants
    const __m128i mask_dist_select = _mm_set_epi8(12, 12, 12, 12,
                                                  8,  8,  8,  8,
                                                  4,  4,  4,  4,
                                                  0,  0,  0,  0);
    const __m128i mask_3FFF = _mm_set1_epi32(0x3FFF);
    const __m128i mask_000F = _mm_set1_epi32(0x000F);
    const __m128i sixteen_8bit = _mm_set1_epi8(16);
    // (0, 0, 0, 0, 0, 0, 0, 0)
    const __m128i zero = _mm_setzero_si128();

    __m128i alpha;
    if (has_alpha)
        // 8x(alpha)
        alpha = _mm_set1_epi16(s.fAlphaScale);

    if (sub_y == 0) {
        // Unroll 4x, interleave bytes, use pmaddubsw (all_x is small)
        while (count > 3) {
            count -= 4;

            int x0[4];
            int x1[4];
            __m128i all_x, sixteen_minus_x;
            PrepareConstantsTwoPixelPairs(xy, mask_3FFF, mask_000F,
                                          sixteen_8bit, mask_dist_select,
                                          &all_x, &sixteen_minus_x, x0, x1);
            xy += 4;

            // First pair of pixel pairs.
            // (4x(x1, 16-x1), 4x(x0, 16-x0))
            __m128i scale_x;
            scale_x = _mm_unpacklo_epi8(sixteen_minus_x, all_x);

            __m128i sum0 = ProcessPixelPairZeroSubY<has_alpha>(
                row0[x0[0]], row0[x1[0]], row0[x0[1]], row0[x1[1]],
                scale_x, alpha);

            // second pair of pixel pairs
            // (4x (x3, 16-x3), 4x (16-x2, x2))
            scale_x = _mm_unpackhi_epi8(sixteen_minus_x, all_x);

            __m128i sum1 = ProcessPixelPairZeroSubY<has_alpha>(
                row0[x0[2]], row0[x1[2]], row0[x0[3]], row0[x1[3]],
                scale_x, alpha);

            // Pack lower 4 16 bit values of sum into lower 4 bytes.
            sum0 = _mm_packus_epi16(sum0, sum1);

            // Extract low int and store.
            _mm_storeu_si128(reinterpret_cast<__m128i *>(colors), sum0);

            colors += 4;
        }

        // handle remainder
        while (count-- > 0) {
            uint32_t xx = *xy++;  // x0:14 | 4 | x1:14
            unsigned x0 = xx >> 18;
            unsigned x1 = xx & 0x3FFF;

            // 16x(x)
            const __m128i all_x = _mm_set1_epi8((xx >> 14) & 0x0F);

            // (16x(16-x))
            __m128i scale_x = _mm_sub_epi8(sixteen_8bit, all_x);

            scale_x = _mm_unpacklo_epi8(scale_x, all_x);

            __m128i sum = ProcessOnePixelZeroSubY<has_alpha>(
                row0[x0], row0[x1],
                scale_x, alpha);

            // Pack lower 4 16 bit values of sum into lower 4 bytes.
            sum = _mm_packus_epi16(sum, zero);

            // Extract low int and store.
            *colors++ = _mm_cvtsi128_si32(sum);
        }
    } else {  // more general case, y != 0
        // 8x(16)
        const __m128i sixteen_16bit = _mm_set1_epi16(16);

        // 8x (y)
        const __m128i all_y = _mm_set1_epi16(sub_y);

        // 8x (16-y)
        const __m128i neg_y = _mm_sub_epi16(sixteen_16bit, all_y);

        // Unroll 4x, interleave bytes, use pmaddubsw (all_x is small)
        while (count > 3) {
            count -= 4;

            int x0[4];
            int x1[4];
            __m128i all_x, sixteen_minus_x;
            PrepareConstantsTwoPixelPairs(xy, mask_3FFF, mask_000F,
                                          sixteen_8bit, mask_dist_select,
                                          &all_x, &sixteen_minus_x, x0, x1);
            xy += 4;

            // First pair of pixel pairs
            // (4x(x1, 16-x1), 4x(x0, 16-x0))
            __m128i scale_x;
            scale_x = _mm_unpacklo_epi8(sixteen_minus_x, all_x);

            __m128i sum0 = ProcessTwoPixelPairs<has_alpha>(
                row0, row1, x0, x1,
                scale_x, all_y, neg_y, alpha);

            // second pair of pixel pairs
            // (4x (x3, 16-x3), 4x (16-x2, x2))
            scale_x = _mm_unpackhi_epi8(sixteen_minus_x, all_x);

            __m128i sum1 = ProcessTwoPixelPairs<has_alpha>(
                row0, row1, x0 + 2, x1 + 2,
                scale_x, all_y, neg_y, alpha);

            // Do the final packing of the two results

            // Pack lower 4 16 bit values of sum into lower 4 bytes.
            sum0 = _mm_packus_epi16(sum0, sum1);

            // Extract low int and store.
            _mm_storeu_si128(reinterpret_cast<__m128i *>(colors), sum0);

            colors += 4;
        }

        // Left over.
        while (count-- > 0) {
            const uint32_t xx = *xy++;  // x0:14 | 4 | x1:14
            const unsigned x0 = xx >> 18;
            const unsigned x1 = xx & 0x3FFF;

            // 16x(x)
            const __m128i all_x = _mm_set1_epi8((xx >> 14) & 0x0F);

            // 16x (16-x)
            __m128i scale_x = _mm_sub_epi8(sixteen_8bit, all_x);

            // (8x (x, 16-x))
            scale_x = _mm_unpacklo_epi8(scale_x, all_x);

            // first row.
            __m128i sum0 = ProcessOnePixel(row0[x0], row0[x1], scale_x, neg_y);
            // second row.
            __m128i sum1 = ProcessOnePixel(row1[x0], row1[x1], scale_x, all_y);

            // Add both rows for full sample
            sum0 = _mm_add_epi16(sum0, sum1);

            sum0 = ScaleFourPixels<has_alpha, 8>(&sum0, alpha);

            // Pack lower 4 16 bit values of sum into lower 4 bytes.
            sum0 = _mm_packus_epi16(sum0, zero);

            // Extract low int and store.
            *colors++ = _mm_cvtsi128_si32(sum0);
        }
    }
}
}  // namepace

void S32_opaque_D32_filter_DX_SSSE3(const SkBitmapProcState& s,
                                    const uint32_t* xy,
                                    int count, uint32_t* colors) {
    S32_generic_D32_filter_DX_SSSE3<false>(s, xy, count, colors);
}

void S32_alpha_D32_filter_DX_SSSE3(const SkBitmapProcState& s,
                                   const uint32_t* xy,
                                   int count, uint32_t* colors) {
    S32_generic_D32_filter_DX_SSSE3<true>(s, xy, count, colors);
}

/*
 * sum  = a00(16-y)(16-x) + a10(y)(16-x)
 *      + a01(16-y)(x)    + a11(y)(x)
 *
 */
extern void S32_Opaque_D32_filter_line_SSSE3(uint32_t* row0, uint32_t* row1,
                                     SkFixed fx, unsigned subY,
                                     uint32_t* colors, SkFixed dx, int count){
    unsigned  subX = (((fx) >> 12) & 0xF);

    unsigned x0 = ((fx) >> 16);
    // ( 0,  0,  0,  0,  0,  0,  0, 16)
    __m128i sixteen = _mm_cvtsi32_si128(16);

    // ( 0,  0,  0,  0, 16, 16, 16, 16)
    sixteen = _mm_shufflelo_epi16(sixteen, 0);

    __m128i allY = _mm_cvtsi32_si128((subY << 8) | (16 - subY));

    // (y, 16-y, y, 16-y,y,16-y,y,16-y)
    allY = _mm_shufflelo_epi16(allY, 0);

    // (y,16-y, y,16-y,y,16-y,y,16-y,y,16-y,y,16-y,y,16-y,y,16-y)
    allY = _mm_shuffle_epi32(allY, 0);

    // ( 0,  0,  0,  0,  0,  0,  0,  0)
    __m128i zero = _mm_setzero_si128();

    __m128i allX = _mm_cvtsi32_si128(subX);
        //(,,,,x,x,x,x)
    allX = _mm_shufflelo_epi16(allX, 0);
    //(,,,,16-x,16-x,16-x,16-x)
    __m128i negX = _mm_sub_epi16(sixteen, allX);
    //(x,x,x,x,16-x,16-x,16-x,16-x)
    negX = _mm_unpacklo_epi64(negX, allX);

    do {
        __m128i a00 = _mm_cvtsi32_si128(row0[x0]);
        __m128i a01 = _mm_cvtsi32_si128(row0[x0+1]);
        __m128i a10 = _mm_cvtsi32_si128(row1[x0]);
        __m128i a11 = _mm_cvtsi32_si128(row1[x0+1]);

        //(0, 0, a10, a00)
        __m128i a01a00 = _mm_unpacklo_epi32(a00, a01);
        //(0, 0, a11, a10)
        __m128i a11a10 = _mm_unpacklo_epi32(a10, a11);

        //(....A10,A00,R10,R00,G10,G00,B10,B00)
        a01a00= _mm_unpacklo_epi8(a01a00, a11a10);
        //[..A00*(16-y)+ A10*y, R00*(16-y)+ R10*y, G00*(16-y)+ G10*y, B00*(16-y)+ B10*y]
        __m128i sum = _mm_maddubs_epi16(a01a00, allY);

        //[...(G00*(16-y)+ G10*y)(16-x),(B00*(16-y)+ B10*y) * (16-x)]
        sum = _mm_mullo_epi16(sum, negX);
        //[...(G01*(16-y)+ G11*y)(x),(B01*(16-y)+ B11*y) * (x)]
        __m128i shifted = _mm_shuffle_epi32(sum, 0xE);
        sum = _mm_add_epi16(sum, shifted);
        sum = _mm_srli_epi16(sum, 8);
        // Pack lower 4 16 bit values of sum into lower 4 bytes.
        sum = _mm_packus_epi16(sum, shifted);
        *colors++ = _mm_cvtsi128_si32(sum);

        fx+= dx;
        x0 = ((fx) >> 16);
        subX = (((fx) >> 12) & 0xF);
        allX = _mm_cvtsi32_si128(subX);
        //(,,,,x,x,x,x)
        allX = _mm_shufflelo_epi16(allX, 0);
        //(,,,,16-x,16-x,16-x,16-x)
        negX = _mm_sub_epi16(sixteen, allX);
        //(x,x,x,x,16-x,16-x,16-x,16-x)
        negX = _mm_unpacklo_epi64(negX, allX);
    } while (--count > 0);
}
