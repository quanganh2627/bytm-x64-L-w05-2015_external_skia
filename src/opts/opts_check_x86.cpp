/*
 * Copyright 2009 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "SkBitmapProcState_opts_SSE2.h"
#include "SkBitmapProcState_opts_SSSE3.h"
#include "SkBitmapFilter_opts_SSE2.h"
#include "SkBlitMask.h"
#include "SkBlitRow.h"
#include "SkBlitRect_opts_SSE2.h"
#include "SkBlitRow_opts_SSE2.h"
#include "SkBlitRow_opts_SSE4.h"
#include "SkUtils_opts_SSE2.h"
#include "SkUtils.h"
#include "SkShader.h"

#include "SkRTConf.h"

#if defined(_MSC_VER) && defined(_WIN64)
#include <intrin.h>
#endif

/* This file must *not* be compiled with -msse or -msse2, otherwise
   gcc may generate sse2 even for scalar ops (and thus give an invalid
   instruction on Pentium3 on the code below).  Only files named *_SSE2.cpp
   in this directory should be compiled with -msse2. */


#ifdef _MSC_VER
static inline void getcpuid(int info_type, int info[4]) {
#if defined(_WIN64)
    __cpuid(info, info_type);
#else
    __asm {
        mov    eax, [info_type]
        cpuid
        mov    edi, [info]
        mov    [edi], eax
        mov    [edi+4], ebx
        mov    [edi+8], ecx
        mov    [edi+12], edx
    }
#endif
}
#else
#if defined(__x86_64__)
static inline void getcpuid(int info_type, int info[4]) {
    asm volatile (
        "cpuid \n\t"
        : "=a"(info[0]), "=b"(info[1]), "=c"(info[2]), "=d"(info[3])
        : "a"(info_type)
    );
}
#else
static inline void getcpuid(int info_type, int info[4]) {
    // We save and restore ebx, so this code can be compatible with -fPIC
    asm volatile (
        "pushl %%ebx      \n\t"
        "cpuid            \n\t"
        "movl %%ebx, %1   \n\t"
        "popl %%ebx       \n\t"
        : "=a"(info[0]), "=r"(info[1]), "=c"(info[2]), "=d"(info[3])
        : "a"(info_type)
    );
}
#endif
#endif

// If the code is built for an architecture that has SSE4+,
// then we shouldn't have to check during run-time.
#if SK_CPU_SSE_LEVEL >= SK_CPU_SSE_LEVEL_SSE42
static inline int getSSELevel() {
    return SK_CPU_SSE_LEVEL_SSE42;
}
#else

static int checkSSELevel() {
    int cpu_info[4] = { 0 };

    getcpuid(1, cpu_info);
    if ((cpu_info[2] & (1<<20)) != 0)
        return SK_CPU_SSE_LEVEL_SSE42;
    else if ((cpu_info[2] & (1<<9)) != 0)
        return SK_CPU_SSE_LEVEL_SSSE3;
    else if ((cpu_info[3] & (1<<26)) != 0)
        return SK_CPU_SSE_LEVEL_SSE2;
    else
        return 0;
}

static inline int getSSELevel() {
    static int gSSELevel = checkSSELevel();
    return gSSELevel;
}
#endif

SK_CONF_DECLARE( bool, c_hqfilter_sse, "bitmap.filter.highQualitySSE", false, "Use SSE optimized version of high quality image filters");

void SkBitmapProcState::platformConvolutionProcs() {
    if (getSSELevel() >= SK_CPU_SSE_LEVEL_SSE2) {
        fConvolutionProcs->fExtraHorizontalReads = 3;
        fConvolutionProcs->fConvolveVertically = &convolveVertically_SSE2;
        fConvolutionProcs->fConvolve4RowsHorizontally = &convolve4RowsHorizontally_SSE2;
        fConvolutionProcs->fConvolveHorizontally = &convolveHorizontally_SSE2;
        fConvolutionProcs->fApplySIMDPadding = &applySIMDPadding_SSE2;
    }
}

void SkBitmapProcState::platformProcs() {
    const int SSELevel = getSSELevel();

    // Check fSampleProc32
    if (fSampleProc32 == S32_opaque_D32_filter_DX) {
        if (SSELevel >= SK_CPU_SSE_LEVEL_SSSE3) {
            fSampleProc32 = S32_opaque_D32_filter_DX_SSSE3;
#if !defined(__x86_64__)
            bool repeatXY = SkShader::kRepeat_TileMode == fTileModeX &&
                            SkShader::kRepeat_TileMode == fTileModeY;
            const unsigned max = fBitmap->width();
            // SSSE3 opted only if more than 4 pixels, dx=non-zero
            if ((fInvSx > 0) && repeatXY && (max > 4) && ((fInvSx & 0xFFFF) != 0)) {
                fShaderProc32 = Repeat_S32_Opaque_D32_filter_DX_shaderproc_opt;    // Not 64-bit compatible
            }
#endif
        } else if (SSELevel >= SK_CPU_SSE_LEVEL_SSE2) {
            fSampleProc32 = S32_opaque_D32_filter_DX_SSE2;
        }
    } else if (fSampleProc32 == S32_opaque_D32_nofilter_DX) {
#if !defined(__x86_64__)
        if (SSELevel >= SK_CPU_SSE_LEVEL_SSE2) {
            fSampleProc32 = S32_opaque_D32_nofilter_DX_SSE2;        // Not 64-bit compatible
        }
#endif
    } else if (fSampleProc32 == S32_opaque_D32_filter_DXDY) {
        if (SSELevel >= SK_CPU_SSE_LEVEL_SSSE3) {
            fSampleProc32 = S32_opaque_D32_filter_DXDY_SSSE3;
        } else if (SSELevel >= SK_CPU_SSE_LEVEL_SSE2) {
#if !defined(__x86_64__)
            fSampleProc32 = S32_opaque_D32_filter_DXDY_SSE2_asm;    // Not 64-bit compatible
#else
            fSampleProc32 = S32_opaque_D32_filter_DXDY_SSE2;
#endif
        }
    } else if (fSampleProc32 == S32_alpha_D32_filter_DX) {
        if (SSELevel >= SK_CPU_SSE_LEVEL_SSSE3) {
            fSampleProc32 = S32_alpha_D32_filter_DX_SSSE3;
        } else if (SSELevel >= SK_CPU_SSE_LEVEL_SSE2) {
            fSampleProc32 = S32_alpha_D32_filter_DX_SSE2;
        }
    } else if (fSampleProc32 == S32_alpha_D32_filter_DXDY) {
        if (SSELevel >= SK_CPU_SSE_LEVEL_SSSE3) {
            fSampleProc32 = S32_alpha_D32_filter_DXDY_SSSE3;
        } else if (SSELevel >= SK_CPU_SSE_LEVEL_SSE2) {
#if !defined(__x86_64__)
            fSampleProc32 = S32_alpha_D32_filter_DXDY_SSE2_asm;    // Not 64-bit compatible
#else
            fSampleProc32 = S32_alpha_D32_filter_DXDY_SSE2;
#endif
        }
    }

    if (SSELevel >= SK_CPU_SSE_LEVEL_SSE2) {
        if (fSampleProc16 == S32_D16_filter_DX) {
            fSampleProc16 = S32_D16_filter_DX_SSE2;
        }

        if (fMatrixProc == ClampX_ClampY_filter_scale) {
            fMatrixProc = ClampX_ClampY_filter_scale_SSE2;
        } else if (fMatrixProc == ClampX_ClampY_nofilter_scale) {
            fMatrixProc = ClampX_ClampY_nofilter_scale_SSE2;
        }

        if (fMatrixProc == ClampX_ClampY_filter_affine) {
            fMatrixProc = ClampX_ClampY_filter_affine_SSE2;
        } else if (fMatrixProc == ClampX_ClampY_nofilter_affine) {
            fMatrixProc = ClampX_ClampY_nofilter_affine_SSE2;
        }

        if (c_hqfilter_sse) {
            if (fShaderProc32 == highQualityFilter) {
                fShaderProc32 = highQualityFilter_SSE2;
            }
        }
    }
}

static SkBlitRow::Proc platform_565_procs[] = {
    // no dither
    S32_D565_Opaque_SSE2,               // S32_D565_Opaque,
    S32_D565_Blend_SSE2,                // S32_D565_Blend,

    S32A_D565_Opaque_SSE2,              // S32A_D565_Opaque
    S32A_D565_Blend_SSE2,               // S32A_D565_Blend

    // dither
    S32_D565_Opaque_Dither_SSE2,        // S32_D565_Opaque_Dither,
    S32_D565_Blend_Dither_SSE2,         // S32_D565_Blend_Dither,

    S32A_D565_Opaque_Dither_SSE2,       // S32A_D565_Opaque_Dither
    NULL                                // S32A_D565_Blend_Dither
};

SkBlitRow::Proc SkBlitRow::PlatformProcs565(unsigned flags) {
    if (getSSELevel() >= SK_CPU_SSE_LEVEL_SSE2) {
        return platform_565_procs[flags];
    } else {
        return NULL;
    }
}

SkBlitRow::ColorProc SkBlitRow::PlatformColorProc() {
    if (getSSELevel() >= SK_CPU_SSE_LEVEL_SSE2) {
        return Color32_SSE2;
    } else {
        return NULL;
    }
}

static SkBlitRow::Proc32 platform_32_procs_SSE2[] = {
    NULL,                               // S32_Opaque,
    S32_Blend_BlitRow32_SSE2,           // S32_Blend,
    S32A_Opaque_BlitRow32_SSE2,         // S32A_Opaque
    S32A_Blend_BlitRow32_SSE2,          // S32A_Blend,
};

static SkBlitRow::Proc32 platform_32_procs_SSE4[] = {
    NULL,                               // S32_Opaque
    S32_Blend_BlitRow32_SSE2,           // S32_Blend
#if !defined(__x86_64__)
    S32A_Opaque_BlitRow32_SSE4_asm,     // S32A_Opaque (32-bit assembly version)
    S32A_Blend_BlitRow32_SSE4_asm       // S32A_Blend (32-bit assembly version)
#else
#warning "Can't use SSE4 assembly optimizations in 64-bit mode. Using old intrinsic version."
    S32A_Opaque_BlitRow32_SSE2,         // S32A_Opaque (Intrinsics fallback version)
    S32A_Blend_BlitRow32_SSE2           // S32A_Blend (Intrinsics fallback version)
#endif
};

SkBlitRow::Proc32 SkBlitRow::PlatformProcs32(unsigned flags) {
    if (getSSELevel() >= SK_CPU_SSE_LEVEL_SSE42) {
        return platform_32_procs_SSE4[flags];
    } else if (getSSELevel() >= SK_CPU_SSE_LEVEL_SSE2) {
        return platform_32_procs_SSE2[flags];
    } else {
        return NULL;
    }
}

SkBlitMask::ColorProc SkBlitMask::PlatformColorProcs(SkBitmap::Config dstConfig,
                                                     SkMask::Format maskFormat,
                                                     SkColor color) {
    if (SkMask::kA8_Format != maskFormat) {
        return NULL;
    }

    ColorProc proc = NULL;
    if (getSSELevel() >= SK_CPU_SSE_LEVEL_SSE2) {
        switch (dstConfig) {
            case SkBitmap::kARGB_8888_Config:
                // The SSE2 version is not (yet) faster for black, so we check
                // for that.
                if (SK_ColorBLACK != color) {
                    proc = SkARGB32_A8_BlitMask_SSE2;
                }
                break;
            default:
                break;
        }
    }
    return proc;
}

SkBlitMask::BlitLCD16RowProc SkBlitMask::PlatformBlitRowProcs16(bool isOpaque) {
    if (getSSELevel() >= SK_CPU_SSE_LEVEL_SSE2) {
        if (isOpaque) {
            return SkBlitLCD16OpaqueRow_SSE2;
        } else {
            return SkBlitLCD16Row_SSE2;
        }
    } else {
        return NULL;
    }

}

SkBlitMask::RowProc SkBlitMask::PlatformRowProcs(SkBitmap::Config dstConfig,
                                                 SkMask::Format maskFormat,
                                                 RowFlags flags) {
    return NULL;
}

SkMemset16Proc SkMemset16GetPlatformProc() {
    if (getSSELevel() >= SK_CPU_SSE_LEVEL_SSE2) {
        return sk_memset16_libcutils;
    } else {
        return NULL;
    }
}

SkMemset32Proc SkMemset32GetPlatformProc() {
    if (getSSELevel() >= SK_CPU_SSE_LEVEL_SSE2) {
        return sk_memset32_libcutils;
    } else {
        return NULL;
    }
}

SkBlitRow::ColorRectProc PlatformColorRectProcFactory(); // suppress warning

SkBlitRow::ColorRectProc PlatformColorRectProcFactory() {
    if (getSSELevel() >= SK_CPU_SSE_LEVEL_SSE2) {
        return ColorRect32_SSE2;
    } else {
        return NULL;
    }
}
