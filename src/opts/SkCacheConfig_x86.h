/*
 * Copyright 2014 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkCacheConfig_x86_DEFINED
#define SkCacheConfig_x86_DEFINED

#if defined(__slm__)
/* Values are optimized for Silvermont */
#define SHARED_CACHE_SIZE   (1024*1024)         /* Silvermont L2 Cache */
#define DATA_CACHE_SIZE     (24*1024)           /* Silvermont L1 Data Cache */
#else
/* Values are optimized for Atom */
#define SHARED_CACHE_SIZE   (512*1024)          /* Atom L2 Cache */
#define DATA_CACHE_SIZE     (24*1024)           /* Atom L1 Data Cache */
#endif

#define SHARED_CACHE_SIZE_HALF  (SHARED_CACHE_SIZE / 2)
#define DATA_CACHE_SIZE_HALF    (DATA_CACHE_SIZE / 2)

#endif
