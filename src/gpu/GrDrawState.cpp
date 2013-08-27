/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "GrDrawState.h"

#include "GrGpuVertex.h"
#include "GrPaint.h"

void GrDrawState::setFromPaint(const GrPaint& paint) {
    for (int i = 0; i < GrPaint::kMaxColorStages; ++i) {
        int s = i + GrPaint::kFirstColorStage;
        if (paint.isColorStageEnabled(i)) {
            fStages[s] = paint.getColorStage(i);
        } else {
            fStages[s].setEffect(NULL);
        }
    }

    this->setFirstCoverageStage(GrPaint::kFirstCoverageStage);

    for (int i = 0; i < GrPaint::kMaxCoverageStages; ++i) {
        int s = i + GrPaint::kFirstCoverageStage;
        if (paint.isCoverageStageEnabled(i)) {
            fStages[s] = paint.getCoverageStage(i);
        } else {
            fStages[s].setEffect(NULL);
        }
    }

    // disable all stages not accessible via the paint
    for (int s = GrPaint::kTotalStages; s < GrDrawState::kNumStages; ++s) {
        this->disableStage(s);
    }

    this->setColor(paint.getColor());

    this->setState(GrDrawState::kDither_StateBit, paint.isDither());
    this->setState(GrDrawState::kHWAntialias_StateBit, paint.isAntiAlias());

    this->setBlendFunc(paint.getSrcBlendCoeff(), paint.getDstBlendCoeff());
    this->setColorFilter(paint.getColorFilterColor(), paint.getColorFilterMode());
    this->setCoverage(paint.getCoverage());
}

////////////////////////////////////////////////////////////////////////////////

namespace {

/**
 * This function generates some masks that we like to have known at compile
 * time. When the number of stages or tex coords is bumped or the way bits
 * are defined in GrDrawState.h changes this function should be rerun to
 * generate the new masks. (We attempted to force the compiler to generate the
 * masks using recursive templates but always wound up with static initializers
 * under gcc, even if they were just a series of immediate->memory moves.)
 *
 */
void gen_mask_arrays(GrVertexLayout* stageTexCoordMasks,
                     GrVertexLayout* texCoordMasks) {
    for (int s = 0; s < GrDrawState::kNumStages; ++s) {
        stageTexCoordMasks[s] = 0;
        for (int t = 0; t < GrDrawState::kMaxTexCoords; ++t) {
            stageTexCoordMasks[s] |= GrDrawState::StageTexCoordVertexLayoutBit(s, t);
        }
    }
    for (int t = 0; t < GrDrawState::kMaxTexCoords; ++t) {
        texCoordMasks[t] = 0;
        for (int s = 0; s < GrDrawState::kNumStages; ++s) {
            texCoordMasks[t] |= GrDrawState::StageTexCoordVertexLayoutBit(s, t);
        }
    }
}

/**
 * Uncomment and run the gen_globals function to generate
 * the code that declares the global masks.
 *
 * #if 0'ed out to avoid unused function warning.
 */

#if 0
void gen_globals() {
    GrVertexLayout stageTexCoordMasks[GrDrawState::kNumStages];
    GrVertexLayout texCoordMasks[GrDrawState::kMaxTexCoords];
    gen_mask_arrays(stageTexCoordMasks, texCoordMasks);

    GrPrintf("const GrVertexLayout gStageTexCoordMasks[] = {\n");
    for (int s = 0; s < GrDrawState::kNumStages; ++s) {
        GrPrintf("    0x%x,\n", stageTexCoordMasks[s]);
    }
    GrPrintf("};\n");
    GrPrintf("GR_STATIC_ASSERT(GrDrawState::kNumStages == GR_ARRAY_COUNT(gStageTexCoordMasks));\n\n");
    GrPrintf("const GrVertexLayout gTexCoordMasks[] = {\n");
    for (int t = 0; t < GrDrawState::kMaxTexCoords; ++t) {
        GrPrintf("    0x%x,\n", texCoordMasks[t]);
    }
    GrPrintf("};\n");
    GrPrintf("GR_STATIC_ASSERT(GrDrawState::kMaxTexCoords == GR_ARRAY_COUNT(gTexCoordMasks));\n");
}
#endif

/* These values were generated by the above function */

const GrVertexLayout gStageTexCoordMasks[] = {
    0x108421,
    0x210842,
    0x421084,
    0x842108,
    0x1084210,
};
GR_STATIC_ASSERT(GrDrawState::kNumStages == GR_ARRAY_COUNT(gStageTexCoordMasks));

const GrVertexLayout gTexCoordMasks[] = {
    0x1f,
    0x3e0,
    0x7c00,
    0xf8000,
    0x1f00000,
};
GR_STATIC_ASSERT(GrDrawState::kMaxTexCoords == GR_ARRAY_COUNT(gTexCoordMasks));

#ifdef SK_DEBUG
bool check_layout(GrVertexLayout layout) {
    // can only have 1 or 0 bits set for each stage.
    for (int s = 0; s < GrDrawState::kNumStages; ++s) {
        int stageBits = layout & gStageTexCoordMasks[s];
        if (stageBits && !GrIsPow2(stageBits)) {
            return false;
        }
    }
    return true;
}
#endif

int num_tex_coords(GrVertexLayout layout) {
    int cnt = 0;
    // figure out how many tex coordinates are present
    for (int t = 0; t < GrDrawState::kMaxTexCoords; ++t) {
        if (gTexCoordMasks[t] & layout) {
            ++cnt;
        }
    }
    return cnt;
}

} //unnamed namespace

size_t GrDrawState::VertexSize(GrVertexLayout vertexLayout) {
    GrAssert(check_layout(vertexLayout));

    size_t vecSize = (vertexLayout & kTextFormat_VertexLayoutBit) ?
                        sizeof(GrGpuTextVertex) :
                        sizeof(GrPoint);

    size_t size = vecSize; // position
    size += num_tex_coords(vertexLayout) * vecSize;
    if (vertexLayout & kColor_VertexLayoutBit) {
        size += sizeof(GrColor);
    }
    if (vertexLayout & kCoverage_VertexLayoutBit) {
        size += sizeof(GrColor);
    }
    if (vertexLayout & kEdge_VertexLayoutBit) {
        size += 4 * sizeof(SkScalar);
    }
    return size;
}

////////////////////////////////////////////////////////////////////////////////

/**
 * Functions for computing offsets of various components from the layout
 * bitfield.
 *
 * Order of vertex components:
 * Position
 * Tex Coord 0
 * ...
 * Tex Coord GrDrawState::kMaxTexCoords-1
 * Color
 * Coverage
 */

int GrDrawState::VertexStageCoordOffset(int stageIdx, GrVertexLayout vertexLayout) {
    GrAssert(check_layout(vertexLayout));

    if (!StageUsesTexCoords(vertexLayout, stageIdx)) {
        return 0;
    }
    int tcIdx = VertexTexCoordsForStage(stageIdx, vertexLayout);
    if (tcIdx >= 0) {

        int vecSize = (vertexLayout & kTextFormat_VertexLayoutBit) ?
                                    sizeof(GrGpuTextVertex) :
                                    sizeof(GrPoint);
        int offset = vecSize; // position
        // figure out how many tex coordinates are present and precede this one.
        for (int t = 0; t < tcIdx; ++t) {
            if (gTexCoordMasks[t] & vertexLayout) {
                offset += vecSize;
            }
        }
        return offset;
    }

    return -1;
}

int GrDrawState::VertexColorOffset(GrVertexLayout vertexLayout) {
    GrAssert(check_layout(vertexLayout));

    if (vertexLayout & kColor_VertexLayoutBit) {
        int vecSize = (vertexLayout & kTextFormat_VertexLayoutBit) ?
                                    sizeof(GrGpuTextVertex) :
                                    sizeof(GrPoint);
        return vecSize * (num_tex_coords(vertexLayout) + 1); //+1 for pos
    }
    return -1;
}

int GrDrawState::VertexCoverageOffset(GrVertexLayout vertexLayout) {
    GrAssert(check_layout(vertexLayout));

    if (vertexLayout & kCoverage_VertexLayoutBit) {
        int vecSize = (vertexLayout & kTextFormat_VertexLayoutBit) ?
                                    sizeof(GrGpuTextVertex) :
                                    sizeof(GrPoint);

        int offset = vecSize * (num_tex_coords(vertexLayout) + 1);
        if (vertexLayout & kColor_VertexLayoutBit) {
            offset += sizeof(GrColor);
        }
        return offset;
    }
    return -1;
}

int GrDrawState::VertexEdgeOffset(GrVertexLayout vertexLayout) {
    GrAssert(check_layout(vertexLayout));

    // edge pts are after the pos, tex coords, and color
    if (vertexLayout & kEdge_VertexLayoutBit) {
        int vecSize = (vertexLayout & kTextFormat_VertexLayoutBit) ?
                                    sizeof(GrGpuTextVertex) :
                                    sizeof(GrPoint);
        int offset = vecSize * (num_tex_coords(vertexLayout) + 1); //+1 for pos
        if (vertexLayout & kColor_VertexLayoutBit) {
            offset += sizeof(GrColor);
        }
        if (vertexLayout & kCoverage_VertexLayoutBit) {
            offset += sizeof(GrColor);
        }
        return offset;
    }
    return -1;
}

int GrDrawState::VertexSizeAndOffsetsByIdx(
        GrVertexLayout vertexLayout,
        int texCoordOffsetsByIdx[kMaxTexCoords],
        int* colorOffset,
        int* coverageOffset,
        int* edgeOffset) {
    GrAssert(check_layout(vertexLayout));

    int vecSize = (vertexLayout & kTextFormat_VertexLayoutBit) ?
                                                    sizeof(GrGpuTextVertex) :
                                                    sizeof(GrPoint);
    int size = vecSize; // position

    for (int t = 0; t < kMaxTexCoords; ++t) {
        if (gTexCoordMasks[t] & vertexLayout) {
            if (NULL != texCoordOffsetsByIdx) {
                texCoordOffsetsByIdx[t] = size;
            }
            size += vecSize;
        } else {
            if (NULL != texCoordOffsetsByIdx) {
                texCoordOffsetsByIdx[t] = -1;
            }
        }
    }
    if (kColor_VertexLayoutBit & vertexLayout) {
        if (NULL != colorOffset) {
            *colorOffset = size;
        }
        size += sizeof(GrColor);
    } else {
        if (NULL != colorOffset) {
            *colorOffset = -1;
        }
    }
    if (kCoverage_VertexLayoutBit & vertexLayout) {
        if (NULL != coverageOffset) {
            *coverageOffset = size;
        }
        size += sizeof(GrColor);
    } else {
        if (NULL != coverageOffset) {
            *coverageOffset = -1;
        }
    }
    if (kEdge_VertexLayoutBit & vertexLayout) {
        if (NULL != edgeOffset) {
            *edgeOffset = size;
        }
        size += 4 * sizeof(SkScalar);
    } else {
        if (NULL != edgeOffset) {
            *edgeOffset = -1;
        }
    }
    return size;
}

int GrDrawState::VertexSizeAndOffsetsByStage(
        GrVertexLayout vertexLayout,
        int texCoordOffsetsByStage[GrDrawState::kNumStages],
        int* colorOffset,
        int* coverageOffset,
        int* edgeOffset) {
    GrAssert(check_layout(vertexLayout));

    int texCoordOffsetsByIdx[kMaxTexCoords];
    int size = VertexSizeAndOffsetsByIdx(vertexLayout,
                                         (NULL == texCoordOffsetsByStage) ?
                                               NULL :
                                               texCoordOffsetsByIdx,
                                         colorOffset,
                                         coverageOffset,
                                         edgeOffset);
    if (NULL != texCoordOffsetsByStage) {
        for (int s = 0; s < GrDrawState::kNumStages; ++s) {
            int tcIdx = VertexTexCoordsForStage(s, vertexLayout);
            texCoordOffsetsByStage[s] =
                tcIdx < 0 ? 0 : texCoordOffsetsByIdx[tcIdx];
        }
    }
    return size;
}

////////////////////////////////////////////////////////////////////////////////

bool GrDrawState::VertexUsesTexCoordIdx(int coordIndex,
                                         GrVertexLayout vertexLayout) {
    GrAssert(coordIndex < kMaxTexCoords);
    GrAssert(check_layout(vertexLayout));
    return !!(gTexCoordMasks[coordIndex] & vertexLayout);
}

int GrDrawState::VertexTexCoordsForStage(int stageIdx,
                                          GrVertexLayout vertexLayout) {
    GrAssert(stageIdx < GrDrawState::kNumStages);
    GrAssert(check_layout(vertexLayout));
    int bit = vertexLayout & gStageTexCoordMasks[stageIdx];
    if (bit) {
        // figure out which set of texture coordates is used
        // bits are ordered T0S0, T0S1, T0S2, ..., T1S0, T1S1, ...
        // and start at bit 0.
        GR_STATIC_ASSERT(sizeof(GrVertexLayout) <= sizeof(uint32_t));
        return (32 - SkCLZ(bit) - 1) / GrDrawState::kNumStages;
    }
    return -1;
}

////////////////////////////////////////////////////////////////////////////////

void GrDrawState::VertexLayoutUnitTest() {
    // Ensure that our globals mask arrays are correct
    GrVertexLayout stageTexCoordMasks[GrDrawState::kNumStages];
    GrVertexLayout texCoordMasks[kMaxTexCoords];
    gen_mask_arrays(stageTexCoordMasks, texCoordMasks);
    for (int s = 0; s < GrDrawState::kNumStages; ++s) {
        GrAssert(stageTexCoordMasks[s] == gStageTexCoordMasks[s]);
    }
    for (int t = 0; t < kMaxTexCoords; ++t) {
        GrAssert(texCoordMasks[t] == gTexCoordMasks[t]);
    }

    // not necessarily exhaustive
    static bool run;
    if (!run) {
        run = true;
        for (int s = 0; s < GrDrawState::kNumStages; ++s) {

            GrVertexLayout stageMask = 0;
            for (int t = 0; t < kMaxTexCoords; ++t) {
                stageMask |= StageTexCoordVertexLayoutBit(s,t);
            }
            GrAssert(1 == kMaxTexCoords ||
                     !check_layout(stageMask));
            GrAssert(gStageTexCoordMasks[s] == stageMask);
            GrAssert(!check_layout(stageMask));
        }
        for (int t = 0; t < kMaxTexCoords; ++t) {
            GrVertexLayout tcMask = 0;
            GrAssert(!VertexUsesTexCoordIdx(t, 0));
            for (int s = 0; s < GrDrawState::kNumStages; ++s) {
                tcMask |= StageTexCoordVertexLayoutBit(s,t);
                GrAssert(sizeof(GrPoint) == VertexStageCoordOffset(s, tcMask));
                GrAssert(VertexUsesTexCoordIdx(t, tcMask));
                GrAssert(2*sizeof(GrPoint) == VertexSize(tcMask));
                GrAssert(t == VertexTexCoordsForStage(s, tcMask));
                for (int s2 = s + 1; s2 < GrDrawState::kNumStages; ++s2) {
                    GrAssert(-1 == VertexTexCoordsForStage(s2, tcMask));

                #if GR_DEBUG
                    GrVertexLayout posAsTex = tcMask;
                #endif
                    GrAssert(0 == VertexStageCoordOffset(s2, posAsTex));
                    GrAssert(2*sizeof(GrPoint) == VertexSize(posAsTex));
                    GrAssert(-1 == VertexTexCoordsForStage(s2, posAsTex));
                    GrAssert(-1 == VertexEdgeOffset(posAsTex));
                }
                GrAssert(-1 == VertexEdgeOffset(tcMask));
                GrAssert(-1 == VertexColorOffset(tcMask));
                GrAssert(-1 == VertexCoverageOffset(tcMask));
            #if GR_DEBUG
                GrVertexLayout withColor = tcMask | kColor_VertexLayoutBit;
            #endif
                GrAssert(-1 == VertexCoverageOffset(withColor));
                GrAssert(2*sizeof(GrPoint) == VertexColorOffset(withColor));
                GrAssert(2*sizeof(GrPoint) + sizeof(GrColor) == VertexSize(withColor));
            #if GR_DEBUG
                GrVertexLayout withEdge = tcMask | kEdge_VertexLayoutBit;
            #endif
                GrAssert(-1 == VertexColorOffset(withEdge));
                GrAssert(2*sizeof(GrPoint) == VertexEdgeOffset(withEdge));
                GrAssert(4*sizeof(GrPoint) == VertexSize(withEdge));
            #if GR_DEBUG
                GrVertexLayout withColorAndEdge = withColor | kEdge_VertexLayoutBit;
            #endif
                GrAssert(2*sizeof(GrPoint) == VertexColorOffset(withColorAndEdge));
                GrAssert(2*sizeof(GrPoint) + sizeof(GrColor) == VertexEdgeOffset(withColorAndEdge));
                GrAssert(4*sizeof(GrPoint) + sizeof(GrColor) == VertexSize(withColorAndEdge));
            #if GR_DEBUG
                GrVertexLayout withCoverage = tcMask | kCoverage_VertexLayoutBit;
            #endif
                GrAssert(-1 == VertexColorOffset(withCoverage));
                GrAssert(2*sizeof(GrPoint) == VertexCoverageOffset(withCoverage));
                GrAssert(2*sizeof(GrPoint) + sizeof(GrColor) == VertexSize(withCoverage));
            #if GR_DEBUG
                GrVertexLayout withCoverageAndColor = tcMask | kCoverage_VertexLayoutBit |
                                                      kColor_VertexLayoutBit;
            #endif
                GrAssert(2*sizeof(GrPoint) == VertexColorOffset(withCoverageAndColor));
                GrAssert(2*sizeof(GrPoint) + sizeof(GrColor) == VertexCoverageOffset(withCoverageAndColor));
                GrAssert(2*sizeof(GrPoint) + 2 * sizeof(GrColor) == VertexSize(withCoverageAndColor));
            }
            GrAssert(gTexCoordMasks[t] == tcMask);
            GrAssert(check_layout(tcMask));

            int stageOffsets[GrDrawState::kNumStages];
            int colorOffset;
            int edgeOffset;
            int coverageOffset;
            int size;
            size = VertexSizeAndOffsetsByStage(tcMask,
                                               stageOffsets, &colorOffset,
                                               &coverageOffset, &edgeOffset);
            GrAssert(2*sizeof(GrPoint) == size);
            GrAssert(-1 == colorOffset);
            GrAssert(-1 == coverageOffset);
            GrAssert(-1 == edgeOffset);
            for (int s = 0; s < GrDrawState::kNumStages; ++s) {
                GrAssert(sizeof(GrPoint) == stageOffsets[s]);
                GrAssert(sizeof(GrPoint) == VertexStageCoordOffset(s, tcMask));
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

bool GrDrawState::StageUsesTexCoords(GrVertexLayout layout, int stageIdx) {
    return SkToBool(layout & gStageTexCoordMasks[stageIdx]);
}

bool GrDrawState::srcAlphaWillBeOne(GrVertexLayout layout) const {

    uint32_t validComponentFlags;
    GrColor  color;
    // Check if per-vertex or constant color may have partial alpha
    if (layout & kColor_VertexLayoutBit) {
        validComponentFlags = 0;
    } else {
        validComponentFlags = GrEffect::kAll_ValidComponentFlags;
        color = this->getColor();
    }

    // Run through the color stages
    int stageCnt = getFirstCoverageStage();
    for (int s = 0; s < stageCnt; ++s) {
        const GrEffectRef* effect = this->getStage(s).getEffect();
        if (NULL != effect) {
            (*effect)->getConstantColorComponents(&color, &validComponentFlags);
        }
    }

    // Check if the color filter could introduce an alpha.
    // We could skip the above work when this is true, but it is rare and the right fix is to make
    // the color filter a GrEffect and implement getConstantColorComponents() for it.
    if (SkXfermode::kDst_Mode != this->getColorFilterMode()) {
        validComponentFlags = 0;
    }

    // Check whether coverage is treated as color. If so we run through the coverage computation.
    if (this->isCoverageDrawing()) {
        GrColor coverageColor = this->getCoverage();
        GrColor oldColor = color;
        color = 0;
        for (int c = 0; c < 4; ++c) {
            if (validComponentFlags & (1 << c)) {
                U8CPU a = (oldColor >> (c * 8)) & 0xff;
                U8CPU b = (coverageColor >> (c * 8)) & 0xff;
                color |= (SkMulDiv255Round(a, b) << (c * 8));
            }
        }
        for (int s = this->getFirstCoverageStage(); s < GrDrawState::kNumStages; ++s) {
            const GrEffectRef* effect = this->getStage(s).getEffect();
            if (NULL != effect) {
                (*effect)->getConstantColorComponents(&color, &validComponentFlags);
            }
        }
    }
    return (GrEffect::kA_ValidComponentFlag & validComponentFlags) && 0xff == GrColorUnpackA(color);
}

bool GrDrawState::hasSolidCoverage(GrVertexLayout layout) const {
    // If we're drawing coverage directly then coverage is effectively treated as color.
    if (this->isCoverageDrawing()) {
        return true;
    }

    GrColor coverage;
    uint32_t validComponentFlags;
    // Initialize to an unknown starting coverage if per-vertex coverage is specified.
    if (layout & kCoverage_VertexLayoutBit) {
        validComponentFlags = 0;
    } else {
        coverage = fCommon.fCoverage;
        validComponentFlags = GrEffect::kAll_ValidComponentFlags;
    }

    // Run through the coverage stages and see if the coverage will be all ones at the end.
    for (int s = this->getFirstCoverageStage(); s < GrDrawState::kNumStages; ++s) {
        const GrEffectRef* effect = this->getStage(s).getEffect();
        if (NULL != effect) {
            (*effect)->getConstantColorComponents(&coverage, &validComponentFlags);
        }
    }
    return (GrEffect::kAll_ValidComponentFlags == validComponentFlags)  && (0xffffffff == coverage);
}

////////////////////////////////////////////////////////////////////////////////

void GrDrawState::AutoViewMatrixRestore::restore() {
    if (NULL != fDrawState) {
        fDrawState->setViewMatrix(fViewMatrix);
        for (int s = 0; s < GrDrawState::kNumStages; ++s) {
            if (fRestoreMask & (1 << s)) {
                fDrawState->fStages[s].restoreCoordChange(fSavedCoordChanges[s]);
            }
        }
    }
    fDrawState = NULL;
}

void GrDrawState::AutoViewMatrixRestore::set(GrDrawState* drawState,
                                             const SkMatrix& preconcatMatrix,
                                             uint32_t explicitCoordStageMask) {
    this->restore();

    fDrawState = drawState;
    if (NULL == drawState) {
        return;
    }

    fRestoreMask = 0;
    fViewMatrix = drawState->getViewMatrix();
    drawState->preConcatViewMatrix(preconcatMatrix);
    for (int s = 0; s < GrDrawState::kNumStages; ++s) {
        if (!(explicitCoordStageMask & (1 << s)) && drawState->isStageEnabled(s)) {
            fRestoreMask |= (1 << s);
            fDrawState->fStages[s].saveCoordChange(&fSavedCoordChanges[s]);
            drawState->fStages[s].preConcatCoordChange(preconcatMatrix);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void GrDrawState::AutoDeviceCoordDraw::restore() {
    if (NULL != fDrawState) {
        fDrawState->setViewMatrix(fViewMatrix);
        for (int s = 0; s < GrDrawState::kNumStages; ++s) {
            if (fRestoreMask & (1 << s)) {
                fDrawState->fStages[s].restoreCoordChange(fSavedCoordChanges[s]);
            }
        }
    }
    fDrawState = NULL;
}

bool GrDrawState::AutoDeviceCoordDraw::set(GrDrawState* drawState,
                                           uint32_t explicitCoordStageMask) {
    GrAssert(NULL != drawState);

    this->restore();

    fDrawState = drawState;
    if (NULL == fDrawState) {
        return false;
    }

    fViewMatrix = drawState->getViewMatrix();
    fRestoreMask = 0;
    SkMatrix invVM;
    bool inverted = false;

    for (int s = 0; s < GrDrawState::kNumStages; ++s) {
        if (!(explicitCoordStageMask & (1 << s)) && drawState->isStageEnabled(s)) {
            if (!inverted && !fViewMatrix.invert(&invVM)) {
                // sad trombone sound
                fDrawState = NULL;
                return false;
            } else {
                inverted = true;
            }
            fRestoreMask |= (1 << s);
            GrEffectStage* stage = drawState->fStages + s;
            stage->saveCoordChange(&fSavedCoordChanges[s]);
            stage->preConcatCoordChange(invVM);
        }
    }
    drawState->viewMatrix()->reset();
    return true;
}
