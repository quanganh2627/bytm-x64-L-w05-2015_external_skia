.globl S32_opaque_D32_filter_DX_SSE2_asm;
#
# extern "C" void S32_opaque_D32_filter_DX_SSE2_asm(const char*     srcAddr,
#                                                   unsigned        rb,
#                                                   const uint32_t* xy,
#                                                   int             count,
#                                                   uint32_t*       colors)
#
# This code was created by manually editing compiler generated code from
# SkBitmapProcState_opts_SSE2.cpp:S32_opaque_D32_filter_DX_SSE2_intrinsic().
# Please see the original function for more comments.
#
# Optimizations that have been made:
# - better register allocation to remove unnecessary loads in the loop
# - re-ordering of multiplications to distribute them more evenly
# - unrolling the loop once and interleaving it
#
# These optimizations showed 40% performance improvement (i.e. 40% fewer
# clock tics) over the compiler generated code on an Atom CPU in the tests
# that were run.
#
# Register allocation in the loop:
#   eax     - row0
#   ebx     - row1
#   ecx     - scratch (x0)
#   edx     - scratch (x1)
#   esi     - source
#   edi     - destination
#
#   xmm0..3 - scratch
#   xmm4    - zero
#   xmm5    - terminal value for source (source + 4 * count)
#   xmm6    - sixteen
#   xmm7    - allY
#

S32_opaque_D32_filter_DX_SSE2_asm:

      push   %ebp
      pxor   %xmm4,%xmm4           # _mm_setzero_si128()
      mov    %esp,%ebp
      push   %esi
      push   %edi
      push   %ebx
      sub    $0xc,%esp

      mov    0x10(%ebp),%esi       # xy
      mov    0xc(%ebp),%edx        # rb

      mov    $0x10, %ebx
      movd   %ebx, %xmm7
      pshuflw $0x0,%xmm7,%xmm6     # sixteen = _mm_shufflelo_epi16(sixteen, 0)

      mov    0x8(%ebp),%ebx        # srcAddr
      pshufd $0x0,%xmm6,%xmm6      # sixteen = _mm_shuffle_epi32(sixteen, 0)
      mov    (%esi),%edi           # XY = *xy
      mov    %edi,%eax             # XY
      shr    $0xe,%eax             # y0 = XY >> 14
      mov    %eax,-0x14(%ebp)      # y0
      mov    %edi,%eax             # XY
      and    $0x3fff,%edi          # XY & 0x3FFF
      shr    $0x12,%eax            # y0 >> 4 == XY >> 18
      imul   %edx,%eax             # (y0 >> 4) * rb
      imul   %edx,%edi             # (XY & 0x3FFF) * rb
      andl   $0xf,-0x14(%ebp)      # subY = y0 & 0xF
      lea    (%ebx,%eax,1),%eax    # row0 = srcAddr + (y0 >> 4) * rb
      lea    (%ebx,%edi,1),%ebx    # row1 = srcAddr + (XY & 0x3FFF) * rb

      movd   -0x14(%ebp),%xmm7     # __m128i allY = _mm_cvtsi32_si128(subY)

      pshuflw $0x0,%xmm7,%xmm7     # allY = _mm_shufflelo_epi16(allY, 0)
      movdqa %xmm6,%xmm0
      psubw  %xmm7,%xmm0           # __m128i negY = _mm_sub_epi16(sixteen, allY)
      punpcklqdq %xmm0,%xmm7       # allY = _mm_unpacklo_epi64(allY, negY)

      mov    0x18(%ebp),%edi       # colors

      lea    -0x4(%edi),%edi

      mov    0x14(%ebp),%ecx       # count
      mov    %ecx,%edx
      shl    $0x2,%ecx             # 4 * count
      add    %esi,%ecx             # xy + 4 * count
      movd   %ecx,%xmm5            # terminal source (xy + 4 * count)

      and    $1,%edx
      test   %edx,%edx
      jz     evenloop              # even count; go to the beginning of
                                   # the unrolled loop


          # odd count; prepare for entering the unrolled loop in the middle
          mov    0x4(%esi),%edx      # *xy == XX

          movdqa %xmm6,%xmm1         # sixteen

          movd   %edx,%xmm3          # XX
          mov    %edx,%ecx           # XX
          and    $0x3fff,%edx        # x1 = XX & 0x3FFF
          shr    $0x12,%ecx          # x0 = XX >> 18
          pslld  $0xe,%xmm3          # XX << 14
          psrld  $0x1c,%xmm3         # (XX >> 14) & 0xF

          movd   (%eax,%ecx,4),%xmm2 # a00 = row0[x0]

          jmp odd                    # enter the unrolled loop
                                     # in the middle


      nop                        # for loop alignment
      nop
      nop

evenloop:
      mov    0x4(%esi),%edx      # *xy == XX

      movdqa %xmm6,%xmm1         # sixteen

      movd   %edx,%xmm3          # XX
      mov    %edx,%ecx           # XX
      and    $0x3fff,%edx        # x1 = XX & 0x3FFF
      shr    $0x12,%ecx          # x0 = XX >> 18
      pslld  $0xe,%xmm3          # XX << 14
      psrld  $0x1c,%xmm3         # (XX >> 14) & 0xF

      movd   (%ebx,%ecx,4),%xmm0 # a10 = row1[x0]
      movd   (%eax,%ecx,4),%xmm2 # a00 = row0[x0]

      pshuflw $0x0,%xmm3,%xmm3   # allX = _mm_shufflelo_epi16(allX, 0)
      pshufd $0x0,%xmm3,%xmm3    # allX = _mm_shuffle_epi32(allX, 0)
      psubw  %xmm3,%xmm1         # negX = _mm_sub_epi16(sixteen, allX)

      pmullw %xmm7,%xmm1         # allY * negX
      punpckldq %xmm2,%xmm0      # a00a10 = _mm_unpacklo_epi32(a10, a00)
      pmullw %xmm7,%xmm3         # allY * allX
      punpcklbw %xmm4,%xmm0      # a00a10 = _mm_unpacklo_epi8(a00a10, zero)

      movd   (%ebx,%edx,4),%xmm2 # a11 = row1[x1]

      pmullw %xmm1,%xmm0         # a00a10 = a0010 * negX * allY

      lea    0x4(%esi),%esi
      movd   (%eax,%edx,4),%xmm1 # a01 = row0[x1]

          mov    0x4(%esi),%edx      # *xy == XX (the first unrolled instrctn)

      punpckldq %xmm1,%xmm2      # a01a11 = _mm_unpacklo_epi32(a11, a01)

          movdqa %xmm6,%xmm1         # sixteen

      punpcklbw %xmm4,%xmm2      # a01a11 = _mm_unpacklo_epi8(a01a11, zero)
      pmullw %xmm3,%xmm2         # a01a11 = a01a11 * allY * allX

          movd   %edx,%xmm3          # XX
          mov    %edx,%ecx           # XX
          and    $0x3fff,%edx        # x1 = XX & 0x3FFF
          shr    $0x12,%ecx          # x0 = XX >> 18
          pslld  $0xe,%xmm3          # XX << 14
          psrld  $0x1c,%xmm3         # (XX >> 14) & 0xF

      paddw  %xmm2,%xmm0         # sum = _mm_add_epi16(a00a10, a01a11)
      pshufd $0xee,%xmm0,%xmm2   # shifted = _mm_shuffle_epi32(sum, 0xEE)
      paddw  %xmm2,%xmm0         # sum = _mm_add_epi16(sum, shifted)

          movd   (%eax,%ecx,4),%xmm2 # a00 = row0[x0]

      lea    0x4(%edi),%edi
      psrlw  $0x8,%xmm0          # sum = _mm_srli_epi16(sum, 8)
      packuswb %xmm4,%xmm0       # sum = _mm_packus_epi16(sum, zero)

      movd   %xmm0,(%edi,1)      # *colors = _mm_cvtsi128_si32(sum)


odd:
          movd   (%ebx,%ecx,4),%xmm0 # a10 = row1[x0]

          pshuflw $0x0,%xmm3,%xmm3   # allX = _mm_shufflelo_epi16(allX, 0)
          pshufd $0x0,%xmm3,%xmm3    # allX = _mm_shuffle_epi32(allX, 0)
          psubw  %xmm3,%xmm1         # negX = _mm_sub_epi16(sixteen, allX)

          pmullw %xmm7,%xmm1         # allY * negX
          punpckldq %xmm2,%xmm0      # a00a10 = _mm_unpacklo_epi32(a10, a00)
          pmullw %xmm7,%xmm3         # allY * allX
          punpcklbw %xmm4,%xmm0      # a00a10 = _mm_unpacklo_epi8(a00a10, zero)

          movd   (%ebx,%edx,4),%xmm2 # a11 = row1[x1]

          pmullw %xmm1,%xmm0         # a00a10 = a0010 * negX * allY

          movd   (%eax,%edx,4),%xmm1 # a01 = row0[x1]
          lea    0x4(%esi),%esi
          punpckldq %xmm1,%xmm2      # a01a11 = _mm_unpacklo_epi32(a11, a01)
          punpcklbw %xmm4,%xmm2      # a01a11 = _mm_unpacklo_epi8(a01a11, zero)

          pmullw %xmm3,%xmm2         # a01a11 = a01a11 * allY * allX

      movd   %xmm5,%ecx          # terminal value for source

          paddw  %xmm2,%xmm0         #  sum = _mm_add_epi16(a00a10, a01a11)
          pshufd $0xee,%xmm0,%xmm2   #  shifted = _mm_shuffle_epi32(sum, 0xEE)
          paddw  %xmm2,%xmm0         #  sum = _mm_add_epi16(sum, shifted)

      sub    %esi,%ecx           # compare source to its terminal value

          lea    0x4(%edi),%edi
          psrlw  $0x8,%xmm0          # sum = _mm_srli_epi16(sum, 8)
          packuswb %xmm4,%xmm0       # sum = _mm_packus_epi16(sum, zero)

          movd   %xmm0,(%edi,1)      # *colors = _mm_cvtsi128_si32(sum)

      jnz    evenloop            # loop if source not yet at its terminal value


      add    $0xc,%esp
      pop    %ebx
      pop    %edi
      pop    %esi
      pop    %ebp
      ret


.globl S32_opaque_D32_nofilter_DX_SSE2_asm;
S32_opaque_D32_nofilter_DX_SSE2_asm:

    push %ebp
    mov  %esp, %ebp
    push %esi
    push %edi
    push %ebx
    sub  $0x8, %esp

    mov 0x08(%ebp), %esi           # xy
    mov 0x0c(%ebp), %ecx           # count
    mov 0x0c(%ebp), %ecx           # count
    mov 0x10(%ebp), %edx           # srcAddr
    mov 0x14(%ebp), %edi           # colors

    sarl $0x2, %ecx                # count / 4
    test %ecx, %ecx
    jle nofilter_done

    sall $0x3, %ecx                # terminal value for index
    movd %ecx, %xmm7

    xor %eax, %eax                 # index = 0

    # prepare for part 1
    movl (%esi,%eax,1), %ebx

    shr $0x4, %ecx                 # count / 8
    jc .Lpart1odd
    jmp .Lpart1even


    .align 16
.Lloop:
    # part 2
    movzx %cx, %ebx
    shr $0x10, %ecx
    movl (%edx,%ebx,4), %ebx
    movl (%edx,%ecx,4), %ecx
    movl %ebx, -0x8(%edi,%eax,2)
    movl (%esi,%eax,1), %ebx       # prepare for part 1
    movl %ecx, -0x4(%edi,%eax,2)

.Lpart1even:
    # part 1
    movzx %bx, %ecx
    shr $0x10, %ebx
    movl (%edx,%ecx,4), %ecx
    movl (%edx,%ebx,4), %ebx
    movl %ecx, (%edi,%eax,2)
    movl 0x4(%esi,%eax,1), %ecx    # prepare for part 2
    movl %ebx, 0x4(%edi,%eax,2)

    lea 0x8(%eax), %eax

    # part 2
    movzx %cx, %ebx
    shr $0x10, %ecx
    movl (%edx,%ebx,4), %ebx
    movl (%edx,%ecx,4), %ecx
    movl %ebx, -0x8(%edi,%eax,2)
    movl (%esi,%eax,1), %ebx       # prepare for part 1
    movl %ecx, -0x4(%edi,%eax,2)

.Lpart1odd:
    # part 1
    movzx %bx, %ecx
    shr $0x10, %ebx
    movl (%edx,%ecx,4), %ecx
    movl (%edx,%ebx,4), %ebx
    movl %ecx, (%edi,%eax,2)
    movl 0x4(%esi,%eax,1), %ecx    # prepare for part 2
    movl %ebx, 0x4(%edi,%eax,2)

    lea 0x8(%eax), %eax

    movd %xmm7, %ebx
    sub %eax, %ebx

    jnz .Lloop


    # part 2
    movzx %cx, %ebx
    shr $0x10, %ecx
    movl (%edx,%ebx,4), %ebx
    movl (%edx,%ecx,4), %ecx
    movl %ebx, -0x8(%edi,%eax,2)
    movl %ecx, -0x4(%edi,%eax,2)


nofilter_done:
    add  $0x8, %esp
    pop  %ebx
    pop  %edi
    pop  %esi
    pop  %ebp
    ret

# extern "C" void S32_Opaque_D32_filter_line_SSSE3_asm(const unsigned int* row0,
#                                       const unsigned int* row1,
#                                       SkFixed fx,
#                                       unsigned int subY,
#                                       unsigned int* colors,
#                                       SkFixed dx,
#                                       int count);

.globl S32_Opaque_D32_filter_line_SSSE3_asm;
S32_Opaque_D32_filter_line_SSSE3_asm:
    push %ebp
    mov  %esp, %ebp
    push    %esi
    push    %edi
    push    %ebx
    lea    -0xc(%esp),%esp


    mov    0x10(%ebp),%eax
    mov    %eax,%edi
    shr    $0xc,%edi
    and    $0xf,%edi
    movd   %edi,%xmm4
    pshuflw $0x0,%xmm4,%xmm6 ## (0, 0, 0, 0,x,x,x,x)

    mov    $0x10,%ecx
    movd   %ecx,%xmm0
    pshuflw $0x0,%xmm0,%xmm1
    movdqa %xmm1,%xmm3      ##xmm1 = sixteen
    psubw  %xmm6,%xmm3      ##(0, 0, 0, 0, 16-x, 16-x, 16-x, 16-x)
    punpcklqdq %xmm6,%xmm3  ##(x,x,x,x, 16-x,16-x,16-x,16-x)

    ##subY << 8 | 16-subY
    mov    0x14(%ebp),%ecx
    mov    %ecx,%edx
    shl    $0x8,%ecx

    neg    %edx
    add    $0x10,%edx
    or     %edx,%ecx
    movd   %ecx,%xmm0 ## allY = _mm_cvtsi32_si128((subY << 8) | (16 - subY))
    pshuflw $0x0,%xmm0,%xmm0
    pshufd $0x0,%xmm0,%xmm0  #xmm0 = allY

    mov    0x8(%ebp),%eax   #eax: row0
    mov    0xc(%ebp),%edx   #edx: row1
    mov    0x18(%ebp),%esi    #esi: colors

    mov    0x20(%ebp),%ebx     #ebp: count
    mov    %ebx, %ecx          #unroll the loop,count = count/2
    shr    $0x1, %ebx          #unroll the loop,count = count/2
    and    $1,%ecx             #if count = odd, go odd first and even loop

    jz     .LevenPrepare

        mov    0x1c(%ebp),%ecx   #ecx: dx
        mov    0x10(%ebp),%ebp
        mov    %ebp,%edi
        sar    $0x10,%edi               #fx>>16
        movdqa %xmm3,%xmm5
        jmp     .Loddloop

.LevenPrepare:
    mov    0x1c(%ebp),%ecx   #ecx: dx
    mov    0x10(%ebp),%ebp
    mov    %ebp,%edi
    sar    $0x10,%edi               #fx>>16

.align 16
.Levenloop:
    movq   (%eax,%edi,4),%xmm4     #a01a00
    dec    %ebx                    #ebx = count --
    movq   (%edx,%edi,4),%xmm2     #a11a10
    punpcklbw %xmm2,%xmm4          #a01a00 = _mm_unpacklo_epi8(a01a00, a11a10);
    pmaddubsw %xmm0,%xmm4          #sum = _mm_maddubs_epi16(a01a00, allY);
    pmullw %xmm3,%xmm4             #sum = _mm_mullo_epi16(sum, negX);
    add    %ecx,%ebp               #fx = fx+dx
    pshufd $0xe,%xmm4,%xmm3        #shifted = _mm_shuffle_epi32(sum, 0xE);

    paddw  %xmm3,%xmm4             #sum = _mm_add_epi16(sum, shifted);
    mov    %ebp,%edi

    psrlw  $0x8,%xmm4              #sum = _mm_srli_epi16(sum, 8)
    shr    $0xc,%edi               #fx>>12 & 0xF
    packuswb %xmm3,%xmm4           #sum = _mm_packus_epi16(sum, shifted)
    and    $0xf,%edi
    movd   %edi,%xmm5              #allX = _mm_cvtsi32_si128(subX);
    movd   %xmm4,(%esi)            #*colors = _mm_cvtsi128_si32(sum)
    add    $0x4,%esi               #colors++

    mov    %ebp,%edi
    sar    $0x10,%edi               #fx>>16
.Loddloop:
    movq   (%eax,%edi,4),%xmm2
    movq   (%edx,%edi,4),%xmm6
    punpcklbw %xmm6,%xmm2

    pmaddubsw %xmm0,%xmm2
    movdqa %xmm1,%xmm6
    pshuflw $0x0,%xmm5,%xmm7       #allX = _mm_shufflelo_epi16(allX, 0)
    psubw  %xmm7,%xmm6             #negX = _mm_sub_epi16(sixteen, allX)
    punpcklqdq %xmm7,%xmm6         #negX = _mm_unpacklo_epi64(negX, allX)
    add    %ecx,%ebp               #fx = fx + dx
    pmullw %xmm6,%xmm2             #sum = _mm_mullo_epi16(sum, negX)
    mov    %ebp,%edi
    shr    $0xc,%edi               #fx>>12 & 0xF

    and    $0xf,%edi
    pshufd $0xe,%xmm2,%xmm7
    paddw  %xmm7,%xmm2
    psrlw  $0x8,%xmm2
    movd   %edi,%xmm3
    mov    %ebp,%edi
    pshuflw $0x0,%xmm3,%xmm4
    movdqa %xmm1,%xmm3
    packuswb %xmm7,%xmm2
    psubw  %xmm4,%xmm3
    movd   %xmm2,(%esi)
    add    $0x4,%esi            #colors = colors + 8
    sar    $0x10,%edi           #fx >> 16
    punpcklqdq %xmm4,%xmm3
    test   %ebx,%ebx
    jg     .Levenloop

    lea    0xc(%esp),%esp
    pop    %ebx
    pop    %edi
    pop    %esi
    pop    %ebp
    ret

