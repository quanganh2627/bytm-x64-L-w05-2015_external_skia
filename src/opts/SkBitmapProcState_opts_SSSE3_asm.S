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
