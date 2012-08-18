//////////////////////////////////////////
// Bootstrap code
//////////////////////////////////////////
// 0x00

boot: 
	// set stack pointer
	mov	$r0 0x1000
	sethi	$r0 0
	mov     $sp $r0
        // set interrupt vector
//	mov	$r0 0x1b // interrupt handler (hardcoded)
	mov	$r0 #ih // interrupt handler (hardcoded)
	sethi	$r0 0x0
	mov	$iv0 $r0
	clear b32 $r0
	// jump to the main function
	call	#main
	// exit the program
	exit
// 0x1b
ih:

//mov $r10 $flags
//call #ihbody
//mov $flags $r10

        push	$r0
	push	$r1
	push	$r2
	push	$r3
	push	$r4
	push	$r5
	push	$r6
	push	$r7
	push	$r8
	push	$r9
	push	$r10
	push	$r11
	push	$r12
	push	$r13
	push	$r14
	push	$r15
	mov $r10 $flags
        push    $r10
        call	#ihbody
        pop $r10
        mov $flags $r10
	pop	$r15
	pop	$r14
	pop	$r13
	pop	$r12
	pop	$r11
	pop	$r10
	pop	$r9
	pop	$r8
	pop	$r7
	pop	$r6
	pop	$r5
	pop	$r4
	pop	$r3
	pop	$r2
	pop	$r1
	pop	$r0
	bset	$flags $p2
	iret

//////////////////////////////////////////
// User code
///////////////////////////////////////////

