	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z11gemm_kernelPK14__hip_bfloat16S1_PS_iii ; -- Begin function _Z11gemm_kernelPK14__hip_bfloat16S1_PS_iii
	.globl	_Z11gemm_kernelPK14__hip_bfloat16S1_PS_iii
	.p2align	8
	.type	_Z11gemm_kernelPK14__hip_bfloat16S1_PS_iii,@function
_Z11gemm_kernelPK14__hip_bfloat16S1_PS_iii: ; @_Z11gemm_kernelPK14__hip_bfloat16S1_PS_iii
; %bb.0:
	s_load_dwordx4 s[4:7], s[0:1], 0x18
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s3, s4, 31
	s_ashr_i32 s7, s5, 31
	s_lshr_b32 s3, s3, 24
	s_lshr_b32 s8, s7, 24
	s_add_i32 s3, s4, s3
	s_ashr_i32 s7, s3, 8
	s_add_i32 s3, s5, s8
	s_ashr_i32 s3, s3, 8
	s_mul_i32 s8, s3, s7
	s_ashr_i32 s9, s8, 31
	s_lshr_b32 s9, s9, 25
	s_add_i32 s8, s8, s9
	s_and_b32 s8, s8, 0xffffff80
	s_cmp_gt_i32 s2, s8
	s_cbranch_scc1 .LBB0_2
; %bb.1:
	s_ashr_i32 s8, s2, 31
	s_lshr_b32 s9, s8, 29
	s_add_i32 s9, s2, s9
	s_ashr_i32 s10, s9, 3
	s_and_b32 s9, s9, 0xffffff8
	s_lshr_b32 s8, s8, 25
	s_sub_i32 s9, s2, s9
	s_add_i32 s2, s2, s8
	s_lshr_b32 s8, s10, 28
	s_add_i32 s8, s10, s8
	s_and_b32 s8, s8, -16
	s_and_b32 s2, s2, 0xffffff80
	s_lshl_b32 s9, s9, 4
	s_sub_i32 s8, s10, s8
	s_add_i32 s2, s2, s9
	s_add_i32 s2, s2, s8
.LBB0_2:                                ; %_ZN7kittens25chiplet_transform_chunkedEiiii.exit
	s_load_dwordx4 s[12:15], s[0:1], 0x0
	s_cmp_lt_i32 s7, s3
	s_cbranch_scc1 .LBB0_4
; %bb.3:
	s_lshl_b32 s8, s7, 2
	s_ashr_i32 s9, s7, 31
	s_add_i32 s10, s8, s9
	s_xor_b32 s10, s10, s9
	v_cvt_f32_u32_e32 v1, s10
	s_ashr_i32 s11, s2, 31
	s_xor_b32 s9, s11, s9
	s_sub_i32 s11, 0, s10
	v_rcp_iflag_f32_e32 v1, v1
	s_abs_i32 s16, s2
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s17, v1
	s_mul_i32 s11, s11, s17
	s_mul_hi_u32 s11, s17, s11
	s_add_i32 s17, s17, s11
	s_mul_hi_u32 s11, s16, s17
	s_mul_i32 s17, s11, s10
	s_sub_i32 s16, s16, s17
	s_add_i32 s18, s11, 1
	s_sub_i32 s17, s16, s10
	s_cmp_ge_u32 s16, s10
	s_cselect_b32 s11, s18, s11
	s_cselect_b32 s16, s17, s16
	s_add_i32 s17, s11, 1
	s_cmp_ge_u32 s16, s10
	s_cselect_b32 s10, s17, s11
	s_xor_b32 s10, s10, s9
	s_sub_i32 s9, s10, s9
	s_lshl_b32 s11, s9, 2
	s_sub_i32 s7, s7, s11
	v_med3_i32 v1, s7, 1, 4
	v_cvt_f32_u32_e32 v2, v1
	v_readfirstlane_b32 s16, v1
	s_sub_i32 s10, 0, s16
	s_mul_i32 s9, s9, s8
	v_rcp_iflag_f32_e32 v2, v2
	s_sub_i32 s7, s2, s9
	s_abs_i32 s9, s7
	s_ashr_i32 s8, s7, 31
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	s_nop 0
	v_readfirstlane_b32 s17, v2
	s_mul_i32 s10, s10, s17
	s_mul_hi_u32 s10, s17, s10
	s_add_i32 s17, s17, s10
	s_mul_hi_u32 s10, s9, s17
	s_mul_i32 s17, s10, s16
	s_sub_i32 s9, s9, s17
	s_add_i32 s18, s10, 1
	s_sub_i32 s17, s9, s16
	s_cmp_ge_u32 s9, s16
	s_cselect_b32 s10, s18, s10
	s_cselect_b32 s9, s17, s9
	s_add_i32 s17, s10, 1
	s_cmp_ge_u32 s9, s16
	s_cselect_b32 s9, s17, s10
	s_xor_b32 s9, s9, s8
	s_sub_i32 s10, s9, s8
	s_mul_i32 s8, s10, s16
	s_sub_i32 s7, s7, s8
	s_add_i32 s7, s7, s11
	s_cbranch_execz .LBB0_5
	s_branch .LBB0_6
.LBB0_4:
                                        ; implicit-def: $sgpr7
                                        ; implicit-def: $sgpr10
.LBB0_5:
	s_lshl_b32 s7, s3, 2
	s_ashr_i32 s8, s3, 31
	s_add_i32 s9, s7, s8
	s_xor_b32 s9, s9, s8
	v_cvt_f32_u32_e32 v1, s9
	s_ashr_i32 s10, s2, 31
	s_xor_b32 s8, s10, s8
	s_sub_i32 s10, 0, s9
	v_rcp_iflag_f32_e32 v1, v1
	s_abs_i32 s11, s2
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s16, v1
	s_mul_i32 s10, s10, s16
	s_mul_hi_u32 s10, s16, s10
	s_add_i32 s16, s16, s10
	s_mul_hi_u32 s10, s11, s16
	s_mul_i32 s16, s10, s9
	s_sub_i32 s11, s11, s16
	s_add_i32 s17, s10, 1
	s_sub_i32 s16, s11, s9
	s_cmp_ge_u32 s11, s9
	s_cselect_b32 s10, s17, s10
	s_cselect_b32 s11, s16, s11
	s_add_i32 s16, s10, 1
	s_cmp_ge_u32 s11, s9
	s_cselect_b32 s9, s16, s10
	s_xor_b32 s9, s9, s8
	s_sub_i32 s8, s9, s8
	s_lshl_b32 s9, s8, 2
	s_sub_i32 s3, s3, s9
	v_med3_i32 v1, s3, 1, 4
	v_cvt_f32_u32_e32 v2, v1
	s_mul_i32 s8, s8, s7
	s_sub_i32 s2, s2, s8
	v_readfirstlane_b32 s8, v1
	v_rcp_iflag_f32_e32 v2, v2
	s_sub_i32 s10, 0, s8
	s_abs_i32 s7, s2
	s_ashr_i32 s3, s2, 31
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	s_nop 0
	v_readfirstlane_b32 s11, v2
	s_mul_i32 s10, s10, s11
	s_mul_hi_u32 s10, s11, s10
	s_add_i32 s11, s11, s10
	s_mul_hi_u32 s10, s7, s11
	s_mul_i32 s11, s10, s8
	s_sub_i32 s7, s7, s11
	s_add_i32 s16, s10, 1
	s_sub_i32 s11, s7, s8
	s_cmp_ge_u32 s7, s8
	s_cselect_b32 s10, s16, s10
	s_cselect_b32 s7, s11, s7
	s_add_i32 s11, s10, 1
	s_cmp_ge_u32 s7, s8
	s_cselect_b32 s7, s11, s10
	s_xor_b32 s7, s7, s3
	s_sub_i32 s7, s7, s3
	s_mul_i32 s3, s7, s8
	s_sub_i32 s2, s2, s3
	s_add_i32 s10, s2, s9
.LBB0_6:                                ; %.preheader456
	s_lshl_b32 s7, s7, 8
	s_mul_i32 s2, s7, s6
	s_ashr_i32 s3, s2, 31
	s_lshl_b32 s18, s10, 8
	s_lshl_b32 s11, s6, 1
	s_lshl_b64 s[2:3], s[2:3], 1
	s_waitcnt lgkmcnt(0)
	s_add_u32 s8, s12, s2
	s_addc_u32 s9, s13, s3
	s_lshl_b32 s2, s6, 17
	s_mul_i32 s16, s18, s6
	s_or_b32 s3, s2, -2.0
	s_mov_b32 s2, 0
	s_ashr_i32 s17, s16, 31
	s_lshl_b32 s10, s6, 9
	s_and_b32 s11, s11, 0x3ffe
	s_or_b64 s[12:13], s[2:3], s[8:9]
	s_lshl_b64 s[16:17], s[16:17], 1
	s_add_u32 s14, s14, s16
	s_addc_u32 s15, s15, s17
	v_lshrrev_b32_e32 v10, 1, v0
	s_or_b64 s[16:17], s[2:3], s[14:15]
	v_lshlrev_b32_e32 v1, 3, v0
	s_cmp_eq_u32 s11, 0
	v_and_b32_e32 v1, 8, v1
	v_mul_lo_u32 v130, s6, v10
	s_cselect_b32 s9, s9, s13
	s_cselect_b32 s8, s8, s12
	s_mov_b32 s11, 0x110000
	v_add_lshl_u32 v1, v130, v1, 1
	s_cselect_b32 s13, s15, s17
	s_cselect_b32 s12, s14, s16
	s_mov_b32 s14, s10
	s_mov_b32 s15, s11
	buffer_load_dwordx4 v[6:9], v1, s[8:11], 0 offen
	buffer_load_dwordx4 v[2:5], v1, s[12:15], 0 offen
	s_movk_i32 s3, 0x200
	v_lshlrev_b32_e32 v142, 4, v0
	v_cmp_gt_u32_e32 vcc, s3, v0
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB0_8
; %bb.7:                                ; %.preheader453
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	ds_write_b64 v142, v[6:7]

	;;#ASMEND
	v_add_u32_e32 v1, 8, v142
	;;#ASMSTART
	ds_write_b64 v1, v[8:9]

	;;#ASMEND
	v_or_b32_e32 v6, 0x4000, v142
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_write_b64 v6, v[2:3]

	;;#ASMEND
	v_add_u32_e32 v1, 0x4008, v142
	;;#ASMSTART
	ds_write_b64 v1, v[4:5]

	;;#ASMEND
.LBB0_8:                                ; %.critedge
	s_or_b64 exec, exec, s[16:17]
	s_load_dwordx2 s[16:17], s[0:1], 0x10
	v_and_b32_e32 v1, 31, v0
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v2, 3, v0
	v_and_b32_e32 v139, 0x80, v10
	v_and_b32_e32 v138, 4, v2
	v_or_b32_e32 v2, v139, v1
	s_movk_i32 s0, 0xdf0
	v_mov_b32_e32 v129, 0
	v_lshl_or_b32 v141, v2, 4, v138
	v_and_or_b32 v140, v142, s0, v138
	s_cmp_gt_i32 s6, 31
	v_mov_b32_e32 v128, v129
	v_mov_b32_e32 v127, v129
	v_mov_b32_e32 v126, v129
	v_mov_b32_e32 v125, v129
	v_mov_b32_e32 v124, v129
	v_mov_b32_e32 v123, v129
	v_mov_b32_e32 v122, v129
	v_mov_b32_e32 v121, v129
	v_mov_b32_e32 v120, v129
	v_mov_b32_e32 v119, v129
	v_mov_b32_e32 v118, v129
	v_mov_b32_e32 v117, v129
	v_mov_b32_e32 v116, v129
	v_mov_b32_e32 v115, v129
	v_mov_b32_e32 v114, v129
	v_mov_b32_e32 v113, v129
	v_mov_b32_e32 v112, v129
	v_mov_b32_e32 v111, v129
	v_mov_b32_e32 v110, v129
	v_mov_b32_e32 v109, v129
	v_mov_b32_e32 v108, v129
	v_mov_b32_e32 v107, v129
	v_mov_b32_e32 v106, v129
	v_mov_b32_e32 v105, v129
	v_mov_b32_e32 v104, v129
	v_mov_b32_e32 v103, v129
	v_mov_b32_e32 v102, v129
	v_mov_b32_e32 v101, v129
	v_mov_b32_e32 v100, v129
	v_mov_b32_e32 v99, v129
	v_mov_b32_e32 v98, v129
	v_mov_b32_e32 v97, v129
	v_mov_b32_e32 v96, v129
	v_mov_b32_e32 v95, v129
	v_mov_b32_e32 v94, v129
	v_mov_b32_e32 v93, v129
	v_mov_b32_e32 v92, v129
	v_mov_b32_e32 v91, v129
	v_mov_b32_e32 v90, v129
	v_mov_b32_e32 v89, v129
	v_mov_b32_e32 v88, v129
	v_mov_b32_e32 v87, v129
	v_mov_b32_e32 v86, v129
	v_mov_b32_e32 v85, v129
	v_mov_b32_e32 v84, v129
	v_mov_b32_e32 v83, v129
	v_mov_b32_e32 v82, v129
	v_mov_b32_e32 v81, v129
	v_mov_b32_e32 v80, v129
	v_mov_b32_e32 v79, v129
	v_mov_b32_e32 v78, v129
	v_mov_b32_e32 v77, v129
	v_mov_b32_e32 v76, v129
	v_mov_b32_e32 v75, v129
	v_mov_b32_e32 v74, v129
	v_mov_b32_e32 v73, v129
	v_mov_b32_e32 v72, v129
	v_mov_b32_e32 v71, v129
	v_mov_b32_e32 v70, v129
	v_mov_b32_e32 v69, v129
	v_mov_b32_e32 v68, v129
	v_mov_b32_e32 v67, v129
	v_mov_b32_e32 v66, v129
	v_mov_b32_e32 v65, v129
	v_mov_b32_e32 v64, v129
	v_mov_b32_e32 v63, v129
	v_mov_b32_e32 v62, v129
	v_mov_b32_e32 v61, v129
	v_mov_b32_e32 v60, v129
	v_mov_b32_e32 v59, v129
	v_mov_b32_e32 v58, v129
	v_mov_b32_e32 v57, v129
	v_mov_b32_e32 v56, v129
	v_mov_b32_e32 v55, v129
	v_mov_b32_e32 v54, v129
	v_mov_b32_e32 v53, v129
	v_mov_b32_e32 v52, v129
	v_mov_b32_e32 v51, v129
	v_mov_b32_e32 v50, v129
	v_mov_b32_e32 v49, v129
	v_mov_b32_e32 v48, v129
	v_mov_b32_e32 v47, v129
	v_mov_b32_e32 v46, v129
	v_mov_b32_e32 v45, v129
	v_mov_b32_e32 v44, v129
	v_mov_b32_e32 v43, v129
	v_mov_b32_e32 v42, v129
	v_mov_b32_e32 v41, v129
	v_mov_b32_e32 v40, v129
	v_mov_b32_e32 v39, v129
	v_mov_b32_e32 v38, v129
	v_mov_b32_e32 v37, v129
	v_mov_b32_e32 v36, v129
	v_mov_b32_e32 v35, v129
	v_mov_b32_e32 v34, v129
	v_mov_b32_e32 v33, v129
	v_mov_b32_e32 v32, v129
	v_mov_b32_e32 v31, v129
	v_mov_b32_e32 v30, v129
	v_mov_b32_e32 v29, v129
	v_mov_b32_e32 v28, v129
	v_mov_b32_e32 v27, v129
	v_mov_b32_e32 v26, v129
	v_mov_b32_e32 v25, v129
	v_mov_b32_e32 v24, v129
	v_mov_b32_e32 v23, v129
	v_mov_b32_e32 v22, v129
	v_mov_b32_e32 v21, v129
	v_mov_b32_e32 v20, v129
	v_mov_b32_e32 v19, v129
	v_mov_b32_e32 v18, v129
	v_mov_b32_e32 v17, v129
	v_mov_b32_e32 v16, v129
	v_mov_b32_e32 v15, v129
	v_mov_b32_e32 v14, v129
	v_mov_b32_e32 v13, v129
	v_mov_b32_e32 v12, v129
	v_mov_b32_e32 v11, v129
	v_mov_b32_e32 v10, v129
	v_mov_b32_e32 v9, v129
	v_mov_b32_e32 v8, v129
	v_mov_b32_e32 v7, v129
	v_mov_b32_e32 v6, v129
	v_mov_b32_e32 v5, v129
	v_mov_b32_e32 v4, v129
	v_mov_b32_e32 v3, v129
	v_mov_b32_e32 v2, v129
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	s_cbranch_scc0 .LBB0_13
; %bb.9:                                ; %.lr.ph
	s_ashr_i32 s0, s6, 31
	s_lshr_b32 s0, s0, 28
	v_mov_b32_e32 v2, 0x4000
	s_add_i32 s0, s6, s0
	v_and_b32_e32 v3, 1, v0
	v_lshl_or_b32 v143, v140, 1, v2
	s_ashr_i32 s0, s0, 4
	v_lshlrev_b32_e32 v2, 1, v130
	v_lshlrev_b32_e32 v3, 4, v3
	s_max_i32 s0, s0, 2
	v_add3_u32 v144, v2, v3, 32
	v_mov_b32_e32 v2, 0
	s_mov_b32 s3, 1
	s_add_i32 s6, s0, -1
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v6, v2
	v_mov_b32_e32 v7, v2
	v_mov_b32_e32 v8, v2
	v_mov_b32_e32 v9, v2
	v_mov_b32_e32 v10, v2
	v_mov_b32_e32 v11, v2
	v_mov_b32_e32 v12, v2
	v_mov_b32_e32 v13, v2
	v_mov_b32_e32 v14, v2
	v_mov_b32_e32 v15, v2
	v_mov_b32_e32 v16, v2
	v_mov_b32_e32 v17, v2
	v_mov_b32_e32 v18, v2
	v_mov_b32_e32 v19, v2
	v_mov_b32_e32 v20, v2
	v_mov_b32_e32 v21, v2
	v_mov_b32_e32 v22, v2
	v_mov_b32_e32 v23, v2
	v_mov_b32_e32 v24, v2
	v_mov_b32_e32 v25, v2
	v_mov_b32_e32 v26, v2
	v_mov_b32_e32 v27, v2
	v_mov_b32_e32 v28, v2
	v_mov_b32_e32 v29, v2
	v_mov_b32_e32 v30, v2
	v_mov_b32_e32 v31, v2
	v_mov_b32_e32 v32, v2
	v_mov_b32_e32 v33, v2
	v_mov_b32_e32 v34, v2
	v_mov_b32_e32 v35, v2
	v_mov_b32_e32 v36, v2
	v_mov_b32_e32 v37, v2
	v_mov_b32_e32 v38, v2
	v_mov_b32_e32 v39, v2
	v_mov_b32_e32 v40, v2
	v_mov_b32_e32 v41, v2
	v_mov_b32_e32 v42, v2
	v_mov_b32_e32 v43, v2
	v_mov_b32_e32 v44, v2
	v_mov_b32_e32 v45, v2
	v_mov_b32_e32 v46, v2
	v_mov_b32_e32 v47, v2
	v_mov_b32_e32 v48, v2
	v_mov_b32_e32 v49, v2
	v_mov_b32_e32 v50, v2
	v_mov_b32_e32 v51, v2
	v_mov_b32_e32 v52, v2
	v_mov_b32_e32 v53, v2
	v_mov_b32_e32 v54, v2
	v_mov_b32_e32 v55, v2
	v_mov_b32_e32 v56, v2
	v_mov_b32_e32 v57, v2
	v_mov_b32_e32 v58, v2
	v_mov_b32_e32 v59, v2
	v_mov_b32_e32 v60, v2
	v_mov_b32_e32 v61, v2
	v_mov_b32_e32 v62, v2
	v_mov_b32_e32 v63, v2
	v_mov_b32_e32 v64, v2
	v_mov_b32_e32 v65, v2
	v_mov_b32_e32 v66, v2
	v_mov_b32_e32 v67, v2
	v_mov_b32_e32 v68, v2
	v_mov_b32_e32 v69, v2
	v_mov_b32_e32 v70, v2
	v_mov_b32_e32 v71, v2
	v_mov_b32_e32 v72, v2
	v_mov_b32_e32 v73, v2
	v_mov_b32_e32 v74, v2
	v_mov_b32_e32 v75, v2
	v_mov_b32_e32 v76, v2
	v_mov_b32_e32 v77, v2
	v_mov_b32_e32 v78, v2
	v_mov_b32_e32 v79, v2
	v_mov_b32_e32 v80, v2
	v_mov_b32_e32 v81, v2
	v_mov_b32_e32 v82, v2
	v_mov_b32_e32 v83, v2
	v_mov_b32_e32 v84, v2
	v_mov_b32_e32 v85, v2
	v_mov_b32_e32 v86, v2
	v_mov_b32_e32 v87, v2
	v_mov_b32_e32 v88, v2
	v_mov_b32_e32 v89, v2
	v_mov_b32_e32 v90, v2
	v_mov_b32_e32 v91, v2
	v_mov_b32_e32 v92, v2
	v_mov_b32_e32 v93, v2
	v_mov_b32_e32 v94, v2
	v_mov_b32_e32 v95, v2
	v_mov_b32_e32 v96, v2
	v_mov_b32_e32 v97, v2
	v_mov_b32_e32 v98, v2
	v_mov_b32_e32 v99, v2
	v_mov_b32_e32 v100, v2
	v_mov_b32_e32 v101, v2
	v_mov_b32_e32 v102, v2
	v_mov_b32_e32 v103, v2
	v_mov_b32_e32 v104, v2
	v_mov_b32_e32 v105, v2
	v_mov_b32_e32 v106, v2
	v_mov_b32_e32 v107, v2
	v_mov_b32_e32 v108, v2
	v_mov_b32_e32 v109, v2
	v_mov_b32_e32 v110, v2
	v_mov_b32_e32 v111, v2
	v_mov_b32_e32 v112, v2
	v_mov_b32_e32 v113, v2
	v_mov_b32_e32 v114, v2
	v_mov_b32_e32 v115, v2
	v_mov_b32_e32 v116, v2
	v_mov_b32_e32 v117, v2
	v_mov_b32_e32 v118, v2
	v_mov_b32_e32 v119, v2
	v_mov_b32_e32 v120, v2
	v_mov_b32_e32 v121, v2
	v_mov_b32_e32 v122, v2
	v_mov_b32_e32 v123, v2
	v_mov_b32_e32 v124, v2
	v_mov_b32_e32 v125, v2
	v_mov_b32_e32 v126, v2
	v_mov_b32_e32 v127, v2
	v_mov_b32_e32 v128, v2
	v_mov_b32_e32 v129, v2
	v_lshlrev_b32_e32 v145, 1, v141
	v_or_b32_e32 v146, 0x4000, v142
	s_branch .LBB0_11
.LBB0_10:                               ; %.critedge1041
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_xor_b32 s2, s2, 1
	s_xor_b32 s3, s3, 1
	s_add_i32 s6, s6, -1
	s_cmp_eq_u32 s6, 0
	v_add_u32_e32 v144, 32, v144
	;;#ASMSTART
	s_waitcnt lgkmcnt(0)
	;;#ASMEND
	s_barrier
	s_cbranch_scc1 .LBB0_13
.LBB0_11:                               ; %.preheader452
                                        ; =>This Inner Loop Header: Depth=1
	buffer_load_dwordx4 v[134:137], v144, s[8:11], 0 offen
	buffer_load_dwordx4 v[130:133], v144, s[12:15], 0 offen
	; sched_group_barrier mask(0x00000020) size(2) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_lshl_b32 s0, s2, 13
	v_or_b32_e32 v147, s0, v145
	ds_read2st64_b64 v[148:151], v147 offset1:2
	v_add_u32_e32 v156, s0, v143
	ds_read2st64_b64 v[152:155], v156 offset1:2
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[114:129], v[148:149], v[152:153], v[114:129]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[154:155], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[150:151], v[152:153], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[150:151], v[154:155], v[66:81]
	ds_read2st64_b64 v[148:151], v147 offset0:4 offset1:6
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[148:149], v[152:153], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[148:149], v[154:155], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[150:151], v[152:153], v[18:33]
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[150:151], v[154:155], v[2:17]
	; sched_barrier mask(0x00000000)
	ds_read2_b64 v[148:151], v147 offset0:2 offset1:130
	ds_read2_b64 v[152:155], v156 offset0:2 offset1:130
	v_add_u32_e32 v147, 16, v147
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[114:129], v[148:149], v[152:153], v[114:129]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[148:149], v[154:155], v[98:113]
	v_mfma_f32_32x32x8_bf16 v[82:97], v[150:151], v[152:153], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[150:151], v[154:155], v[66:81]
	ds_read2st64_b64 v[148:151], v147 offset0:4 offset1:6
	; sched_group_barrier mask(0x00000100) size(6) SyncID(0)
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[148:149], v[152:153], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[148:149], v[154:155], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[150:151], v[152:153], v[18:33]
	; sched_group_barrier mask(0x00000008) size(8) SyncID(0)
	v_mfma_f32_32x32x8_bf16 v[2:17], v[150:151], v[154:155], v[2:17]
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_10
; %bb.12:                               ; %.preheader451
                                        ;   in Loop: Header=BB0_11 Depth=1
	s_lshl_b32 s19, s3, 13
	v_or_b32_e32 v147, s19, v142
	s_waitcnt vmcnt(1)
	;;#ASMSTART
	ds_write_b64 v147, v[134:135]

	;;#ASMEND
	v_add_u32_e32 v148, 8, v147
	;;#ASMSTART
	ds_write_b64 v148, v[136:137]

	;;#ASMEND
	v_add_u32_e32 v134, s19, v146
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	ds_write_b64 v134, v[130:131]

	;;#ASMEND
	v_add_u32_e32 v135, 8, v134
	;;#ASMSTART
	ds_write_b64 v135, v[132:133]

	;;#ASMEND
	s_branch .LBB0_10
.LBB0_13:                               ; %Flow1733
	s_lshl_b32 s0, s2, 13
	v_lshl_or_b32 v148, v141, 1, s0
	s_waitcnt vmcnt(0)
	ds_read2_b64 v[130:133], v148 offset1:2
	v_lshl_or_b32 v134, v140, 1, s0
	v_add_u32_e32 v140, 0x4000, v134
	ds_read2_b64 v[134:137], v140 offset1:2
	ds_read2_b64 v[140:143], v140 offset0:128 offset1:130
	ds_read2_b64 v[144:147], v148 offset0:128 offset1:130
	v_and_b32_e32 v0, 0xc0, v0
	v_or3_b32 v0, v0, s18, v1
	v_add_u32_e32 v1, s7, v139
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[114:129], v[130:131], v[134:135], v[114:129]
	v_cmp_gt_i32_e32 vcc, s5, v0
	v_mfma_f32_32x32x8_bf16 v[98:113], v[130:131], v[140:141], v[98:113]
	v_add_u32_e32 v130, 0x800, v148
	ds_read2_b64 v[148:151], v130 offset1:2
	ds_read2_b64 v[152:155], v130 offset0:128 offset1:130
	v_mfma_f32_32x32x8_bf16 v[82:97], v[144:145], v[134:135], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[144:145], v[140:141], v[66:81]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_32x32x8_bf16 v[50:65], v[148:149], v[134:135], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[148:149], v[140:141], v[34:49]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_32x32x8_bf16 v[18:33], v[152:153], v[134:135], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[152:153], v[140:141], v[2:17]
	v_mfma_f32_32x32x8_bf16 v[114:129], v[132:133], v[136:137], v[114:129]
	v_mfma_f32_32x32x8_bf16 v[98:113], v[132:133], v[142:143], v[98:113]
	v_or_b32_e32 v132, v1, v138
	v_ashrrev_i32_e32 v1, 31, v0
	v_mfma_f32_32x32x8_bf16 v[82:97], v[146:147], v[136:137], v[82:97]
	v_mfma_f32_32x32x8_bf16 v[66:81], v[146:147], v[142:143], v[66:81]
	v_mfma_f32_32x32x8_bf16 v[50:65], v[150:151], v[136:137], v[50:65]
	v_mfma_f32_32x32x8_bf16 v[34:49], v[150:151], v[142:143], v[34:49]
	v_mfma_f32_32x32x8_bf16 v[18:33], v[154:155], v[136:137], v[18:33]
	v_mfma_f32_32x32x8_bf16 v[2:17], v[154:155], v[142:143], v[2:17]
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_46
; %bb.14:
	v_lshl_add_u64 v[130:131], v[0:1], 1, s[16:17]
	v_cmp_gt_i32_e64 s[0:1], s4, v132
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_16
; %bb.15:
	v_bfe_u32 v133, v114, 16, 1
	s_movk_i32 s0, 0x7fff
	v_add3_u32 v133, v133, v114, s0
	v_or_b32_e32 v134, 0x400000, v114
	v_cmp_u_f32_e64 s[0:1], v114, v114
	s_nop 1
	v_cndmask_b32_e64 v114, v133, v134, s[0:1]
	v_mul_lo_u32 v134, v132, s5
	v_ashrrev_i32_e32 v135, 31, v134
	v_lshl_add_u64 v[134:135], v[134:135], 1, v[130:131]
	global_store_short_d16_hi v[134:135], v114, off
.LBB0_16:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v114, 1, v132
	v_cmp_gt_i32_e64 s[0:1], s4, v114
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_18
; %bb.17:
	v_bfe_u32 v133, v115, 16, 1
	s_movk_i32 s0, 0x7fff
	v_mul_lo_u32 v114, v114, s5
	v_add3_u32 v133, v133, v115, s0
	v_or_b32_e32 v134, 0x400000, v115
	v_cmp_u_f32_e64 s[0:1], v115, v115
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, v[130:131]
	v_cndmask_b32_e64 v133, v133, v134, s[0:1]
	global_store_short_d16_hi v[114:115], v133, off
.LBB0_18:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v114, 2, v132
	v_cmp_gt_i32_e64 s[0:1], s4, v114
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_20
; %bb.19:
	v_bfe_u32 v115, v116, 16, 1
	s_movk_i32 s0, 0x7fff
	v_add3_u32 v115, v115, v116, s0
	v_or_b32_e32 v133, 0x400000, v116
	v_cmp_u_f32_e64 s[0:1], v116, v116
	v_mul_lo_u32 v114, v114, s5
	s_nop 0
	v_cndmask_b32_e64 v116, v115, v133, s[0:1]
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, v[130:131]
	global_store_short_d16_hi v[114:115], v116, off
.LBB0_20:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v114, 3, v132
	v_cmp_gt_i32_e64 s[0:1], s4, v114
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_22
; %bb.21:
	v_bfe_u32 v115, v117, 16, 1
	s_movk_i32 s0, 0x7fff
	v_add3_u32 v115, v115, v117, s0
	v_or_b32_e32 v116, 0x400000, v117
	v_cmp_u_f32_e64 s[0:1], v117, v117
	v_mul_lo_u32 v114, v114, s5
	s_nop 0
	v_cndmask_b32_e64 v116, v115, v116, s[0:1]
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, v[130:131]
	global_store_short_d16_hi v[114:115], v116, off
.LBB0_22:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v114, 8, v132
	v_cmp_gt_i32_e64 s[0:1], s4, v114
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_24
; %bb.23:
	v_bfe_u32 v115, v118, 16, 1
	s_movk_i32 s0, 0x7fff
	v_add3_u32 v115, v115, v118, s0
	v_or_b32_e32 v116, 0x400000, v118
	v_cmp_u_f32_e64 s[0:1], v118, v118
	v_mul_lo_u32 v114, v114, s5
	s_nop 0
	v_cndmask_b32_e64 v116, v115, v116, s[0:1]
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, v[130:131]
	global_store_short_d16_hi v[114:115], v116, off
.LBB0_24:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v114, 9, v132
	v_cmp_gt_i32_e64 s[0:1], s4, v114
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_26
; %bb.25:
	v_bfe_u32 v115, v119, 16, 1
	s_movk_i32 s0, 0x7fff
	v_add3_u32 v115, v115, v119, s0
	v_or_b32_e32 v116, 0x400000, v119
	v_cmp_u_f32_e64 s[0:1], v119, v119
	v_mul_lo_u32 v114, v114, s5
	s_nop 0
	v_cndmask_b32_e64 v116, v115, v116, s[0:1]
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, v[130:131]
	global_store_short_d16_hi v[114:115], v116, off
.LBB0_26:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v114, 10, v132
	v_cmp_gt_i32_e64 s[0:1], s4, v114
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_28
; %bb.27:
	v_bfe_u32 v115, v120, 16, 1
	s_movk_i32 s0, 0x7fff
	v_add3_u32 v115, v115, v120, s0
	v_or_b32_e32 v116, 0x400000, v120
	v_cmp_u_f32_e64 s[0:1], v120, v120
	v_mul_lo_u32 v114, v114, s5
	s_nop 0
	v_cndmask_b32_e64 v116, v115, v116, s[0:1]
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, v[130:131]
	global_store_short_d16_hi v[114:115], v116, off
.LBB0_28:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v114, 11, v132
	v_cmp_gt_i32_e64 s[0:1], s4, v114
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_30
; %bb.29:
	v_bfe_u32 v115, v121, 16, 1
	s_movk_i32 s0, 0x7fff
	v_add3_u32 v115, v115, v121, s0
	v_or_b32_e32 v116, 0x400000, v121
	v_cmp_u_f32_e64 s[0:1], v121, v121
	v_mul_lo_u32 v114, v114, s5
	s_nop 0
	v_cndmask_b32_e64 v116, v115, v116, s[0:1]
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, v[130:131]
	global_store_short_d16_hi v[114:115], v116, off
.LBB0_30:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v114, 16, v132
	v_cmp_gt_i32_e64 s[0:1], s4, v114
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_32
; %bb.31:
	v_bfe_u32 v115, v122, 16, 1
	s_movk_i32 s0, 0x7fff
	v_add3_u32 v115, v115, v122, s0
	v_or_b32_e32 v116, 0x400000, v122
	v_cmp_u_f32_e64 s[0:1], v122, v122
	v_mul_lo_u32 v114, v114, s5
	s_nop 0
	v_cndmask_b32_e64 v116, v115, v116, s[0:1]
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, v[130:131]
	global_store_short_d16_hi v[114:115], v116, off
.LBB0_32:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v114, 17, v132
	v_cmp_gt_i32_e64 s[0:1], s4, v114
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_34
; %bb.33:
	v_bfe_u32 v115, v123, 16, 1
	s_movk_i32 s0, 0x7fff
	v_add3_u32 v115, v115, v123, s0
	v_or_b32_e32 v116, 0x400000, v123
	v_cmp_u_f32_e64 s[0:1], v123, v123
	v_mul_lo_u32 v114, v114, s5
	s_nop 0
	v_cndmask_b32_e64 v116, v115, v116, s[0:1]
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, v[130:131]
	global_store_short_d16_hi v[114:115], v116, off
.LBB0_34:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v114, 18, v132
	v_cmp_gt_i32_e64 s[0:1], s4, v114
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_36
; %bb.35:
	v_bfe_u32 v115, v124, 16, 1
	s_movk_i32 s0, 0x7fff
	v_add3_u32 v115, v115, v124, s0
	v_or_b32_e32 v116, 0x400000, v124
	v_cmp_u_f32_e64 s[0:1], v124, v124
	v_mul_lo_u32 v114, v114, s5
	s_nop 0
	v_cndmask_b32_e64 v116, v115, v116, s[0:1]
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, v[130:131]
	global_store_short_d16_hi v[114:115], v116, off
.LBB0_36:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v114, 19, v132
	v_cmp_gt_i32_e64 s[0:1], s4, v114
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_38
; %bb.37:
	v_bfe_u32 v115, v125, 16, 1
	s_movk_i32 s0, 0x7fff
	v_add3_u32 v115, v115, v125, s0
	v_or_b32_e32 v116, 0x400000, v125
	v_cmp_u_f32_e64 s[0:1], v125, v125
	v_mul_lo_u32 v114, v114, s5
	s_nop 0
	v_cndmask_b32_e64 v116, v115, v116, s[0:1]
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, v[130:131]
	global_store_short_d16_hi v[114:115], v116, off
.LBB0_38:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v114, 24, v132
	v_cmp_gt_i32_e64 s[0:1], s4, v114
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_40
; %bb.39:
	v_bfe_u32 v115, v126, 16, 1
	s_movk_i32 s0, 0x7fff
	v_add3_u32 v115, v115, v126, s0
	v_or_b32_e32 v116, 0x400000, v126
	v_cmp_u_f32_e64 s[0:1], v126, v126
	v_mul_lo_u32 v114, v114, s5
	s_nop 0
	v_cndmask_b32_e64 v116, v115, v116, s[0:1]
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, v[130:131]
	global_store_short_d16_hi v[114:115], v116, off
.LBB0_40:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v114, 25, v132
	v_cmp_gt_i32_e64 s[0:1], s4, v114
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_42
; %bb.41:
	v_bfe_u32 v115, v127, 16, 1
	s_movk_i32 s0, 0x7fff
	v_add3_u32 v115, v115, v127, s0
	v_or_b32_e32 v116, 0x400000, v127
	v_cmp_u_f32_e64 s[0:1], v127, v127
	v_mul_lo_u32 v114, v114, s5
	s_nop 0
	v_cndmask_b32_e64 v116, v115, v116, s[0:1]
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, v[130:131]
	global_store_short_d16_hi v[114:115], v116, off
.LBB0_42:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v114, 26, v132
	v_cmp_gt_i32_e64 s[0:1], s4, v114
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_44
; %bb.43:
	v_bfe_u32 v115, v128, 16, 1
	s_movk_i32 s0, 0x7fff
	v_add3_u32 v115, v115, v128, s0
	v_or_b32_e32 v116, 0x400000, v128
	v_cmp_u_f32_e64 s[0:1], v128, v128
	v_mul_lo_u32 v114, v114, s5
	s_nop 0
	v_cndmask_b32_e64 v116, v115, v116, s[0:1]
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, v[130:131]
	global_store_short_d16_hi v[114:115], v116, off
.LBB0_44:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v114, 27, v132
	v_cmp_gt_i32_e64 s[0:1], s4, v114
	s_and_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_46
; %bb.45:
	v_bfe_u32 v115, v129, 16, 1
	s_movk_i32 s0, 0x7fff
	v_add3_u32 v115, v115, v129, s0
	v_or_b32_e32 v116, 0x400000, v129
	v_cmp_u_f32_e64 s[0:1], v129, v129
	v_mul_lo_u32 v114, v114, s5
	s_nop 0
	v_cndmask_b32_e64 v116, v115, v116, s[0:1]
	v_ashrrev_i32_e32 v115, 31, v114
	v_lshl_add_u64 v[114:115], v[114:115], 1, v[130:131]
	global_store_short_d16_hi v[114:115], v116, off
.LBB0_46:                               ; %Flow1731
	s_or_b64 exec, exec, s[2:3]
	v_or_b32_e32 v114, 32, v0
	v_cmp_gt_i32_e64 s[0:1], s5, v114
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_79
; %bb.47:
	v_lshl_add_u64 v[114:115], v[0:1], 1, s[16:17]
	v_cmp_gt_i32_e64 s[2:3], s4, v132
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_49
; %bb.48:
	v_bfe_u32 v116, v98, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v116, v116, v98, s2
	v_or_b32_e32 v117, 0x400000, v98
	v_cmp_u_f32_e64 s[2:3], v98, v98
	s_nop 1
	v_cndmask_b32_e64 v98, v116, v117, s[2:3]
	v_mul_lo_u32 v116, v132, s5
	v_ashrrev_i32_e32 v117, 31, v116
	v_lshl_add_u64 v[116:117], v[116:117], 1, v[114:115]
	global_store_short_d16_hi v[116:117], v98, off offset:64
.LBB0_49:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v98, 1, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v98
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_51
; %bb.50:
	v_bfe_u32 v116, v99, 16, 1
	s_movk_i32 s2, 0x7fff
	v_mul_lo_u32 v98, v98, s5
	v_add3_u32 v116, v116, v99, s2
	v_or_b32_e32 v117, 0x400000, v99
	v_cmp_u_f32_e64 s[2:3], v99, v99
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 1, v[114:115]
	v_cndmask_b32_e64 v116, v116, v117, s[2:3]
	global_store_short_d16_hi v[98:99], v116, off offset:64
.LBB0_51:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v98, 2, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v98
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_53
; %bb.52:
	v_bfe_u32 v99, v100, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v99, v99, v100, s2
	v_or_b32_e32 v116, 0x400000, v100
	v_cmp_u_f32_e64 s[2:3], v100, v100
	v_mul_lo_u32 v98, v98, s5
	s_nop 0
	v_cndmask_b32_e64 v100, v99, v116, s[2:3]
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 1, v[114:115]
	global_store_short_d16_hi v[98:99], v100, off offset:64
.LBB0_53:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v98, 3, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v98
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_55
; %bb.54:
	v_bfe_u32 v99, v101, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v99, v99, v101, s2
	v_or_b32_e32 v100, 0x400000, v101
	v_cmp_u_f32_e64 s[2:3], v101, v101
	v_mul_lo_u32 v98, v98, s5
	s_nop 0
	v_cndmask_b32_e64 v100, v99, v100, s[2:3]
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 1, v[114:115]
	global_store_short_d16_hi v[98:99], v100, off offset:64
.LBB0_55:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v98, 8, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v98
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_57
; %bb.56:
	v_bfe_u32 v99, v102, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v99, v99, v102, s2
	v_or_b32_e32 v100, 0x400000, v102
	v_cmp_u_f32_e64 s[2:3], v102, v102
	v_mul_lo_u32 v98, v98, s5
	s_nop 0
	v_cndmask_b32_e64 v100, v99, v100, s[2:3]
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 1, v[114:115]
	global_store_short_d16_hi v[98:99], v100, off offset:64
.LBB0_57:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v98, 9, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v98
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_59
; %bb.58:
	v_bfe_u32 v99, v103, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v99, v99, v103, s2
	v_or_b32_e32 v100, 0x400000, v103
	v_cmp_u_f32_e64 s[2:3], v103, v103
	v_mul_lo_u32 v98, v98, s5
	s_nop 0
	v_cndmask_b32_e64 v100, v99, v100, s[2:3]
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 1, v[114:115]
	global_store_short_d16_hi v[98:99], v100, off offset:64
.LBB0_59:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v98, 10, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v98
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_61
; %bb.60:
	v_bfe_u32 v99, v104, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v99, v99, v104, s2
	v_or_b32_e32 v100, 0x400000, v104
	v_cmp_u_f32_e64 s[2:3], v104, v104
	v_mul_lo_u32 v98, v98, s5
	s_nop 0
	v_cndmask_b32_e64 v100, v99, v100, s[2:3]
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 1, v[114:115]
	global_store_short_d16_hi v[98:99], v100, off offset:64
.LBB0_61:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v98, 11, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v98
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_63
; %bb.62:
	v_bfe_u32 v99, v105, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v99, v99, v105, s2
	v_or_b32_e32 v100, 0x400000, v105
	v_cmp_u_f32_e64 s[2:3], v105, v105
	v_mul_lo_u32 v98, v98, s5
	s_nop 0
	v_cndmask_b32_e64 v100, v99, v100, s[2:3]
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 1, v[114:115]
	global_store_short_d16_hi v[98:99], v100, off offset:64
.LBB0_63:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v98, 16, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v98
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_65
; %bb.64:
	v_bfe_u32 v99, v106, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v99, v99, v106, s2
	v_or_b32_e32 v100, 0x400000, v106
	v_cmp_u_f32_e64 s[2:3], v106, v106
	v_mul_lo_u32 v98, v98, s5
	s_nop 0
	v_cndmask_b32_e64 v100, v99, v100, s[2:3]
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 1, v[114:115]
	global_store_short_d16_hi v[98:99], v100, off offset:64
.LBB0_65:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v98, 17, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v98
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_67
; %bb.66:
	v_bfe_u32 v99, v107, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v99, v99, v107, s2
	v_or_b32_e32 v100, 0x400000, v107
	v_cmp_u_f32_e64 s[2:3], v107, v107
	v_mul_lo_u32 v98, v98, s5
	s_nop 0
	v_cndmask_b32_e64 v100, v99, v100, s[2:3]
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 1, v[114:115]
	global_store_short_d16_hi v[98:99], v100, off offset:64
.LBB0_67:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v98, 18, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v98
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_69
; %bb.68:
	v_bfe_u32 v99, v108, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v99, v99, v108, s2
	v_or_b32_e32 v100, 0x400000, v108
	v_cmp_u_f32_e64 s[2:3], v108, v108
	v_mul_lo_u32 v98, v98, s5
	s_nop 0
	v_cndmask_b32_e64 v100, v99, v100, s[2:3]
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 1, v[114:115]
	global_store_short_d16_hi v[98:99], v100, off offset:64
.LBB0_69:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v98, 19, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v98
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_71
; %bb.70:
	v_bfe_u32 v99, v109, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v99, v99, v109, s2
	v_or_b32_e32 v100, 0x400000, v109
	v_cmp_u_f32_e64 s[2:3], v109, v109
	v_mul_lo_u32 v98, v98, s5
	s_nop 0
	v_cndmask_b32_e64 v100, v99, v100, s[2:3]
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 1, v[114:115]
	global_store_short_d16_hi v[98:99], v100, off offset:64
.LBB0_71:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v98, 24, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v98
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_73
; %bb.72:
	v_bfe_u32 v99, v110, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v99, v99, v110, s2
	v_or_b32_e32 v100, 0x400000, v110
	v_cmp_u_f32_e64 s[2:3], v110, v110
	v_mul_lo_u32 v98, v98, s5
	s_nop 0
	v_cndmask_b32_e64 v100, v99, v100, s[2:3]
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 1, v[114:115]
	global_store_short_d16_hi v[98:99], v100, off offset:64
.LBB0_73:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v98, 25, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v98
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_75
; %bb.74:
	v_bfe_u32 v99, v111, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v99, v99, v111, s2
	v_or_b32_e32 v100, 0x400000, v111
	v_cmp_u_f32_e64 s[2:3], v111, v111
	v_mul_lo_u32 v98, v98, s5
	s_nop 0
	v_cndmask_b32_e64 v100, v99, v100, s[2:3]
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 1, v[114:115]
	global_store_short_d16_hi v[98:99], v100, off offset:64
.LBB0_75:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v98, 26, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v98
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_77
; %bb.76:
	v_bfe_u32 v99, v112, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v99, v99, v112, s2
	v_or_b32_e32 v100, 0x400000, v112
	v_cmp_u_f32_e64 s[2:3], v112, v112
	v_mul_lo_u32 v98, v98, s5
	s_nop 0
	v_cndmask_b32_e64 v100, v99, v100, s[2:3]
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 1, v[114:115]
	global_store_short_d16_hi v[98:99], v100, off offset:64
.LBB0_77:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v98, 27, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v98
	s_and_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB0_79
; %bb.78:
	v_bfe_u32 v99, v113, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v99, v99, v113, s2
	v_or_b32_e32 v100, 0x400000, v113
	v_cmp_u_f32_e64 s[2:3], v113, v113
	v_mul_lo_u32 v98, v98, s5
	s_nop 0
	v_cndmask_b32_e64 v100, v99, v100, s[2:3]
	v_ashrrev_i32_e32 v99, 31, v98
	v_lshl_add_u64 v[98:99], v[98:99], 1, v[114:115]
	global_store_short_d16_hi v[98:99], v100, off offset:64
.LBB0_79:                               ; %Flow1729
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v100, 32, v132
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_112
; %bb.80:
	v_lshl_add_u64 v[98:99], v[0:1], 1, s[16:17]
	v_cmp_gt_i32_e64 s[2:3], s4, v100
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_82
; %bb.81:
	v_bfe_u32 v101, v82, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v101, v101, v82, s2
	v_or_b32_e32 v102, 0x400000, v82
	v_cmp_u_f32_e64 s[2:3], v82, v82
	s_nop 1
	v_cndmask_b32_e64 v82, v101, v102, s[2:3]
	v_mul_lo_u32 v102, v100, s5
	v_ashrrev_i32_e32 v103, 31, v102
	v_lshl_add_u64 v[102:103], v[102:103], 1, v[98:99]
	global_store_short_d16_hi v[102:103], v82, off
.LBB0_82:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v82, 33, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v82
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_84
; %bb.83:
	v_bfe_u32 v101, v83, 16, 1
	s_movk_i32 s2, 0x7fff
	v_mul_lo_u32 v82, v82, s5
	v_add3_u32 v101, v101, v83, s2
	v_or_b32_e32 v102, 0x400000, v83
	v_cmp_u_f32_e64 s[2:3], v83, v83
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshl_add_u64 v[82:83], v[82:83], 1, v[98:99]
	v_cndmask_b32_e64 v101, v101, v102, s[2:3]
	global_store_short_d16_hi v[82:83], v101, off
.LBB0_84:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v82, 34, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v82
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_86
; %bb.85:
	v_bfe_u32 v83, v84, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v83, v83, v84, s2
	v_or_b32_e32 v101, 0x400000, v84
	v_cmp_u_f32_e64 s[2:3], v84, v84
	v_mul_lo_u32 v82, v82, s5
	s_nop 0
	v_cndmask_b32_e64 v84, v83, v101, s[2:3]
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshl_add_u64 v[82:83], v[82:83], 1, v[98:99]
	global_store_short_d16_hi v[82:83], v84, off
.LBB0_86:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v82, 35, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v82
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_88
; %bb.87:
	v_bfe_u32 v83, v85, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v83, v83, v85, s2
	v_or_b32_e32 v84, 0x400000, v85
	v_cmp_u_f32_e64 s[2:3], v85, v85
	v_mul_lo_u32 v82, v82, s5
	s_nop 0
	v_cndmask_b32_e64 v84, v83, v84, s[2:3]
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshl_add_u64 v[82:83], v[82:83], 1, v[98:99]
	global_store_short_d16_hi v[82:83], v84, off
.LBB0_88:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v82, 40, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v82
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_90
; %bb.89:
	v_bfe_u32 v83, v86, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v83, v83, v86, s2
	v_or_b32_e32 v84, 0x400000, v86
	v_cmp_u_f32_e64 s[2:3], v86, v86
	v_mul_lo_u32 v82, v82, s5
	s_nop 0
	v_cndmask_b32_e64 v84, v83, v84, s[2:3]
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshl_add_u64 v[82:83], v[82:83], 1, v[98:99]
	global_store_short_d16_hi v[82:83], v84, off
.LBB0_90:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v82, 41, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v82
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_92
; %bb.91:
	v_bfe_u32 v83, v87, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v83, v83, v87, s2
	v_or_b32_e32 v84, 0x400000, v87
	v_cmp_u_f32_e64 s[2:3], v87, v87
	v_mul_lo_u32 v82, v82, s5
	s_nop 0
	v_cndmask_b32_e64 v84, v83, v84, s[2:3]
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshl_add_u64 v[82:83], v[82:83], 1, v[98:99]
	global_store_short_d16_hi v[82:83], v84, off
.LBB0_92:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v82, 42, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v82
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_94
; %bb.93:
	v_bfe_u32 v83, v88, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v83, v83, v88, s2
	v_or_b32_e32 v84, 0x400000, v88
	v_cmp_u_f32_e64 s[2:3], v88, v88
	v_mul_lo_u32 v82, v82, s5
	s_nop 0
	v_cndmask_b32_e64 v84, v83, v84, s[2:3]
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshl_add_u64 v[82:83], v[82:83], 1, v[98:99]
	global_store_short_d16_hi v[82:83], v84, off
.LBB0_94:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v82, 43, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v82
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_96
; %bb.95:
	v_bfe_u32 v83, v89, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v83, v83, v89, s2
	v_or_b32_e32 v84, 0x400000, v89
	v_cmp_u_f32_e64 s[2:3], v89, v89
	v_mul_lo_u32 v82, v82, s5
	s_nop 0
	v_cndmask_b32_e64 v84, v83, v84, s[2:3]
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshl_add_u64 v[82:83], v[82:83], 1, v[98:99]
	global_store_short_d16_hi v[82:83], v84, off
.LBB0_96:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v82, 48, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v82
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_98
; %bb.97:
	v_bfe_u32 v83, v90, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v83, v83, v90, s2
	v_or_b32_e32 v84, 0x400000, v90
	v_cmp_u_f32_e64 s[2:3], v90, v90
	v_mul_lo_u32 v82, v82, s5
	s_nop 0
	v_cndmask_b32_e64 v84, v83, v84, s[2:3]
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshl_add_u64 v[82:83], v[82:83], 1, v[98:99]
	global_store_short_d16_hi v[82:83], v84, off
.LBB0_98:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v82, 49, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v82
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_100
; %bb.99:
	v_bfe_u32 v83, v91, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v83, v83, v91, s2
	v_or_b32_e32 v84, 0x400000, v91
	v_cmp_u_f32_e64 s[2:3], v91, v91
	v_mul_lo_u32 v82, v82, s5
	s_nop 0
	v_cndmask_b32_e64 v84, v83, v84, s[2:3]
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshl_add_u64 v[82:83], v[82:83], 1, v[98:99]
	global_store_short_d16_hi v[82:83], v84, off
.LBB0_100:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v82, 50, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v82
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_102
; %bb.101:
	v_bfe_u32 v83, v92, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v83, v83, v92, s2
	v_or_b32_e32 v84, 0x400000, v92
	v_cmp_u_f32_e64 s[2:3], v92, v92
	v_mul_lo_u32 v82, v82, s5
	s_nop 0
	v_cndmask_b32_e64 v84, v83, v84, s[2:3]
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshl_add_u64 v[82:83], v[82:83], 1, v[98:99]
	global_store_short_d16_hi v[82:83], v84, off
.LBB0_102:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v82, 51, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v82
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_104
; %bb.103:
	v_bfe_u32 v83, v93, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v83, v83, v93, s2
	v_or_b32_e32 v84, 0x400000, v93
	v_cmp_u_f32_e64 s[2:3], v93, v93
	v_mul_lo_u32 v82, v82, s5
	s_nop 0
	v_cndmask_b32_e64 v84, v83, v84, s[2:3]
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshl_add_u64 v[82:83], v[82:83], 1, v[98:99]
	global_store_short_d16_hi v[82:83], v84, off
.LBB0_104:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v82, 56, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v82
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_106
; %bb.105:
	v_bfe_u32 v83, v94, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v83, v83, v94, s2
	v_or_b32_e32 v84, 0x400000, v94
	v_cmp_u_f32_e64 s[2:3], v94, v94
	v_mul_lo_u32 v82, v82, s5
	s_nop 0
	v_cndmask_b32_e64 v84, v83, v84, s[2:3]
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshl_add_u64 v[82:83], v[82:83], 1, v[98:99]
	global_store_short_d16_hi v[82:83], v84, off
.LBB0_106:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v82, 57, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v82
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_108
; %bb.107:
	v_bfe_u32 v83, v95, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v83, v83, v95, s2
	v_or_b32_e32 v84, 0x400000, v95
	v_cmp_u_f32_e64 s[2:3], v95, v95
	v_mul_lo_u32 v82, v82, s5
	s_nop 0
	v_cndmask_b32_e64 v84, v83, v84, s[2:3]
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshl_add_u64 v[82:83], v[82:83], 1, v[98:99]
	global_store_short_d16_hi v[82:83], v84, off
.LBB0_108:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v82, 58, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v82
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_110
; %bb.109:
	v_bfe_u32 v83, v96, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v83, v83, v96, s2
	v_or_b32_e32 v84, 0x400000, v96
	v_cmp_u_f32_e64 s[2:3], v96, v96
	v_mul_lo_u32 v82, v82, s5
	s_nop 0
	v_cndmask_b32_e64 v84, v83, v84, s[2:3]
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshl_add_u64 v[82:83], v[82:83], 1, v[98:99]
	global_store_short_d16_hi v[82:83], v84, off
.LBB0_110:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v82, 59, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v82
	s_and_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB0_112
; %bb.111:
	v_bfe_u32 v83, v97, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v83, v83, v97, s2
	v_or_b32_e32 v84, 0x400000, v97
	v_cmp_u_f32_e64 s[2:3], v97, v97
	v_mul_lo_u32 v82, v82, s5
	s_nop 0
	v_cndmask_b32_e64 v84, v83, v84, s[2:3]
	v_ashrrev_i32_e32 v83, 31, v82
	v_lshl_add_u64 v[82:83], v[82:83], 1, v[98:99]
	global_store_short_d16_hi v[82:83], v84, off
.LBB0_112:                              ; %Flow1727
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_145
; %bb.113:
	v_lshl_add_u64 v[82:83], v[0:1], 1, s[16:17]
	v_cmp_gt_i32_e64 s[2:3], s4, v100
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_115
; %bb.114:
	v_bfe_u32 v84, v66, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v84, v84, v66, s2
	v_or_b32_e32 v85, 0x400000, v66
	v_cmp_u_f32_e64 s[2:3], v66, v66
	s_nop 1
	v_cndmask_b32_e64 v66, v84, v85, s[2:3]
	v_mul_lo_u32 v84, v100, s5
	v_ashrrev_i32_e32 v85, 31, v84
	v_lshl_add_u64 v[84:85], v[84:85], 1, v[82:83]
	global_store_short_d16_hi v[84:85], v66, off offset:64
.LBB0_115:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v66, 33, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v66
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_117
; %bb.116:
	v_bfe_u32 v84, v67, 16, 1
	s_movk_i32 s2, 0x7fff
	v_mul_lo_u32 v66, v66, s5
	v_add3_u32 v84, v84, v67, s2
	v_or_b32_e32 v85, 0x400000, v67
	v_cmp_u_f32_e64 s[2:3], v67, v67
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 1, v[82:83]
	v_cndmask_b32_e64 v84, v84, v85, s[2:3]
	global_store_short_d16_hi v[66:67], v84, off offset:64
.LBB0_117:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v66, 34, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v66
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_119
; %bb.118:
	v_bfe_u32 v67, v68, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v67, v67, v68, s2
	v_or_b32_e32 v84, 0x400000, v68
	v_cmp_u_f32_e64 s[2:3], v68, v68
	v_mul_lo_u32 v66, v66, s5
	s_nop 0
	v_cndmask_b32_e64 v68, v67, v84, s[2:3]
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 1, v[82:83]
	global_store_short_d16_hi v[66:67], v68, off offset:64
.LBB0_119:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v66, 35, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v66
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_121
; %bb.120:
	v_bfe_u32 v67, v69, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v67, v67, v69, s2
	v_or_b32_e32 v68, 0x400000, v69
	v_cmp_u_f32_e64 s[2:3], v69, v69
	v_mul_lo_u32 v66, v66, s5
	s_nop 0
	v_cndmask_b32_e64 v68, v67, v68, s[2:3]
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 1, v[82:83]
	global_store_short_d16_hi v[66:67], v68, off offset:64
.LBB0_121:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v66, 40, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v66
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_123
; %bb.122:
	v_bfe_u32 v67, v70, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v67, v67, v70, s2
	v_or_b32_e32 v68, 0x400000, v70
	v_cmp_u_f32_e64 s[2:3], v70, v70
	v_mul_lo_u32 v66, v66, s5
	s_nop 0
	v_cndmask_b32_e64 v68, v67, v68, s[2:3]
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 1, v[82:83]
	global_store_short_d16_hi v[66:67], v68, off offset:64
.LBB0_123:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v66, 41, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v66
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_125
; %bb.124:
	v_bfe_u32 v67, v71, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v67, v67, v71, s2
	v_or_b32_e32 v68, 0x400000, v71
	v_cmp_u_f32_e64 s[2:3], v71, v71
	v_mul_lo_u32 v66, v66, s5
	s_nop 0
	v_cndmask_b32_e64 v68, v67, v68, s[2:3]
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 1, v[82:83]
	global_store_short_d16_hi v[66:67], v68, off offset:64
.LBB0_125:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v66, 42, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v66
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_127
; %bb.126:
	v_bfe_u32 v67, v72, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v67, v67, v72, s2
	v_or_b32_e32 v68, 0x400000, v72
	v_cmp_u_f32_e64 s[2:3], v72, v72
	v_mul_lo_u32 v66, v66, s5
	s_nop 0
	v_cndmask_b32_e64 v68, v67, v68, s[2:3]
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 1, v[82:83]
	global_store_short_d16_hi v[66:67], v68, off offset:64
.LBB0_127:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v66, 43, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v66
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_129
; %bb.128:
	v_bfe_u32 v67, v73, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v67, v67, v73, s2
	v_or_b32_e32 v68, 0x400000, v73
	v_cmp_u_f32_e64 s[2:3], v73, v73
	v_mul_lo_u32 v66, v66, s5
	s_nop 0
	v_cndmask_b32_e64 v68, v67, v68, s[2:3]
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 1, v[82:83]
	global_store_short_d16_hi v[66:67], v68, off offset:64
.LBB0_129:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v66, 48, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v66
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_131
; %bb.130:
	v_bfe_u32 v67, v74, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v67, v67, v74, s2
	v_or_b32_e32 v68, 0x400000, v74
	v_cmp_u_f32_e64 s[2:3], v74, v74
	v_mul_lo_u32 v66, v66, s5
	s_nop 0
	v_cndmask_b32_e64 v68, v67, v68, s[2:3]
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 1, v[82:83]
	global_store_short_d16_hi v[66:67], v68, off offset:64
.LBB0_131:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v66, 49, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v66
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_133
; %bb.132:
	v_bfe_u32 v67, v75, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v67, v67, v75, s2
	v_or_b32_e32 v68, 0x400000, v75
	v_cmp_u_f32_e64 s[2:3], v75, v75
	v_mul_lo_u32 v66, v66, s5
	s_nop 0
	v_cndmask_b32_e64 v68, v67, v68, s[2:3]
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 1, v[82:83]
	global_store_short_d16_hi v[66:67], v68, off offset:64
.LBB0_133:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v66, 50, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v66
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_135
; %bb.134:
	v_bfe_u32 v67, v76, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v67, v67, v76, s2
	v_or_b32_e32 v68, 0x400000, v76
	v_cmp_u_f32_e64 s[2:3], v76, v76
	v_mul_lo_u32 v66, v66, s5
	s_nop 0
	v_cndmask_b32_e64 v68, v67, v68, s[2:3]
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 1, v[82:83]
	global_store_short_d16_hi v[66:67], v68, off offset:64
.LBB0_135:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v66, 51, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v66
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_137
; %bb.136:
	v_bfe_u32 v67, v77, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v67, v67, v77, s2
	v_or_b32_e32 v68, 0x400000, v77
	v_cmp_u_f32_e64 s[2:3], v77, v77
	v_mul_lo_u32 v66, v66, s5
	s_nop 0
	v_cndmask_b32_e64 v68, v67, v68, s[2:3]
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 1, v[82:83]
	global_store_short_d16_hi v[66:67], v68, off offset:64
.LBB0_137:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v66, 56, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v66
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_139
; %bb.138:
	v_bfe_u32 v67, v78, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v67, v67, v78, s2
	v_or_b32_e32 v68, 0x400000, v78
	v_cmp_u_f32_e64 s[2:3], v78, v78
	v_mul_lo_u32 v66, v66, s5
	s_nop 0
	v_cndmask_b32_e64 v68, v67, v68, s[2:3]
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 1, v[82:83]
	global_store_short_d16_hi v[66:67], v68, off offset:64
.LBB0_139:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v66, 57, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v66
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_141
; %bb.140:
	v_bfe_u32 v67, v79, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v67, v67, v79, s2
	v_or_b32_e32 v68, 0x400000, v79
	v_cmp_u_f32_e64 s[2:3], v79, v79
	v_mul_lo_u32 v66, v66, s5
	s_nop 0
	v_cndmask_b32_e64 v68, v67, v68, s[2:3]
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 1, v[82:83]
	global_store_short_d16_hi v[66:67], v68, off offset:64
.LBB0_141:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v66, 58, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v66
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_143
; %bb.142:
	v_bfe_u32 v67, v80, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v67, v67, v80, s2
	v_or_b32_e32 v68, 0x400000, v80
	v_cmp_u_f32_e64 s[2:3], v80, v80
	v_mul_lo_u32 v66, v66, s5
	s_nop 0
	v_cndmask_b32_e64 v68, v67, v68, s[2:3]
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 1, v[82:83]
	global_store_short_d16_hi v[66:67], v68, off offset:64
.LBB0_143:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v66, 59, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v66
	s_and_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB0_145
; %bb.144:
	v_bfe_u32 v67, v81, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v67, v67, v81, s2
	v_or_b32_e32 v68, 0x400000, v81
	v_cmp_u_f32_e64 s[2:3], v81, v81
	v_mul_lo_u32 v66, v66, s5
	s_nop 0
	v_cndmask_b32_e64 v68, v67, v68, s[2:3]
	v_ashrrev_i32_e32 v67, 31, v66
	v_lshl_add_u64 v[66:67], v[66:67], 1, v[82:83]
	global_store_short_d16_hi v[66:67], v68, off offset:64
.LBB0_145:                              ; %Flow1725
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v68, 64, v132
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_178
; %bb.146:
	v_lshl_add_u64 v[66:67], v[0:1], 1, s[16:17]
	v_cmp_gt_i32_e64 s[2:3], s4, v68
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_148
; %bb.147:
	v_bfe_u32 v69, v50, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v69, v69, v50, s2
	v_or_b32_e32 v70, 0x400000, v50
	v_cmp_u_f32_e64 s[2:3], v50, v50
	s_nop 1
	v_cndmask_b32_e64 v50, v69, v70, s[2:3]
	v_mul_lo_u32 v70, v68, s5
	v_ashrrev_i32_e32 v71, 31, v70
	v_lshl_add_u64 v[70:71], v[70:71], 1, v[66:67]
	global_store_short_d16_hi v[70:71], v50, off
.LBB0_148:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v50, 0x41, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v50
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_150
; %bb.149:
	v_bfe_u32 v69, v51, 16, 1
	s_movk_i32 s2, 0x7fff
	v_mul_lo_u32 v50, v50, s5
	v_add3_u32 v69, v69, v51, s2
	v_or_b32_e32 v70, 0x400000, v51
	v_cmp_u_f32_e64 s[2:3], v51, v51
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 1, v[66:67]
	v_cndmask_b32_e64 v69, v69, v70, s[2:3]
	global_store_short_d16_hi v[50:51], v69, off
.LBB0_150:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v50, 0x42, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v50
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_152
; %bb.151:
	v_bfe_u32 v51, v52, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v51, v51, v52, s2
	v_or_b32_e32 v69, 0x400000, v52
	v_cmp_u_f32_e64 s[2:3], v52, v52
	v_mul_lo_u32 v50, v50, s5
	s_nop 0
	v_cndmask_b32_e64 v52, v51, v69, s[2:3]
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 1, v[66:67]
	global_store_short_d16_hi v[50:51], v52, off
.LBB0_152:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v50, 0x43, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v50
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_154
; %bb.153:
	v_bfe_u32 v51, v53, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v51, v51, v53, s2
	v_or_b32_e32 v52, 0x400000, v53
	v_cmp_u_f32_e64 s[2:3], v53, v53
	v_mul_lo_u32 v50, v50, s5
	s_nop 0
	v_cndmask_b32_e64 v52, v51, v52, s[2:3]
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 1, v[66:67]
	global_store_short_d16_hi v[50:51], v52, off
.LBB0_154:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v50, 0x48, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v50
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_156
; %bb.155:
	v_bfe_u32 v51, v54, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v51, v51, v54, s2
	v_or_b32_e32 v52, 0x400000, v54
	v_cmp_u_f32_e64 s[2:3], v54, v54
	v_mul_lo_u32 v50, v50, s5
	s_nop 0
	v_cndmask_b32_e64 v52, v51, v52, s[2:3]
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 1, v[66:67]
	global_store_short_d16_hi v[50:51], v52, off
.LBB0_156:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v50, 0x49, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v50
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_158
; %bb.157:
	v_bfe_u32 v51, v55, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v51, v51, v55, s2
	v_or_b32_e32 v52, 0x400000, v55
	v_cmp_u_f32_e64 s[2:3], v55, v55
	v_mul_lo_u32 v50, v50, s5
	s_nop 0
	v_cndmask_b32_e64 v52, v51, v52, s[2:3]
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 1, v[66:67]
	global_store_short_d16_hi v[50:51], v52, off
.LBB0_158:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v50, 0x4a, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v50
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_160
; %bb.159:
	v_bfe_u32 v51, v56, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v51, v51, v56, s2
	v_or_b32_e32 v52, 0x400000, v56
	v_cmp_u_f32_e64 s[2:3], v56, v56
	v_mul_lo_u32 v50, v50, s5
	s_nop 0
	v_cndmask_b32_e64 v52, v51, v52, s[2:3]
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 1, v[66:67]
	global_store_short_d16_hi v[50:51], v52, off
.LBB0_160:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v50, 0x4b, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v50
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_162
; %bb.161:
	v_bfe_u32 v51, v57, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v51, v51, v57, s2
	v_or_b32_e32 v52, 0x400000, v57
	v_cmp_u_f32_e64 s[2:3], v57, v57
	v_mul_lo_u32 v50, v50, s5
	s_nop 0
	v_cndmask_b32_e64 v52, v51, v52, s[2:3]
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 1, v[66:67]
	global_store_short_d16_hi v[50:51], v52, off
.LBB0_162:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v50, 0x50, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v50
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_164
; %bb.163:
	v_bfe_u32 v51, v58, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v51, v51, v58, s2
	v_or_b32_e32 v52, 0x400000, v58
	v_cmp_u_f32_e64 s[2:3], v58, v58
	v_mul_lo_u32 v50, v50, s5
	s_nop 0
	v_cndmask_b32_e64 v52, v51, v52, s[2:3]
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 1, v[66:67]
	global_store_short_d16_hi v[50:51], v52, off
.LBB0_164:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v50, 0x51, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v50
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_166
; %bb.165:
	v_bfe_u32 v51, v59, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v51, v51, v59, s2
	v_or_b32_e32 v52, 0x400000, v59
	v_cmp_u_f32_e64 s[2:3], v59, v59
	v_mul_lo_u32 v50, v50, s5
	s_nop 0
	v_cndmask_b32_e64 v52, v51, v52, s[2:3]
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 1, v[66:67]
	global_store_short_d16_hi v[50:51], v52, off
.LBB0_166:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v50, 0x52, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v50
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_168
; %bb.167:
	v_bfe_u32 v51, v60, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v51, v51, v60, s2
	v_or_b32_e32 v52, 0x400000, v60
	v_cmp_u_f32_e64 s[2:3], v60, v60
	v_mul_lo_u32 v50, v50, s5
	s_nop 0
	v_cndmask_b32_e64 v52, v51, v52, s[2:3]
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 1, v[66:67]
	global_store_short_d16_hi v[50:51], v52, off
.LBB0_168:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v50, 0x53, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v50
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_170
; %bb.169:
	v_bfe_u32 v51, v61, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v51, v51, v61, s2
	v_or_b32_e32 v52, 0x400000, v61
	v_cmp_u_f32_e64 s[2:3], v61, v61
	v_mul_lo_u32 v50, v50, s5
	s_nop 0
	v_cndmask_b32_e64 v52, v51, v52, s[2:3]
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 1, v[66:67]
	global_store_short_d16_hi v[50:51], v52, off
.LBB0_170:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v50, 0x58, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v50
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_172
; %bb.171:
	v_bfe_u32 v51, v62, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v51, v51, v62, s2
	v_or_b32_e32 v52, 0x400000, v62
	v_cmp_u_f32_e64 s[2:3], v62, v62
	v_mul_lo_u32 v50, v50, s5
	s_nop 0
	v_cndmask_b32_e64 v52, v51, v52, s[2:3]
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 1, v[66:67]
	global_store_short_d16_hi v[50:51], v52, off
.LBB0_172:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v50, 0x59, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v50
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_174
; %bb.173:
	v_bfe_u32 v51, v63, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v51, v51, v63, s2
	v_or_b32_e32 v52, 0x400000, v63
	v_cmp_u_f32_e64 s[2:3], v63, v63
	v_mul_lo_u32 v50, v50, s5
	s_nop 0
	v_cndmask_b32_e64 v52, v51, v52, s[2:3]
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 1, v[66:67]
	global_store_short_d16_hi v[50:51], v52, off
.LBB0_174:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v50, 0x5a, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v50
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_176
; %bb.175:
	v_bfe_u32 v51, v64, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v51, v51, v64, s2
	v_or_b32_e32 v52, 0x400000, v64
	v_cmp_u_f32_e64 s[2:3], v64, v64
	v_mul_lo_u32 v50, v50, s5
	s_nop 0
	v_cndmask_b32_e64 v52, v51, v52, s[2:3]
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 1, v[66:67]
	global_store_short_d16_hi v[50:51], v52, off
.LBB0_176:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v50, 0x5b, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v50
	s_and_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB0_178
; %bb.177:
	v_bfe_u32 v51, v65, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v51, v51, v65, s2
	v_or_b32_e32 v52, 0x400000, v65
	v_cmp_u_f32_e64 s[2:3], v65, v65
	v_mul_lo_u32 v50, v50, s5
	s_nop 0
	v_cndmask_b32_e64 v52, v51, v52, s[2:3]
	v_ashrrev_i32_e32 v51, 31, v50
	v_lshl_add_u64 v[50:51], v[50:51], 1, v[66:67]
	global_store_short_d16_hi v[50:51], v52, off
.LBB0_178:                              ; %Flow1723
	s_or_b64 exec, exec, s[6:7]
	s_and_saveexec_b64 s[6:7], s[0:1]
	s_cbranch_execz .LBB0_211
; %bb.179:
	v_lshl_add_u64 v[50:51], v[0:1], 1, s[16:17]
	v_cmp_gt_i32_e64 s[2:3], s4, v68
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_181
; %bb.180:
	v_bfe_u32 v52, v34, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v52, v52, v34, s2
	v_or_b32_e32 v53, 0x400000, v34
	v_cmp_u_f32_e64 s[2:3], v34, v34
	s_nop 1
	v_cndmask_b32_e64 v34, v52, v53, s[2:3]
	v_mul_lo_u32 v52, v68, s5
	v_ashrrev_i32_e32 v53, 31, v52
	v_lshl_add_u64 v[52:53], v[52:53], 1, v[50:51]
	global_store_short_d16_hi v[52:53], v34, off offset:64
.LBB0_181:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v34, 0x41, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v34
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_183
; %bb.182:
	v_bfe_u32 v52, v35, 16, 1
	s_movk_i32 s2, 0x7fff
	v_mul_lo_u32 v34, v34, s5
	v_add3_u32 v52, v52, v35, s2
	v_or_b32_e32 v53, 0x400000, v35
	v_cmp_u_f32_e64 s[2:3], v35, v35
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[34:35], v[34:35], 1, v[50:51]
	v_cndmask_b32_e64 v52, v52, v53, s[2:3]
	global_store_short_d16_hi v[34:35], v52, off offset:64
.LBB0_183:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v34, 0x42, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v34
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_185
; %bb.184:
	v_bfe_u32 v35, v36, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v35, v35, v36, s2
	v_or_b32_e32 v52, 0x400000, v36
	v_cmp_u_f32_e64 s[2:3], v36, v36
	v_mul_lo_u32 v34, v34, s5
	s_nop 0
	v_cndmask_b32_e64 v36, v35, v52, s[2:3]
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[34:35], v[34:35], 1, v[50:51]
	global_store_short_d16_hi v[34:35], v36, off offset:64
.LBB0_185:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v34, 0x43, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v34
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_187
; %bb.186:
	v_bfe_u32 v35, v37, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v35, v35, v37, s2
	v_or_b32_e32 v36, 0x400000, v37
	v_cmp_u_f32_e64 s[2:3], v37, v37
	v_mul_lo_u32 v34, v34, s5
	s_nop 0
	v_cndmask_b32_e64 v36, v35, v36, s[2:3]
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[34:35], v[34:35], 1, v[50:51]
	global_store_short_d16_hi v[34:35], v36, off offset:64
.LBB0_187:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v34, 0x48, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v34
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_189
; %bb.188:
	v_bfe_u32 v35, v38, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v35, v35, v38, s2
	v_or_b32_e32 v36, 0x400000, v38
	v_cmp_u_f32_e64 s[2:3], v38, v38
	v_mul_lo_u32 v34, v34, s5
	s_nop 0
	v_cndmask_b32_e64 v36, v35, v36, s[2:3]
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[34:35], v[34:35], 1, v[50:51]
	global_store_short_d16_hi v[34:35], v36, off offset:64
.LBB0_189:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v34, 0x49, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v34
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_191
; %bb.190:
	v_bfe_u32 v35, v39, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v35, v35, v39, s2
	v_or_b32_e32 v36, 0x400000, v39
	v_cmp_u_f32_e64 s[2:3], v39, v39
	v_mul_lo_u32 v34, v34, s5
	s_nop 0
	v_cndmask_b32_e64 v36, v35, v36, s[2:3]
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[34:35], v[34:35], 1, v[50:51]
	global_store_short_d16_hi v[34:35], v36, off offset:64
.LBB0_191:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v34, 0x4a, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v34
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_193
; %bb.192:
	v_bfe_u32 v35, v40, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v35, v35, v40, s2
	v_or_b32_e32 v36, 0x400000, v40
	v_cmp_u_f32_e64 s[2:3], v40, v40
	v_mul_lo_u32 v34, v34, s5
	s_nop 0
	v_cndmask_b32_e64 v36, v35, v36, s[2:3]
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[34:35], v[34:35], 1, v[50:51]
	global_store_short_d16_hi v[34:35], v36, off offset:64
.LBB0_193:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v34, 0x4b, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v34
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_195
; %bb.194:
	v_bfe_u32 v35, v41, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v35, v35, v41, s2
	v_or_b32_e32 v36, 0x400000, v41
	v_cmp_u_f32_e64 s[2:3], v41, v41
	v_mul_lo_u32 v34, v34, s5
	s_nop 0
	v_cndmask_b32_e64 v36, v35, v36, s[2:3]
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[34:35], v[34:35], 1, v[50:51]
	global_store_short_d16_hi v[34:35], v36, off offset:64
.LBB0_195:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v34, 0x50, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v34
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_197
; %bb.196:
	v_bfe_u32 v35, v42, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v35, v35, v42, s2
	v_or_b32_e32 v36, 0x400000, v42
	v_cmp_u_f32_e64 s[2:3], v42, v42
	v_mul_lo_u32 v34, v34, s5
	s_nop 0
	v_cndmask_b32_e64 v36, v35, v36, s[2:3]
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[34:35], v[34:35], 1, v[50:51]
	global_store_short_d16_hi v[34:35], v36, off offset:64
.LBB0_197:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v34, 0x51, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v34
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_199
; %bb.198:
	v_bfe_u32 v35, v43, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v35, v35, v43, s2
	v_or_b32_e32 v36, 0x400000, v43
	v_cmp_u_f32_e64 s[2:3], v43, v43
	v_mul_lo_u32 v34, v34, s5
	s_nop 0
	v_cndmask_b32_e64 v36, v35, v36, s[2:3]
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[34:35], v[34:35], 1, v[50:51]
	global_store_short_d16_hi v[34:35], v36, off offset:64
.LBB0_199:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v34, 0x52, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v34
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_201
; %bb.200:
	v_bfe_u32 v35, v44, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v35, v35, v44, s2
	v_or_b32_e32 v36, 0x400000, v44
	v_cmp_u_f32_e64 s[2:3], v44, v44
	v_mul_lo_u32 v34, v34, s5
	s_nop 0
	v_cndmask_b32_e64 v36, v35, v36, s[2:3]
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[34:35], v[34:35], 1, v[50:51]
	global_store_short_d16_hi v[34:35], v36, off offset:64
.LBB0_201:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v34, 0x53, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v34
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_203
; %bb.202:
	v_bfe_u32 v35, v45, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v35, v35, v45, s2
	v_or_b32_e32 v36, 0x400000, v45
	v_cmp_u_f32_e64 s[2:3], v45, v45
	v_mul_lo_u32 v34, v34, s5
	s_nop 0
	v_cndmask_b32_e64 v36, v35, v36, s[2:3]
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[34:35], v[34:35], 1, v[50:51]
	global_store_short_d16_hi v[34:35], v36, off offset:64
.LBB0_203:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v34, 0x58, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v34
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_205
; %bb.204:
	v_bfe_u32 v35, v46, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v35, v35, v46, s2
	v_or_b32_e32 v36, 0x400000, v46
	v_cmp_u_f32_e64 s[2:3], v46, v46
	v_mul_lo_u32 v34, v34, s5
	s_nop 0
	v_cndmask_b32_e64 v36, v35, v36, s[2:3]
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[34:35], v[34:35], 1, v[50:51]
	global_store_short_d16_hi v[34:35], v36, off offset:64
.LBB0_205:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v34, 0x59, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v34
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_207
; %bb.206:
	v_bfe_u32 v35, v47, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v35, v35, v47, s2
	v_or_b32_e32 v36, 0x400000, v47
	v_cmp_u_f32_e64 s[2:3], v47, v47
	v_mul_lo_u32 v34, v34, s5
	s_nop 0
	v_cndmask_b32_e64 v36, v35, v36, s[2:3]
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[34:35], v[34:35], 1, v[50:51]
	global_store_short_d16_hi v[34:35], v36, off offset:64
.LBB0_207:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v34, 0x5a, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v34
	s_and_saveexec_b64 s[8:9], s[2:3]
	s_cbranch_execz .LBB0_209
; %bb.208:
	v_bfe_u32 v35, v48, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v35, v35, v48, s2
	v_or_b32_e32 v36, 0x400000, v48
	v_cmp_u_f32_e64 s[2:3], v48, v48
	v_mul_lo_u32 v34, v34, s5
	s_nop 0
	v_cndmask_b32_e64 v36, v35, v36, s[2:3]
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[34:35], v[34:35], 1, v[50:51]
	global_store_short_d16_hi v[34:35], v36, off offset:64
.LBB0_209:
	s_or_b64 exec, exec, s[8:9]
	v_or_b32_e32 v34, 0x5b, v132
	v_cmp_gt_i32_e64 s[2:3], s4, v34
	s_and_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB0_211
; %bb.210:
	v_bfe_u32 v35, v49, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v35, v35, v49, s2
	v_or_b32_e32 v36, 0x400000, v49
	v_cmp_u_f32_e64 s[2:3], v49, v49
	v_mul_lo_u32 v34, v34, s5
	s_nop 0
	v_cndmask_b32_e64 v36, v35, v36, s[2:3]
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[34:35], v[34:35], 1, v[50:51]
	global_store_short_d16_hi v[34:35], v36, off offset:64
.LBB0_211:                              ; %Flow1721
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v36, 0x60, v132
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_244
; %bb.212:
	v_lshl_add_u64 v[34:35], v[0:1], 1, s[16:17]
	v_cmp_gt_i32_e32 vcc, s4, v36
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_214
; %bb.213:
	v_bfe_u32 v37, v18, 16, 1
	s_movk_i32 s8, 0x7fff
	v_add3_u32 v37, v37, v18, s8
	v_or_b32_e32 v38, 0x400000, v18
	v_cmp_u_f32_e32 vcc, v18, v18
	s_nop 1
	v_cndmask_b32_e32 v18, v37, v38, vcc
	v_mul_lo_u32 v38, v36, s5
	v_ashrrev_i32_e32 v39, 31, v38
	v_lshl_add_u64 v[38:39], v[38:39], 1, v[34:35]
	global_store_short_d16_hi v[38:39], v18, off
.LBB0_214:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v18, 0x61, v132
	v_cmp_gt_i32_e32 vcc, s4, v18
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_216
; %bb.215:
	v_bfe_u32 v37, v19, 16, 1
	s_movk_i32 s8, 0x7fff
	v_mul_lo_u32 v18, v18, s5
	v_add3_u32 v37, v37, v19, s8
	v_or_b32_e32 v38, 0x400000, v19
	v_cmp_u_f32_e32 vcc, v19, v19
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[34:35]
	v_cndmask_b32_e32 v37, v37, v38, vcc
	global_store_short_d16_hi v[18:19], v37, off
.LBB0_216:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v18, 0x62, v132
	v_cmp_gt_i32_e32 vcc, s4, v18
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_218
; %bb.217:
	v_bfe_u32 v19, v20, 16, 1
	s_movk_i32 s8, 0x7fff
	v_add3_u32 v19, v19, v20, s8
	v_or_b32_e32 v37, 0x400000, v20
	v_cmp_u_f32_e32 vcc, v20, v20
	v_mul_lo_u32 v18, v18, s5
	s_nop 0
	v_cndmask_b32_e32 v20, v19, v37, vcc
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[34:35]
	global_store_short_d16_hi v[18:19], v20, off
.LBB0_218:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v18, 0x63, v132
	v_cmp_gt_i32_e32 vcc, s4, v18
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_220
; %bb.219:
	v_bfe_u32 v19, v21, 16, 1
	s_movk_i32 s8, 0x7fff
	v_add3_u32 v19, v19, v21, s8
	v_or_b32_e32 v20, 0x400000, v21
	v_cmp_u_f32_e32 vcc, v21, v21
	v_mul_lo_u32 v18, v18, s5
	s_nop 0
	v_cndmask_b32_e32 v20, v19, v20, vcc
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[34:35]
	global_store_short_d16_hi v[18:19], v20, off
.LBB0_220:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v18, 0x68, v132
	v_cmp_gt_i32_e32 vcc, s4, v18
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_222
; %bb.221:
	v_bfe_u32 v19, v22, 16, 1
	s_movk_i32 s8, 0x7fff
	v_add3_u32 v19, v19, v22, s8
	v_or_b32_e32 v20, 0x400000, v22
	v_cmp_u_f32_e32 vcc, v22, v22
	v_mul_lo_u32 v18, v18, s5
	s_nop 0
	v_cndmask_b32_e32 v20, v19, v20, vcc
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[34:35]
	global_store_short_d16_hi v[18:19], v20, off
.LBB0_222:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v18, 0x69, v132
	v_cmp_gt_i32_e32 vcc, s4, v18
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_224
; %bb.223:
	v_bfe_u32 v19, v23, 16, 1
	s_movk_i32 s8, 0x7fff
	v_add3_u32 v19, v19, v23, s8
	v_or_b32_e32 v20, 0x400000, v23
	v_cmp_u_f32_e32 vcc, v23, v23
	v_mul_lo_u32 v18, v18, s5
	s_nop 0
	v_cndmask_b32_e32 v20, v19, v20, vcc
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[34:35]
	global_store_short_d16_hi v[18:19], v20, off
.LBB0_224:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v18, 0x6a, v132
	v_cmp_gt_i32_e32 vcc, s4, v18
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_226
; %bb.225:
	v_bfe_u32 v19, v24, 16, 1
	s_movk_i32 s8, 0x7fff
	v_add3_u32 v19, v19, v24, s8
	v_or_b32_e32 v20, 0x400000, v24
	v_cmp_u_f32_e32 vcc, v24, v24
	v_mul_lo_u32 v18, v18, s5
	s_nop 0
	v_cndmask_b32_e32 v20, v19, v20, vcc
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[34:35]
	global_store_short_d16_hi v[18:19], v20, off
.LBB0_226:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v18, 0x6b, v132
	v_cmp_gt_i32_e32 vcc, s4, v18
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_228
; %bb.227:
	v_bfe_u32 v19, v25, 16, 1
	s_movk_i32 s8, 0x7fff
	v_add3_u32 v19, v19, v25, s8
	v_or_b32_e32 v20, 0x400000, v25
	v_cmp_u_f32_e32 vcc, v25, v25
	v_mul_lo_u32 v18, v18, s5
	s_nop 0
	v_cndmask_b32_e32 v20, v19, v20, vcc
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[34:35]
	global_store_short_d16_hi v[18:19], v20, off
.LBB0_228:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v18, 0x70, v132
	v_cmp_gt_i32_e32 vcc, s4, v18
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_230
; %bb.229:
	v_bfe_u32 v19, v26, 16, 1
	s_movk_i32 s8, 0x7fff
	v_add3_u32 v19, v19, v26, s8
	v_or_b32_e32 v20, 0x400000, v26
	v_cmp_u_f32_e32 vcc, v26, v26
	v_mul_lo_u32 v18, v18, s5
	s_nop 0
	v_cndmask_b32_e32 v20, v19, v20, vcc
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[34:35]
	global_store_short_d16_hi v[18:19], v20, off
.LBB0_230:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v18, 0x71, v132
	v_cmp_gt_i32_e32 vcc, s4, v18
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_232
; %bb.231:
	v_bfe_u32 v19, v27, 16, 1
	s_movk_i32 s8, 0x7fff
	v_add3_u32 v19, v19, v27, s8
	v_or_b32_e32 v20, 0x400000, v27
	v_cmp_u_f32_e32 vcc, v27, v27
	v_mul_lo_u32 v18, v18, s5
	s_nop 0
	v_cndmask_b32_e32 v20, v19, v20, vcc
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[34:35]
	global_store_short_d16_hi v[18:19], v20, off
.LBB0_232:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v18, 0x72, v132
	v_cmp_gt_i32_e32 vcc, s4, v18
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_234
; %bb.233:
	v_bfe_u32 v19, v28, 16, 1
	s_movk_i32 s8, 0x7fff
	v_add3_u32 v19, v19, v28, s8
	v_or_b32_e32 v20, 0x400000, v28
	v_cmp_u_f32_e32 vcc, v28, v28
	v_mul_lo_u32 v18, v18, s5
	s_nop 0
	v_cndmask_b32_e32 v20, v19, v20, vcc
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[34:35]
	global_store_short_d16_hi v[18:19], v20, off
.LBB0_234:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v18, 0x73, v132
	v_cmp_gt_i32_e32 vcc, s4, v18
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_236
; %bb.235:
	v_bfe_u32 v19, v29, 16, 1
	s_movk_i32 s8, 0x7fff
	v_add3_u32 v19, v19, v29, s8
	v_or_b32_e32 v20, 0x400000, v29
	v_cmp_u_f32_e32 vcc, v29, v29
	v_mul_lo_u32 v18, v18, s5
	s_nop 0
	v_cndmask_b32_e32 v20, v19, v20, vcc
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[34:35]
	global_store_short_d16_hi v[18:19], v20, off
.LBB0_236:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v18, 0x78, v132
	v_cmp_gt_i32_e32 vcc, s4, v18
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_238
; %bb.237:
	v_bfe_u32 v19, v30, 16, 1
	s_movk_i32 s8, 0x7fff
	v_add3_u32 v19, v19, v30, s8
	v_or_b32_e32 v20, 0x400000, v30
	v_cmp_u_f32_e32 vcc, v30, v30
	v_mul_lo_u32 v18, v18, s5
	s_nop 0
	v_cndmask_b32_e32 v20, v19, v20, vcc
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[34:35]
	global_store_short_d16_hi v[18:19], v20, off
.LBB0_238:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v18, 0x79, v132
	v_cmp_gt_i32_e32 vcc, s4, v18
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_240
; %bb.239:
	v_bfe_u32 v19, v31, 16, 1
	s_movk_i32 s8, 0x7fff
	v_add3_u32 v19, v19, v31, s8
	v_or_b32_e32 v20, 0x400000, v31
	v_cmp_u_f32_e32 vcc, v31, v31
	v_mul_lo_u32 v18, v18, s5
	s_nop 0
	v_cndmask_b32_e32 v20, v19, v20, vcc
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[34:35]
	global_store_short_d16_hi v[18:19], v20, off
.LBB0_240:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v18, 0x7a, v132
	v_cmp_gt_i32_e32 vcc, s4, v18
	s_and_saveexec_b64 s[6:7], vcc
	s_cbranch_execz .LBB0_242
; %bb.241:
	v_bfe_u32 v19, v32, 16, 1
	s_movk_i32 s8, 0x7fff
	v_add3_u32 v19, v19, v32, s8
	v_or_b32_e32 v20, 0x400000, v32
	v_cmp_u_f32_e32 vcc, v32, v32
	v_mul_lo_u32 v18, v18, s5
	s_nop 0
	v_cndmask_b32_e32 v20, v19, v20, vcc
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[34:35]
	global_store_short_d16_hi v[18:19], v20, off
.LBB0_242:
	s_or_b64 exec, exec, s[6:7]
	v_or_b32_e32 v18, 0x7b, v132
	v_cmp_gt_i32_e32 vcc, s4, v18
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_244
; %bb.243:
	v_bfe_u32 v19, v33, 16, 1
	s_movk_i32 s6, 0x7fff
	v_add3_u32 v19, v19, v33, s6
	v_or_b32_e32 v20, 0x400000, v33
	v_cmp_u_f32_e32 vcc, v33, v33
	v_mul_lo_u32 v18, v18, s5
	s_nop 0
	v_cndmask_b32_e32 v20, v19, v20, vcc
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[34:35]
	global_store_short_d16_hi v[18:19], v20, off
.LBB0_244:                              ; %Flow1719
	s_or_b64 exec, exec, s[2:3]
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_277
; %bb.245:
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[16:17]
	v_cmp_gt_i32_e32 vcc, s4, v36
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_247
; %bb.246:
	v_bfe_u32 v18, v2, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v18, v18, v2, s2
	v_or_b32_e32 v19, 0x400000, v2
	v_cmp_u_f32_e32 vcc, v2, v2
	s_nop 1
	v_cndmask_b32_e32 v2, v18, v19, vcc
	v_mul_lo_u32 v18, v36, s5
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[0:1]
	global_store_short_d16_hi v[18:19], v2, off offset:64
.LBB0_247:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 0x61, v132
	v_cmp_gt_i32_e32 vcc, s4, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_249
; %bb.248:
	v_bfe_u32 v18, v3, 16, 1
	s_movk_i32 s2, 0x7fff
	v_mul_lo_u32 v2, v2, s5
	v_add3_u32 v18, v18, v3, s2
	v_or_b32_e32 v19, 0x400000, v3
	v_cmp_u_f32_e32 vcc, v3, v3
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	v_cndmask_b32_e32 v18, v18, v19, vcc
	global_store_short_d16_hi v[2:3], v18, off offset:64
.LBB0_249:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 0x62, v132
	v_cmp_gt_i32_e32 vcc, s4, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_251
; %bb.250:
	v_bfe_u32 v3, v4, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v3, v3, v4, s2
	v_or_b32_e32 v18, 0x400000, v4
	v_cmp_u_f32_e32 vcc, v4, v4
	v_mul_lo_u32 v2, v2, s5
	s_nop 0
	v_cndmask_b32_e32 v4, v3, v18, vcc
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], v4, off offset:64
.LBB0_251:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 0x63, v132
	v_cmp_gt_i32_e32 vcc, s4, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_253
; %bb.252:
	v_bfe_u32 v3, v5, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v3, v3, v5, s2
	v_or_b32_e32 v4, 0x400000, v5
	v_cmp_u_f32_e32 vcc, v5, v5
	v_mul_lo_u32 v2, v2, s5
	s_nop 0
	v_cndmask_b32_e32 v4, v3, v4, vcc
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], v4, off offset:64
.LBB0_253:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 0x68, v132
	v_cmp_gt_i32_e32 vcc, s4, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_255
; %bb.254:
	v_bfe_u32 v3, v6, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v3, v3, v6, s2
	v_or_b32_e32 v4, 0x400000, v6
	v_cmp_u_f32_e32 vcc, v6, v6
	v_mul_lo_u32 v2, v2, s5
	s_nop 0
	v_cndmask_b32_e32 v4, v3, v4, vcc
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], v4, off offset:64
.LBB0_255:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 0x69, v132
	v_cmp_gt_i32_e32 vcc, s4, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_257
; %bb.256:
	v_bfe_u32 v3, v7, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v3, v3, v7, s2
	v_or_b32_e32 v4, 0x400000, v7
	v_cmp_u_f32_e32 vcc, v7, v7
	v_mul_lo_u32 v2, v2, s5
	s_nop 0
	v_cndmask_b32_e32 v4, v3, v4, vcc
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], v4, off offset:64
.LBB0_257:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 0x6a, v132
	v_cmp_gt_i32_e32 vcc, s4, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_259
; %bb.258:
	v_bfe_u32 v3, v8, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v3, v3, v8, s2
	v_or_b32_e32 v4, 0x400000, v8
	v_cmp_u_f32_e32 vcc, v8, v8
	v_mul_lo_u32 v2, v2, s5
	s_nop 0
	v_cndmask_b32_e32 v4, v3, v4, vcc
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], v4, off offset:64
.LBB0_259:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 0x6b, v132
	v_cmp_gt_i32_e32 vcc, s4, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_261
; %bb.260:
	v_bfe_u32 v3, v9, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v3, v3, v9, s2
	v_or_b32_e32 v4, 0x400000, v9
	v_cmp_u_f32_e32 vcc, v9, v9
	v_mul_lo_u32 v2, v2, s5
	s_nop 0
	v_cndmask_b32_e32 v4, v3, v4, vcc
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], v4, off offset:64
.LBB0_261:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 0x70, v132
	v_cmp_gt_i32_e32 vcc, s4, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_263
; %bb.262:
	v_bfe_u32 v3, v10, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v3, v3, v10, s2
	v_or_b32_e32 v4, 0x400000, v10
	v_cmp_u_f32_e32 vcc, v10, v10
	v_mul_lo_u32 v2, v2, s5
	s_nop 0
	v_cndmask_b32_e32 v4, v3, v4, vcc
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], v4, off offset:64
.LBB0_263:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 0x71, v132
	v_cmp_gt_i32_e32 vcc, s4, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_265
; %bb.264:
	v_bfe_u32 v3, v11, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v3, v3, v11, s2
	v_or_b32_e32 v4, 0x400000, v11
	v_cmp_u_f32_e32 vcc, v11, v11
	v_mul_lo_u32 v2, v2, s5
	s_nop 0
	v_cndmask_b32_e32 v4, v3, v4, vcc
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], v4, off offset:64
.LBB0_265:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 0x72, v132
	v_cmp_gt_i32_e32 vcc, s4, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_267
; %bb.266:
	v_bfe_u32 v3, v12, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v3, v3, v12, s2
	v_or_b32_e32 v4, 0x400000, v12
	v_cmp_u_f32_e32 vcc, v12, v12
	v_mul_lo_u32 v2, v2, s5
	s_nop 0
	v_cndmask_b32_e32 v4, v3, v4, vcc
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], v4, off offset:64
.LBB0_267:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 0x73, v132
	v_cmp_gt_i32_e32 vcc, s4, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_269
; %bb.268:
	v_bfe_u32 v3, v13, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v3, v3, v13, s2
	v_or_b32_e32 v4, 0x400000, v13
	v_cmp_u_f32_e32 vcc, v13, v13
	v_mul_lo_u32 v2, v2, s5
	s_nop 0
	v_cndmask_b32_e32 v4, v3, v4, vcc
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], v4, off offset:64
.LBB0_269:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 0x78, v132
	v_cmp_gt_i32_e32 vcc, s4, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_271
; %bb.270:
	v_bfe_u32 v3, v14, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v3, v3, v14, s2
	v_or_b32_e32 v4, 0x400000, v14
	v_cmp_u_f32_e32 vcc, v14, v14
	v_mul_lo_u32 v2, v2, s5
	s_nop 0
	v_cndmask_b32_e32 v4, v3, v4, vcc
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], v4, off offset:64
.LBB0_271:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 0x79, v132
	v_cmp_gt_i32_e32 vcc, s4, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_273
; %bb.272:
	v_bfe_u32 v3, v15, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v3, v3, v15, s2
	v_or_b32_e32 v4, 0x400000, v15
	v_cmp_u_f32_e32 vcc, v15, v15
	v_mul_lo_u32 v2, v2, s5
	s_nop 0
	v_cndmask_b32_e32 v4, v3, v4, vcc
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], v4, off offset:64
.LBB0_273:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 0x7a, v132
	v_cmp_gt_i32_e32 vcc, s4, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_275
; %bb.274:
	v_bfe_u32 v3, v16, 16, 1
	s_movk_i32 s2, 0x7fff
	v_add3_u32 v3, v3, v16, s2
	v_or_b32_e32 v4, 0x400000, v16
	v_cmp_u_f32_e32 vcc, v16, v16
	v_mul_lo_u32 v2, v2, s5
	s_nop 0
	v_cndmask_b32_e32 v4, v3, v4, vcc
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], v4, off offset:64
.LBB0_275:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 0x7b, v132
	v_cmp_gt_i32_e32 vcc, s4, v2
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_277
; %bb.276:
	v_bfe_u32 v3, v17, 16, 1
	s_movk_i32 s0, 0x7fff
	v_add3_u32 v3, v3, v17, s0
	v_or_b32_e32 v4, 0x400000, v17
	v_cmp_u_f32_e32 vcc, v17, v17
	v_mul_lo_u32 v2, v2, s5
	s_nop 0
	v_cndmask_b32_e32 v4, v3, v4, vcc
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[0:1], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[0:1], v4, off offset:64
.LBB0_277:                              ; %.loopexit.1.3
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z11gemm_kernelPK14__hip_bfloat16S1_PS_iii
		.amdhsa_group_segment_fixed_size 32768
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 36
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 157
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 160
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_Z11gemm_kernelPK14__hip_bfloat16S1_PS_iii, .Lfunc_end0-_Z11gemm_kernelPK14__hip_bfloat16S1_PS_iii
                                        ; -- End function
	.set _Z11gemm_kernelPK14__hip_bfloat16S1_PS_iii.num_vgpr, 157
	.set _Z11gemm_kernelPK14__hip_bfloat16S1_PS_iii.num_agpr, 0
	.set _Z11gemm_kernelPK14__hip_bfloat16S1_PS_iii.numbered_sgpr, 20
	.set _Z11gemm_kernelPK14__hip_bfloat16S1_PS_iii.num_named_barrier, 0
	.set _Z11gemm_kernelPK14__hip_bfloat16S1_PS_iii.private_seg_size, 0
	.set _Z11gemm_kernelPK14__hip_bfloat16S1_PS_iii.uses_vcc, 1
	.set _Z11gemm_kernelPK14__hip_bfloat16S1_PS_iii.uses_flat_scratch, 0
	.set _Z11gemm_kernelPK14__hip_bfloat16S1_PS_iii.has_dyn_sized_stack, 0
	.set _Z11gemm_kernelPK14__hip_bfloat16S1_PS_iii.has_recursion, 0
	.set _Z11gemm_kernelPK14__hip_bfloat16S1_PS_iii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 15500
; TotalNumSgprs: 26
; NumVgprs: 157
; NumAgprs: 0
; TotalNumVgprs: 157
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 32768 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 19
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 157
; AccumOffset: 160
; Occupancy: 3
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 39
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_43ecab5ed6d1a4a5,@object ; @__hip_cuid_43ecab5ed6d1a4a5
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_43ecab5ed6d1a4a5
__hip_cuid_43ecab5ed6d1a4a5:
	.byte	0                               ; 0x0
	.size	__hip_cuid_43ecab5ed6d1a4a5, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_43ecab5ed6d1a4a5
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 32768
    .kernarg_segment_align: 8
    .kernarg_segment_size: 36
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z11gemm_kernelPK14__hip_bfloat16S1_PS_iii
    .private_segment_fixed_size: 0
    .sgpr_count:     26
    .sgpr_spill_count: 0
    .symbol:         _Z11gemm_kernelPK14__hip_bfloat16S1_PS_iii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     157
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
