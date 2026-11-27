// Opcode spellings and fields follow the repository gfx1013 disassembler.
static const char* mnemonic(Encoding enc, unsigned op) {
    switch (enc) {
    case Encoding::Sopp:
        switch (op) {
        case 0x0: return "s_nop";
        case 0x1: return "s_endpgm";
        case 0x2: return "s_branch";
        case 0x4: return "s_cbranch_scc0";
        case 0x5: return "s_cbranch_scc1";
        case 0x6: return "s_cbranch_vccz";
        case 0x7: return "s_cbranch_vccnz";
        case 0x8: return "s_cbranch_execz";
        case 0x9: return "s_cbranch_execnz";
        case 0xa: return "s_barrier";
        case 0xc: return "s_waitcnt";
        case 0x20: return "s_inst_prefetch";
        case 0x21: return "s_clause";
        default: break;
        }
        break;
    case Encoding::Sop1:
        switch (op) {
        case 0x3: return "s_mov_b32";
        case 0x4: return "s_mov_b64";
        case 0x7: return "s_not_b32";
        case 0x3c: return "s_and_saveexec_b32";
        case 0x47: return "s_andn2_wrexec_b32";
        default: break;
        }
        break;
    case Encoding::Sop2:
        switch (op) {
        case 0x1: return "s_sub_u32";
        case 0x7: return "s_min_u32";
        case 0x9: return "s_max_u32";
        case 0x20: return "s_lshr_b32";
        case 0x35: return "s_mul_hi_u32";
        case 0: return "s_add_u32";
        case 0x26: return "s_mul_i32";
        case 0xa: return "s_cselect_b32";
        case 0xe: return "s_and_b32";
        case 0x10: return "s_or_b32";
        case 0x14: return "s_andn2_b32";
        case 0x1a: return "s_nor_b32";
        case 0x1e: return "s_lshl_b32";
        case 0x24: return "s_bfm_b32";
        case 0x27: return "s_bfe_u32";
        case 0x2c: return "s_absdiff_i32";
        case 0x2e: return "s_lshl1_add_u32";
        case 0x2f: return "s_lshl2_add_u32";
        case 0x30: return "s_lshl3_add_u32";
        case 0x31: return "s_lshl4_add_u32";
        default: break;
        }
        break;
    case Encoding::Sopc:
        switch (op) {
        case 0x6: return "s_cmp_eq_u32";
        case 0x7: return "s_cmp_lg_u32";
        case 0xa: return "s_cmp_lt_u32";
        case 0x0: return "s_cmp_eq_i32";
        case 0x1: return "s_cmp_lg_i32";
        case 0x4: return "s_cmp_lt_i32";
        case 0xd: return "s_bitcmp1_b32";
        default: break;
        }
        break;
    case Encoding::Sopk:
        switch (op) {
        case 0x4: return "s_cmpk_lg_i32";
        case 0x10: return "s_mulk_i32";
        case 0x0: return "s_movk_i32";
        default: break;
        }
        break;
    case Encoding::Vop1:
        switch (op) {
        case 0x0b: return "v_cvt_f32_f16";
        case 0x36: return "v_cos_f32";
        case 0x65: return "v_swap_b32";
        case 2: return "v_readfirstlane_b32";
        case 0x1: return "v_mov_b32";
        case 0x5: return "v_cvt_f32_i32";
        case 0x6: return "v_cvt_f32_u32";
        case 0x8: return "v_cvt_i32_f32";
        case 0x11: return "v_cvt_f32_ubyte0";
        case 0x23: return "v_rndne_f32";
        case 0x2e: return "v_rsq_f32";
        case 0x7: return "v_cvt_u32_f32";
        case 0x24: return "v_floor_f32";
        case 0x25: return "v_exp_f32";
        case 0x27: return "v_log_f32";
        case 0x2a: return "v_rcp_f32";
        case 0x33: return "v_sqrt_f32";
        case 0x35: return "v_sin_f32";
        default: break;
        }
        break;
    case Encoding::Vop2:
        switch (op) {
        case 0x1d: return "v_xor_b32";
        case 0x13: return "v_min_u32";
        case 0x1: return "v_cndmask_b32";
        case 0x3: return "v_add_f32";
        case 0x4: return "v_sub_f32";
        case 0x5: return "v_subrev_f32";
        case 0x8: return "v_mul_f32";
        case 0xb: return "v_mul_u32_u24";
        case 0xf: return "v_min_f32";
        case 0x10: return "v_max_f32";
        case 0x16: return "v_lshrrev_b32";
        case 0x18: return "v_ashrrev_i32";
        case 0x1a: return "v_lshlrev_b32";
        case 0x1b: return "v_and_b32";
        case 0x1c: return "v_or_b32";
        case 0x25: return "v_add_nc_u32";
        case 0x26: return "v_sub_nc_u32";
        case 0x2b: return "v_fmac_f32";
        case 0x2c: return "v_fmamk_f32";
        case 0x2d: return "v_fmaak_f32";
        default: break;
        }
        break;
    case Encoding::Vopc:
        switch (op) {
        case 0x0d: return "v_cmp_neq_f32";
        case 0x81: return "v_cmp_lt_i32";
        case 0x84: return "v_cmp_gt_i32";
        case 0x86: return "v_cmp_ge_i32";
        case 0x91: return "v_cmpx_lt_i32";
        case 0x92: return "v_cmpx_eq_i32";
        case 0x93: return "v_cmpx_le_i32";
        case 0x94: return "v_cmpx_gt_i32";
        case 0x95: return "v_cmpx_ne_i32";
        case 0x96: return "v_cmpx_ge_i32";
        case 0xc1: return "v_cmp_lt_u32";
        case 0xc2: return "v_cmp_eq_u32";
        case 0xc3: return "v_cmp_le_u32";
        case 0xc4: return "v_cmp_gt_u32";
        case 0xc5: return "v_cmp_ne_u32";
        case 0xc6: return "v_cmp_ge_u32";
        case 0xd1: return "v_cmpx_lt_u32";
        case 0xd2: return "v_cmpx_eq_u32";
        case 0xd3: return "v_cmpx_le_u32";
        case 0xd4: return "v_cmpx_gt_u32";
        case 0xd5: return "v_cmpx_ne_u32";
        case 0xd6: return "v_cmpx_ge_u32";
        case 0x1: return "v_cmp_lt_f32";
        case 0x2: return "v_cmp_eq_f32";
        case 0x3: return "v_cmp_le_f32";
        case 0x4: return "v_cmp_gt_f32";
        case 0x6: return "v_cmp_ge_f32";
        case 0xc: return "v_cmp_nle_f32";
        case 0x82: return "v_cmp_eq_i32";
        case 0x83: return "v_cmp_le_i32";
        case 0x85: return "v_cmp_ne_i32";
        case 0x88: return "v_cmp_class_f32";
        default: break;
        }
        break;
    case Encoding::Vop3:
        switch (op) {
        case 0x148: return "v_bfe_u32";
        case 0x14a: return "v_bfi_b32";
        case 0x14b: return "v_fma_f32";
        case 0x151: return "v_min3_f32";
        case 0x154: return "v_max3_f32";
        case 0x157: return "v_med3_f32";
        case 0x158: return "v_med3_i32";
        case 0x16a: return "v_mul_hi_u32";
        case 0x169: return "v_mul_lo_u32";
        case 0x16c: return "v_mul_hi_i32";
        case 0x300: return "v_lshrrev_b64";
        case 0x340: return "v_mad_u16";
        case 0x346: return "v_lshl_add_u32";
        case 0x347: return "v_add_lshl_u32";
        case 0x352: return "v_min3_i16";
        case 0x353: return "v_min3_u16";
        case 0x355: return "v_max3_i16";
        case 0x356: return "v_max3_u16";
        case 0x358: return "v_med3_i16";
        case 0x359: return "v_med3_u16";
        case 0x35e: return "v_mad_i16";
        case 0x365: return "v_mbcnt_lo_u32_b32";
        case 0x366: return "v_mbcnt_hi_u32_b32";
        case 0x36f: return "v_lshl_or_b32";
        case 0x371: return "v_and_or_b32";
        default: break;
        }
        break;
    case Encoding::Mix:
        switch (op) {
        case 0x20: return "v_fma_mix_f32";
        default: break;
        }
        break;
    default: break;
    }
    return nullptr;
}
