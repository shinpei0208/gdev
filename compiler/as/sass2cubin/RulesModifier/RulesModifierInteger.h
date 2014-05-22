#ifndef RulesModifierIntegerDefined
#define RulesModifierIntegerDefined


struct ModifierRuleIMUL0U32;
extern ModifierRuleIMUL0U32 MRIMUL0U32;

struct ModifierRuleIMUL1U32;
extern ModifierRuleIMUL1U32 MRIMUL1U32;


struct ModifierRuleIMUL0S32;
extern ModifierRuleIMUL0S32 MRIMUL0S32;

struct ModifierRuleIMUL1S32;
extern ModifierRuleIMUL1S32 MRIMUL1S32;

struct ModifierRuleIMULHI;
extern ModifierRuleIMULHI MRIMULHI;

struct ModifierRuleIMULSAT;
extern ModifierRuleIMULSAT MRIMULSAT;

struct ModifierRuleIADD32ISAT;
extern ModifierRuleIADD32ISAT MRIADD32ISAT;

struct ModifierRuleIADD32IX;
extern ModifierRuleIADD32IX MRIADD32IX;

struct ModifierRuleIMADX;
extern ModifierRuleIMADX MRIMADX;


struct ModifierRuleISETPU32;
extern ModifierRuleISETPU32 MRISETPU32;

struct ModifierRuleISETPX;
extern ModifierRuleISETPX MRISETPX;

struct ModifierRuleVADD_UD;
extern ModifierRuleVADD_UD MRVADD_UD, MRVABSDIFF4_UD;

struct ModifierRuleVADD_OpType;
extern ModifierRuleVADD_OpType MRVADD_Op1_U8, MRVADD_Op1_U16, MRVADD_Op1_U32, MRVADD_Op1_S8, MRVADD_Op1_S16, MRVADD_Op1_S32;
extern ModifierRuleVADD_OpType MRVADD_Op2_U8, MRVADD_Op2_U16, MRVADD_Op2_U32, MRVADD_Op2_S8, MRVADD_Op2_S16, MRVADD_Op2_S32;

struct ModifierRuleVADD_SAT;
extern ModifierRuleVADD_SAT MRVADD_SAT;

struct ModifierRuleVADD_SecOp;
extern ModifierRuleVADD_SecOp MRVADD_SecOp_MRG_16H, MRVADD_SecOp_MRG_16L, MRVADD_SecOp_MRG_8B0, MRVADD_SecOp_MRG_8B2, MRVADD_SecOp_ACC, MRVADD_SecOp_MIN, MRVADD_SecOp_MAX;

struct ModifierRuleISETBF;
extern ModifierRuleISETBF MRISETBF;

struct ModifierRuleIMNMXHI;
extern ModifierRuleIMNMXHI MRIMNMXHI;

struct ModifierRuleIMNMXLO;
extern ModifierRuleIMNMXLO MRIMNMXLO;

struct ModifierRuleVABSDIFF4_OpType;
extern ModifierRuleVABSDIFF4_OpType MRVABSDIFF4_Op1_U8, MRVABSDIFF4_Op2_U8, MRVABSDIFF4_Op1_S8, MRVABSDIFF4_Op2_S8;

struct ModifierRuleVABSDIFF4_SecOp;
extern ModifierRuleVABSDIFF4_SecOp MRVABSDIFF4_SIMD_MIN, MRVABSDIFF4_SIMD_MAX, MRVABSDIFF4_ACC, MRVABSDIFF4_MIN, MRVABSDIFF4_MAX;

#else
#endif
