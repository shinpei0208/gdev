#ifndef RulesOperandConstantDefined
#define RulesOperandConstantDefined




//24-bit Hexadecimal Constant Operand
struct OperandRuleImmediate24HexConstant;
extern OperandRuleImmediate24HexConstant OPRImmediate24HexConstant;


struct OperandRuleImmediate32HexConstant;
extern OperandRuleImmediate32HexConstant OPRImmediate32HexConstant;



//32-bit Integer Constant Operand
struct OperandRuleImmediate32IntConstant;
extern OperandRuleImmediate32IntConstant OPRImmediate32IntConstant;

//32-bit Floating Number Constant Operand
struct OperandRuleImmediate32FloatConstant;
extern OperandRuleImmediate32FloatConstant OPRImmediate32FloatConstant;

//32-bit Constant: Hex || Int || Float
struct OperandRuleImmediate32AnyConstant;
extern OperandRuleImmediate32AnyConstant OPRImmediate32AnyConstant;

struct OperandRuleImmediate16HexOrInt;
extern OperandRuleImmediate16HexOrInt OPRImmediate16HexOrInt, OPRImmediate16HexOrIntOptional;

struct OperandRuleS2R;
extern OperandRuleS2R OPRS2R;

#else
#endif