// Reuse upstream test oracles without modifying the GLX checkout.
#define main glx_upstream_reduce_main
#include <diagonal_reduce_test.cu>
#undef main

int main() {
    bool ok = true;
    ok &= run_reduce_case<16, 32, UnaryElement, AddOp, F32x2, false>("false F32x2 AddOp");
    ok &= run_reduce_case<16, 32, UnaryElement, MulOp, F32x2, false>("false F32x2 MulOp");
    ok &= run_reduce_case<16, 32, BinaryElement, AffineComposeOp, F32x2, false>("false F32x2 AffineComposeOp");
    ok &= run_reduce_case<16, 32, UnaryElement, AddOp, F32x2, true>("true F32x2 AddOp");
    ok &= run_reduce_case<16, 32, UnaryElement, MulOp, F32x2, true>("true F32x2 MulOp");
    ok &= run_reduce_case<16, 32, BinaryElement, AffineComposeOp, F32x2, true>("true F32x2 AffineComposeOp");
    ok &= run_reduce_case<16, 32, UnaryElement, AddOp, BF16x2, false>("false BF16x2 AddOp");
    ok &= run_reduce_case<16, 32, UnaryElement, MulOp, BF16x2, false>("false BF16x2 MulOp");
    ok &= run_reduce_case<16, 32, BinaryElement, AffineComposeOp, BF16x2, false>("false BF16x2 AffineComposeOp");
    ok &= run_reduce_case<16, 32, UnaryElement, AddOp, BF16x2, true>("true BF16x2 AddOp");
    ok &= run_reduce_case<16, 32, UnaryElement, MulOp, BF16x2, true>("true BF16x2 MulOp");
    ok &= run_reduce_case<16, 32, BinaryElement, AffineComposeOp, BF16x2, true>("true BF16x2 AffineComposeOp");
    return ok ? 0 : 1;
}

