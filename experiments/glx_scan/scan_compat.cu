// Reuse upstream test oracles without modifying the GLX checkout.
#define main glx_upstream_scan_main
#include <diagonal_scan_test.cu>
#undef main

int main() {
    bool ok = true;
    ok &= run_case<16, 32, UnaryElement, AddOp, false, F32x2>("F32x2 AddOp");
    ok &= run_case<16, 32, UnaryElement, MulOp, false, F32x2>("F32x2 MulOp");
    ok &= run_case<16, 32, BinaryElement, AffineComposeOp, false, F32x2>("F32x2 AffineComposeOp");
    ok &= run_case<16, 32, UnaryElement, AddOp, true, F32x2>("F32x2 AddOp");
    ok &= run_case<16, 32, UnaryElement, MulOp, true, F32x2>("F32x2 MulOp");
    ok &= run_case<16, 32, BinaryElement, AffineComposeOp, true, F32x2>("F32x2 AffineComposeOp");
    ok &= run_case<16, 32, UnaryElement, AddOp, false, BF16x2>("BF16x2 AddOp");
    ok &= run_case<16, 32, UnaryElement, MulOp, false, BF16x2>("BF16x2 MulOp");
    ok &= run_case<16, 32, BinaryElement, AffineComposeOp, false, BF16x2>("BF16x2 AffineComposeOp");
    ok &= run_case<16, 32, UnaryElement, AddOp, true, BF16x2>("BF16x2 AddOp");
    ok &= run_case<16, 32, UnaryElement, MulOp, true, BF16x2>("BF16x2 MulOp");
    ok &= run_case<16, 32, BinaryElement, AffineComposeOp, true, BF16x2>("BF16x2 AffineComposeOp");
    return ok ? 0 : 1;
}

