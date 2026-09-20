// The ReLU circuit on the card, with FIDESlib's OpenFHE interop layer.
//
// The envelope makes the context, the keys and the ciphertexts with OpenFHE
// before the clock starts; this file takes them as they are. Once per point
// (gpu_init, not measured) the device context is adapted from the envelope's
// own and the relinearisation key is moved over. Per case (gpu_relu, which
// is what the clock sees) the ciphertext goes to the card, the circuit runs
// there, and the answer comes back as an OpenFHE ciphertext the envelope can
// decrypt. Moving the data is inside the measurement on purpose: it is part
// of what answering on a GPU costs.
//
// The circuit is polycircuit's ReLUFunction, operation for operation, as
// the CPU answer adapted it (coefficients as plaintext scalars, every slot
// treated alike):
//
//   relu(x) ≈ a_0 + Σ_{k=1}^{15} a_k x^k  −  54 x^16
//
// with x^2..x^8 by squaring and products, x^9..x^15 as a low power times
// x^8, and the leading integer term folded in as 54 subtractions of x^16 —
// which is what keeps it at four levels, since a multiplication by 54 would
// cost a fifth. Four levels, like the CPU answer, on the same parameters.
#include "gpu.h"

#include <memory>
#include <stdexcept>
#include <vector>

#include "openfhe.h"
#ifdef duration
#undef duration  // OpenFHE defines the word; CUDA's headers use it.
#endif
#include "CKKS/Ciphertext.cuh"
#include "CKKS/Context.cuh"
#include "CKKS/KeySwitchingKey.cuh"
#include "CKKS/openfhe-interface/RawCiphertext.cuh"

namespace {

using GpuCiphertext = FIDESlib::CKKS::Ciphertext;

struct Device {
    FIDESlib::CKKS::Context gpu;
};

// The MILP-optimised coefficients a_0..a_15; the leading term is the integer B.
const double A[16] = {
    0.0323949878919212,   0.500001412106499,   2.13483086933591,   -4.78160418051218e-05,
    -13.9205486530553,    0.00061641818435605, 70.0957556465309,   -0.00388040016141976,
    -213.087053403128,    0.0129145434432087,  385.924971250905,   -0.0230082806472531,
    -407.029727261512,    0.0206280915579812,  230.348436664049,   -0.00728306945833855,
};
const int B = -54;

// dst = a · b, and dst = a · c as a scalar, without touching a.
GpuCiphertext product(FIDESlib::CKKS::Context& gpu, const GpuCiphertext& a, const GpuCiphertext& b) {
    GpuCiphertext out(gpu);
    out.mult(a, b);
    return out;
}
GpuCiphertext scaled(FIDESlib::CKKS::Context& gpu, const GpuCiphertext& a, double c) {
    GpuCiphertext out(gpu);
    out.multScalar(a, c);
    return out;
}

}  // namespace

void* gpu_init(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc) {
    // The device context, adapted from the envelope's: its primes, its
    // digits, its scaling technique. FIDESlib's own parameters only add the
    // batch it processes limbs in.
    FIDESlib::CKKS::RawParams raw = FIDESlib::CKKS::GetRawParams(cc);
    FIDESlib::CKKS::Parameters seed{};
    seed.batch = 100;
    FIDESlib::CKKS::Context gpu = FIDESlib::CKKS::GenCryptoContextGPU(seed.adaptTo(raw), {0});

    // The relinearisation key, read out of the context's own key store. The
    // envelope made one key pair, so the store holds one entry; the key is
    // the first of its vector, as it is in OpenFHE's own EvalMult.
    auto& store = lbcrypto::CryptoContextImpl<lbcrypto::DCRTPoly>::GetAllEvalMultKeys();
    if (store.empty()) throw std::runtime_error("gpu_init: the context holds no evaluation key");
    auto relin = std::dynamic_pointer_cast<lbcrypto::EvalKeyRelinImpl<lbcrypto::DCRTPoly>>(
        store.begin()->second.at(0));
    if (!relin) throw std::runtime_error("gpu_init: the evaluation key is not a relinearisation key");

    FIDESlib::CKKS::RawKeySwitchKey rawKey = FIDESlib::CKKS::GetKeySwitchKey(relin);
    FIDESlib::CKKS::KeySwitchingKey key(gpu);
    key.Initialize(rawKey);
    gpu->AddEvalKey(std::move(key));

    return new Device{std::move(gpu)};
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> gpu_relu(
    void* state,
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc,
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& ct) {
    Device& device = *static_cast<Device*>(state);
    FIDESlib::CKKS::Context& gpu = device.gpu;

    // To the card.
    FIDESlib::CKKS::RawCipherText raw = FIDESlib::CKKS::GetRawCipherText(cc, ct);
    GpuCiphertext x1(gpu, raw);

    // The powers, in the component's order.
    GpuCiphertext x2 = product(gpu, x1, x1);
    GpuCiphertext x3 = product(gpu, x1, x2);
    GpuCiphertext x4 = product(gpu, x2, x2);
    GpuCiphertext x5 = product(gpu, x1, x4);
    GpuCiphertext x6 = product(gpu, x4, x2);
    GpuCiphertext x7 = product(gpu, x3, x4);
    GpuCiphertext x8 = product(gpu, x4, x4);

    // a_k x^k for k ≤ 8: the power times its coefficient.
    GpuCiphertext a1x1 = scaled(gpu, x1, A[1]);
    GpuCiphertext a2x2 = scaled(gpu, x2, A[2]);
    GpuCiphertext a3x3 = scaled(gpu, x3, A[3]);
    GpuCiphertext a4x4 = scaled(gpu, x4, A[4]);
    GpuCiphertext a5x5 = scaled(gpu, x5, A[5]);
    GpuCiphertext a6x6 = scaled(gpu, x6, A[6]);
    GpuCiphertext a7x7 = scaled(gpu, x7, A[7]);
    GpuCiphertext a8x8 = scaled(gpu, x8, A[8]);

    // a_k x^k for k > 8: the coefficient goes onto a low power first, and
    // the rest of the power is multiplied on — the component's level plan.
    GpuCiphertext a9x9 = product(gpu, scaled(gpu, x1, A[9]), x8);
    GpuCiphertext a10x10 = product(gpu, scaled(gpu, x2, A[10]), x8);
    GpuCiphertext a11x11 = product(gpu, scaled(gpu, x3, A[11]), x8);
    GpuCiphertext a12x12 = product(gpu, scaled(gpu, x4, A[12]), x8);
    GpuCiphertext a13x13 = product(gpu, product(gpu, scaled(gpu, x1, A[13]), x4), x8);
    GpuCiphertext a14x14 = product(gpu, product(gpu, scaled(gpu, x2, A[14]), x4), x8);
    GpuCiphertext a15x15 =
        product(gpu, product(gpu, product(gpu, scaled(gpu, x1, A[15]), x2), x4), x8);

    // The sum, in the component's order, then the constant term.
    GpuCiphertext out(gpu);
    out.add(a1x1, a2x2);
    out.add(a3x3);
    out.add(a4x4);
    out.add(a5x5);
    out.add(a6x6);
    out.add(a7x7);
    out.add(a8x8);
    out.add(a9x9);
    out.add(a10x10);
    out.add(a11x11);
    out.add(a12x12);
    out.add(a13x13);
    out.add(a14x14);
    out.add(a15x15);
    out.addScalar(A[0]);

    // The leading integer coefficient, by repeated addition or subtraction:
    // no level spent on it.
    GpuCiphertext x16 = product(gpu, x8, x8);
    for (int m = 0; m < (B >= 0 ? B : -B); ++m) {
        if (B >= 0) out.add(x16);
        else out.sub(x16);
    }

    // Back, as an OpenFHE ciphertext the envelope can decrypt.
    FIDESlib::CKKS::RawCipherText rawOut;
    out.store(rawOut);
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> answer = ct->Clone();
    FIDESlib::CKKS::GetOpenFHECipherText(answer, rawOut);
    return answer;
}

void gpu_free(void* state) { delete static_cast<Device*>(state); }
