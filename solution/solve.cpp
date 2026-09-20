// max(0, x) for every element, over CKKS, computed on the GPU with FIDESlib.
//
// A port. The circuit is the one the CPU answer runs — the ReLUFunction
// component from fairmath/polycircuit (Apache-2.0), the winning entry of the
// FHERMA ReLU challenge's depth-constrained track by Janis Adamek, Dieter
// Teichrib, Philipp Binfet and Moritz Schulze Darup (Control and
// Cyberphysical Systems Group, TU Dortmund): a MILP-optimised 16th-order
// polynomial whose integer leading term is folded in by repeated
// subtraction, so the whole evaluation fits in four levels. Moved operation
// for operation onto FIDESlib's device ciphertexts; the mathematics is
// theirs and untouched.
//
// This file is the plain C++ half: how the vector is packed and read back.
// The circuit itself is in gpu.cu, which is compiled by nvcc.
#include "solve.h"

#include "gpu.h"

void* solve_init(const fherma::Point& p, CryptoContext<DCRTPoly> cc) {
    // The device context and the relinearisation key, before the clock.
    return gpu_init(cc);
}

std::vector<Plaintext> solve_encoding(CryptoContext<DCRTPoly> cc,
                                      const fherma::Inputs& inp) {
    // The whole vector in one packing, slot i holding element i.
    std::vector<double> xs(inp.xs.data.begin(), inp.xs.data.end());
    return { cc->MakeCKKSPackedPlaintext(xs) };
}

std::vector<Ciphertext<DCRTPoly>> solve_run(
    void* state,
    CryptoContext<DCRTPoly> cc,
    const std::vector<Ciphertext<DCRTPoly>>& cts) {
    return { gpu_relu(state, cc, cts[0]) };
}

fherma::Outputs solve_decoding(const fherma::Point& p,
                               CryptoContext<DCRTPoly> cc,
                               const std::vector<Plaintext>& pts) {
    auto values = pts[0]->GetRealPackedValue();

    fherma::Outputs out;
    out.r.shape = { static_cast<int64_t>(p.N) };
    out.r.data.assign(values.begin(), values.begin() + p.N);
    return out;
}

void solve_free(void* state) { gpu_free(state); }
