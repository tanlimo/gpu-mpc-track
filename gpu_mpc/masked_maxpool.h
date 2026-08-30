//
// DeepDTAGen affinity path — masked global max-pool for GPU-MPC.
//
// Reduces the per-node GCN features H (Nmax x F) to a single graph embedding
// (1 x F): for each feature channel, the maximum over the REAL nodes only,
// padding nodes excluded via the secret binary mask. Faithful to the golden
// reference reference/masked_maxpool.py / fixed_forward.py (sentinel select
// then column-wise max).
//
// Equivalence note: the reference forces padding rows to a NEG sentinel then
// takes the max. Since the preceding GCN layer applies ReLU, every real entry
// is >= 0, so zeroing padding rows (mask-multiply) is EXACTLY equivalent:
// padding contributes 0, which can only tie the max when the true masked max is
// also 0. We use mask-multiply because it is a single Hadamard product.
//
// SHAPE CONTRACTS discovered from sytorch source (layers/layers.h):
//   * _Mul asserts identical shapes on every axis — NO broadcast. Hence the
//     mask must arrive pre-tiled to (Nmax x F), not (Nmax x 1). The Python
//     sharer (share_data.py) emits maskTiled as literal {0,1} ring bits at scale 0.
//   * View(idx) selects row idx AND erases axis 0, yielding a 1-D (F,) tensor.
//     The pairwise fold therefore runs on 1-D vectors (add/relu/scalarmul are
//     all shape-preserving elementwise ops, so this is fine), and we unsqueeze
//     the (F,) result back to (1 x F) before the downstream FC, whose _resize
//     hard-asserts a 2-D input.
//
// The reduction is a STATIC left-fold of pairwise maxima, unrolled at graph
// build time (the sytorch graph is data-independent; a fixed-trip-count loop is
// legal). Cost: Nmax-1 ReLUs over an F-vector.
//     max(a, b) = a + ReLU(b - a)
//
// No upstream file is modified; this only composes the public functional API.
//
#pragma once

#include <sytorch/module.h>
#include "utils/gpu_data_types.h"
#include <chrono>
#include <cstdlib>
#include <cstdio>

template <typename T>
struct MaskedGlobalMaxPool
{
    SytorchModule<T> *owner;   // supplies functional ops so nodes join its graph
    u64 Nmax;
    u64 F;

    MaskedGlobalMaxPool(SytorchModule<T> *owner_, u64 Nmax_, u64 F_)
        : owner(owner_), Nmax(Nmax_), F(F_) {}

    // b - a  via pure ring subtraction (no truncation, no MPC protocol).
    // owner->sub() routes through the _Sub functional layer (doTruncationForward
    // = false), so no Sigma DPF TrFloor truncate protocol runs — just a local
    // gpuLinearComb [1, -1]. Replaces the old scalarmul(-1/2^scale)+add, which
    // created a _ScalarMul node that truncated needlessly.
    Tensor<T> &sub(Tensor<T> &b, Tensor<T> &a)
    {
        return owner->sub(b, a);
    }

    // max(a, b) = a + ReLU(b - a), elementwise over the (F,) vectors.
    Tensor<T> &pairwise_max(Tensor<T> &a, Tensor<T> &b)
    {
        auto &d = sub(b, a);
        auto &r = owner->relu(d);
        return owner->add(a, r);
    }

    // H:        (Nmax x F) secret post-ReLU GCN features
    // maskTiled: (Nmax x F) secret {0,1} node mask tiled across F channels,
    //            encoded as literal ring values {0,1} at scale 0.
    // Returns (1 x F) graph embedding.
    Tensor<T> &forward(Tensor<T> &H, Tensor<T> &maskTiled)
    {
        auto &masked = owner->mul(H, maskTiled);   // (Nmax x F), padding rows -> 0

        Tensor<T> *acc = &owner->view(masked, 0);  // (F,)
        for (u64 i = 1; i < Nmax; ++i)
        {
            auto &row = owner->view(masked, (i64)i);   // (F,)
            acc = &pairwise_max(*acc, row);            // (F,)
        }
        return owner->unsqueeze(*acc);   // (F,) -> (1 x F) for downstream FC
    }

    // Gather all B samples' row `i` into one (B, F) tensor.
    //   masked is (B*Nmax, F) sample-major: sample b's row i is at index b*Nmax+i.
    //   view() extracts that (F,) row (memcpy only — no crypto, no comm).
    //   concat() along the last axis stacks B rows → (1, B*F), reshaped to (B, F).
    // The gather is cheap (memcpy); the expensive crypto/comm runs later, ONCE, on
    // the assembled (B, F).
    Tensor<T> &gatherRow(Tensor<T> &masked, u64 i, u64 B)
    {
        std::vector<Tensor<T>*> rows;
        rows.reserve(B);
        for (u64 b = 0; b < B; ++b)
            rows.push_back(&owner->view(masked, (i64)(b * Nmax + i)));  // (F,)
        auto &row = owner->concat(rows);   // (1, B*F)  (host+GPU concat)
        row.shape = {B, F};                // reinterpret as (B, F)
        return row;
    }

    // Batched forward with TRUE amortization: H, maskTiled are (B*Nmax, F).
    // Instead of B independent 137-deep folds (16×137 crypto ops, no amortization),
    // run ONE 137-deep fold over (B, F) accumulators. Each of the 137 pairwise-max
    // steps launches its truncate/relu/comm ONCE for all B samples — the fixed
    // per-op costs (kernel launch, comm round-trip, key read) are paid 137× total
    // rather than 16×137×. Element count (and thus key SIZE) is unchanged.
    // Output: (B, F).
    Tensor<T> &forward_batched(Tensor<T> &H, Tensor<T> &maskTiled, u64 B)
    {
        auto &masked = owner->mul(H, maskTiled);   // (B*Nmax, F), padding rows -> 0

        const bool prof = (getenv("DDG_POOL_PROF") != nullptr);
        double t_gather = 0, t_crypto = 0;
        auto now = []() {
            cudaDeviceSynchronize();
            return std::chrono::duration<double>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        };

        // acc = row 0 across all B samples → (B, F)
        Tensor<T> *acc = &gatherRow(masked, 0, B);
        for (u64 i = 1; i < Nmax; ++i) {
            double t0 = prof ? now() : 0;
            auto &row = gatherRow(masked, i, B);       // (B, F), cheap memcpy gather
            double t1 = prof ? now() : 0;
            acc = &pairwise_max(*acc, row);            // ONE crypto op over B*F elems
            double t2 = prof ? now() : 0;
            if (prof) { t_gather += (t1 - t0); t_crypto += (t2 - t1); }
        }
        if (prof)
            printf("[POOL_PROF] gather=%.3fs crypto=%.3fs (Nmax=%lu B=%lu)\n",
                   t_gather, t_crypto, (unsigned long)Nmax, (unsigned long)B);
        return *acc;   // (B, F)
    }
};
