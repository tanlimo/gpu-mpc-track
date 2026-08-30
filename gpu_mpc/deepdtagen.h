//
// DeepDTAGen affinity path — full model for GPU-MPC (iDASH 2024 Track 3).
//
// Scope: ONLY the drug-target affinity PREDICTION path. The drug-generation
// branch (VAE + Transformer decoder) is intentionally excluded.
//
// This mirrors, layer-for-layer, the Python fixed-point reference
// (idash/mpc/reference/fixedpoint.py). The two MUST stay in lockstep: the
// exported weight blob (reference/export_weights.py) is laid out in this exact
// forward order, and the accuracy gate is validated against this graph.
//
// Layout (Nmax = 138 padded nodes, FEAT_DIM = 94):
//
//   Drug branch  (secret, MPC):
//     GCN1 : ReLU(A_hat @ (X   @ W1 + b1))   94  -> 188
//     GCN2 : ReLU(A_hat @ (H1  @ W2 + b2))   188 -> 282
//     GCN3 : ReLU(A_hat @ (H2  @ W3 + b3))   282 -> 376
//     pool : masked global max over nodes    (138 x 376) -> (1 x 376)
//     dfc1 : ReLU(p   @ Wd1 + bd1)           376  -> 1024
//     dfc2 :      d1  @ Wd2 + bd2            1024 -> 128   (drug embedding)
//
//   Protein branch (public sequence): GatedCNN evaluated OUTSIDE MPC by P2,
//     entering here as a 128-vector secret share (proteinEmb).
//
//   Fusion branch (secret, MPC):
//     concat[d2, proteinEmb]                 -> 256
//     ffc1 : ReLU(. @ Wf1 + bf1)             256  -> 1024
//     ffc2 : ReLU(. @ Wf2 + bf2)             1024 -> 512
//     ffc3 : ReLU(. @ Wf3 + bf3)             512  -> 256
//     out  :      . @ Wo  + bo               256  -> 1     (affinity, revealed)
//
#pragma once

#include <sytorch/module.h>
#include "utils/gpu_data_types.h"
#include "nn/orca/fc_layer.h"
#include "nn/orca/relu_layer.h"

#include "ddg_orca_base.h"   // DDGOrcaBase<T> for revealPublicLeaf in GPUConcat
#include "gcn_layer.h"
#include "masked_maxpool.h"
#include "masked_maxpool_layer.h"   // fused tree-reduction pool (Path B+A)

// GPU-aware Concat layer: concatenates d_data (GPU) in addition to host data
template <typename T>
class GPUConcat : public Concat<T>
{
public:
    GPUConcat() : Concat<T>() { this->name = "Concat"; }  // keep "Concat" so optimizer recognizes it

    void _forward(std::vector<Tensor<T> *> &arr) override
    {
        // Parent fills host .data
        Concat<T>::_forward(arr);

        // Reveal any public leaves (e.g. proteinEmb) before concatenation.
        // proteinEmb is an additive share (party0=0, party1=P) with r=0 in keygen.
        // Reconstruct to get P on both parties before concat, so the fused tensor
        // is fully masked-public: [d2+r_d2 | P+0].
        auto *backend = dynamic_cast<DDGOrcaBase<T>*>(this->backend);
        if (backend) {
            for (auto &t : arr) {
                if (t->d_data) {
                    backend->revealPublicLeaf(t->d_data, backend->bw, t->size());
                }
            }
        }

        // Populate GPU .d_data by concatenating input GPU buffers.
        // Concat is along the LAST axis, so for B>1 rows the layout must be
        // per-row interleaved: out[b] = [t0[b] | t1[b] | ...]. A flat contiguous
        // copy ([all t0 | all t1]) is only correct when B==1; for B>1 it scrambles
        // samples (each fused row would mix two different samples' halves).
        size_t total_size = this->activation.size();

        // Allocate GPU buffer only if not already allocated (activation persists across calls)
        if (!this->activation.d_data) {
            this->activation.d_data = (T*)gpuMalloc(total_size * sizeof(T));
        }

        // Number of rows B = outer dim of the (B, sum_of_widths) output.
        u64 outW = this->activation.shape.back();
        u64 B = (outW > 0) ? (total_size / outW) : 1;

        // Per-row interleaved copy: for each row b, copy each input's row b slice
        // into the fused row at the running column offset.
        for (u64 b = 0; b < B; ++b) {
            size_t colOff = 0;
            for (auto &t : arr) {
                u64 w = t->shape.back();          // this input's width (cols)
                if (t->d_data) {
                    checkCudaErrors(cudaMemcpy(
                        this->activation.d_data + b * outW + colOff,
                        t->d_data + b * w,
                        w * sizeof(T), cudaMemcpyDeviceToDevice));
                }
                colOff += w;
            }
        }
    }
};

#ifndef DDG_NMAX
#define DDG_NMAX 138
#endif
#ifndef DDG_FEAT
#define DDG_FEAT 94
#endif

template <typename T>
class DeepDTAGenAffinity : public SytorchModule<T>
{
    using SytorchModule<T>::concat;
    using SytorchModule<T>::view;

public:
    // Drug GCN stack (weights are P2's private, held host-side by FC<T>).
    GCNLayer<T> *gcn1;
    GCNLayer<T> *gcn2;
    GCNLayer<T> *gcn3;

    // Drug embedding FCs.
    FC<T> *dfc1; ReLU<T> *drelu1;
    FC<T> *dfc2;

    // Fusion FCs.
    FC<T> *ffc1; ReLU<T> *frelu1;
    FC<T> *ffc2; ReLU<T> *frelu2;
    FC<T> *ffc3; ReLU<T> *frelu3;
    FC<T> *fout;

    u64 Nmax  = DDG_NMAX;
    u64 feat  = DDG_FEAT;
    u64 BATCH = 1;   // set via setBatch() before forward

    // Secret side-inputs, set per-sample before forward (see setSample()).
    //  * A_hat, maskTiled : P1's drug-graph secrets.
    //  * proteinEmb       : P2's public GatedCNN output injected as a share.
    // GCN biases fold into P2's private FC weights (associativity rewrite in
    // gcn_layer.h), so no tiled bias leaf is needed. maskTiled carries literal
    // ring values {0,1} at scale 0; H*mask therefore preserves H's scale.
    // Batched mode: A_hat (B*Nmax x Nmax), maskTiled (B*Nmax x 376), proteinEmb (B x 128).
    Tensor<T> *A_hat       = nullptr;   // single: (Nmax x Nmax), batched: (B*Nmax x Nmax)
    Tensor<T> *maskTiled   = nullptr;   // single: (Nmax x 376), batched: (B*Nmax x 376)
    Tensor<T> *proteinEmb  = nullptr;   // single: (1 x 128), batched: (B x 128)

    GPUConcat<T> *gpu_concat = nullptr;  // GPU-aware concat for fusion

    // Fused masked max-pool (Path B+A): one graph node running the whole
    // tree-reduction fold internally via batched backend crypto calls, replacing
    // the ~1233-node functional fold. Lazily constructed in _forward once BATCH
    // is known. Only used when BATCH > 1.
    MaskedMaxPoolLayer<T> *mmpool = nullptr;

    DeepDTAGenAffinity()
    {
        gcn1 = new GCNLayer<T>(feat, 188);
        gcn2 = new GCNLayer<T>(188, 282);
        gcn3 = new GCNLayer<T>(282, 376);

        dfc1 = new FC<T>(376, 1024, true);  drelu1 = new ReLU<T>();
        dfc2 = new FC<T>(1024, 128, true);

        gpu_concat = new GPUConcat<T>();

        ffc1 = new FC<T>(256, 1024, true);  frelu1 = new ReLU<T>();
        ffc2 = new FC<T>(1024, 512, true);  frelu2 = new ReLU<T>();
        ffc3 = new FC<T>(512, 256, true);   frelu3 = new ReLU<T>();
        fout = new FC<T>(256, 1, true);
    }

    void setSample(Tensor<T> *A_hat_, Tensor<T> *maskTiled_, Tensor<T> *proteinEmb_)
    {
        A_hat = A_hat_;
        maskTiled = maskTiled_;
        proteinEmb = proteinEmb_;
    }

    void setBatch(u64 B) { BATCH = B; }

    // input = X, the (Nmax x FEAT) padded node-feature matrix (P1's secret).
    // Side-inputs must have been set via setSample().
    Tensor<T> &_forward(Tensor<T> &X)
    {
        // ---- drug GCN stack (aggregate-first; bias folds into FC) ----
        auto &h1 = gcn1->forward(X,  *A_hat);   // (Nmax x 188)
        auto &h2 = gcn2->forward(h1, *A_hat);   // (Nmax x 282)
        auto &h3 = gcn3->forward(h2, *A_hat);   // (Nmax x 376)

        // ---- masked global max-pool over nodes ----
        MaskedGlobalMaxPool<T> pool(this, Nmax, 376);

        Tensor<T> *pooledPtr;
        if (BATCH > 1) {
            // Path B+A: mask-multiply stays a functional _Mul node (unchanged
            // leaf-reveal semantics), then ONE fused layer runs the entire
            // tree-reduction fold internally — replacing the ~1233-node fold.
            auto &masked = this->mul(h3, *maskTiled);        // (B*Nmax x 376)
            if (mmpool == nullptr)
                mmpool = new MaskedMaxPoolLayer<T>(BATCH, Nmax, 376);
            mmpool->B = BATCH;                               // refresh if changed
            pooledPtr = &mmpool->forward(masked);            // (B x 376)
        } else {
            pooledPtr = &pool.forward(h3, *maskTiled);       // (1 x 376)
        }
        auto &pooled = *pooledPtr;

        // ---- drug embedding ----
        auto &d1  = dfc1->forward(pooled);       // (1 x 1024)
        auto &d1r = drelu1->forward(d1);
        auto &d2  = dfc2->forward(d1r);          // (1 x 128) drug embedding

        // ---- fusion ----
        auto &fused = gpu_concat->forward(d2, *proteinEmb);   // (1 x 256) GPU-aware concat
        auto &e1 = ffc1->forward(fused);
        auto &f1 = frelu1->forward(e1);   // (1 x 1024)
        auto &e2 = ffc2->forward(f1);
        auto &f2 = frelu2->forward(e2);      // (1 x 512)
        auto &f3 = frelu3->forward(ffc3->forward(f2));      // (1 x 256)
        auto &out = fout->forward(f3);                      // (1 x 1) affinity
        return out;
    }
};
