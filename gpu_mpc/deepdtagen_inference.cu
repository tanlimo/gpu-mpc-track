//
// DeepDTAGen affinity path — 2PC inference driver (iDASH 2024 Track 3).
//
// Packaged after experiments/sigma/sigma_offline_online.cu: a single binary
// with a `role` switch — role 0 is the dealer (FSS key generation written to
// disk, the offline phase), role 1 is the evaluator (online protocol). The
// crypto BACKEND is Orca (backend/orca.h), the only GPU-MPC backend providing
// the relu / select / maxPool2D gates this graph needs; the SIGMA backend
// lacks them, so we borrow only Sigma's offline/online *packaging*, not its
// backend. The model graph is DeepDTAGenAffinity<T> (see deepdtagen.h).
//
// Inputs (produced by the Python offline side, fixed-point scale = 12, bw = 32
// — the Q20.12 ring proven to clear the accuracy gate; build BW=64 for a
// 64-bit production ring). Ring element width = InfType (u32 for BW=32, u64
// for BW=64), selected by the Makefile's -DInfType flag:
//   * X, A_hat, mask : P0/P1 drug-graph secret shares  (reference/share_data.py)
//       files {x,adj,mask}_share{0,1}.dat, headerless little-endian InfType,
//       emitted by reference/offline_prepare.py
//   * proteinEmb     : GatedCNN(protein) fixed-point constant, protein_emb.dat
//       — public sequence, loaded on party 1 only (party 0 holds zero)
//   * model weights  : public weights blob weights.bin (+ weights.bin.json
//       sidecar), reference/export_weights.py. NOTE: the weight blob is int64
//       REGARDLESS of the ring bw (export_weights hardcodes BITWIDTH=64), so
//       the weight loader must read int64 even when shares are u32.
//
// BUILD/RUN (CUDA 12.1 toolkit installed at /usr/local/cuda-12.1; nvcc is not
// on the login PATH — prepend it). From this dir, for the local RTX 4060
// (SM 8.9), 32-bit ring:
//     export PATH=/usr/local/cuda-12.1/bin:$PATH
//     make GPU_MPC_ROOT=$HOME/EzPC/GPU-MPC BW=32 GPU_ARCH=89 deepdtagen_inference
// The EzPC/GPU-MPC checkout is used strictly read-only (headers + util TUs).
//
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <omp.h>

#include "ddg_orca.h"  // Forked Orca backend with mul/scalarmul for graph models
#include "ddg_orca_batched.h"  // Optimized backend with LocalARS truncation
#include "deepdtagen.h"

#ifndef InfType
#define InfType u64
#endif

// Load a length-n fixed-point share file (headerless little-endian InfType —
// u32 for BW=32, u64 for BW=64 — matching reference/share_data.py's bw mode)
// into a host Tensor<InfType>.
static void loadShare(const std::string &path, Tensor<InfType> &t)
{
    std::ifstream f(path, std::ios::binary);
    assert(f.good() && "missing share file");
    f.read((char *)t.data, t.size() * sizeof(InfType));
    assert(f.gcount() == (std::streamsize)(t.size() * sizeof(InfType)));
}

// ── int64 weight loader ────────────────────────────────────────────────────
// Reads the headerless little-endian int64 blob produced by
// reference/export_weights.py and fills the 9 FC layers of
// DeepDTAGenAffinity in forward order.
//
// Blob layout (fixed — mirrors export_weights.py _MPC_GROUPS order):
//   gcn.0   W(94,188)  b(188)
//   gcn.1   W(188,282) b(282)
//   gcn.2   W(282,376) b(376)
//   drug_fc.0 W(376,1024) b(1024)
//   drug_fc.1 W(1024,128) b(128)
//   fusion.0  W(256,1024) b(1024)
//   fusion.1  W(1024,512) b(512)
//   fusion.2  W(512,256)  b(256)
//   fusion.3  W(256,1)    b(1)
//
// Scaling:
//   weights: stored at scale s  → fill directly: (InfType)(u64)w_i64[j]
//   biases:  stored at scale s  → Orca matmul adds bias BEFORE the
//            truncation node (TruncateType::None), so bias must be at
//            scale 2s: (InfType)(u64)((i64)b_i64[j] << scale)
//
static void loadWeightsI64(DeepDTAGenAffinity<InfType> *model,
                            const std::string &path, u64 scale)
{
    // Layer table in blob order: {FC pointer, in_features, out_features}
    struct LayerSpec { FC<InfType> *fc; int in_feat; int out_feat; };
    LayerSpec layers[] = {
        { model->gcn1->lin, 94,   188  },
        { model->gcn2->lin, 188,  282  },
        { model->gcn3->lin, 282,  376  },
        { model->dfc1,      376,  1024 },
        { model->dfc2,      1024, 128  },
        { model->ffc1,      256,  1024 },
        { model->ffc2,      1024, 512  },
        { model->ffc3,      512,  256  },
        { model->fout,      256,  1    },
    };
    constexpr int N_LAYERS = 9;

    // Compute total expected int64 elements for pre-check
    size_t total_elems = 0;
    for (int l = 0; l < N_LAYERS; l++)
        total_elems += (size_t)layers[l].in_feat * layers[l].out_feat
                     + (size_t)layers[l].out_feat;

    // Read the entire blob into a vector
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good()) {
        fprintf(stderr, "[loadWeightsI64] Cannot open weights file: %s\n", path.c_str());
        exit(1);
    }
    std::streamsize file_bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    size_t n_elems = (size_t)file_bytes / sizeof(int64_t);
    if (n_elems < total_elems) {
        fprintf(stderr, "[loadWeightsI64] weights.bin too small: got %zu int64 elements, expected %zu\n",
                n_elems, total_elems);
        exit(1);
    }
    std::vector<int64_t> buf(n_elems);
    f.read((char *)buf.data(), (std::streamsize)(n_elems * sizeof(int64_t)));

    size_t offset = 0;
    for (int l = 0; l < N_LAYERS; l++) {
        int w_size = layers[l].in_feat * layers[l].out_feat;
        int b_size = layers[l].out_feat;

        // Fill weights: stored at scale s, use directly (low bw bits)
        TensorRef<InfType> wref = layers[l].fc->getweights();
        assert((int)wref.size == w_size && "weight size mismatch");
        for (int j = 0; j < w_size; j++)
            wref.data[j] = (InfType)(uint64_t)(int64_t)buf[offset + j];
        offset += w_size;

        // Fill biases: stored at scale s, must be at scale 2s for Orca matmul
        TensorRef<InfType> bref = layers[l].fc->getbias();
        assert((int)bref.size == b_size && "bias size mismatch");
        for (int j = 0; j < b_size; j++)
            bref.data[j] = (InfType)(uint64_t)((int64_t)buf[offset + j] << (int)scale);
        offset += b_size;
    }
}

int main(int argc, char *argv[])
{
    // argv: bw scale role party keyDir shareDir [batch] [ip]
    sytorch_init();
    int bw       = atoi(argv[1]);
    u64 scale    = strtoul(argv[2], 0, 10);
    int role     = atoi(argv[3]);   // 0 = dealer, 1 = evaluator
    int party    = atoi(argv[4]);
    auto keyDir  = std::string(argv[5]);
    auto shareDir = std::string(argv[6]);
    int BATCH    = (argc > 7 && atoi(argv[7]) > 0) ? atoi(argv[7]) : 1;
    assert(bw <= 8 * (int)sizeof(InfType));
    assert(scale < (u64)bw);
    assert(BATCH > 0 && BATCH <= 128);

    const u64 Nmax = DDG_NMAX, FEAT = DDG_FEAT;

    auto model = new DeepDTAGenAffinity<InfType>();

    // Primary input X (Nmax x FEAT). Secret side-inputs:
    //   A_hat      (Nmax x Nmax)  P1 graph
    //   maskTiled  (Nmax x 376)   P1 node mask, tiled across pooled channels
    //   proteinEmb (1 x 128)      P2 GatedCNN output (public seq) as a share
    // GCN biases fold into P2's FC weights (see gcn_layer.h) — no bias leaf here.
    // Batched mode folds the batch dim into rows: X (B*Nmax x FEAT),
    // A_hat (B*Nmax x Nmax), maskTiled (B*Nmax x 376), proteinEmb (B x 128).
    // For B=1 this reduces to the original single-sample shapes.
    Tensor<InfType> X({(u64)BATCH * Nmax, FEAT});
    Tensor<InfType> A_hat({(u64)BATCH * Nmax, Nmax});
    Tensor<InfType> maskTiled({(u64)BATCH * Nmax, 376});
    Tensor<InfType> proteinEmb({(u64)BATCH, 128});

    // Dealer: zero-init for now; actual random masks generated after backend init
    // Evaluator: zero-init (overwritten by loadShare below)
    X.zero(); A_hat.zero(); maskTiled.zero(); proteinEmb.zero();
    model->setSample(&A_hat, &maskTiled, &proteinEmb);
    model->setBatch((u64)BATCH);

    // ── side-input graph-gen setup ──────────────────────────────────────────
    // SytorchModule::init(scale, X) sets X.graphGenMode=true and creates X's
    // PlaceHolder graphNode pointing to &allNodesInExecutionOrder.  But
    // functionalGraphGen (used by matmul/mul/concat/etc.) fires
    // always_assert(a->graphGenMode) for EVERY argument, so side-inputs that
    // participate in functional ops (A_hat, maskTiled, proteinEmb) must also
    // have graphGenMode=true and a valid graphNode before init() is called.
    //
    // We mirror exactly what genGraphAndExecutionOrder does for the primary
    // input (module.h lines 106-116), and point allNodesInExecutionOrderRef
    // at the same vector so the topological order is built consistently.
    //
    // currTensor must be set to the live host tensor so the execution loop
    // (forward non-graph-gen path) can find the real data via
    //   p->currTensor  for each parent p of a functional node.
    {
        auto *aref = &model->allNodesInExecutionOrder;
        auto prepareSideInput = [&](Tensor<InfType> &t) {
            t.graphGenMode = true;
            t.graphNode    = new LayerGraphNode<InfType>();
            t.graphNode->layer    = new PlaceHolderLayer<InfType>("Input");
            t.graphNode->currTensor = &t;           // execution: live data lives here
            t.graphNode->allNodesInExecutionOrderRef = aref;
        };
        prepareSideInput(A_hat);
        prepareSideInput(maskTiled);
        prepareSideInput(proteinEmb);
    }

    model->init(scale, X);

    // genGraphAndExecutionOrder resets X.graphGenMode=false at line 115 of
    // module.h, but it never touches the side-inputs.  Reset them now so the
    // execution path doesn't confuse them with graph-gen tensors.
    A_hat.graphGenMode    = false;
    maskTiled.graphGenMode = false;
    proteinEmb.graphGenMode = false;

    model->zero();

    // Load trained weights from the int64 blob produced by reference/export_weights.py.
    // Path is taken from env var DDG_WEIGHTS_BIN; falls back to ./weights.bin.
    // Loading on BOTH dealer and evaluator: dealer needs only shapes (harmless to load);
    // evaluator needs real values for correct arithmetic.
    // MUST run AFTER model->zero() so loaded values are not wiped.
    {
        const char *wenv = getenv("DDG_WEIGHTS_BIN");
        std::string wpath = (wenv && wenv[0]) ? std::string(wenv) : std::string("./weights.bin");
        loadWeightsI64(model, wpath, scale);
    }

    auto expName = std::string("DeepDTAGen_") + std::to_string(bw) + "_" + std::to_string(scale);
    auto keyFileName = keyDir + expName + "_party" + std::to_string(party);

    if (role == 0)
    {
        // Dealer: generate FSS keys for the whole graph.
        // The DDGOrcaKeygen constructor calls initGPURandomness() (seed 12345),
        // so randomGEOnGpu is ready AFTER this line.

        // Select backend based on optimization flags
        DDGOrcaBaseKeygen<InfType> *fss = nullptr;
        if (std::getenv("DDG_SLACK_TRUNC") || std::getenv("DDG_LOCAL_TRUNC")) {
            fss = new DDGOrcaKeygenBatched<InfType>(party, bw, scale, keyFileName);
        } else {
            fss = new DDGOrcaKeygen<InfType>(party, bw, scale, keyFileName);
        }
        model->setBackend(fss);
        model->optimize();

        // ORCA SMALL-MASK SCHEME (solves large-mask truncation precision loss).
        // The mask r_X is share0 (the pad from offline split_shares). To avoid
        // FSS truncation errors from wrap-around at the 2^31 sign boundary, the
        // OFFLINE share generator MUST use a small mask (mask_bw≈14) rather than
        // a full-range 32-bit pad. Both dealers load the SAME share0, so
        // writeShares produces consistent complementary key shares.
        //
        // DIAGNOSTIC CONFIRMED: bw=32 mask → 11.9x error (64.15 vs 5.37);
        //                       bw=14 mask → 13% error  (4.66 vs 5.37, optimal).
        // NOTE: proteinEmb is P2's public value (party 1 owns it, party 0 zero).
        // Keygen uses r_protein = 0 for both parties (proteinEmb.d_data stays 0),
        // making the fused mask [r_d2 | 0]. Eval reconstructs the additive share
        // (0, P) to get P on both parties before concat.
        loadShare(shareDir + "/x_share0.dat", X);
        loadShare(shareDir + "/adj_share0.dat", A_hat);
        loadShare(shareDir + "/mask_share0.dat", maskTiled);
        // proteinEmb NOT loaded in keygen → d_data stays 0 → r_protein = 0

        X.d_data         = (InfType *)moveToGPU((u8 *)X.data, X.size() * sizeof(InfType), nullptr);
        A_hat.d_data     = (InfType *)moveToGPU((u8 *)A_hat.data, A_hat.size() * sizeof(InfType), nullptr);
        maskTiled.d_data = (InfType *)moveToGPU((u8 *)maskTiled.data, maskTiled.size() * sizeof(InfType), nullptr);
        proteinEmb.d_data = (InfType *)moveToGPU((u8 *)proteinEmb.data, proteinEmb.size() * sizeof(InfType), nullptr);

        auto &out = model->forward(X);
        fss->output(out);
        fss->close();
    }
    else
    {
        // Evaluator: load this party's real secret shares, then run online.
        //  P1 secrets (drug graph): X, A_hat, maskTiled — both parties hold a share.
        loadShare(shareDir + "/x_share"    + std::to_string(party) + ".dat", X);
        loadShare(shareDir + "/adj_share"  + std::to_string(party) + ".dat", A_hat);
        loadShare(shareDir + "/mask_share" + std::to_string(party) + ".dat", maskTiled);
        //  P2 public constant (GatedCNN output): P2 holds the value, P1 holds
        //  zero. Load only on party 1; party 0's tensor stays zeroed.
        if (party == 1)
            loadShare(shareDir + "/protein_emb.dat", proteinEmb);

        auto ip = (argc > 8) ? argv[8] : "127.0.0.1";

        // Select backend based on optimization flags
        DDGOrcaBase<InfType> *fss = nullptr;
        if (std::getenv("DDG_SLACK_TRUNC") || std::getenv("DDG_LOCAL_TRUNC")) {
            fss = new DDGOrcaBatched<InfType>(party, ip, bw, (int)scale, keyFileName);
        } else {
            fss = new DDGOrca<InfType>(party, ip, bw, (int)scale, keyFileName);
        }
        model->setBackend(fss);
        model->optimize();

        std::vector<u64> times;
        u64 commBytes = 0;
        lseek(fss->fd, 0, SEEK_SET);
        readKey(fss->fd, fss->keySize, fss->startPtr, NULL);

        // ── OPTIMIZATION: One-time H2D upload + per-iteration D2D refresh ──────
        // revealIfLeaf mutates GPU data in-place (share → revealed value), so we
        // can't reuse the same buffer across iterations. Instead: upload pristine
        // copies once (H2D, slow), then use fast D2D memcpy each iteration to
        // refresh working buffers from the pristine copies.
        //
        // Baseline: 11× H2D upload (moveToGPU) of ~4 MB → ~3 ms total on PCIe 3.0×16
        // Optimized: 1× H2D + 10× D2D refresh → ~0.3 ms H2D + ~0.05 ms D2D (50× faster)
        InfType *d_X_pristine          = (InfType *)moveToGPU((u8 *)X.data,          X.size()          * sizeof(InfType), NULL);
        InfType *d_A_hat_pristine      = (InfType *)moveToGPU((u8 *)A_hat.data,      A_hat.size()      * sizeof(InfType), NULL);
        InfType *d_maskTiled_pristine  = (InfType *)moveToGPU((u8 *)maskTiled.data,  maskTiled.size()  * sizeof(InfType), NULL);
        InfType *d_proteinEmb_pristine = (InfType *)moveToGPU((u8 *)proteinEmb.data, proteinEmb.size() * sizeof(InfType), NULL);

        // Allocate working buffers (will be overwritten by D2D copy each iteration)
        InfType *d_X_work          = (InfType *)gpuMalloc(X.size()          * sizeof(InfType));
        InfType *d_A_hat_work      = (InfType *)gpuMalloc(A_hat.size()      * sizeof(InfType));
        InfType *d_maskTiled_work  = (InfType *)gpuMalloc(maskTiled.size()  * sizeof(InfType));
        InfType *d_proteinEmb_work = (InfType *)gpuMalloc(proteinEmb.size() * sizeof(InfType));

        // Production inference should execute the MPC graph exactly once.
        // Historical benchmarking used 11 forwards (1 warmup + 10 measured).
        // Keep that mode available explicitly via DDG_INFERENCE_ITERS=11.
        int inferenceIters = 1;
        if (const char *env = std::getenv("DDG_INFERENCE_ITERS")) {
            inferenceIters = std::atoi(env);
            if (inferenceIters <= 0) {
                fprintf(stderr,
                        "[DeepDTAGen] DDG_INFERENCE_ITERS must be >= 1\n");
                exit(1);
            }
        }

        printf("[DeepDTAGen] inference iterations = %d\n",
               inferenceIters);

        Tensor<InfType> *out_ptr = nullptr;   // capture last forward output
        for (int i = 0; i < inferenceIters; i++)
        {
            fss->keyBuf = fss->startPtr;
            fss->s.reset();
            fss->sxsMatmulIdx = 0;   // reset per-forward SxS matmul counter (reveal only fires on #0)
            fss->resetLeaves();      // clear leaf registry from prior iteration
            fss->peer->sync();
            auto commStart = fss->peer->bytesSent() + fss->peer->bytesReceived();
            auto start = std::chrono::high_resolution_clock::now();

            // Fast D2D refresh from pristine → working buffers (replaces slow H2D moveToGPU)
            checkCudaErrors(cudaMemcpy(d_X_work,          d_X_pristine,          X.size()          * sizeof(InfType), cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemcpy(d_A_hat_work,      d_A_hat_pristine,      A_hat.size()      * sizeof(InfType), cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemcpy(d_maskTiled_work,  d_maskTiled_pristine,  maskTiled.size()  * sizeof(InfType), cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemcpy(d_proteinEmb_work, d_proteinEmb_pristine, proteinEmb.size() * sizeof(InfType), cudaMemcpyDeviceToDevice));

            X.d_data          = d_X_work;
            A_hat.d_data      = d_A_hat_work;
            maskTiled.d_data  = d_maskTiled_work;
            proteinEmb.d_data = d_proteinEmb_work;

            // Register secret leaves for reveal-on-first-use. proteinEmb is P2's
            // public value (party 1 holds it, party 0 holds zero), but it still
            // needs to be revealed so both parties hold the same masked-public value.
            //
            // Batched mode: A_hat and X (first GCN input) use per-slice reveal in
            // _MatMul. Register each slice pointer so revealIfLeaf matches.
            if (BATCH == 1) {
                fss->registerLeaf(X.d_data);
                fss->registerLeaf(A_hat.d_data);
                fss->registerLeaf(maskTiled.d_data);
                fss->registerLeaf(proteinEmb.d_data);
            } else {
                // X: register each (Nmax×FEAT) slice for batched _MatMul reveal in gcn1.
                for (int b = 0; b < BATCH; ++b) {
                    fss->registerLeaf(X.d_data + b * Nmax * FEAT);
                }

                // A_hat: register each (Nmax×Nmax) slice for batched _MatMul reveal.
                for (int b = 0; b < BATCH; ++b) {
                    fss->registerLeaf(A_hat.d_data + b * Nmax * Nmax);
                }

                // maskTiled: register full folded pointer (pool uses full tensor).
                fss->registerLeaf(maskTiled.d_data);

                // proteinEmb: batched (B×128), register full pointer for GPUConcat reveal.
                fss->registerLeaf(proteinEmb.d_data);
            }

            auto &out = model->forward(X);
            fss->output(out);
            out_ptr = &out;   // model holds this tensor; stable across iters
            auto end = std::chrono::high_resolution_clock::now();
            // Benchmark mode (>1 iteration): iteration 0 is warmup.
            // Production mode (1 iteration): measure that single execution.
            if (inferenceIters == 1 || i > 0)
                times.push_back(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        end - start
                    ).count()
                );
            auto commEnd = fss->peer->bytesSent() + fss->peer->bytesReceived();
            if (i == 0)
                commBytes = commEnd - commStart;
        }

        // Clean up pristine and working GPU buffers
        gpuFree(d_X_pristine);
        gpuFree(d_A_hat_pristine);
        gpuFree(d_maskTiled_pristine);
        gpuFree(d_proteinEmb_pristine);
        gpuFree(d_X_work);
        gpuFree(d_A_hat_work);
        gpuFree(d_maskTiled_work);
        gpuFree(d_proteinEmb_work);

        fss->close();
        auto avgTime = std::reduce(times.begin(), times.end()) / (float)times.size();
        printf("Average time taken (microseconds)=%f\n", avgTime);
        printf("Comm (B)=%lu\n", commBytes);

        // Per-operation timing breakdown (microseconds)
        if (party == 0) {
            printf("\n=== Timing breakdown (us) ===\n");
            printf("  matmul:    %10lu  (comm: %10lu)\n", fss->s.matmul_time, fss->s.matmul_comm_time);
            printf("  relu:      %10lu\n", fss->s.relu_time);
            printf("  reluext:   %10lu  (comm: %10lu)\n", fss->s.reluext_time, fss->s.reluext_comm_time);
            printf("  maxpool:   %10lu  (comm: %10lu)\n", fss->s.maxpool_time, fss->s.maxpool_comm_time);
            printf("  truncate:  %10lu  (comm: %10lu)\n", fss->s.truncate_time, fss->s.truncate_comm_time);
            printf("  signext:   %10lu\n", fss->s.signext_time);
            printf("  transfer:  %10lu\n", fss->s.transfer_time);
            printf("  compute:   %10lu\n", fss->s.compute_time);
            printf("  total_comm:%10lu\n", fss->s.comm_time);
            printf("=============================\n\n");
        }

        // Reveal the final affinity scalar.
        // fss->output() reconstructs shares onto party 0's host tensor.
        // Party 0 prints; party 1 stays silent to avoid duplicate output.
        if (party == 0 && out_ptr != nullptr) {
            // Reinterpret each ring value as signed int32, divide by 2^scale.
            // Batched mode: output is (B, 1) → print B affinities (one per sample).
            int n_out = (int)out_ptr->size();
            for (int j = 0; j < n_out; j++) {
                int32_t sv = (int32_t)(uint32_t)(uint64_t)out_ptr->data[j];
                double  aff = (double)sv / (double)(1LL << scale);
                if (n_out == 1)
                    printf("AFFINITY=%.6f\n", aff);
                else
                    printf("AFFINITY[%d]=%.6f\n", j, aff);
            }
        }
    }
    return 0;
}
