// Forked from EzPC/GPU-MPC backend/orca_base.h (commit 66d9cddc, Aug 2024)
// Extended for DeepDTAGen graph model: added mul()/scalarmul() methods for
// element-wise secret×secret and public-scalar multiplication (GCN + MaskedMaxPool).
// Also: keyBufSize capped to 2 GiB (upstream 20 GiB OOMs on 7.6 GiB WSL2 box).
//
// Original Authors: Neha Jawalkar
// Copyright (c) 2024 Microsoft Research
// Licensed under MIT (see original file header for full text).
//
// Modifications Copyright (c) 2026 iDASH Track 3 submission team.

#pragma once

#include <omp.h>
#include <cstdlib>
#include <set>

#include <sytorch/backend/backend.h>
#include <sytorch/backend/llama_base.h>
#include <llama/comms.h>
#include <llama/api.h>

#include "ddg_orca_opt.h"

#include "utils/gpu_random.h"
#include "utils/gpu_mem.h"

#include "fss/gpu_matmul.h"
#include "fss/gpu_conv2d.h"
#include "fss/gpu_relu.h"
#include "fss/gpu_maxpool.h"
#include "fss/gpu_avgpool.h"
#include "fss/gpu_add.h"
#include "fss/gpu_mul.h"
#include "fss/gpu_scalarmul.h"

template <typename T>
class DDGOrcaBase : public Backend<T>
{
public:
    u8 *startPtr = NULL;
    u8 *keyBuf = NULL;
    size_t keySize = 0;
    int fd = -1;
    GpuPeer *peer = NULL;
    int party = -1;
    Stats s;
    int bw;
    int scale;
    AESGlobalContext g;

    // DEPRECATED: sxsMatmulIdx was used by the old reveal-once heuristic.
    // Now unused (Sigma-native SxS matmul has no reveal step), but kept for
    // backward compat with inference.cu:294 reset line.
    int sxsMatmulIdx = 0;

    // Sigma-native leaf tracking: register secret leaves before each forward pass,
    // reveal each on first use. Intermediates are already masked-public from their
    // source ops, so only leaves need explicit reveal.
    std::set<void*> unrevealedLeaves;

    void registerLeaf(void *d_ptr) {
        unrevealedLeaves.insert(d_ptr);
    }

    void resetLeaves() {
        unrevealedLeaves.clear();
    }

    // Reveal helper: if d_data is an unrevealed leaf, add mask and reconstruct.
    // Returns true if revealed, false if already masked-public (intermediate or pre-revealed leaf).
    bool revealIfLeaf(T *d_data, T *d_mask, int bw, u64 size) {
        if (unrevealedLeaves.count(d_data)) {
            gpuLinearComb(bw, size, d_data, T(1), d_data, T(1), d_mask);
            peer->reconstructInPlace(d_data, bw, size, &s);
            unrevealedLeaves.erase(d_data);
            return true;
        }
        return false;
    }

    // Reveal public leaf: reconstruct additive share (no mask, just reconstruct).
    // Use for proteinEmb (party0=0, party1=P, r=0) → P on both parties.
    void revealPublicLeaf(T *d_data, int bw, u64 size) {
        if (unrevealedLeaves.count(d_data)) {
            peer->reconstructInPlace(d_data, bw, size, &s);
            unrevealedLeaves.erase(d_data);
        }
    }

    DDGOrcaBase() {}

    DDGOrcaBase(int party, std::string ip, int bw, int scale, std::string keyFile = "", bool compress = true) : party(party), bw(bw), scale(scale)
    {
        initAESContext(&g);
        initGPUMemPool();
        if (keyFile.compare("") != 0)
        {
            auto filename =
                keyFile + "_inference_key" +
                std::to_string(party) + ".dat";

            const char *explicitBytesEnv =
                std::getenv("DDG_EVAL_KEY_CHUNK_BYTES");

            const bool externalKeyIO =
                std::getenv("DDG_EVAL_EXTERNAL_KEY_IO") != nullptr;

            if (explicitBytesEnv && explicitBytesEnv[0]) {
                char *endPtr = nullptr;

                unsigned long long parsed =
                    std::strtoull(
                        explicitBytesEnv,
                        &endPtr,
                        10
                    );

                if (
                    parsed == 0 ||
                    endPtr == explicitBytesEnv ||
                    *endPtr != '\0'
                ) {
                    fprintf(
                        stderr,
                        "[DDGOrcaEval] invalid "
                        "DDG_EVAL_KEY_CHUNK_BYTES=%s\n",
                        explicitBytesEnv
                    );
                    exit(1);
                }

                keySize = static_cast<size_t>(parsed);

                if (keySize % 4096 != 0) {
                    fprintf(
                        stderr,
                        "[DDGOrcaEval] explicit key chunk "
                        "size %zu is not 4096-byte aligned\n",
                        keySize
                    );
                    exit(1);
                }

                printf(
                    "[DDGOrcaEval] explicit key chunk "
                    "bytes=%zu external_io=%d\n",
                    keySize,
                    externalKeyIO ? 1 : 0
                );

                // Compatibility/test mode:
                // explicit chunk size, but still consume the normal key file.
                if (!externalKeyIO) {
                    if (!std::filesystem::exists(filename)) {
                        fprintf(
                            stderr,
                            "[DDGOrcaEval] missing key file: %s\n",
                            filename.c_str()
                        );
                        exit(1);
                    }

                    size_t totalKeySize =
                        static_cast<size_t>(
                            std::filesystem::file_size(filename)
                        );

                    if (totalKeySize < keySize) {
                        fprintf(
                            stderr,
                            "[DDGOrcaEval] key file too small: "
                            "total=%zu chunk=%zu\n",
                            totalKeySize,
                            keySize
                        );
                        exit(1);
                    }

                    fd = openForReading(filename);
                }
                else {
                    // D2 pipeline mode:
                    // no key file is required at constructor time.
                    // Persistent evaluator code will fill startPtr
                    // from a bounded key slot for each chunk.
                    fd = -1;
                }
            }
            else {
                if (externalKeyIO) {
                    fprintf(
                        stderr,
                        "[DDGOrcaEval] "
                        "DDG_EVAL_EXTERNAL_KEY_IO requires "
                        "DDG_EVAL_KEY_CHUNK_BYTES\n"
                    );
                    exit(1);
                }

                size_t totalKeySize =
                    static_cast<size_t>(
                        std::filesystem::file_size(filename)
                    );

                // Default one-shot evaluator:
                // the whole file is one key chunk.
                keySize = totalKeySize;

                // Existing persistent sequential-file mode.
                if (const char *chunksEnv =
                        std::getenv("DDG_EVAL_CHUNKS")) {

                    long nChunks =
                        std::strtol(
                            chunksEnv,
                            nullptr,
                            10
                        );

                    if (nChunks <= 0) {
                        fprintf(
                            stderr,
                            "[DDGOrcaEval] DDG_EVAL_CHUNKS "
                            "must be >= 1\n"
                        );
                        exit(1);
                    }

                    if (
                        totalKeySize %
                        static_cast<size_t>(nChunks) != 0
                    ) {
                        fprintf(
                            stderr,
                            "[DDGOrcaEval] key file size %zu "
                            "is not divisible by chunks=%ld\n",
                            totalKeySize,
                            nChunks
                        );
                        exit(1);
                    }

                    keySize =
                        totalKeySize /
                        static_cast<size_t>(nChunks);

                    if (keySize % 4096 != 0) {
                        fprintf(
                            stderr,
                            "[DDGOrcaEval] chunk key size %zu "
                            "is not 4096-byte aligned\n",
                            keySize
                        );
                        exit(1);
                    }

                    printf(
                        "[DDGOrcaEval] sequential key: "
                        "total=%zu chunks=%ld chunk=%zu\n",
                        totalKeySize,
                        nChunks,
                        keySize
                    );
                }

                fd = openForReading(filename);
            }

            // Always allocate exactly one key chunk.
            getAlignedBuf(
                &keyBuf,
                keySize,
                false
            );

            startPtr = keyBuf;
        }
        peer = new GpuPeer(compress);
        peer->connect(party, ip);
    }

    void close()
    {
        peer->close();
    }

    void conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, bool useBias, const Tensor1D<T> &bias, Tensor4D<T> &output, bool isFirst)
    {
        auto comm_start = s.comm_time;
        auto start = std::chrono::high_resolution_clock::now();
        GPUConv2DKey<T> k;
        k.p = {
            bw, bw, (int)input.d1, (int)input.d2, (int)input.d3, (int)ci,
            (int)fh, (int)fw, (int)co, (int)padding, (int)padding, (int)padding, (int)padding,
            (int)stride, (int)stride, 0, 0, 0, 0, 0};
        fillConv2DParams(&(k.p));
        k.mem_size_I = k.p.size_I * sizeof(T);
        k.mem_size_F = k.p.size_F * sizeof(T);
        k.mem_size_O = k.p.size_O * sizeof(T);

        k.I = (T *)keyBuf;
        keyBuf += k.mem_size_I;
        k.F = (T *)keyBuf;
        keyBuf += k.mem_size_F;
        k.O = (T *)keyBuf;
        keyBuf += k.mem_size_O;

        auto d_mask_I = (T *)moveToGPU((u8 *)k.I, k.mem_size_I, &s);
        if (isFirst)
        {
            gpuLinearComb(bw, k.p.size_I, input.d_data, T(1), input.d_data, T(1), d_mask_I);
            peer->reconstructInPlace(input.d_data, bw, k.p.size_I, &s);
        }
        auto d_F = (T *)moveToGPU((u8 *)filter.data, k.mem_size_F, &s);
        auto d_mask_F = (T *)moveToGPU((u8 *)k.F, k.mem_size_F, &s);
        auto d_C = gpuConv2DBeaver<T>(k, party, input.d_data, d_F, d_mask_I, d_mask_F, useBias && party == SERVER0 ? bias.data : (T *)NULL, &s, 0);

        gpuFree(d_F);
        gpuFree(d_mask_I);
        gpuFree(d_mask_F);
        peer->reconstructInPlace(d_C, k.p.bout, k.p.size_O, &s);
        output.d_data = d_C;

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        s.conv_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        auto comm_end = s.comm_time;
        s.conv_comm_time += (comm_end - comm_start);
    }

    // ── 3-arg matmul: secret×secret (eval, SIGMA NATIVE) ────────────────────
    // Returns MASKED-PUBLIC output. Truncation handled by _MatMul's doTruncationForward.
    // Inputs a.d_data, b.d_data are SHARES that get revealed to masked-public.
    void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c)
    {
        auto comm_start = s.comm_time;
        auto start = std::chrono::high_resolution_clock::now();

        MatmulParams p;
        p.M = a.d1;
        p.K = a.d2;
        p.N = b.d2;
        p.batchSz = 1;
        stdInit(p, bw, 0);  // Truncation by _MatMul's doTruncationForward=true node

        auto k = readGPUMatmulKey<T>(p, TruncateType::None, &keyBuf);

        // Load mask share for operand A (gpuMatmul reloads k.B internally).
        T *d_mask_A = (T *)moveToGPU((u8 *)k.A, k.mem_size_A, &s);
        T *d_mask_B = (T *)moveToGPU((u8 *)k.B, k.mem_size_B, &s);

        // Reveal inputs if they're leaves (first use). After reveal, both parties
        // hold the masked-public value (e.g., X + r_X). Intermediates are already
        // masked-public (previous matmul reconstructed + truncated them).
        revealIfLeaf(a.d_data, d_mask_A, bw, p.size_A);
        revealIfLeaf(b.d_data, d_mask_B, bw, p.size_B);
        gpuFree(d_mask_B);  // gpuMatmul reloads k.B internally as d_mask_W

        // gpuMatmul-full: Beaver + reconstruct (no internal truncate; _MatMul node does it).
        // Returns MASKED-PUBLIC (A·B) with output mask r_Z baked in. Scale = 2s.
        c.d_data = gpuMatmul(peer, party, p, k, a.d_data, b.d_data, (T *)NULL,
                             TruncateType::None, &g, &s, /*wIsOnGpu=*/true, d_mask_A);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        s.matmul_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        auto comm_end = s.comm_time;
        s.matmul_comm_time += (comm_end - comm_start);
    }

    // ── 4-arg matmul: secret×public (FC layers, with M=1 padding fix) ──────
    void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c, bool useBias, Tensor1D<T> &d, bool isFirst)
    {
        auto comm_start = s.comm_time;
        auto start = std::chrono::high_resolution_clock::now();

        // CUTLASS fails with M=1 for integer GEMMs. Pad to M=128 to match keygen.
        if (a.d1 == 1) {
            const int M_PADDED = 128;
            const u64 padded_size_A = M_PADDED * a.d2;
            T *d_A_padded = (T *)gpuMalloc(padded_size_A * sizeof(T));
            checkCudaErrors(cudaMemcpy(d_A_padded, a.d_data, a.d2 * sizeof(T), cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemset(d_A_padded + a.d2, 0, (M_PADDED - 1) * a.d2 * sizeof(T)));

            MatmulParams p;
            p.M = M_PADDED;
            p.K = a.d2;
            p.N = b.d2;
            p.batchSz = 1;
            stdInit(p, bw, 0);  // FC truncation handled by separate truncateForward node
            auto k = readGPUMatmulKey<T>(p, TruncateType::None, &keyBuf);

            auto d_mask_A = (T *)moveToGPU((u8 *)k.A, k.mem_size_A, &s);
            // Leaf reveal on the ORIGINAL (unpadded) leaf pointer a.d_data. The
            // pad copies row 0 into d_A_padded, so reveal a.d_data first then copy.
            if (isFirst || unrevealedLeaves.count(a.d_data))
            {
                // Reveal padded operand (mask covers M=128 rows; row0 is the leaf).
                gpuLinearComb(bw, p.size_A, d_A_padded, T(1), d_A_padded, T(1), d_mask_A);
                peer->reconstructInPlace(d_A_padded, bw, p.size_A, &s);
                unrevealedLeaves.erase(a.d_data);
            }
            T *d_C_padded = gpuMatmul(peer, party, p, k, d_A_padded, b.data, useBias ? d.data : (T *)NULL, TruncateType::None, &g, &s, false, d_mask_A);

            // Extract row 0 from padded output
            c.d_data = (T *)gpuMalloc(b.d2 * sizeof(T));
            checkCudaErrors(cudaMemcpy(c.d_data, d_C_padded, b.d2 * sizeof(T), cudaMemcpyDeviceToDevice));

            gpuFree(d_A_padded);
            gpuFree(d_C_padded);
        } else {
            MatmulParams p;
            p.M = a.d1;
            p.K = a.d2;
            p.N = b.d2;
            p.batchSz = 1;
            stdInit(p, bw, 0);  // FC truncation handled by separate truncateForward node
            auto k = readGPUMatmulKey<T>(p, TruncateType::None, &keyBuf);

            auto d_mask_A = (T *)moveToGPU((u8 *)k.A, k.mem_size_A, &s);
            if (isFirst || unrevealedLeaves.count(a.d_data))
            {
                gpuLinearComb(bw, p.size_A, a.d_data, T(1), a.d_data, T(1), d_mask_A);
                peer->reconstructInPlace(a.d_data, bw, p.size_A, &s);
                unrevealedLeaves.erase(a.d_data);
            }
            c.d_data = gpuMatmul(peer, party, p, k, a.d_data, b.data, useBias ? d.data : (T *)NULL, TruncateType::None, &g, &s, false, d_mask_A);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        s.matmul_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        auto comm_end = s.comm_time;
        s.matmul_comm_time += (comm_end - comm_start);
    }

    // ── ADDED: element-wise secret×secret multiply (eval) ───────────────────
    virtual void mul(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out)
    {
        // Override in derived class (DDGOrca)
        throw std::runtime_error("mul() not implemented in DDGOrcaBase");
    }

    // ── ADDED: public-scalar multiply (eval) ────────────────────────────────
    virtual void scalarmul(Tensor<T> &x, T scalar, Tensor<T> &y)
    {
        // Override in derived class (DDGOrca)
        throw std::runtime_error("scalarmul() not implemented in DDGOrcaBase");
    }

    void avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, u64 scale)
    {
        AvgPoolParams p = {
            bw, bw, (int)scale, (int)scale, 0, (int)in.d1, (int)in.d2, (int)in.d3, (int)in.d4,
            (int)ks, (int)ks, (int)stride, (int)stride, (int)padding, (int)padding, (int)padding, (int)padding, 0, 0, false};
        initPoolParams(p);
        out.d_data = gpuAddPool(p, in.d_data, &s);
    }

    void output(Tensor<T> &a)
    {
        int N = a.size();
        unmaskValues(bw, N, a.d_data, (T *)keyBuf, &s);
        gpuLocalTr<T, T, ars>(party, bw, scale, N, a.d_data, true);
        moveIntoCPUMem((u8 *)a.data, (u8 *)a.d_data, N * sizeof(T), &s);
    }

    void add(const std::vector<Tensor<T> *> &in, Tensor<T> &out)
    {
        int tmpBw = bw;  // Fixed-point add preserves scale: use full ring
        int N = in[0]->size();
        std::vector<T *> gpuInp;
        for (int i = 0; i < in.size(); i++)
        {
            gpuInp.push_back(in[i]->d_data);
        }
        out.d_data = gpuAdd(tmpBw, N, gpuInp);
    }

    void optimize(LayerGraphNode<T> *root)
    {
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { ddgOrcaOpt<T>(n, r); });
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { pinCpuMem(n, r); });
    }
};

template <typename T>
class DDGOrcaBaseKeygen : public Backend<T>
{
public:
    u8 *startPtr;
    u8 *keyBuf = NULL;
    size_t keyBufSize = 0;
    int party = -1;
    std::string keyFile;
    int scale;
    int bw;
    AESGlobalContext g;

    DDGOrcaBaseKeygen(int party, int bw, int scale, std::string keyFile) : party(party), bw(bw), scale(scale), keyFile(keyFile)
    {
        initAESContext(&g);
        initGPURandomness();
        initCPURandomness();
        initGPUMemPool();

        // Dealer key buffer.
        //
        // The previous 20 GiB hard cap was introduced for an older
        // small-memory development machine.  BW64 batched DeepDTAGen
        // needs substantially more key material, so scalability
        // experiments allow up to 64 GiB.
        //
        // 64 GiB is only the hard upper bound.  The actual allocation is
        // controlled by DDG_KEYBUF_CAP_GB and remains 2 GiB by default.
        constexpr size_t HARD_CAP_GB = 64;

        const char *cap_env = std::getenv("DDG_KEYBUF_CAP_GB");
        size_t cap_gb = cap_env
            ? std::strtoull(cap_env, nullptr, 10)
            : 2;

        if (cap_gb < 1 || cap_gb > HARD_CAP_GB) {
            printf("[DDGOrcaKeygen] invalid DDG_KEYBUF_CAP_GB=%zu; "
                   "valid range is 1..%zu GiB\n",
                   cap_gb, HARD_CAP_GB);
            exit(1);
        }

        keyBufSize = cap_gb * OneGB;

        printf("[DDGOrcaKeygen] key buffer cap = %zu GiB\n",
               cap_gb);

        getAlignedBuf(&keyBuf, keyBufSize, true);
        startPtr = keyBuf;
    }

    // Write one generated key chunk and reuse the same pinned buffer.
    // Randomness state deliberately remains alive across chunks.
    size_t flushChunk(int fd)
    {
        size_t keySize = keyBuf - startPtr;
        size_t padding = 4096 - (keySize % 4096);

        char *zeros = new char[padding];
        memset(zeros, 0, padding);

        memcpy(keyBuf, zeros, padding);
        keyBuf += padding;
        keySize += padding;

        delete[] zeros;

        assert(keySize < keyBufSize &&
               "Key buffer overflow — raise DDG_KEYBUF_CAP_GB");

        writeKeyBuf(fd, keySize, startPtr);

        // Persistent dealer: next chunk starts from the same allocation.
        keyBuf = startPtr;

        return keySize;
    }

    // Destroy resources only when the dealer process is really finished.
    void finalize()
    {
        if (startPtr != nullptr) {
            cpuFree(startPtr, true);
            startPtr = nullptr;
            keyBuf = nullptr;
        }

        destroyGPURandomness();
        destroyCPURandomness();
    }

    // Backward-compatible one-chunk behaviour.
    void close()
    {
        int fd = openForWriting(
            keyFile + "_inference_key" +
            std::to_string(party) + ".dat"
        );

        flushChunk(fd);

        assert(0 == fsync(fd) && "sync error!");
        closeFile(fd);

        finalize();
    }

    void conv2D(u64 fh, u64 fw, u64 padding, u64 stride, u64 ci, u64 co, const Tensor4D<T> &input, const Tensor2D<T> &filter, Tensor4D<T> &output, bool isFirst)
    {
        GPUConv2DKey<T> k;
        k.p = {
            bw, bw, (int)input.d1, (int)input.d2, (int)input.d3, (int)ci,
            (int)fh, (int)fw, (int)co, (int)padding, (int)padding, (int)padding, (int)padding,
            (int)stride, (int)stride, 0, 0, 0, 0, 0};
        fillConv2DParams(&(k.p));
        k.mem_size_I = k.p.size_I * sizeof(T);
        k.mem_size_F = k.p.size_F * sizeof(T);
        k.mem_size_O = k.p.size_O * sizeof(T);
        output.d_data = gpuKeygenConv2D<T>(&keyBuf, party, k, input.d_data, filter.data, true);
    }

    // ── 3-arg matmul: secret×secret (FIXED from upstream bug) ──────────────
    // Upstream Orca/SIGMA incorrectly pass b.data (host ptr) to gpuKeygenMatmul,
    // treating the 2nd operand as public. For GCN's A_hat @ X (both secret),
    // we must pass b.d_data (GPU ptr) with wIsOnGpu=true.
    void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c)
    {
        MatmulParams p;
        p.M = a.d1;
        p.K = a.d2;
        p.N = b.d2;
        p.batchSz = 1;
        stdInit(p, bw, 0);  // Truncation by _MatMul's doTruncationForward=true node
        // Pass b.d_data (GPU ptr) with wIsOnGpu=true instead of b.data (host ptr)
        c.d_data = gpuKeygenMatmul<T>(&keyBuf, party, p, a.d_data, b.d_data, (T *)NULL, TruncateType::None, &g, true);
        { cudaError_t e = cudaDeviceSynchronize();
          if (e != cudaSuccess) {
              fprintf(stderr, "  ERROR: CUDA error after gpuKeygenMatmul: %s\n", cudaGetErrorString(e));
              fflush(stderr);
          }
        }
    }

    // ── 4-arg matmul: secret×public (FC layers, with M=1 padding fix) ──────
    void matmul(const Tensor2D<T> &a, const Tensor2D<T> &b, Tensor2D<T> &c, bool useBias, Tensor1D<T> &d, bool isFirst)
    {
        // For PUBLIC weights W (both parties load identical model files):
        // Weight mask r_W = 0, so gpuKeygenMatmul receives a zero buffer as h_mask_W.
        // This makes k.B = share(0) and output mask = r_A · W + r_Z (no r_W term).
        // Eval then reconstructs W - 0 = W, producing correct (A + r_A) · W output.
        u64 weight_size = (u64)b.d1 * b.d2;
        T *h_mask_W_zero = (T *)calloc(weight_size, sizeof(T));  // zero mask for public W

        // CUTLASS fails with M=1 for integer GEMMs ("Error Internal" at gpu_matmul.cu:119).
        // Pad M=1 to M=128 (typical CUTLASS SIMT tile size).
        // Compute on padded matrix, extract row 0.
        if (a.d1 == 1) {
            const int M_PADDED = 128;
            const u64 padded_size_A = M_PADDED * a.d2;
            T *d_A_padded = (T *)gpuMalloc(padded_size_A * sizeof(T));
            // Copy row 0; zero-fill rows 1..127
            checkCudaErrors(cudaMemcpy(d_A_padded, a.d_data, a.d2 * sizeof(T), cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemset(d_A_padded + a.d2, 0, (M_PADDED - 1) * a.d2 * sizeof(T)));

            // Clear any sticky CUDA errors and force device sync before CUTLASS
            cudaGetLastError();  // consume any prior error
            checkCudaErrors(cudaDeviceSynchronize());

            MatmulParams p;
            p.M = M_PADDED;
            p.K = a.d2;
            p.N = b.d2;
            p.batchSz = 1;
            stdInit(p, bw, 0);  // FC truncation handled by separate truncateForward node
            T *d_C_padded = gpuKeygenMatmul<T>(&keyBuf, party, p, d_A_padded, h_mask_W_zero, (T *)NULL, TruncateType::None, &g, false);

            // Check for errors immediately after matmul (sticky CUDA errors propagate)
            cudaError_t err_post_matmul = cudaDeviceSynchronize();
            if (err_post_matmul != cudaSuccess) {
                fprintf(stderr, "  ERROR: CUDA error after gpuKeygenMatmul: %s\n", cudaGetErrorString(err_post_matmul));
                fflush(stderr);
            }

            // Extract row 0 from padded output (size: M_PADDED x N -> 1 x N)
            c.d_data = (T *)gpuMalloc(b.d2 * sizeof(T));
            checkCudaErrors(cudaMemcpy(c.d_data, d_C_padded, b.d2 * sizeof(T), cudaMemcpyDeviceToDevice));

            gpuFree(d_A_padded);
            gpuFree(d_C_padded);
        } else {
            MatmulParams p;
            p.M = a.d1;
            p.K = a.d2;
            p.N = b.d2;
            p.batchSz = 1;
            stdInit(p, bw, 0);  // FC truncation handled by separate truncateForward node
            c.d_data = gpuKeygenMatmul<T>(&keyBuf, party, p, a.d_data, h_mask_W_zero, (T *)NULL, TruncateType::None, &g, false);
        }

        free(h_mask_W_zero);  // clean up zero mask buffer
    }

    // ── ADDED: element-wise secret×secret multiply (keygen) ─────────────────
    virtual void mul(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out)
    {
        // Override in derived class (DDGOrcaKeygen)
        throw std::runtime_error("mul() not implemented in DDGOrcaBaseKeygen");
    }

    // ── ADDED: public-scalar multiply (keygen) ──────────────────────────────
    virtual void scalarmul(Tensor<T> &x, T scalar, Tensor<T> &y)
    {
        // Override in derived class (DDGOrcaKeygen)
        throw std::runtime_error("scalarmul() not implemented in DDGOrcaBaseKeygen");
    }

    void avgPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, u64 scale)
    {
        AvgPoolParams p = {
            bw, bw, (int)scale, (int)scale, 0, (int)in.d1, (int)in.d2, (int)in.d3, (int)in.d4,
            (int)ks, (int)ks, (int)stride, (int)stride, (int)padding, (int)padding, (int)padding, (int)padding, 0, 0, false};
        initPoolParams(p);
        out.d_data = gpuAddPool(p, in.d_data, (Stats *)NULL);
    }

    void add(const std::vector<Tensor<T> *> &in, Tensor<T> &out)
    {
        int tmpBw = this->bw;  // Fixed-point add preserves scale: use full ring
        int N = in[0]->size();
        std::vector<T *> gpuInp;
        for (int i = 0; i < in.size(); i++)
        {
            gpuInp.push_back(in[i]->d_data);
        }
        out.d_data = gpuAdd(tmpBw, N, gpuInp);
    }

    void addbias(Tensor<T> &x, const Tensor1D<T> &bias)
    {
        gpuAddBias(1, x.size() / bias.d1, bias.d1, bw, x.d_data, bias.data, NULL);
    }

    void output(Tensor<T> &a)
    {
        int N = a.size();
        size_t memSz = N * sizeof(T);
        moveIntoCPUMem((u8 *)keyBuf, (u8 *)a.d_data, memSz, (Stats *)NULL);
        keyBuf += memSz;
    }

    void optimize(LayerGraphNode<T> *root)
    {
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { ddgOrcaOpt<T>(n, r); });
        topologicalApply(root, [&](LayerGraphNode<T> *n, LayerGraphNode<T> *r)
                         { pinCpuMem(n, r); });
    }
};
