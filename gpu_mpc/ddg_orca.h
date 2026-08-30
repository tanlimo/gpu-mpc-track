// Forked from EzPC/GPU-MPC backend/orca.h (commit 66d9cddc, Aug 2024)
// Extended for DeepDTAGen graph model: implements mul()/scalarmul() using
// Beaver triples (fss/gpu_mul.h) and public-scalar linear combination.
//
// Original Authors: Neha Jawalkar
// Copyright (c) 2024 Microsoft Research
// Licensed under MIT (see original file header for full text).
//
// Modifications Copyright (c) 2026 iDASH Track 3 submission team.
//
// MODIFIED: Replaced old DCF (fss/dcf/*) with DPF-based DCF (Algorithm 1 from FSS-DT.pdf)
// via dpf_dcf_adapter.h. The new implementation uses DPF tree traversal for comparison.

#pragma once

#include "utils/gpu_random.h"
#include "utils/gpu_mem.h"

#include "ddg_orca_base.h"

// CHANGED: Use DPF-based DCF adapter instead of old DCF
#include "dpf_dcf_adapter.h"
// Sigma native DPF-based ReLU + deterministic TrFloor truncation / sign-extension.
// These replace the hand-written stochastic-truncation + signext in dpf_dcf_adapter
// (whose masks failed to cancel → non-deterministic FC outputs). Keygen and eval
// both use the identical Sigma protocol so keys align.
#include "fss/gpu_relu.h"
#include "fss/gpu_truncate.h"
#include "fss/gpu_mul.h"
#include "fss/gpu_scalarmul.h"

template <typename T>
class DDGOrca : public DDGOrcaBase<T>
{
public:
    DDGOrca() : DDGOrcaBase<T>() {}

    DDGOrca(int party, std::string ip, int bw, int scale, std::string keyFile = "") : DDGOrcaBase<T>(party, ip, bw, scale, keyFile, false)
    {
    }

    // ── ADDED: element-wise secret×secret multiply (online eval) ────────────
// CURRENT DEEPDTAGEN CONTRACT:
// This _Mul is used only for H(scale=s) * nodeMask(scale=0, values {0,1}).
// Therefore the output remains at scale=s and no fixed-point truncation is
// required. If generic scale-s * scale-s elementwise multiplication is added
// later, it must use a separate truncating path.
    void mul(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out) override
    {
        int N = (int)a.size();
        assert(a.is_same_shape(b) && "mul: shape mismatch");
        assert(a.is_same_shape(out) && "mul: output shape mismatch");

        // Read Beaver-triple key written by DDGOrcaKeygen::mul
        auto k = readGPUMulKey<T>(&this->keyBuf, (u64)N, (u64)N, (u64)N, TruncateType::None);

        // Reveal leaf operands (e.g. maskTiled in maxpool) to masked-public.
        auto d_mask_a = (T *)moveToGPU((u8 *)k.a, N * sizeof(T), &this->s);
        auto d_mask_b = (T *)moveToGPU((u8 *)k.b, N * sizeof(T), &this->s);
        this->revealIfLeaf(a.d_data, d_mask_a, this->bw, N);
        this->revealIfLeaf(b.d_data, d_mask_b, this->bw, N);
        gpuFree(d_mask_a);
        gpuFree(d_mask_b);

        out.d_data = gpuMul(this->peer, this->party, this->bw, this->scale, N, k,
                            a.d_data, b.d_data, TruncateType::None,
                            &this->g, &this->s);
    }

    // ── ADDED: public-scalar multiply (online eval) ─────────────────────────
    void scalarmul(Tensor<T> &x, T scalar, Tensor<T> &y) override
    {
        int N = (int)x.size();
        y.d_data = (T *)gpuMalloc(N * sizeof(T));
        gpuLinearComb(this->bw, N, y.d_data, scalar, x.d_data);
    }

    void sub(Tensor<T> &b, Tensor<T> &a, Tensor<T> &out)
    {
        // Pure ring subtraction: out = b - a (mod 2^tmpBw), no truncation, no comm.
        // tmpBw = bw - scale matches Add's modulus (Sigma mode-0 full-32: 32-12=20).
        int tmpBw = this->bw - this->scale;
        int N = (int)b.size();
        out.d_data = (T *)gpuMalloc(N * sizeof(T));
        gpuLinearComb(tmpBw, N, out.d_data, (T)1, b.d_data, (T)(-1), a.d_data);
    }

    void relu(Tensor<T> &in, Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode)
    {
        if (mode == 2)
        {
            auto start = std::chrono::high_resolution_clock::now();

            auto k = dpf_dcf::readGPUReluExtendKey<T>(&(this->keyBuf));
            auto d_temp = dpf_dcf::gpuReluExtend(this->peer, this->party, k, in.d_data, &(this->g), &(this->s));
            auto d_drelu = d_temp.first;
            gpuFree(d_drelu);
            out.d_data = d_temp.second;
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = end - start;
            this->s.reluext_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        }
        else
        {
            auto start = std::chrono::high_resolution_clock::now();

            // Sigma native DPF TwoRoundRelu (returns single relu-share pointer).
            auto k = readReluKey<T>(&(this->keyBuf));
            out.d_data = gpuRelu<T, T, 0, 0, false>(this->peer, this->party, k, in.d_data, &(this->g), &(this->s));
            auto end = std::chrono::high_resolution_clock::now();
            auto elapsed = end - start;
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
            this->s.relu_time += us;
        }
    }

    void truncateForward(Tensor<T> &in, u64 shift, u8 mode = 0)
    {
        // Sigma native deterministic TrFloor (mode ignored — always deterministic).
        auto start = std::chrono::high_resolution_clock::now();
        auto k = readGPUTruncateKey<T>(TruncateType::TrFloor, &(this->keyBuf));
        in.d_data = gpuTruncate<T, T>(this->bw, this->bw, TruncateType::TrFloor, k, (int)shift,
                                       this->peer, this->party, in.size(), in.d_data, &(this->g), &(this->s));
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        this->s.truncate_time += us;
    }

    void signext(Tensor<T> &x, u64 scale)
    {
        auto start = std::chrono::high_resolution_clock::now();
        // genGPUSignExtendKey writes a 3-int header (bin,bout,N) before the TrCorrKey.
        int bin = *((int *)this->keyBuf); this->keyBuf += sizeof(int);
        int bout = *((int *)this->keyBuf); this->keyBuf += sizeof(int);
        int N = *((int *)this->keyBuf); this->keyBuf += sizeof(int);
        assert(bin == this->bw - (int)scale && bout == this->bw && N == (int)x.size());
        auto k = readGPUTrCorrKey<T>(&(this->keyBuf));
        x.d_data = gpuSignExtend<T, T>(this->party, this->peer, bin, bout, x.size(), k, x.d_data, &(this->g), &(this->s));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        this->s.signext_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    }

    void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale, u8 mode)
    {
        auto start = std::chrono::high_resolution_clock::now();

        assert(in.d1 == out.d1);
        assert(in.d4 == out.d4);
        int tmpBw = this->bw;
        if (mode == 3)
            tmpBw -= scale;
        MaxpoolParams p = {
            tmpBw, tmpBw, 0, 0, this->bw,
            (int)in.d1, (int)in.d2, (int)in.d3, (int)in.d4,
            (int)ks, (int)ks,
            (int)stride, (int)stride,
            (int)padding, (int)padding,
            (int)padding, (int)padding,
            0, 0, false};
        initPoolParams(p);
        auto k = dpf_dcf::readGPUMaxpoolKey<T>(p, &(this->keyBuf));
        out.d_data = dpf_dcf::gpuMaxPool(this->peer, this->party, p, k, in.d_data, (u32 *)NULL, &(this->g), &(this->s));

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;
        this->s.maxpool_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    }
};

template <typename T>
class DDGOrcaKeygen : public DDGOrcaBaseKeygen<T>
{
public:
    DDGOrcaKeygen(int party, int bw, int scale, std::string keyFile) : DDGOrcaBaseKeygen<T>(party, bw, scale, keyFile)
    {
    }

    // ── ADDED: element-wise secret×secret multiply (keygen) ─────────────────
// Mirrors eval: node-mask multiplication requires TruncateType::None.
    void mul(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &out) override
    {
        int N = (int)a.size();
        assert(a.is_same_shape(b) && "mul keygen: shape mismatch");
        assert(a.is_same_shape(out) && "mul keygen: output shape mismatch");

        // Generate Beaver-triple key: A, B, C where C = A*B + random mask
        out.d_data = gpuKeygenMul<T>(
            &this->keyBuf,
            this->party,
            this->bw,
            this->scale,
            N,
            a.d_data,
            b.d_data,
            TruncateType::None,
            &this->g);
    }

    // ── ADDED: public-scalar multiply (keygen) ──────────────────────────────
    void scalarmul(Tensor<T> &x, T scalar, Tensor<T> &y) override
    {
        int N = (int)x.size();
        // Public-scalar multiply: no FSS key needed, just linear combination
        y.d_data = (T *)gpuMalloc(N * sizeof(T));
        gpuLinearComb(this->bw, N, y.d_data, scalar, x.d_data);
    }

    // ── ADDED: pure ring subtraction b - a (keygen) ──────────────────────────
    void sub(Tensor<T> &b, Tensor<T> &a, Tensor<T> &out)
    {
        // No FSS key needed — local linear combination. Mirrors eval exactly.
        // tmpBw = bw - scale matches Add's modulus semantics.
        int tmpBw = this->bw - this->scale;
        int N = (int)b.size();
        out.d_data = (T *)gpuMalloc(N * sizeof(T));
        gpuLinearComb(tmpBw, N, out.d_data, (T)1, b.d_data, (T)(-1), a.d_data);
    }

    void relu(Tensor<T> &in, Tensor<T> &out, const Tensor<T> &drelu, u64 scale, int mode)
    {
        assert(in.is_same_shape(out));
        assert(in.is_same_shape(drelu));
        if (mode == 2)
        {
            auto d_tempMask = dpf_dcf::gpuKeygenReluExtend<T>(&(this->keyBuf), this->party, this->bw - scale, this->bw, in.size(), in.d_data, &(this->g));
            auto d_dreluMask = d_tempMask.first;
            if (d_dreluMask) gpuFree(d_dreluMask);
            auto d_reluMask = d_tempMask.second;
            out.d_data = d_reluMask;
        }
        else
        {
            int tmpBw = this->bw;
            if (mode == 3)
                tmpBw -= scale;
            out.d_data = gpuGenReluKey<T, T, 0, 0, false>(&(this->keyBuf), this->party, tmpBw, tmpBw, in.size(), in.d_data, &(this->g));
        }
    }

    void truncateForward(Tensor<T> &in, u64 shift, u8 mode = 0)
    {
        // Sigma native deterministic TrFloor keygen (mode ignored).
        in.d_data = genGPUTruncateKey<T, T>(&(this->keyBuf), this->party, TruncateType::TrFloor,
                                             this->bw, this->bw, (int)shift, in.size(), in.d_data, &(this->g));
    }

    void signext(Tensor<T> &x, u64 scale)
    {
        int bin = this->bw - scale;
        int bout = this->bw;
        // Sigma native genGPUSignExtendKey writes bin,bout,N + TrCorrKey.
        x.d_data = genGPUSignExtendKey<T, T>(&(this->keyBuf), this->party, bin, bout, x.size(), x.d_data, &(this->g));
    }

    void maxPool2D(u64 ks, u64 padding, u64 stride, const Tensor4D<T> &in, Tensor4D<T> &out, Tensor4D<u64> &maxIdx, u64 scale, u8 mode)
    {
        int tmpBw = this->bw;
        if (mode == 3)
            tmpBw -= scale;
        MaxpoolParams p = {
            tmpBw, tmpBw, 0, 0, this->bw,
            (int)in.d1, (int)in.d2, (int)in.d3, (int)in.d4,
            (int)ks, (int)ks,
            (int)stride, (int)stride,
            (int)padding, (int)padding,
            (int)padding, (int)padding,
            0, 0, false};
        initPoolParams(p);
        out.d_data = dpf_dcf::gpuKeygenMaxpool(&(this->keyBuf), this->party, p, in.d_data, (u8 *)NULL, &(this->g));
    }
};
