#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "utils/gpu_data_types.h"
#include "utils/gpu_mem.h"
#include "utils/gpu_comms.h"

#include "fss/gpu_lut.h"
#include "fss/gpu_mul.h"


// ============================================================================
// DeepDTAGen secure adjacency normalization
//
// Input:
//   A_raw : secret additive shares of dense 0/1 adjacency,
//           INCLUDING real-node self loops.
//           Fixed-point scale = 0.
//
// Output:
//   A_norm = D^{-1/2} A_raw D^{-1/2}
//           masked-public under the Orca/SIGMA preprocessing mask.
//           Fixed-point scale = `scale` (normally Q12).
//
// Security:
//   degree is NEVER reconstructed in plaintext.
//   Evaluators reconstruct only:
//
//       degree + r_degree
//
//   before performing the DPF-based LUT.
//
// Degree range:
//   DeepDTAGen Nmax=138, therefore degree <= 138.
//   We use an 8-bit LUT domain [0,255].
// ============================================================================


template <typename T>
__global__ void ddgAdjRowSumKernel(
    const T *A,
    T *degree,
    int batch,
    int nmax,
    int bw
)
{
    const u64 row =
        static_cast<u64>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    const u64 rows =
        static_cast<u64>(batch) *
        static_cast<u64>(nmax);

    if (row >= rows)
        return;

    T acc = 0;

    const u64 base =
        row * static_cast<u64>(nmax);

    for (int j = 0; j < nmax; ++j) {
        acc += A[
            base + static_cast<u64>(j)
        ];
    }

    gpuMod(acc, bw);
    degree[row] = acc;
}


template <typename T>
__global__ void ddgTileInvDegreeKernel(
    const T *invDegree,
    T *rowInv,
    T *colInv,
    int batch,
    int nmax
)
{
    const u64 idx =
        static_cast<u64>(blockIdx.x) *
        blockDim.x +
        threadIdx.x;

    const u64 graphElems =
        static_cast<u64>(nmax) *
        static_cast<u64>(nmax);

    const u64 total =
        static_cast<u64>(batch) *
        graphElems;

    if (idx >= total)
        return;

    const u64 b = idx / graphElems;
    const u64 rem = idx % graphElems;

    const u64 i =
        rem / static_cast<u64>(nmax);

    const u64 j =
        rem % static_cast<u64>(nmax);

    rowInv[idx] =
        invDegree[
            b * static_cast<u64>(nmax) + i
        ];

    colInv[idx] =
        invDegree[
            b * static_cast<u64>(nmax) + j
        ];
}


// Public 256-entry table:
//
//   table[0] = 0
//
//   table[d] = round(
//       2^scale / sqrt(d)
//   )
//
// for 1 <= d <= nmax.
//
// Values outside the valid degree domain are zero.
template <typename T>
static T *ddgMakeDegreeInvSqrtTable(
    int scale,
    int nmax,
    Stats *s
)
{
    if (nmax > 255) {
        fprintf(
            stderr,
            "[DDG_ADJNORM] nmax=%d exceeds "
            "8-bit degree LUT domain\n",
            nmax
        );
        exit(1);
    }

    std::vector<T> table(
        256,
        static_cast<T>(0)
    );

    const double q =
        std::ldexp(1.0, scale);

    for (int d = 1; d <= nmax; ++d) {
        table[d] =
            static_cast<T>(
                std::llround(
                    q /
                    std::sqrt(
                        static_cast<double>(d)
                    )
                )
            );
    }

    return reinterpret_cast<T *>(
        moveToGPU(
            reinterpret_cast<u8 *>(
                table.data()
            ),
            table.size() * sizeof(T),
            s
        )
    );
}


// ============================================================================
// Dealer / key-generation side
// ============================================================================
//
// d_A_mask is the preprocessing mask corresponding to A_raw.
//
// Key layout emitted into *keyBuf:
//
//   [degree-mask additive share]
//   [DPF-LUT key]
//   [Mul #1 key: A_raw * rowInv]
//   [Mul #2 key: left * colInv, TrFloor(scale)]
//
// Returned pointer is the preprocessing mask of A_norm.
// This becomes the input mask consumed by the existing GCN keygen.
// ============================================================================
template <typename T>
T *ddgSecureAdjNormKeygen(
    u8 **keyBuf,
    int party,
    int bw,
    int scale,
    AESGlobalContext *g,
    T *d_A_mask,
    int batch,
    int nmax
)
{
    const int degreeN =
        batch * nmax;

    const int adjN =
        batch * nmax * nmax;

    // ------------------------------------------------------------------------
    // 1. r_degree = rowSum(r_A)
    // ------------------------------------------------------------------------
    T *d_degree_mask =
        reinterpret_cast<T *>(
            gpuMalloc(
                static_cast<u64>(degreeN) *
                sizeof(T)
            )
        );

    ddgAdjRowSumKernel<T>
        <<<(
            degreeN + 255
        ) / 256, 256>>>(
            d_A_mask,
            d_degree_mask,
            batch,
            nmax,
            bw
        );

    checkCudaErrors(
        cudaDeviceSynchronize()
    );

    // Evaluators need party-specific shares of
    // r_degree so they can reconstruct:
    //
    //     degree + r_degree
    //
    // without revealing degree.
    writeShares<T, T>(
        keyBuf,
        party,
        degreeN,
        d_degree_mask,
        bw
    );

    // ------------------------------------------------------------------------
    // 2. Secure 8-bit LUT:
    //
    //       degree -> round(2^scale/sqrt(degree))
    //
    // gpuKeyGenLUT returns the output mask r_inv.
    // ------------------------------------------------------------------------
    constexpr int DEGREE_BIN = 8;

    T *d_inv_mask =
        gpuKeyGenLUT<T, T>(
            keyBuf,
            party,
            DEGREE_BIN,
            bw,
            degreeN,
            d_degree_mask,
            g
        );

    // ------------------------------------------------------------------------
    // 3. Tile inverse-degree masks to full adjacency shape.
    //
    // rowInv[b,i,j] = inv[b,i]
    // colInv[b,i,j] = inv[b,j]
    // ------------------------------------------------------------------------
    T *d_row_inv_mask =
        reinterpret_cast<T *>(
            gpuMalloc(
                static_cast<u64>(adjN) *
                sizeof(T)
            )
        );

    T *d_col_inv_mask =
        reinterpret_cast<T *>(
            gpuMalloc(
                static_cast<u64>(adjN) *
                sizeof(T)
            )
        );

    ddgTileInvDegreeKernel<T>
        <<<(
            adjN + 255
        ) / 256, 256>>>(
            d_inv_mask,
            d_row_inv_mask,
            d_col_inv_mask,
            batch,
            nmax
        );

    checkCudaErrors(
        cudaDeviceSynchronize()
    );

    // ------------------------------------------------------------------------
    // 4. First multiplication:
    //
    //    A_raw(scale 0) * rowInv(scale s)
    //        -> scale s
    //
    // No truncation required.
    // ------------------------------------------------------------------------
    T *d_left_mask =
        gpuKeygenMul<T>(
            keyBuf,
            party,
            bw,
            scale,
            adjN,
            d_A_mask,
            d_row_inv_mask,
            TruncateType::None,
            g
        );

    // ------------------------------------------------------------------------
    // 5. Second multiplication:
    //
    //    left(scale s) * colInv(scale s)
    //        -> scale 2s
    //
    // Deterministic TrFloor(scale)
    //        -> scale s
    // ------------------------------------------------------------------------
    T *d_norm_mask =
        gpuKeygenMul<T>(
            keyBuf,
            party,
            bw,
            scale,
            adjN,
            d_left_mask,
            d_col_inv_mask,
            TruncateType::TrFloor,
            g
        );

    gpuFree(d_degree_mask);
    gpuFree(d_inv_mask);
    gpuFree(d_row_inv_mask);
    gpuFree(d_col_inv_mask);
    gpuFree(d_left_mask);

    return d_norm_mask;
}


// ============================================================================
// Evaluator side
// ============================================================================
//
// d_A_share initially contains this evaluator's additive share of A_raw.
//
// The function mutates d_A_share when revealing the masked-public A_raw:
//
//     A_raw + r_A
//
// It returns masked-public A_norm:
//
//     A_norm + r_norm
//
// This output must NOT subsequently be registered as an unrevealed secret leaf.
// ============================================================================
template <typename T>
T *ddgSecureAdjNormEval(
    u8 **keyBuf,
    GpuPeer *peer,
    int party,
    int bw,
    int scale,
    AESGlobalContext *g,
    Stats *s,
    T *d_A_share,
    int batch,
    int nmax
)
{
    const int degreeN =
        batch * nmax;

    const int adjN =
        batch * nmax * nmax;

    // ------------------------------------------------------------------------
    // 1. Local linear row sum on this party's A share.
    //
    // No MPC communication is needed:
    //
    //   sum(A0) + sum(A1)
    //       = sum(A)
    // ------------------------------------------------------------------------
    T *d_degree_share =
        reinterpret_cast<T *>(
            gpuMalloc(
                static_cast<u64>(degreeN) *
                sizeof(T)
            )
        );

    ddgAdjRowSumKernel<T>
        <<<(
            degreeN + 255
        ) / 256, 256>>>(
            d_A_share,
            d_degree_share,
            batch,
            nmax,
            bw
        );

    checkCudaErrors(
        cudaDeviceSynchronize()
    );

    // ------------------------------------------------------------------------
    // 2. Read this party's share of r_degree.
    // ------------------------------------------------------------------------
    T *h_degree_mask_share =
        reinterpret_cast<T *>(
            *keyBuf
        );

    *keyBuf +=
        static_cast<u64>(degreeN) *
        sizeof(T);

    T *d_degree_mask_share =
        reinterpret_cast<T *>(
            moveToGPU(
                reinterpret_cast<u8 *>(
                    h_degree_mask_share
                ),
                static_cast<u64>(degreeN) *
                sizeof(T),
                s
            )
        );

    // Locally form:
    //
    //     degree_party + r_degree_party
    //
    gpuLinearComb(
        bw,
        degreeN,
        d_degree_share,
        static_cast<T>(1),
        d_degree_share,
        static_cast<T>(1),
        d_degree_mask_share
    );

    gpuFree(d_degree_mask_share);

    // Reconstruct ONLY masked degree:
    //
    //     degree + r_degree
    //
    peer->reconstructInPlace(
        d_degree_share,
        bw,
        degreeN,
        s
    );

    // ------------------------------------------------------------------------
    // 3. DPF LUT on masked degree.
    // ------------------------------------------------------------------------
    auto lutKey =
        readGPULUTKey<T>(
            keyBuf
        );

    T *d_table =
        ddgMakeDegreeInvSqrtTable<T>(
            scale,
            nmax,
            s
        );

    T *d_inv =
        gpuDpfLUT<T, T>(
            lutKey,
            peer,
            party,
            d_degree_share,
            d_table,
            g,
            s,
            true
        );

    gpuFree(d_degree_share);
    gpuFree(d_table);

    // d_inv is now masked-public Q(scale)
    // inverse-square-root degree.

    // ------------------------------------------------------------------------
    // 4. Tile masked-public inverse degrees.
    // ------------------------------------------------------------------------
    T *d_row_inv =
        reinterpret_cast<T *>(
            gpuMalloc(
                static_cast<u64>(adjN) *
                sizeof(T)
            )
        );

    T *d_col_inv =
        reinterpret_cast<T *>(
            gpuMalloc(
                static_cast<u64>(adjN) *
                sizeof(T)
            )
        );

    ddgTileInvDegreeKernel<T>
        <<<(
            adjN + 255
        ) / 256, 256>>>(
            d_inv,
            d_row_inv,
            d_col_inv,
            batch,
            nmax
        );

    checkCudaErrors(
        cudaDeviceSynchronize()
    );

    // ------------------------------------------------------------------------
    // 5. First secure multiplication.
    //
    // The first Mul key contains the preprocessing mask share
    // corresponding to A_raw. Use it to turn the raw additive
    // A share into masked-public A before gpuMul().
    // ------------------------------------------------------------------------
    auto mul1Key =
        readGPUMulKey<T>(
            keyBuf,
            adjN,
            adjN,
            adjN,
            TruncateType::None
        );

    T *d_A_mask_share =
        reinterpret_cast<T *>(
            moveToGPU(
                reinterpret_cast<u8 *>(
                    mul1Key.a
                ),
                static_cast<u64>(adjN) *
                sizeof(T),
                s
            )
        );

    gpuLinearComb(
        bw,
        adjN,
        d_A_share,
        static_cast<T>(1),
        d_A_share,
        static_cast<T>(1),
        d_A_mask_share
    );

    gpuFree(d_A_mask_share);

    peer->reconstructInPlace(
        d_A_share,
        bw,
        adjN,
        s
    );

    // A_raw(scale0) * rowInv(scale s)
    // -> scale s, no truncation.
    T *d_left =
        gpuMul<T>(
            peer,
            party,
            bw,
            scale,
            adjN,
            mul1Key,
            d_A_share,
            d_row_inv,
            TruncateType::None,
            g,
            s
        );

    // ------------------------------------------------------------------------
    // 6. Second secure multiplication + deterministic Q-scale truncation.
    // ------------------------------------------------------------------------
    auto mul2Key =
        readGPUMulKey<T>(
            keyBuf,
            adjN,
            adjN,
            adjN,
            TruncateType::TrFloor
        );

    T *d_norm =
        gpuMul<T>(
            peer,
            party,
            bw,
            scale,
            adjN,
            mul2Key,
            d_left,
            d_col_inv,
            TruncateType::TrFloor,
            g,
            s
        );

    gpuFree(d_inv);
    gpuFree(d_row_inv);
    gpuFree(d_col_inv);
    gpuFree(d_left);

    return d_norm;
}
