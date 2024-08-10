#include "gmxpre.h"

#include "gromacs/gpu_utils/cuda_arch_utils.cuh"
#include "gromacs/gpu_utils/cuda_kernel_utils.cuh"
#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/gpu_utils.h"
#include "gromacs/gpu_utils/typecasts.cuh"
#include "gromacs/gpu_utils/vectype_ops.cuh"
#include "gromacs/hardware/device_information.h"
#include "gromacs/math/units.h"
#include "gromacs/math/utilities.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/nbnxm/gpu_common.h"
#include "gromacs/nbnxm/gpu_common_utils.h"
#include "gromacs/nbnxm/gpu_data_mgmt.h"
#include "gromacs/nbnxm/gpu_types_common.h"
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/nbnxm_gpu.h"
#include "gromacs/nbnxm/pairlist.h"
#include "gromacs/pbcutil/ishift.h"

#include "nbnxm_cuda.h"
#include "nbnxm_cuda_kernel_utils.cuh"
#include "nbnxm_cuda_types.h"

#define NTHREAD_Z (1)
#define MIN_BLOCKS_PER_MP (16)
#define THREADS_PER_BLOCK (c_clSize * c_clSize * NTHREAD_Z)

__launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP) __global__ void nbnxn_F_cuda_test_kernel
                (NBAtomDataGpu atdat, NBParamGpu nbparam, Nbnxm::gpu_plist plist, bool bCalcFshift)

{
    /* convenience variables */
    /* the sorted list has been generated using data from a previous call to this kernel */
    const nbnxn_sci_t* pl_sci = plist.sorting.sciSorted;
    const
    nbnxn_cj_packed_t*  pl_cjPacked = plist.cjPacked;
    const nbnxn_excl_t* excl        = plist.excl;
    const int* atom_types = atdat.atomTypes;
    int        ntypes     = atdat.numTypes;
    const float4* xq          = atdat.xq;
    float3*       f           = asFloat3(atdat.f);
    const float3* shift_vec   = asFloat3(atdat.shiftVec);
    float         rcoulomb_sq = nbparam.rcoulomb_sq;
    float beta2 = nbparam.ewald_beta * nbparam.ewald_beta;
    float beta3 = nbparam.ewald_beta * nbparam.ewald_beta * nbparam.ewald_beta;

    /* thread/block/warp id-s */
    unsigned int tidxi = threadIdx.x;
    unsigned int tidxj = threadIdx.y;
    unsigned int tidx  = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int tidxz = 0;
    unsigned int bidx = blockIdx.x;
    unsigned int widx = tidx / warp_size; /* warp index */
    int sci, ci, cj, ai, aj, cijPackedBegin, cijPackedEnd;
    int typei, typej;
    int   i, jm, jPacked, wexcl_idx;
    float qi, qj_f, r2, inv_r, inv_r2;
    float inv_r6, c6, c12;
    float int_bit, F_invr;
    unsigned int wexcl, imask, mask_ji;
    float4       xqbuf;
    float3       xi, xj, rv, f_ij, fcj_buf;
    float3       fci_buf[c_nbnxnGpuNumClusterPerSupercluster]; /* i force buffer */
    nbnxn_sci_t  nb_sci;

    /*! i-cluster interaction mask for a super-cluster with all c_nbnxnGpuNumClusterPerSupercluster=8 bits set */
    const unsigned superClInteractionMask = ((1U << c_nbnxnGpuNumClusterPerSupercluster) - 1U);

    // cj preload is off in the following cases:
    // - sm_70 (V100), sm_80 (A100), sm_86 (GA02)
    // - for future arch (> 8.6 at the time of writing) we assume it is better to keep it off
    // cj preload is left on for:
    // - sm_75: improvements +/- very small
    // - sm_61: tested and slower without preload
    // - sm_6x and earlier not tested to
    constexpr bool c_preloadCj = (GMX_PTX_ARCH < 700 || GMX_PTX_ARCH == 750);


    // Full or partial unroll on Ampere (and later) GPUs is beneficial given the increased L1
    // instruction cache. Tested with CUDA 11-12.
    static constexpr int jmLoopUnrollFactor = 4;
    /*********************************************************************
     * Set up shared memory pointers.
     * sm_nextSlotPtr should always be updated to point to the "next slot",
     * that is past the last point where data has been stored.
     */
    // NOLINTNEXTLINE(readability-redundant-declaration)
    extern __shared__ char sm_dynamicShmem[];
    char*                  sm_nextSlotPtr = sm_dynamicShmem;
    static_assert(sizeof(char) == 1,
                  "The shared memory offset calculation assumes that char is 1 byte");

    /* shmem buffer for i x+q pre-loading */
    float4* xqib = reinterpret_cast<float4*>(sm_nextSlotPtr);
    sm_nextSlotPtr += (c_nbnxnGpuNumClusterPerSupercluster * c_clSize * sizeof(*xqib));

    /* shmem buffer for cj, for each warp separately */
    int* cjs = reinterpret_cast<int*>(sm_nextSlotPtr);
    if (c_preloadCj)
    {
        /* the cjs buffer's use expects a base pointer offset for pairs of warps in the j-concurrent execution */
        cjs += tidxz * c_nbnxnGpuClusterpairSplit * c_nbnxnGpuJgroupSize;
        sm_nextSlotPtr += (NTHREAD_Z * c_nbnxnGpuClusterpairSplit * c_nbnxnGpuJgroupSize * sizeof(*cjs));
    }

    /* shmem buffer for i atom-type pre-loading */
    int* atib = reinterpret_cast<int*>(sm_nextSlotPtr);
    sm_nextSlotPtr += (c_nbnxnGpuNumClusterPerSupercluster * c_clSize * sizeof(*atib));
    /*********************************************************************/

    nb_sci         = pl_sci[bidx];         /* my i super-cluster's index = current bidx */
    sci            = nb_sci.sci;           /* super-cluster */
    cijPackedBegin = nb_sci.cjPackedBegin; /* first ...*/
    cijPackedEnd   = nb_sci.cjPackedEnd;   /* and last index of j clusters */

    // We may need only a subset of threads active for preloading i-atoms
    // depending on the super-cluster and cluster / thread-block size.
    constexpr bool c_loadUsingAllXYThreads = (c_clSize == c_nbnxnGpuNumClusterPerSupercluster);
    if (tidxz == 0 && (c_loadUsingAllXYThreads || tidxj < c_nbnxnGpuNumClusterPerSupercluster))
    {
        /* Pre-load i-atom x and q into shared memory */
        ci = sci * c_nbnxnGpuNumClusterPerSupercluster + tidxj;
        ai = ci * c_clSize + tidxi;

        const float* shiftptr = reinterpret_cast<const float*>(&shift_vec[nb_sci.shift]);
        xqbuf = xq[ai] + make_float4(LDG(shiftptr), LDG(shiftptr + 1), LDG(shiftptr + 2), 0.0F);
        xqbuf.w *= nbparam.epsfac;
        xqib[tidxj * c_clSize + tidxi] = xqbuf;

        /* Pre-load the i-atom types into shared memory */
        atib[tidxj * c_clSize + tidxi] = atom_types[ai];
    }

    __syncthreads();

    for (i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i++)
    {
        fci_buf[i] = make_float3(0.0F);
    }



    const int nonSelfInteraction = !(nb_sci.shift == gmx::c_centralShiftIndex & tidxj <= tidxi);

    /* loop over the j clusters = seen by any of the atoms in the current super-cluster;
     * The loop stride NTHREAD_Z ensures that consecutive warps-pairs are assigned
     * consecutive jPacked's entries.
     */
    for (jPacked = cijPackedBegin + tidxz; jPacked < cijPackedEnd; jPacked += NTHREAD_Z)
    {
        wexcl_idx = pl_cjPacked[jPacked].imei[widx].excl_ind;
        imask     = pl_cjPacked[jPacked].imei[widx].imask;
        wexcl     = excl[wexcl_idx].pair[(tidx) & (warp_size - 1)];

        if (imask)
        {
            if (c_preloadCj)
            {
                /* Pre-load cj into shared memory on both warps separately */
                if ((tidxj == 0 | tidxj == 4) & (tidxi < c_nbnxnGpuJgroupSize))
                {
                    cjs[tidxi + tidxj * c_nbnxnGpuJgroupSize / c_splitClSize] =
                            pl_cjPacked[jPacked].cj[tidxi];
                }
                __syncwarp(c_fullWarpMask);
            }

#    pragma unroll jmLoopUnrollFactor
            for (jm = 0; jm < c_nbnxnGpuJgroupSize; jm++)
            {
                if (imask & (superClInteractionMask << (jm * c_nbnxnGpuNumClusterPerSupercluster)))
                {
                    mask_ji = (1U << (jm * c_nbnxnGpuNumClusterPerSupercluster));

                    cj = c_preloadCj ? cjs[jm + (tidxj & 4) * c_nbnxnGpuJgroupSize / c_splitClSize]
                                     : cj = pl_cjPacked[jPacked].cj[jm];

                    aj = cj * c_clSize + tidxj;

                    /* load j atom data */
                    xqbuf = xq[aj];
                    xj    = make_float3(xqbuf.x, xqbuf.y, xqbuf.z);
                    qj_f  = xqbuf.w;
                    typej = atom_types[aj];

                    fcj_buf = make_float3(0.0F);

#    pragma unroll c_nbnxnGpuNumClusterPerSupercluster
                    for (i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i++)
                    {
                        if (imask & mask_ji)
                        {
                            ci = sci * c_nbnxnGpuNumClusterPerSupercluster + i; /* i cluster index */

                            /* all threads load an atom from i cluster ci into shmem! */
                            xqbuf = xqib[i * c_clSize + tidxi];
                            xi    = make_float3(xqbuf.x, xqbuf.y, xqbuf.z);

                            /* distance between i and j atoms */
                            rv = xi - xj;
                            r2 = norm2(rv);

                            int_bit = (wexcl & mask_ji) ? 1.0F : 0.0F;

                            /* cutoff & exclusion check */
                            if ((r2 < rcoulomb_sq) * (nonSelfInteraction | (ci != cj)))
                            {
                                /* load the rest of the i-atom parameters */
                                qi = xqbuf.w;

                                /* LJ 6*C6 and 12*C12 */
                                typei = atib[i * c_clSize + tidxi];
                                fetch_nbfp_c6_c12(c6, c12, nbparam, ntypes * typei + typej);

                                // Ensure distance do not become so small that r^-12 overflows
                                r2 = max(r2, c_nbnxnMinDistanceSquared);

                                inv_r  = rsqrt(r2);
                                inv_r2 = inv_r * inv_r;
                                inv_r6 = inv_r2 * inv_r2 * inv_r2;
                                /* We could mask inv_r2, but with Ewald
                                 * masking both inv_r6 and F_invr is faster */
                                inv_r6 *= int_bit;

                                F_invr = inv_r6 * (c12 * inv_r6 - c6) * inv_r2;

                                calculate_force_switch_F(nbparam, c6, c12, inv_r, r2, &F_invr);


                                F_invr += qi * qj_f
                                          * (int_bit * inv_r2 * inv_r + pmecorrF(beta2 * r2) * beta3);

                                f_ij = rv * F_invr;

                                /* accumulate j forces in registers */
                                fcj_buf -= f_ij;

                                /* accumulate i forces in registers */
                                fci_buf[i] += f_ij;
                            }
                        }

                        /* shift the mask bit by 1 */
                        mask_ji += mask_ji;
                    }

                    /* reduce j forces */
                    reduce_force_j_warp_shfl(fcj_buf, f, tidxi, aj, c_fullWarpMask);
                }
            }
        }
        if (c_preloadCj)
        {
            // avoid shared memory WAR hazards on sm_cjs between loop iterations
            __syncwarp(c_fullWarpMask);
        }
    }

    /* skip central shifts when summing shift forces */
    if (nb_sci.shift == gmx::c_centralShiftIndex)
    {
        bCalcFshift = false;
    }

    float fshift_buf = 0.0F;

    /* reduce i forces */
    for (i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i++)
    {
        ai = (sci * c_nbnxnGpuNumClusterPerSupercluster + i) * c_clSize + tidxi;
        reduce_force_i_warp_shfl(fci_buf[i], f, &fshift_buf, bCalcFshift, tidxj, ai, c_fullWarpMask);
    }

    /* add up local shift forces into global mem, tidxj indexes x,y,z */
    if (bCalcFshift && (tidxj & 3) < 3)
    {
        float3* fShift = asFloat3(atdat.fShift);
        atomicAdd(&(fShift[nb_sci.shift].x) + (tidxj & 3), fshift_buf);
    }
}
