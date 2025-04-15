#include <cooperative_groups.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvtx3/nvToolsExt.h>

#include "all_to_all/internode.h"
#include "core/device_utils.h"
#include "core/nvshmem_utils.h"
#include "core/utils.h"

#include "../../ThunderKittens/include/kittens.cuh"
#include <cooperative_groups.h>

using namespace pplx;
using namespace kittens;
namespace cg = cooperative_groups;

constexpr int WORLD_SIZE = 8;
constexpr int DP_SIZE = 1;
constexpr int NUM_EXPERTS = 256;

constexpr int HIDDEN_DIM_SIZE = 7168;
// Force HIDDEN_DIM_SCALE_SIZE to be multiple of 16
constexpr int HIDDEN_DIM_SCALE_SIZE = 64; // HIDDEN_DIM_SIZE / BLOCK_SIZE => HIDDEN_DIM_SIZE / 128
constexpr int TOKEN_DIM = HIDDEN_DIM_SIZE + HIDDEN_DIM_SCALE_SIZE;

constexpr int NUM_WARPS = 10;

using g_pgl = pgl<gl<int, -1, -1, -1, -1>>;
using g_gl = gl<int, -1, -1, -1, -1>;

using g_buffer = pgl<gl<int, -1, -1, -1, -1>>;
// using s_vec = sv<bf16, NUM_EXPERTS>;

// 736 * 10 = 7360 => Only first TOKEN DIM elements are used though
// using tok_rt = rt<float, 16, 46>;
using tok_rv = rv<int, 736>;
using tok_sv = sv<int, HIDDEN_DIM_SIZE>;

/*
Grid size is 132
*/

struct globals {
    // expertX: numLocalExperts, maxNumTokens * numDPGroups, perTokenBytes
    g_gl expertX; // destination array for token data 
    // g_pgl expertXScale;

    // dpX: numTokens * hiddenDimBytes
    g_gl dpX; // src array for token data to send off 
    // g_pgl dpXScale;

    
    /*
    numTokensBuffer: numDPGroups, numLocalExperts
    */
    g_buffer numTokensBuffer;
    /*
    numRecvBuffer: numDPGroups, numLocalExperts
    */
    g_buffer numRecvBuffer;
    /*
    xBufferOut: 8, 32, 128, TOKEN_DIM
    xBufferOut: numDPGroups, numLocalExperts, maxNumTokens, perTokenBytes
    */
    g_buffer xBufferOut;
    
    
    int32_t *outNumTokensPerExpert; // ensure this is zero'd out before calling t
    // inidices is m * numExpertsPerToken
    // Reasonable to load into a r
    uint32_t *indices;
    uint32_t *numTokensPerDP; // make non pgl

    // uint32_t *sourceExpert;
    // uint32_t *sourceIndex;
    // uint32_t *sourceOffset;
    // uint32_t *sourceGroup;
    
    size_t maxNumTokens;
    size_t numExperts;
    size_t numExpertsPerToken;

    int numTokens; // num tokens cur device needs to send (m from pplx)
    int dev_idx;
};

__global__ void dispatch_kernel(const __grid_constant__ globals g) {
    using everyone = kittens::group<NUM_WARPS>;
    using send_group = kittens::group<NUM_WARPS - 1>;
    using recv_group = kittens::group<NUM_WARPS>;
    const unsigned num_local_experts = NUM_EXPERTS / WORLD_SIZE;

    /*
    Send Phase
    */
    extern __shared__ uint32_t tokenIndex[];
    for (uint32_t i = threadIdx.x; i < g.numExperts; i += blockDim.x) {
      tokenIndex[i] = 0;
    }
    __syncthreads();

    extern __shared__ kittens::alignment_dummy __shm[]; 
    kittens::shared_allocator al((int*)&__shm[0]);
    // s_vec &s_v = al.allocate<s_vec>();
    // everyone::zero(s_v);
    // everyone::sync(0);

    // Do last warp here so can use "groups" for shared memory loads
    if (kittens::warpid() == NUM_WARPS - 1) {
        /*
        There are 256 experts in total, each device is responsible for 32 experts 

        Last warp needs to determine how many tokens each expert is responsible for
            - To do this, each warp across all devices loop through all 256 experts
                - Each device (in this scenario) has 132 blocks

        Starting g_expert_idx is between 0 and 132:
            - Block Idx 0 handles expert 0, 132, 
            ...
            - Block Idx 124 handles expert 124, 256
        */
        for (size_t g_expert_idx = blockIdx.x; g_expert_idx < NUM_EXPERTS; g_expert_idx += gridDim.x) {
            int dst_gpu = g_expert_idx / num_local_experts;
            int dst_expert_idx = g_expert_idx % num_local_experts;

            int count = 0;
            
            /*
            The loop is looping through all tokens for current device and checking if it 
            is equal to the global expert
                - all threads in a block look through all routings given by indices 
                and if it equals the current expert block is handling, increment count
            

            How do we make this kittens? 
            - load all of indices into an rt of equivalent size
            - want to create tile then that is all zero's unless is equal to dst_expert_idx
            - then 
            - some sort of shared buffer 

            how are we going to 
            - Would be nice if we can just reduce add and send... how can we reduce add to correct 
            location 
            - how large of a tile would that be? 

            have tile that represents all experts => JUST 256
            have local register tile of 256 => keep local count, logic, and then atomic add RT to 

            want to use kittens load functions to optimize loads? 

            
            */
            #pragma unroll
            for (int i = laneid(); i < g.numTokens * g.numExpertsPerToken; i += WARP_THREADS) {
                int expert = __ldg(&g.indices[i]);
                if (expert == dst_expert_idx) count++;
            }
            
            count += __shfl_xor_sync(0xffffffff, count, 16);
            count += __shfl_xor_sync(0xffffffff, count, 8);
            count += __shfl_xor_sync(0xffffffff, count, 4);
            count += __shfl_xor_sync(0xffffffff, count, 2);
            count += __shfl_xor_sync(0xffffffff, count, 1);
        
            if (laneid() == 0) {
                // g.numTokensBuffer.mc_vas[g.dev_idx][dst_expert_idx] = count + 1;
                // Could be faster to just store directly instead of multimem here => test difference
                // asm volatile ("{multimem.red.release.sys.global.add.u32 [%0], %1;}" 
                //     :: "l"(g.numTokensBuffer.mc_vas[g.dev_idx][dst_expert_idx]), "n"(count + 1) : "memory");
                // unsigned int value = count + 1;
                // asm volatile ("{multimem.red.release.sys.global.add.u32 [%0], %1;}" 
                //     :: "l"((unsigned long long)(g.numTokensBuffer.mc_vas[g.dev_idx][dst_expert_idx])), 
                //       "r"(value) : "memory");
                asm volatile ("{multimem.red.release.sys.global.add.u32 [%0], %1;}" 
                  :: "l"(&(g.numTokensBuffer.mc_vas[g.dev_idx][dst_expert_idx])), 
                     "r"(count + 1) : "memory");
            }
        }
    } else {
        for (int i = 0; i < g.numTokens; ++i) {
            // Each block handles a different token
            // In example, max numTokens is 128 so only 128 / 132 blocks will be active
            if (i % gridDim.x == blockIdx.x) {
                /*
                Here, we want to take data from dpX which is the local token data
                currently on device and place it into xBufferOut 

                Need to send both dpX and dpXScale to the expert

                dpX is hiddenDim size (7168)
                dpXScale is hiddenDimScale (7168 / 128) => 56
                    - dpXScale is hiddenDimScale (7168 / 112) => 64

                Have 10 Warps to load => Just load straight to shared memory

                Use TMA here eventually??
                */
                tok_sv &tok_dst_data = al.allocate<tok_sv>();
                send_group::load(tok_dst_data, g.dpX, {i});
                send_group::sync(0);

                for (int j = 0; j < g.numExpertsPerToken; ++j) {
                    int dst_expert = __ldg(&g.indices[i * g.numExpertsPerToken + j]);
                    int dst_gpu = dst_expert / num_local_experts;
                    int dst_expert_idx = dst_expert % num_local_experts;
                    
                    // Get the number of tokens that have currently been sent to the expert
                    int index = tokenIndex[dst_expert];

                    // Experiment with just using store here??
                    send_group::broadcast(g.xBufferOut, tok_dst_data, g.dev_idx, 
                        {g.dev_idx, dst_expert_idx, index, 0});
                    if (warpid() == 0 && laneid() == 0) {
                        asm volatile ("{multimem.red.release.sys.global.add.u32 [%0], %1;}" 
                            :: "l"(&(g.numRecvBuffer.mc_vas[g.dev_idx][dst_expert_idx])), "n"(1) : "memory");
                    }
                }

                // Replicate token count across all blocks
                if (warpid() == 0 && laneid() < g.numExpertsPerToken) {
                    int dst_expert = __ldg(&g.indices[i * g.numExpertsPerToken + laneid()]);
                    tokenIndex[dst_expert]++;
                }
            }


        }


    }

    /*
    Receive Phase
    */
    /*
    first 32 threads of device stall untill device receives all tokens expected
    */
    for (size_t expert = blockIdx.x * blockDim.x + threadIdx.x;
         expert < num_local_experts; expert += blockDim.x * gridDim.x) {
        int src_local_expert = expert / WORLD_SIZE;
        // Stall until numTokensBuffer is updated 
        while (g.numTokensBuffer[g.dev_idx].raw_ptr[src_local_expert] == 0);
        size_t numTokens = g.numTokensBuffer[g.dev_idx].raw_ptr[src_local_expert] - 1;
        // Stall here until we know that all tokens have been received
        while (g.numRecvBuffer[g.dev_idx].raw_ptr[src_local_expert] < numTokens);
        
        int src_gpu = expert % WORLD_SIZE;
        int slot = src_local_expert * WORLD_SIZE + src_gpu;
        g.numTokensPerDP[slot] = numTokens;
        atomicAdd(&g.outNumTokensPerExpert[src_local_expert], numTokens);
        
        g.numTokensBuffer[g.dev_idx].raw_ptr[src_local_expert] = 0;
        g.numRecvBuffer[g.dev_idx].raw_ptr[src_local_expert] = 0;
    }
    
    cg::this_grid().sync();
    int expert = 0;
    int device = 0;
    int offset = 0;
    int start = 0;
    int max_batch_tokens = num_local_experts * WORLD_SIZE * g.maxNumTokens;

    /*
    Each block handles it's own token
    */
    for (int token = blockIdx.x; token < max_batch_tokens; token += gridDim.x) {
        int j = token - offset;
        while (offset + __ldg(&g.numTokensPerDP[expert * WORLD_SIZE + device]) <= token) {
            // Since current token not in current group, add number of tokens in group to offset
            offset += __ldg(&g.numTokensPerDP[expert * WORLD_SIZE + device]);
            j = token - offset;
            if (++device == WORLD_SIZE) {
                device = 0;
                start = offset;
                if (++expert == num_local_experts) {
                break;
                }
            }
            }
            if (expert >= num_local_experts) {
            break;
            }

            // Copy the token to the output buffer.
            int group = expert * WORLD_SIZE + device;
            int loc = token - start;

            tok_sv &tok_data = al.allocate<tok_sv>();
            
            // recv_group::load(tok_data, g.xBufferOut)
            recv_group::sync(0);
            recv_group::store(g.expertX, tok_data, {expert, loc});
    }
}

void AllToAllInterNode::dispatch(
    const Strided1D<int32_t> &outNumTokensPerExpert,
    const Strided2D<std::byte> &expertX,
    const Strided2D<std::byte> &expertXScale,
    const Strided1D<std::byte> &dpX,
    const Strided1D<std::byte> &dpXScale,
    const Strided2D<uint32_t> &indices,
    unsigned m,
    const unsigned *boundM,
    SplitMode splitMode,
    cudaStream_t stream
) {
    CUCHECK(cuInit(0));
    int device_ids[WORLD_SIZE];
    for (int i = 0; i < WORLD_SIZE; ++i) device_ids[i] = i;
    KittensClub club(device_ids, WORLD_SIZE);

    unsigned long smem_size = kittens::MAX_SHARED_MEMORY - 1024; // MAX_SHARED_MEMORY = 227KB for Hopper
    club.execute([smem_size](int dev_idx) {
        CUDACHECK(cudaFuncSetAttribute(dispatch_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, 
                                        smem_size));
    });

    constexpr unsigned NUM_WARPS = 10;
    const unsigned numBlocks = std::min( // min( max((256 / 10), (128 * 8)), 132) = 132
        std::max(
            ceil_div<unsigned>(numExperts, NUM_WARPS), (unsigned)(maxNumTokens * expertsPerToken)
        ),
        132u
    );
    
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(NUM_WARPS * 32, 1, 1);


    uint32_t **device_num_tokens_ptrs = new uint32_t*[WORLD_SIZE];
    size_t num_tokens_size = WORLD_SIZE * (NUM_EXPERTS / WORLD_SIZE) * sizeof(uint32_t);

    uint32_t **device_num_recv_ptrs = new uint32_t*[WORLD_SIZE];
    size_t num_recv_size = WORLD_SIZE * (NUM_EXPERTS / WORLD_SIZE) * sizeof(uint32_t);

    uint32_t **device_num_x_buffer_ptrs = new uint32_t*[WORLD_SIZE];
    size_t num_x_buffer_size = WORLD_SIZE * (NUM_EXPERTS / WORLD_SIZE) * sizeof(uint32_t);
    for (int dev_idx = 0; dev_idx < WORLD_SIZE; ++dev_idx) {
        pglCudaMalloc<true>(WORLD_SIZE, device_ids, dev_idx, &device_num_tokens_ptrs[dev_idx], num_tokens_size);
        pglCudaMalloc<true>(WORLD_SIZE, device_ids, dev_idx, &device_num_recv_ptrs[dev_idx], num_recv_size);
        pglCudaMalloc<true>(WORLD_SIZE, device_ids, dev_idx, &device_num_x_buffer_ptrs[dev_idx], num_x_buffer_size);
    }

    // This is more than needed => Can optimize
    g_gl dpX_gl{expertX.data, 1, 1, maxNumTokens, hiddenDimBytes}
    g_gl expertX_gl{expertX.data, 1, outNumTokensPerExpert.data, maxNumTokens * WORLD_SIZE, hiddenDimBytes}
    
    globals G(
        outNumTokensPerExpert.data,
        expertX_gl,
        dpX_gl,
        

    );

    club.execute([&](int dev_idx) { // warmup
        dispatch_kernel<<<dimGrid, dimBlock, kittens::MAX_SHARED_MEMORY - 1024>>>(sm, dev_idx);
        CUDACHECK(cudaDeviceSynchronize());
    });

  
}


