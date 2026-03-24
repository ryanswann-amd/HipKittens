// FP32 TT GEMM via fused transpose+NT: C = A^T @ B^T where B is KxN
// Phase 1: Transpose B (KxN → NxK) via 32x32 shared tile
// Phase 2: FP32 NT GEMM on transposed data

#ifndef BLOCK_SIZE_VAL
#define BLOCK_SIZE_VAL 128
#endif
#include "kittens.cuh"
#include <pybind11/pybind11.h>
using namespace kittens;

constexpr int BS=BLOCK_SIZE_VAL, KS=16, NW=8, WS=64, NT_THR=NW*WS, T=16, DK=4, RB=BS/4, KI=KS/DK;
typedef __attribute__((__vector_size__(16))) float f4v;

// ========== Transpose Kernel (float) ==========
constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;

__global__ __launch_bounds__(TILE_DIM * BLOCK_ROWS)
void transpose_f32(float* __restrict__ dst, const float* __restrict__ src, int rows, int cols) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        if ((y+j) < rows && x < cols) tile[threadIdx.y+j][threadIdx.x] = src[(y+j)*cols+x];
    __syncthreads();
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        if ((y+j) < cols && x < rows) dst[(y+j)*rows+x] = tile[threadIdx.x][threadIdx.y+j];
}

// ========== FP32 NT GEMM Kernel (copy from fp32_native_nt_kernel.cpp) ==========
__global__ __launch_bounds__(NT_THR, 4)
void fp32_gemm_nt(float* __restrict__ C, const float* __restrict__ A,
                  const float* __restrict__ B, int M, int N, int K) {
    __shared__ float As[BS][KS], Bs[BS][KS];
    const int nn=N/BS; int wgid=blockIdx.x;
    {int W=4,ch=W*W,nc=(gridDim.x+ch-1)/ch; wgid=(wgid%nc)*ch+wgid/nc;}
    int tm=wgid/nn, tn=wgid%nn;
    const int wid=threadIdx.x/WS, ln=threadIdx.x%WS, wr=wid/4, wc=wid%4, mr=ln%16, mk=ln/16;
    float c0[2][2][4]={}, c1[2][2][4]={};
    float* as=&As[0][0]; float* bs=&Bs[0][0];
    int ao[4]={(wr*RB+0*T+mr)*KS,(wr*RB+1*T+mr)*KS,((wr+2)*RB+0*T+mr)*KS,((wr+2)*RB+1*T+mr)*KS};
    int bo[2]={(wc*RB+0*T+mr)*KS,(wc*RB+1*T+mr)*KS};
    const float* A_base=A+tm*BS*K; const float* B_base=B+tn*BS*K;
    i32x4 a_sr=make_srsrc(A_base,BS*K*4,K*4), b_sr=make_srsrc(B_base,BS*K*4,K*4);

    {constexpr int LPR=KS/4, TOTAL=BS*LPR;
    for(int idx=threadIdx.x;idx<TOTAL;idx+=NT_THR){int r=idx/LPR,c4=(idx%LPR)*4;int bo2=(r*K+c4)*4;
        __uint128_t ra=llvm_amdgcn_raw_buffer_load_b128(a_sr,bo2,0,0);
        __uint128_t rb=llvm_amdgcn_raw_buffer_load_b128(b_sr,bo2,0,0);
        *(float4*)&As[r][c4]=*(float4*)&ra;*(float4*)&Bs[r][c4]=*(float4*)&rb;}}
    __syncthreads();

    for(int kt=0;kt<K/KS-1;kt++){
        float4 ab,bb;int idx=threadIdx.x;
        if(idx<BS*KS/4){int r=idx/(KS/4),c4=(idx%(KS/4))*4;int bo2=(r*K+(kt+1)*KS+c4)*4;
            __uint128_t ra=llvm_amdgcn_raw_buffer_load_b128(a_sr,bo2,0,0);
            __uint128_t rb=llvm_amdgcn_raw_buffer_load_b128(b_sr,bo2,0,0);
            ab=*(float4*)&ra;bb=*(float4*)&rb;}
        #pragma unroll
        for(int ki=0;ki<KI;ki++){int koff=ki*DK+mk;
            float a0=as[ao[0]+koff],a1=as[ao[1]+koff],a2=as[ao[2]+koff],a3=as[ao[3]+koff];
            float bv0=bs[bo[0]+koff],bv1=bs[bo[1]+koff];
            asm volatile("s_waitcnt lgkmcnt(2)");
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c0[0][0]):"v"(a0),"v"(bv0));
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c0[0][1]):"v"(a0),"v"(bv1));
            asm volatile("s_waitcnt lgkmcnt(0)");
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c0[1][0]):"v"(a1),"v"(bv0));
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c0[1][1]):"v"(a1),"v"(bv1));
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c1[0][0]):"v"(a2),"v"(bv0));
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c1[0][1]):"v"(a2),"v"(bv1));
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c1[1][0]):"v"(a3),"v"(bv0));
            asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c1[1][1]):"v"(a3),"v"(bv1));}
        asm volatile("s_waitcnt vmcnt(0)");__syncthreads();
        if(idx<BS*KS/4){int r=idx/(KS/4),c4=(idx%(KS/4))*4;*(float4*)&As[r][c4]=ab;*(float4*)&Bs[r][c4]=bb;}
        __syncthreads();}

    #pragma unroll
    for(int ki=0;ki<KI;ki++){int koff=ki*DK+mk;
        float a0=as[ao[0]+koff],a1=as[ao[1]+koff],a2=as[ao[2]+koff],a3=as[ao[3]+koff];
        float bv0=bs[bo[0]+koff],bv1=bs[bo[1]+koff];
        asm volatile("s_waitcnt lgkmcnt(0)");
        asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c0[0][0]):"v"(a0),"v"(bv0));
        asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c0[0][1]):"v"(a0),"v"(bv1));
        asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c0[1][0]):"v"(a1),"v"(bv0));
        asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c0[1][1]):"v"(a1),"v"(bv1));
        asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c1[0][0]):"v"(a2),"v"(bv0));
        asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c1[0][1]):"v"(a2),"v"(bv1));
        asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c1[1][0]):"v"(a3),"v"(bv0));
        asm volatile("v_mfma_f32_16x16x4_f32 %0,%1,%2,%0":"+v"(*(f4v*)&c1[1][1]):"v"(a3),"v"(bv1));}

    for(int bi=0;bi<2;bi++) for(int bj=0;bj<2;bj++){int col=tn*BS+wc*RB+bj*T+mr;
        for(int k=0;k<4;k++){C[(tm*BS+wr*RB+bi*T+mk*4+k)*N+col]=c0[bi][bj][k];C[(tm*BS+(wr+2)*RB+bi*T+mk*4+k)*N+col]=c1[bi][bj][k];}}
}

// ========== Dispatch ==========
#ifndef HK_MODULE_NAME
#define HK_MODULE_NAME hk_fp32_tt_fused
#endif
PYBIND11_MODULE(HK_MODULE_NAME, m) {
    m.def("dispatch", [](pybind11::object A, pybind11::object B, pybind11::object C) {
        auto sa=A.attr("shape").cast<pybind11::tuple>(); auto sb=B.attr("shape").cast<pybind11::tuple>();
        int Ksz=sa[0].cast<int>(), Msz=sa[1].cast<int>(), Nsz=sb[0].cast<int>(); // A=KxM, B=NxK
        uint64_t a_ptr=A.attr("data_ptr")().cast<uint64_t>(),b_ptr=B.attr("data_ptr")().cast<uint64_t>(),c_ptr=C.attr("data_ptr")().cast<uint64_t>();
        static float* ws=nullptr; static size_t ws_sz=0;
        size_t need=(size_t)Msz*Ksz*sizeof(float); // workspace for transposed A
        if(need>ws_sz){if(ws)hipFree(ws);hipMalloc(&ws,need);ws_sz=need;}
        dim3 tg((Msz+TILE_DIM-1)/TILE_DIM,(Ksz+TILE_DIM-1)/TILE_DIM);
        transpose_f32<<<tg,dim3(TILE_DIM,BLOCK_ROWS)>>>(ws,(const float*)a_ptr,Ksz,Msz); // transpose A KxM→MxK
        fp32_gemm_nt<<<dim3((Nsz/BS)*(Msz/BS)),dim3(NT_THR),0,(hipStream_t)0>>>((float*)c_ptr,ws,(const float*)b_ptr,Msz,Nsz,Ksz); // transposed A, original B(NxK)
    }, "FP32 TT GEMM via transpose+NT: C = A^T @ B^T");
}
