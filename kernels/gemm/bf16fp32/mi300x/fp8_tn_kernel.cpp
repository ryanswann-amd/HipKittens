// FP8 TN GEMM: C = A^T @ B where A is KxM, B is KxN
// Both A and B need in-kernel transpose to shared memory.

#ifndef BLOCK_SIZE_VAL
#define BLOCK_SIZE_VAL 128
#endif
#ifndef K_STEP_VAL
#define K_STEP_VAL 32
#endif
#include "kittens.cuh"
#include <pybind11/pybind11.h>
using namespace kittens;
constexpr int BS=BLOCK_SIZE_VAL, KS=K_STEP_VAL, NW=8, WS=64, NT=NW*WS;
constexpr int T=32, DK=16, KI=KS/DK;
constexpr int NROW = BS / (T * 2);
constexpr int NCOL = BS / (T * 4);
constexpr int NACC = NROW * NCOL;
static_assert(NROW * 2 * T == BS && NCOL * 4 * T == BS, "Tile config");
constexpr int SH_PAD = KS + 4;  // LDS padding for bank conflict elimination
typedef __attribute__((__vector_size__(16*sizeof(float)))) float f16v;
#define MFMA(acc, a, b) acc = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a, b, acc, 0, 0, 0)

// Transpose load: src[k][n] → shared[n][k]
template<int STRIDE>
__device__ void load_transpose_fp8(
    char shared[][STRIDE], const char* __restrict__ base,
    int stride, int k_offset, int tile_offset)
{
    constexpr int N_GROUP=16, K_GROUP=4;
    constexpr int N_BATCHES=BS/N_GROUP, K_BATCHES=KS/K_GROUP;
    constexpr int TOTAL=K_BATCHES*N_BATCHES;
    #pragma unroll 1
    for(int batch=threadIdx.x; batch<TOTAL; batch+=NT) {
        int kb=batch/N_BATCHES, nb=batch%N_BATCHES;
        int k=kb*K_GROUP, n=nb*N_GROUP;
        float4 r0=*(const float4*)&base[(k_offset+k+0)*stride+tile_offset+n];
        float4 r1=*(const float4*)&base[(k_offset+k+1)*stride+tile_offset+n];
        float4 r2=*(const float4*)&base[(k_offset+k+2)*stride+tile_offset+n];
        float4 r3=*(const float4*)&base[(k_offset+k+3)*stride+tile_offset+n];
        const char *p0=(const char*)&r0,*p1=(const char*)&r1,*p2=(const char*)&r2,*p3=(const char*)&r3;
        
        for(int i=0;i<N_GROUP;i++){
            union{char c[4];int w;} col;
            col.c[0]=p0[i];col.c[1]=p1[i];col.c[2]=p2[i];col.c[3]=p3[i];
            *(int*)&shared[n+i][k]=col.w;
        }
    }
}

__global__ __launch_bounds__(NT, 2)
void fp8_gemm_tn(float* __restrict__ C, const char* __restrict__ A,
                 const char* __restrict__ B, int M, int N, int K) {
    __shared__ char As[BS][SH_PAD], Bs[BS][SH_PAD];  // Both padded (both transposed)
    const int nn=N/BS; int wgid=blockIdx.x;
    {int W=4,ch=W*W,nc=(gridDim.x+ch-1)/ch; wgid=(wgid%nc)*ch+wgid/nc;}
    int tm=wgid/nn, tn=wgid%nn;
    const int wid=threadIdx.x/WS, ln=threadIdx.x%WS, wr=wid/4, wc=wid%4;
    int lk=(ln/32)*8;
    int ar[NROW], br[NCOL];
    
    for(int i=0;i<NROW;i++) ar[i]=(wr+i*2)*T+(ln%32);
    
    for(int i=0;i<NCOL;i++) br[i]=(wc+i*4)*T+(ln%32);
    f16v acc[NACC]; for(int i=0;i<NACC;i++) acc[i]={};

    // A: KxM, B: KxN — both need transpose
    int A_m_offset = tm * BS;
    int B_n_offset = tn * BS;

    // Prologue
    load_transpose_fp8<SH_PAD>(As, A, M, 0, A_m_offset);
    load_transpose_fp8<SH_PAD>(Bs, B, N, 0, B_n_offset);
    asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();

    const int num_tiles=K/KS;
    for(int kt=0;kt<num_tiles-1;kt++){
        
        for(int ki=0;ki<KI;ki++){int koff=ki*DK+lk;
            long av[NROW],bv[NCOL];
            
            for(int r=0;r<NROW;r++) av[r]=*(const long*)&As[ar[r]][koff];
            
            for(int c=0;c<NCOL;c++) bv[c]=*(const long*)&Bs[br[c]][koff];
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            
            for(int c=0;c<NCOL;c++) 
                for(int r=0;r<NROW;r++) MFMA(acc[r*NCOL+c],av[r],bv[c]);
            __builtin_amdgcn_s_setprio(0);}

        __builtin_amdgcn_s_barrier();
        load_transpose_fp8<SH_PAD>(As, A, M, (kt+1)*KS, A_m_offset);
        load_transpose_fp8<SH_PAD>(Bs, B, N, (kt+1)*KS, B_n_offset);
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
    }

    
    for(int ki=0;ki<KI;ki++){int koff=ki*DK+lk;
        long av[NROW],bv[NCOL];
        
        for(int r=0;r<NROW;r++) av[r]=*(const long*)&As[ar[r]][koff];
        
        for(int c=0;c<NCOL;c++) bv[c]=*(const long*)&Bs[br[c]][koff];
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        
        for(int c=0;c<NCOL;c++) 
            for(int r=0;r<NROW;r++) MFMA(acc[r*NCOL+c],av[r],bv[c]);
        __builtin_amdgcn_s_setprio(0);}

    
    for(int c=0;c<NCOL;c++){int col_base=tn*BS+(wc+c*4)*T+(ln%32);
        
        for(int r=0;r<NROW;r++){float*cp=(float*)&acc[r*NCOL+c];int row_base=tm*BS+(wr+r*2)*T;
            
            for(int g=0;g<4;g++) 
                for(int j=0;j<4;j++) C[(row_base+g*8+(ln/32)*4+j)*N+col_base]=cp[g*4+j];}}
}
#undef MFMA

#ifndef HK_MODULE_NAME
#define HK_MODULE_NAME hk_fp8_tn
#endif
PYBIND11_MODULE(HK_MODULE_NAME, m) {
    m.def("dispatch", [](pybind11::object A, pybind11::object B, pybind11::object C) {
        auto sa=A.attr("shape").cast<pybind11::tuple>(); auto sb=B.attr("shape").cast<pybind11::tuple>();
        int Ksz=sa[0].cast<int>(), Msz=sa[1].cast<int>(), Nsz=sb[1].cast<int>();
        fp8_gemm_tn<<<dim3((Nsz/BS)*(Msz/BS)),dim3(NT),0,(hipStream_t)0>>>(
            (float*)C.attr("data_ptr")().cast<uint64_t>(),
            (const char*)A.attr("data_ptr")().cast<uint64_t>(),
            (const char*)B.attr("data_ptr")().cast<uint64_t>(), Msz, Nsz, Ksz);
    }, "FP8 TN GEMM: C = A^T @ B (in-kernel both transpose)");
}
