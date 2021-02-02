#include <iostream>

#include "legion.h"

using namespace Legion;

static constexpr int WarpSize = 32;

enum TaskID {
  TOP_LEVEL_ID,
  CUDA_ID
};

__device__ inline int lane_id()
{
  int id;
  asm("mov.s32 %0, %laneid;" : "=r"(id));
  return id;
}

template <typename T>
__device__ inline T warp_reduce(T val)
{
#pragma unroll
  for (int i = WarpSize / 2; i > 0; i >>= 1) {
    T tmp = __shfl_sync(val, lane_id() + i, WarpSize);
    val += tmp;
  }
  return val;
}

template <typename T, int TPB>
__device__ T block_reduce(T val, T* smem) {
  static_assert(TPB % WarpSize == 0, "Threads per block must be multiple of warp size");
  int lane = lane_id();
  int warp = threadIdx.x / WarpSize;
  T sum    = warp_reduce(val);
  if (TPB > WarpSize) {
    if (lane == 0) { smem[warp] = sum; }
    __syncthreads();
    if (warp == 0) {
      val = lane < TPB / WarpSize ? smem[lane] : T(0);
      sum = warp_reduce(val);
    }
  }
  return sum;
}

template <int TPB>
__global__ void cuda_kernel() {
  if (threadIdx.x <= 1)
    printf("I am thread %d:%d\n", blockIdx.x, threadIdx.x);
  float val = 1.f;
  __shared__ float tmp[WarpSize];
  float out = block_reduce<float, TPB>(val, &*tmp);
  if (threadIdx.x == 0)
    printf("Block: %d, reduced value: %f\n", blockIdx.x, out);
}

void cuda_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime)
{
  static constexpr int TPB = 4 * WarpSize;
  dim3 block(TPB, 1);
  dim3 grid(8, 1);
  cuda_kernel<TPB><<<grid, block>>>();
}

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  TaskLauncher launcher(CUDA_ID, TaskArgument(nullptr, 0));
  Future cuda_future = runtime->execute_task(ctx, launcher);
  cuda_future.wait();
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_ID);
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(CUDA_ID, "cuda_task");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    Runtime::preregister_task_variant<cuda_task>(registrar, "cuda_task");
  }

  return Runtime::start(argc, argv);
}
