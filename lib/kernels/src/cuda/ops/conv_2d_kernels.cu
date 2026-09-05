#include "internal/device.h"
#include "kernels/conv_2d_kernels_gpu.h"
#include "kernels/device_aux_streams.h"
#include "kernels/device_scratch.h"
#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/tensor_dims.h"
#include <atomic>
#include <cstdlib>
#include <string>
#include <vector>

namespace FlexFlow {

/**
 * \brief Whether convolution algorithms are chosen by measuring them.
 *
 * On by default; set \c FF_CUDNN_BENCHMARK=0 to fall back to cuDNN's
 * heuristics. The name follows PyTorch's \c cudnn.benchmark, which is the same
 * switch, though its default is the other way round.
 *
 * Measuring is worth about 6ms of a 172ms YOLOv10x iteration. What it costs is
 * roughly 8 seconds of startup and, less obviously, reproducibility: the
 * measured-fastest algorithm is sometimes one that accumulates with atomics, so
 * two runs of the same program on the same input stop agreeing to the last bit.
 * Turning this off is the way to get a run that repeats exactly.
 */
static bool conv_algorithms_are_measured() {
  static bool const measured = []() {
    char const *value = std::getenv("FF_CUDNN_BENCHMARK");
    return value == nullptr || std::string{value} != "0";
  }();
  return measured;
}

// Picks the fastest algorithm reported by cuDNN's heuristics that both
// succeeds and fits in the available workspace.
template <typename PerfResult>
static int select_algorithm_idx(std::vector<PerfResult> const &perf_results,
                                int num_results,
                                size_t workspace_size) {
  for (int i = 0; i < num_results; i++) {
    if (perf_results.at(i).status == CUDNN_STATUS_SUCCESS &&
        perf_results.at(i).memory <= workspace_size) {
      return i;
    }
  }

  PANIC("No cuDNN convolution algorithm fits within the available workspace",
        num_results,
        workspace_size);
}

Conv2DPerDeviceState conv_2d_gpu_init_kernel(PerDeviceFFHandle const &handle,
                                             Conv2DAttrs const &attrs,
                                             TensorShape const &input_shape,
                                             TensorShape const &output_shape) {
  // Applying an activation as part of Conv2D is not currently implemented. The
  // previous implementation hardcoded relu (regardless of which activation was
  // requested) and computed the backward pass by destructively modifying the
  // output gradient. If you need it, please create an issue.
  ASSERT(!attrs.activation.has_value(),
         "Conv2D does not currently support fused activations",
         attrs.activation);

  ASSERT(get_num_dims(input_shape.dims) == num_tensor_dims_t{4_n},
         "Conv2D expects 4-dimensional (i.e., NCHW) input tensors",
         input_shape);
  ASSERT(conv2d_get_output_shape(attrs, input_shape) == output_shape,
         "Conv2D output shape does not match the shape implied by its "
         "attributes and input shape",
         attrs,
         input_shape,
         output_shape);

  TensorShape filter_shape = conv2d_get_kernel_shape(attrs, input_shape);

  positive_int input_c = dim_at_idx(input_shape.dims, ff_dim_t{1_n});
  positive_int output_c = dim_at_idx(output_shape.dims, ff_dim_t{1_n});

  ASSERT(input_c % attrs.groups == 0,
         "Conv2D requires the number of input channels to be divisible by the "
         "number of groups",
         input_c,
         attrs.groups);

  ffTensorDescriptor_t inputTensor;
  ffTensorDescriptor_t biasTensor;
  ffTensorDescriptor_t outputTensor;
  ffFilterDescriptor_t filterDesc;
  ffConvolutionDescriptor_t convDesc;

  checkCUDNN(cudnnCreateTensorDescriptor(&inputTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&biasTensor));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputTensor));
  checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

  checkCUDNN(cudnnSetTensorDescriptorFromTensorShape(inputTensor, input_shape));
  checkCUDNN(
      cudnnSetTensorDescriptorFromTensorShape(outputTensor, output_shape));

  checkCUDNN(
      cudnnSetTensor4dDescriptor(biasTensor,
                                 CUDNN_TENSOR_NCHW,
                                 ff_to_cudnn_datatype(output_shape.data_type),
                                 /*n=*/1,
                                 /*c=*/output_c.int_from_positive_int(),
                                 /*h=*/1,
                                 /*w=*/1));

  checkCUDNN(cudnnSetFilter4dDescriptor(
      filterDesc,
      ff_to_cudnn_datatype(filter_shape.data_type),
      CUDNN_TENSOR_NCHW,
      dim_at_idx(filter_shape.dims, ff_dim_t{0_n}).int_from_positive_int(),
      dim_at_idx(filter_shape.dims, ff_dim_t{1_n}).int_from_positive_int(),
      dim_at_idx(filter_shape.dims, ff_dim_t{2_n}).int_from_positive_int(),
      dim_at_idx(filter_shape.dims, ff_dim_t{3_n}).int_from_positive_int()));

  checkCUDNN(
      cudnnSetConvolution2dDescriptor(convDesc,
                                      attrs.padding_h.unwrap_nonnegative(),
                                      attrs.padding_w.unwrap_nonnegative(),
                                      attrs.stride_h.int_from_positive_int(),
                                      attrs.stride_w.int_from_positive_int(),
                                      /*dilation_h=*/1,
                                      /*dilation_w=*/1,
                                      CUDNN_CROSS_CORRELATION,
                                      CUDNN_DATA_FLOAT));

  checkCUDNN(cudnnSetConvolutionGroupCount(
      convDesc, attrs.groups.int_from_positive_int()));

  // enable tensor core when possible
  if (handle.allowTensorOpMathConversion) {
    checkCUDNN(cudnnSetConvolutionMathType(
        convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  } else {
    checkCUDNN(cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));
  }

  {
    int n, c, h, w;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
        convDesc, inputTensor, filterDesc, &n, &c, &h, &w));
    ASSERT(dim_at_idx(output_shape.dims, ff_dim_t{0_n}) == positive_int{n});
    ASSERT(dim_at_idx(output_shape.dims, ff_dim_t{1_n}) == positive_int{c});
    ASSERT(dim_at_idx(output_shape.dims, ff_dim_t{2_n}) == positive_int{h});
    ASSERT(dim_at_idx(output_shape.dims, ff_dim_t{3_n}) == positive_int{w});
  }

  // NOTE: algorithms are chosen by measuring them (cudnnFind*, which runs each
  // candidate on scratch buffers of its own) rather than by cuDNN's heuristics
  // (cudnnGet*_v7, which predicts). What the heuristics predict is not what
  // runs fastest here: measuring instead is worth about 6ms of a 172ms
  // YOLOv10x iteration, for about 8 seconds of startup.
  //
  // The reason this is a reasonable default here, where the equivalent is
  // opt-in in frameworks that do the same thing, is that the shapes are fixed
  // for the life of a \ref PCGInstance. There is no risk of paying for the
  // measurement again on the next call with a different shape.
  //
  // Note this is cudnnFind*, not cudnnFind*Ex: the Ex family takes the real
  // input, output and gradient buffers and writes over them, and we have
  // nothing to hand at this point.
  ffConvolutionFwdAlgo_t fwdAlgo;
  size_t fwdWorkspaceSize;
  {
    int max_num_results;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(handle.dnn,
                                                           &max_num_results));
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(max_num_results);
    int num_results = 0;
    if (conv_algorithms_are_measured()) {
      checkCUDNN(cudnnFindConvolutionForwardAlgorithm(handle.dnn,
                                                      inputTensor,
                                                      filterDesc,
                                                      convDesc,
                                                      outputTensor,
                                                      max_num_results,
                                                      &num_results,
                                                      perf_results.data()));
    } else {
      checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(handle.dnn,
                                                        inputTensor,
                                                        filterDesc,
                                                        convDesc,
                                                        outputTensor,
                                                        max_num_results,
                                                        &num_results,
                                                        perf_results.data()));
    }
    int idx =
        select_algorithm_idx(perf_results, num_results, handle.workSpaceSize);
    fwdAlgo = perf_results.at(idx).algo;
    fwdWorkspaceSize = perf_results.at(idx).memory;
  }

  ffConvolutionBwdFilterAlgo_t bwdFilterAlgo;
  size_t bwdFilterWorkspaceSize;
  {
    int max_num_results;
    checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
        handle.dnn, &max_num_results));
    std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results(
        max_num_results);
    int num_results = 0;
    if (conv_algorithms_are_measured()) {
      checkCUDNN(
          cudnnFindConvolutionBackwardFilterAlgorithm(handle.dnn,
                                                      inputTensor,
                                                      outputTensor,
                                                      convDesc,
                                                      filterDesc,
                                                      max_num_results,
                                                      &num_results,
                                                      perf_results.data()));
    } else {
      checkCUDNN(
          cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle.dnn,
                                                        inputTensor,
                                                        outputTensor,
                                                        convDesc,
                                                        filterDesc,
                                                        max_num_results,
                                                        &num_results,
                                                        perf_results.data()));
    }
    int idx =
        select_algorithm_idx(perf_results, num_results, handle.workSpaceSize);
    bwdFilterAlgo = perf_results.at(idx).algo;
    bwdFilterWorkspaceSize = perf_results.at(idx).memory;
  }

  ffConvolutionBwdDataAlgo_t bwdDataAlgo;
  size_t bwdDataWorkspaceSize;
  {
    int max_num_results;
    checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
        handle.dnn, &max_num_results));
    std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results(
        max_num_results);
    int num_results = 0;
    if (conv_algorithms_are_measured()) {
      checkCUDNN(
          cudnnFindConvolutionBackwardDataAlgorithm(handle.dnn,
                                                    filterDesc,
                                                    outputTensor,
                                                    convDesc,
                                                    inputTensor,
                                                    max_num_results,
                                                    &num_results,
                                                    perf_results.data()));
    } else {
      checkCUDNN(
          cudnnGetConvolutionBackwardDataAlgorithm_v7(handle.dnn,
                                                      filterDesc,
                                                      outputTensor,
                                                      convDesc,
                                                      inputTensor,
                                                      max_num_results,
                                                      &num_results,
                                                      perf_results.data()));
    }
    int idx =
        select_algorithm_idx(perf_results, num_results, handle.workSpaceSize);
    bwdDataAlgo = perf_results.at(idx).algo;
    bwdDataWorkspaceSize = perf_results.at(idx).memory;
  }

  return Conv2DPerDeviceState{
      /*inputTensor=*/inputTensor,
      /*biasTensor=*/biasTensor,
      /*outputTensor=*/outputTensor,
      /*filterDesc=*/filterDesc,
      /*convDesc=*/convDesc,
      /*fwdAlgo=*/fwdAlgo,
      /*bwdFilterAlgo=*/bwdFilterAlgo,
      /*bwdDataAlgo=*/bwdDataAlgo,
      /*fwdWorkspaceSize=*/fwdWorkspaceSize,
      /*bwdFilterWorkspaceSize=*/bwdFilterWorkspaceSize,
      /*bwdDataWorkspaceSize=*/bwdDataWorkspaceSize,
  };
}

void conv_2d_gpu_forward_kernel(
    cudaStream_t stream,
    PerDeviceFFHandle const &handle,
    Conv2DPerDeviceState const &per_device_state,
    Conv2DAttrs const &attrs,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorR const &filter,
    std::optional<GenericTensorAccessorR> const &bias,
    GenericTensorAccessorW const &output) {
  ASSERT(bias.has_value() == attrs.use_bias);

  checkCUDNN(cudnnSetStream(handle.dnn, stream));

  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnConvolutionForward(
      handle.dnn,
      &alpha,
      per_device_state.inputTensor,
      input.ptr,
      per_device_state.filterDesc,
      filter.ptr,
      per_device_state.convDesc,
      per_device_state.fwdAlgo,
      get_device_scratch_for_stream(stream, per_device_state.fwdWorkspaceSize),
      per_device_state.fwdWorkspaceSize,
      &beta,
      per_device_state.outputTensor,
      output.ptr));

  if (bias.has_value()) {
    checkCUDNN(cudnnAddTensor(handle.dnn,
                              &alpha,
                              per_device_state.biasTensor,
                              bias.value().ptr,
                              &alpha,
                              per_device_state.outputTensor,
                              output.ptr));
  }
}

void conv_2d_gpu_backward_kernel(
    cudaStream_t stream,
    PerDeviceFFHandle const &handle,
    Conv2DPerDeviceState const &per_device_state,
    Conv2DAttrs const &attrs,
    GenericTensorAccessorR const &output,
    GenericTensorAccessorR const &output_grad,
    GenericTensorAccessorR const &input,
    GenericTensorAccessorW const &input_grad,
    GenericTensorAccessorR const &filter,
    GenericTensorAccessorW const &filter_grad,
    std::optional<GenericTensorAccessorW> const &bias_grad) {
  ASSERT(bias_grad.has_value() == attrs.use_bias);

  // The weight gradient has nothing waiting on it until the update pass, so it
  // does not have to sit on the critical path the rest of the backward pass
  // runs down. Put it on a stream of its own where there is one; the data
  // gradient, which the next layer's backward pass waits for, stays on the
  // task's own stream.
  ffStream_t wgrad_stream = stream;
  ffHandle_t wgrad_dnn = handle.dnn;
  if (num_aux_streams() > 0) {
    static std::atomic<int> next_aux{0};
    int index = next_aux++ % num_aux_streams();
    wgrad_stream = get_aux_stream(stream, index);
    wgrad_dnn = get_aux_dnn_handle(stream, index);
    fork_to_aux_stream(stream, wgrad_stream);
  }

  checkCUDNN(cudnnSetStream(wgrad_dnn, wgrad_stream));

  float alpha = 1.0f, beta = 0.0f;

  checkCUDNN(cudnnConvolutionBackwardFilter(
      wgrad_dnn,
      &alpha,
      per_device_state.inputTensor,
      input.ptr,
      per_device_state.outputTensor,
      output_grad.ptr,
      per_device_state.convDesc,
      per_device_state.bwdFilterAlgo,
      get_device_scratch_for_stream(wgrad_stream,
                                    per_device_state.bwdFilterWorkspaceSize),
      per_device_state.bwdFilterWorkspaceSize,
      &beta,
      per_device_state.filterDesc,
      filter_grad.ptr));

  if (bias_grad.has_value()) {
    checkCUDNN(cudnnConvolutionBackwardBias(wgrad_dnn,
                                            &alpha,
                                            per_device_state.outputTensor,
                                            output_grad.ptr,
                                            &beta,
                                            per_device_state.biasTensor,
                                            bias_grad.value().ptr));
  }

  checkCUDNN(cudnnSetStream(handle.dnn, stream));
  checkCUDNN(cudnnConvolutionBackwardData(
      handle.dnn,
      &alpha,
      per_device_state.filterDesc,
      filter.ptr,
      per_device_state.outputTensor,
      output_grad.ptr,
      per_device_state.convDesc,
      per_device_state.bwdDataAlgo,
      get_device_scratch_for_stream(stream,
                                    per_device_state.bwdDataWorkspaceSize),
      per_device_state.bwdDataWorkspaceSize,
      &beta,
      per_device_state.inputTensor,
      input_grad.ptr));
}

void conv_2d_gpu_cleanup_kernel(Conv2DPerDeviceState &per_device_state) {
  checkCUDNN(cudnnDestroyTensorDescriptor(per_device_state.inputTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(per_device_state.biasTensor));
  checkCUDNN(cudnnDestroyTensorDescriptor(per_device_state.outputTensor));
  checkCUDNN(cudnnDestroyFilterDescriptor(per_device_state.filterDesc));
  checkCUDNN(cudnnDestroyConvolutionDescriptor(per_device_state.convDesc));
}

} // namespace FlexFlow
