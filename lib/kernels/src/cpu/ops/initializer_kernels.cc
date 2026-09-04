#include "kernels/initializer_kernels.h"
#include "kernels/accessor.h"
#include "kernels/datatype_dispatch.h"
#include "kernels/device.h"
#include "op-attrs/initializers/kaiming_initializer_mode.h"
#include "op-attrs/initializers/kaiming_initializer_nonlinearity.h"
#include "op-attrs/tensor_dims.h"
#include "utils/overload.h"
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>

namespace FlexFlow {

template <DataType DT>
struct ZeroInitKernel {
  void operator()(GenericTensorAccessorW const &tensor) const {
    auto arr = get<DT>(tensor);
    for (size_t i = 0; i < get_num_elements(tensor.shape.dims); i++) {
      arr[i] = 0.0f;
    }
  }
};

void zero_init_kernel_cpu(GenericTensorAccessorW const &tensor) {
  DataTypeDispatch1<ZeroInitKernel>{}(tensor.shape.data_type, tensor);
}

template <DataType DT>
struct ConstantInitKernel {
  void operator()(GenericTensorAccessorW const &tensor,
                  DataTypeValue value) const {
    auto arr = get<DT>(tensor);
    auto unwrapped_value = value.get<real_type_t<DT>>();
    for (size_t i = 0; i < get_num_elements(tensor.shape.dims); i++) {
      arr[i] = unwrapped_value;
    }
  }
};

void constant_init_kernel_cpu(GenericTensorAccessorW const &tensor,
                              DataTypeValue value) {
  DataTypeDispatch1<ConstantInitKernel>{}(
      tensor.shape.data_type, tensor, value);
}

void zero_init_kernel(TaskLocation const &loc,
                      GenericTensorAccessorW const &tensor) {
  if (loc == TaskLocation::CPU) {
    return zero_init_kernel_cpu(tensor);
  } else if (loc == TaskLocation::GPU) {
    return zero_init_kernel_gpu(tensor);
  }
}

void zero_init_kernel_gpu(GenericTensorAccessorW const &tensor) {
  NOT_IMPLEMENTED();
}

/**
 * \brief The source of randomness used by the random initializers.
 *
 * The distributions are computed here rather than with
 * \c std::uniform_real_distribution and \c std::normal_distribution because
 * those are not specified to produce the same sequence across standard library
 * implementations, and an initialization that changes with the toolchain is
 * not much use for reproducing a training run.
 */
class InitializerRng {
public:
  InitializerRng(int seed, size_t salt) : engine(make_engine(seed, salt)) {}

  float next_uniform(float min_val, float max_val) {
    return min_val +
           static_cast<float>(this->next_canonical()) * (max_val - min_val);
  }

  float next_normal(float mean, float stddev) {
    // Box-Muller, which produces two independent samples at a time.
    if (this->has_spare) {
      this->has_spare = false;
      return mean + stddev * this->spare;
    }

    double u1 = 0.0;
    while (u1 <= 0.0) {
      u1 = this->next_canonical();
    }
    double u2 = this->next_canonical();

    double radius = std::sqrt(-2.0 * std::log(u1));
    double angle = 2.0 * M_PI * u2;

    this->spare = static_cast<float>(radius * std::sin(angle));
    this->has_spare = true;
    return mean + stddev * static_cast<float>(radius * std::cos(angle));
  }

  float next_truncated_normal(float mean,
                              float stddev,
                              float min_cutoff,
                              float max_cutoff) {
    // Rejection sampling. The attempt limit keeps a cutoff range that the
    // distribution almost never lands in from hanging; the value it falls back
    // to is at least still within the requested range.
    for (int attempt = 0; attempt < MAX_TRUNCATION_ATTEMPTS; attempt++) {
      float result = this->next_normal(mean, stddev);
      if (result >= min_cutoff && result <= max_cutoff) {
        return result;
      }
    }
    return std::min(std::max(mean, min_cutoff), max_cutoff);
  }

private:
  /**
   * \brief The next value in [0, 1).
   */
  double next_canonical() {
    return static_cast<double>(this->engine() - std::mt19937::min()) /
           (static_cast<double>(std::mt19937::max() - std::mt19937::min()) +
            1.0);
  }

  static std::mt19937 make_engine(int seed, size_t salt) {
    std::seed_seq seed_seq = {static_cast<uint32_t>(seed),
                              static_cast<uint32_t>(salt),
                              static_cast<uint32_t>(salt >> 32)};
    return std::mt19937{seed_seq};
  }

  static constexpr int MAX_TRUNCATION_ATTEMPTS = 100;

  std::mt19937 engine;
  bool has_spare = false;
  float spare = 0.0f;
};

template <DataType DT>
struct FillWithGeneratedValues {
  void operator()(GenericTensorAccessorW const &tensor,
                  std::function<float()> const &generate) const {
    if constexpr (std::is_floating_point_v<real_type_t<DT>>) {
      real_type_t<DT> *arr = get<DT>(tensor);
      for (int i = 0; i < get_num_elements(tensor.shape.dims); i++) {
        arr[i] = static_cast<real_type_t<DT>>(generate());
      }
    } else {
      PANIC("random initialization requires a floating-point tensor",
            tensor.shape.data_type);
    }
  }
};

static void fill_with_generated_values(GenericTensorAccessorW const &tensor,
                                       std::function<float()> const &generate) {
  DataTypeDispatch1<FillWithGeneratedValues>{}(
      tensor.shape.data_type, tensor, generate);
}

/**
 * \brief The standard deviation Glorot (a.k.a. Xavier) initialization draws
 * with, for a gain of 1.
 *
 * see
 * https://github.com/pytorch/pytorch/blob/bd019c0bb485904a99fb38589444b1461ab1e486/torch/nn/init.py#L387-L397
 */
static float glorot_stddev(TensorDims const &dims) {
  positive_int fan_in =
      calculate_fan_for_mode(dims, KaimingInitializerMode::FAN_IN);
  positive_int fan_out =
      calculate_fan_for_mode(dims, KaimingInitializerMode::FAN_OUT);

  return std::sqrt(
      2.0f / static_cast<float>((fan_in + fan_out).int_from_positive_int()));
}

void initialize_tensor_cpu(GenericTensorAccessorW const &tensor,
                           InitializerAttrs const &attrs,
                           size_t salt) {
  ASSERT(tensor.device_type == DeviceType::CPU,
         "initialize_tensor_cpu requires a CPU-allocated tensor");

  TensorDims const &dims = tensor.shape.dims;

  attrs.visit<std::monostate>(overload{
      [&](ZeroInitializerAttrs const &) {
        zero_init_kernel_cpu(tensor);
        return std::monostate{};
      },
      [&](ConstantInitializerAttrs const &constant) {
        constant_init_kernel_cpu(tensor, constant.value);
        return std::monostate{};
      },
      [&](UniformInitializerAttrs const &uniform) {
        InitializerRng rng = InitializerRng{uniform.seed, salt};
        fill_with_generated_values(tensor, [&] {
          return rng.next_uniform(uniform.min_val, uniform.max_val);
        });
        return std::monostate{};
      },
      [&](NormInitializerAttrs const &norm) {
        InitializerRng rng = InitializerRng{norm.seed, salt};
        fill_with_generated_values(
            tensor, [&] { return rng.next_normal(norm.mean, norm.stddev); });
        return std::monostate{};
      },
      [&](TruncatedNormalInitializerAttrs const &truncated) {
        InitializerRng rng = InitializerRng{truncated.seed, salt};
        fill_with_generated_values(tensor, [&] {
          return rng.next_truncated_normal(truncated.mean,
                                           truncated.stddev,
                                           truncated.min_cutoff,
                                           truncated.max_cutoff);
        });
        return std::monostate{};
      },
      [&](GlorotUniformAttrs const &glorot) {
        // A uniform distribution over [-bound, bound] has standard deviation
        // bound / sqrt(3), so this is the uniform distribution with the
        // Glorot standard deviation.
        float bound = std::sqrt(3.0f) * glorot_stddev(dims);
        InitializerRng rng = InitializerRng{glorot.seed, salt};
        fill_with_generated_values(
            tensor, [&] { return rng.next_uniform(-bound, bound); });
        return std::monostate{};
      },
      [&](GlorotNormalAttrs const &glorot) {
        float stddev = glorot_stddev(dims);
        InitializerRng rng = InitializerRng{glorot.seed, salt};
        fill_with_generated_values(
            tensor, [&] { return rng.next_normal(/*mean=*/0.0f, stddev); });
        return std::monostate{};
      },
      [&](KaimingNormalAttrs const &kaiming) {
        positive_int fan = calculate_fan_for_mode(dims, kaiming.mode);
        float gain = gain_for_nonlinearity(kaiming.nonlinearity, kaiming.a);
        float stddev =
            gain / std::sqrt(static_cast<float>(fan.int_from_positive_int()));

        InitializerRng rng = InitializerRng{kaiming.seed, salt};
        fill_with_generated_values(
            tensor, [&] { return rng.next_normal(/*mean=*/0.0f, stddev); });
        return std::monostate{};
      },
  });
}

} // namespace FlexFlow
