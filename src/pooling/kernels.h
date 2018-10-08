/*
 * Copyright 2018 Codeplay Software Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SYCLDNN_SRC_POOLING_KERNELS_H_
#define SYCLDNN_SRC_POOLING_KERNELS_H_

#include <CL/sycl.hpp>

#include "src/helpers/tensor_index.h"
#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"

#include "sycldnn/accessor_types.h"

#include "sycldnn/helpers/minmax.h"

#include "sycldnn/pooling/operators.h"
#include "sycldnn/pooling/params.h"

namespace sycldnn {
namespace pooling {

template <typename T>
struct Max {
  T max;
  Max() : max(std::numeric_limits<T>::lowest()) {}
  void accumulate(T val) { val > max ? max = val : T(0); }
  T value() { return max; }
};

/** Template that will average a sequence of accumulated values. */
template <typename T>
struct Average {
  /** The number of values accumulated. */
  int tally;
  /** The sum of the accumulated values. */
  T sum;

  Average() : tally(0), sum(0) {}

  /** Increases the running total of the struct's accumulator.
   * \param val The next value to be added to the accumulator. */
  void accumulate(T val) {
    tally++;
    sum += val;
  }

  /** Observes the average, by dividing the sum  by the number of tallies.
   * \return The average of all accumulated values. */
  T value() { return sum / T(tally); }
};

template <typename T, typename Index, template <typename U> class Op,
          typename Direction>
class PoolingOp;

template <typename T, typename Index, template <typename U> class Op>
class PoolingOp<T, Index, Op, Forward> {
  ReadAccessor<T const> in_data_;
  WriteAccessor<T> out_data_;
  PoolingParams params_;

 public:
  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    if (index < params_.batch * params_.out_rows * params_.out_cols *
                    params_.channels) {
      Op<T> op;
      const auto tensor_id =
          helpers::TensorIndexHelper<Index, false>::unflatten4d(
              index, params_.out_rows, params_.out_rows, params_.out_cols,
              params_.out_cols, params_.channels, params_.channels);
      const auto feature = tensor_id.s3;
      const auto col = tensor_id.s2;
      const auto row = tensor_id.s1;
      const auto batch = tensor_id.s0;

      auto row_start = row * params_.stride_rows - params_.pad_rows;
      const auto row_end =
          helpers::min(row_start + params_.window_rows, params_.in_rows);
      row_start = helpers::max(row_start, 0);

      auto col_start = col * params_.stride_cols - params_.pad_cols;
      const auto col_end =
          helpers::min(col_start + params_.window_cols, params_.in_cols);
      col_start = helpers::max(col_start, 0);

      const auto input_data_offset =
          in_data_.get_pointer() +
          batch * params_.in_cols * params_.in_rows * params_.channels;
      for (Index r = row_start; r < row_end; r++) {
        for (Index c = col_start; c < col_end; c++) {
          Index loc = (r * params_.in_cols + c) * params_.channels + feature;
          op.accumulate(input_data_offset.get()[loc]);
        }
      }
      out_data_[index] = op.value();
    }
  }

  PoolingOp(ReadAccessor<T const> in_data, WriteAccessor<T> out_data,
            const PoolingParams& pp)
      : in_data_(in_data), out_data_(out_data), params_(pp) {}
};

/**
 * Max pooling gradient kernel.
 *
 * Expects to be run with one thread per output value in the backprop kernel.
 */
template <typename T, typename Index>
class PoolingOp<T, Index, Max, Backpropagate> {
  using DataType = typename helpers::VectorType<T, 1>::type;
  using LoadData = helpers::io::Load<DataType>;
  using StoreData = helpers::io::Store<DataType>;

 public:
  PoolingOp(ReadAccessor<T const> const& in_data,
            ReadAccessor<T const> const& out_data,
            ReadAccessor<T const> const& in_backprop,
            WriteAccessor<T> const& out_backprop, PoolingParams const& pp)
      : in_data_{in_data},
        out_data_{out_data},
        in_backprop_{in_backprop},
        out_backprop_{out_backprop},
        n_items_{pp.batch * pp.in_rows * pp.in_cols * pp.channels},
        params_{pp} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    if (index < n_items_) {
      auto in_data = in_data_.get_pointer();
      auto out_data = out_data_.get_pointer();
      auto in_backprop = in_backprop_.get_pointer();
      auto out_backprop = out_backprop_.get_pointer();

      auto const tensor_id =
          helpers::TensorIndexHelper<Index, false>::unflatten4d(
              index, params_.in_rows, params_.in_rows, params_.in_cols,
              params_.in_cols, params_.channels, params_.channels);
      auto const channel = tensor_id.s3;
      auto const col_idx = tensor_id.s2 + params_.pad_cols;
      auto const row_idx = tensor_id.s1 + params_.pad_rows;
      auto const batch = tensor_id.s0;
      DataType gradient{0};
      auto const input_value = LoadData()(in_data, index);

      auto const col_input = get_input_window(
          col_idx, params_.out_cols, params_.window_cols, params_.stride_cols);
      auto const row_input = get_input_window(
          row_idx, params_.out_rows, params_.window_rows, params_.stride_rows);

      Index const index_no_n =
          index - batch * params_.in_cols * params_.in_rows * params_.channels -
          channel;

      auto const input_data_n =
          in_data +
          batch * params_.in_cols * params_.in_rows * params_.channels +
          channel;
      auto const output_data_n =
          out_data +
          batch * params_.out_cols * params_.out_rows * params_.channels +
          channel;
      auto const input_backprop_n =
          in_backprop +
          batch * params_.out_cols * params_.out_rows * params_.channels +
          channel;

      for (Index poolr = row_input.begin; poolr < row_input.end; ++poolr) {
        auto const row_output =
            get_output_window(poolr, params_.in_rows, params_.window_rows,
                              params_.stride_rows, params_.pad_rows);
        for (Index poolc = col_input.begin; poolc < col_input.end; ++poolc) {
          auto const col_output =
              get_output_window(poolc, params_.in_cols, params_.window_cols,
                                params_.stride_cols, params_.pad_cols);

          Index const output_data_idx =
              (poolr * params_.out_cols + poolc) * params_.channels;
          auto const output_value = LoadData()(output_data_n, output_data_idx);

          bool is_max = input_value == output_value;
          bool should_continue = is_max;

          // Even if this thread's output value is the maximum, we cannot say
          // for sure that the input gradient should be assigned to this
          // thread's output gradient. It could be the case that another input
          // in the output value's pool is also the maximum, and then we need to
          // assign the gradient to the first maximum value.
          //
          // To ensure that we assign the gradient to the correct output, loop
          // through the pool values which appear before this thread's value and
          // check that none of those values are a maximum.
          //
          // This is unlikely to occur in real life, as the chances of two
          // random floats in a max pool coinciding is rare, however Tensorflow
          // contains tests which explicitly set input values to be the same.
          // Perhaps this check should be optional to allow a user to choose
          // performance in the general case over correctness in the rare case
          // that this is needed.
          // TODO(jwlawson): Add option to disable max pool correctness check.
          for (Index win_r = row_output.begin;
               win_r < row_output.end && should_continue; ++win_r) {
            for (Index win_c = col_output.begin;
                 win_c < col_output.end && should_continue; ++win_c) {
              Index const input_data_idx =
                  (win_r * params_.in_cols + win_c) * params_.channels;

              if (input_data_idx == index_no_n) {
                // Only check up to the input index
                should_continue = false;
              } else if (LoadData()(input_data_n, input_data_idx) ==
                         output_value) {
                // Found another maximum value before this thread's value
                should_continue = false;
                is_max = false;
              }
            }
          }
          if (is_max) {
            gradient += LoadData()(input_backprop_n, output_data_idx);
          }
        }
      }
      StoreData()(out_backprop, index, gradient);
    }
  }

 private:
  /** Struct defining a window in one dimension of a tensor. */
  struct Window {
    /** First index into the window. */
    Index begin;
    /** One past the last index into the window. */
    Index end;
  };

  /** Get the input window corresponding to the given index.  */
  Window SNN_ALWAYS_INLINE get_input_window(Index idx, Index max_idx,
                                            Index window_size, Index stride) {
    Index const begin =
        (idx < window_size) ? 0 : (idx - window_size) / stride + 1;
    Index const end = helpers::min(idx / stride + 1, max_idx);
    return Window{begin, end};
  }

  /** Get the output window corresponding to the given index.  */
  Window SNN_ALWAYS_INLINE get_output_window(Index idx, Index max_idx,
                                             Index window_size, Index stride,
                                             Index pad) {
    Index begin = idx * stride - pad;
    Index end = helpers::min(begin + window_size, max_idx);
    begin = helpers::max(begin, 0);
    return Window{begin, end};
  }

  ReadAccessor<T const> in_data_;
  ReadAccessor<T const> out_data_;
  ReadAccessor<T const> in_backprop_;
  WriteAccessor<T> out_backprop_;
  Index n_items_;
  PoolingParams params_;
};

/**
 * Average pooling gradient kernel.
 *
 * Expects to be run with one thread per output value in the backprop kernel.
 */
template <typename T, typename Index>
class PoolingOp<T, Index, Average, Backpropagate> {
  using DataType = typename helpers::VectorType<T, 1>::type;
  using LoadData = helpers::io::Load<DataType>;
  using StoreData = helpers::io::Store<DataType>;

 public:
  PoolingOp(ReadAccessor<T const> const& in_data,
            WriteAccessor<T> const& out_data, PoolingParams const& pp)
      : in_backprop_{in_data},
        out_backprop_{out_data},
        n_items_{pp.batch * pp.in_rows * pp.in_cols * pp.channels},
        params_{pp} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    if (index < n_items_) {
      auto const tensor_id =
          helpers::TensorIndexHelper<Index, false>::unflatten4d(
              index, params_.in_rows, params_.in_rows, params_.in_cols,
              params_.in_cols, params_.channels, params_.channels);
      auto const channel = tensor_id.s3;
      auto const col_idx = tensor_id.s2 + params_.pad_cols;
      auto const row_idx = tensor_id.s1 + params_.pad_rows;
      auto const batch = tensor_id.s0;

      auto input_backprop = in_backprop_.get_pointer();
      auto output_backprop = out_backprop_.get_pointer();

      auto const col_input = get_input_window(
          col_idx, params_.out_cols, params_.window_cols, params_.stride_cols);
      auto const row_input = get_input_window(
          row_idx, params_.out_rows, params_.window_rows, params_.stride_rows);

      DataType gradient{0};
      auto input_backprop_n =
          input_backprop +
          batch * params_.out_cols * params_.out_rows * params_.channels +
          channel;

      // For each element in the input window compute the size of the
      // corresponding average pool window. The pool window may include some
      // padding, which are discounted from the pooling, so the window size may
      // not correspond exactly to the parameter's window sizes.
      //
      // Each input gradient needs to be spread out across all the indicies
      // which contributed to that average pool output value, and so we divide
      // the input gradients by the window size before accumulating in the
      // output.
      for (Index poolr = row_input.begin; poolr < row_input.end; ++poolr) {
        Index const row_window_size =
            get_actual_window_size(poolr, params_.in_rows, params_.window_rows,
                                   params_.stride_rows, params_.pad_rows);

        for (Index poolc = col_input.begin; poolc < col_input.end; ++poolc) {
          Index const col_window_size = get_actual_window_size(
              poolc, params_.in_cols, params_.window_cols, params_.stride_cols,
              params_.pad_cols);

          Index const idx =
              (poolr * params_.out_cols + poolc) * params_.channels;
          Index const window_size = row_window_size * col_window_size;
          gradient +=
              LoadData()(input_backprop_n, idx) / static_cast<T>(window_size);
        }
      }
      StoreData()(output_backprop, index, gradient);
    }
  }

 private:
  /**
   * Compute the actual size of a window for the given index.
   *
   * The window may fall off the start or end of the tensor, so the size may be
   * smaller than just the value of `window_size`.
   */
  Index SNN_ALWAYS_INLINE get_actual_window_size(Index idx, Index max_idx,
                                                 Index window_size,
                                                 Index stride, Index pad) {
    Index start = idx * stride - pad;
    Index const end = helpers::min(start + window_size, max_idx);
    start = helpers::max(start, 0);
    Index const size = end - start;
    return size;
  }

  /** Struct defining a window in one dimension of the input tensor. */
  struct InputWindow {
    /** First index into the window. */
    Index begin;
    /** One past the last index into the window. */
    Index end;
  };

  /**
   * Get the input window corresponding to the given index.
   */
  InputWindow SNN_ALWAYS_INLINE get_input_window(Index idx, Index max_idx,
                                                 Index window_size,
                                                 Index stride) {
    Index const begin =
        (idx < window_size) ? 0 : (idx - window_size) / stride + 1;
    Index const end = helpers::min(idx / stride + 1, max_idx);
    return InputWindow{begin, end};
  }

  ReadAccessor<T const> in_backprop_;
  WriteAccessor<T> out_backprop_;
  Index n_items_;
  PoolingParams params_;
};

}  // namespace pooling
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_POOLING_KERNELS_H_