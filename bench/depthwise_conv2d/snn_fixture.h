/*
 * Copyright 2019 Codeplay Software Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use these files except in compliance with the License.
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
#ifndef SYCLDNN_BENCH_DEPTHWISE_CONV2D_SNN_FIXTURE_H_
#define SYCLDNN_BENCH_DEPTHWISE_CONV2D_SNN_FIXTURE_H_

#include "base_depthwise_convolution_fixture.h"
#include "snn_depthwise_conv2d_executor.h"

#include "src/backend/backend_provider.h"

#include "bench/fixture/add_computecpp_info.h"
#include "bench/fixture/add_datatype_info.h"
#include "bench/fixture/add_sycl_device_info.h"
#include "bench/fixture/statistic.h"
#include "bench/fixture/string_reporter.h"
#include "bench/fixture/typenames.h"

template <typename Backend, typename DataType, typename ParamGen,
          typename ConvType>
class SNNDepthwiseConvolutionBenchmark
    : public sycldnn::bench::SNNDepthwiseConv2DExecutor<
          SNNDepthwiseConvolutionBenchmark<Backend, DataType, ParamGen,
                                           ConvType>,
          ConvType>,
      public sycldnn::backend::BackendProvider<Backend>,
      public sycldnn::bench::StringReporter,
      public BaseDepthwiseConvolutionBenchmark {
 private:
  using State = benchmark::State;

 protected:
  void run(State& state) {
    auto params = ParamGen()();
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::MaxStatistic{}});
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::MinStatistic{}});
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::StdDevStatistic{}});
    this->execute(state, params);

    // Get the SYCL device, and add device and driver info to the benchmark.
    auto& backend = this->get_backend();
    auto dev = backend.get_queue().get_device();
    sycldnn::bench::device_info::add_opencl_device_info(dev, *this);
    sycldnn::bench::computecpp_info::add_computecpp_version(*this);
    sycldnn::bench::datatype_info::add_datatype_info<DataType>(*this);

    this->add_to_label("@conv_type", sycldnn::bench::TypeName<ConvType>::name);
    this->add_to_label("@backend", backend.name());
    this->add_to_label("short_name", "Depthwise Convolution");
    this->add_to_label("git_hash", commit_hash);
    this->set_label(state);
  }

  void set_model(const char* model_name) {
    this->add_to_label("@model_name", model_name);
  }
};

#define DEPTHWISE_CONVOLUTION_BENCHMARK(model, name, ...)             \
  BENCHMARK_TEMPLATE_DEFINE_F(SNNDepthwiseConvolutionBenchmark, name, \
                              __VA_ARGS__)                            \
  (benchmark::State & state) {                                        \
    this->set_model(model);                                           \
    this->run(state);                                                 \
  }                                                                   \
  BENCHMARK_REGISTER_F(SNNDepthwiseConvolutionBenchmark, name)        \
      ->UseManualTime()                                               \
      ->Unit(benchmark::kNanosecond);

#endif  // SYCLDNN_BENCH_DEPTHWISE_CONV2D_SNN_FIXTURE_H_
