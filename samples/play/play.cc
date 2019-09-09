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

// This sample makes use of the Eigen backend, and so we need to include the
// relevant Eigen header.
#include <unsupported/Eigen/CXX11/Tensor>

#include "sycldnn/backend/eigen_backend.h"

#include "sycldnn/conv2d/conv_type.h"
#include "sycldnn/conv2d/launch.h"
#include "sycldnn/conv2d/params.h"
#include "sycldnn/conv2d/selector/direct_selector.h"
#include "sycldnn/conv2d/selector/im2col_selector.h"
#include "sycldnn/conv2d/selector/winograd_selector.h"
#include "sycldnn/conv2d/sizes.h"
#include "sycldnn/pointwise/direction.h"
#include "sycldnn/pointwise/launch.h"
#include "sycldnn/pointwise/operators.h"
#include "sycldnn/status.h"

#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <CL/sycl.hpp>

/**
 * Basic SYCL-DNN convolution-operation sample (walk-through)
 */

// custom additions to the sycldnn namespace
namespace sycldnn {

enum class TensorType : int { Conv2D = 0, DepthwiseConv2D = 1 };

template <TensorType>
struct Tensor {};

template <>
struct Tensor<TensorType::Conv2D> {
  conv2d::Conv2DParams params;
  conv2d::ConvSizes sizes;
};

// template over Selector
enum class TensorMode : int {
  Matmul = 0,
  Tiled = 1,
  Winograd = 2,
  WinogradLarge = 3
};

template <TensorType>
class TensorDescriptor {};

template <>
class TensorDescriptor<TensorType::Conv2D> {
 public:
  explicit TensorDescriptor(conv2d::Conv2DParams const& params)
      : m_tensor{.params = params, .sizes = {}} {}
  explicit TensorDescriptor(conv2d::Conv2DParams&& params)
      : m_tensor{.params = std::move(params), .sizes = {}} {}

  template <typename ConvType>
  inline auto getTensor() -> Tensor<TensorType::Conv2D> {
    m_tensor.sizes = getTensorSize<ConvType>();
    return m_tensor;
  }

 private:
  Tensor<TensorType::Conv2D> m_tensor;

  template <typename ConvType>
  inline conv2d::ConvSizes getTensorSize() {
    // i.e., sycldnn::conv2d::conv_type::Forward
    return sycldnn::conv2d::get_sizes<ConvType>(m_tensor.params);
  }
};

}  // namespace sycldnn

// ...
class OpenCVManager {
  using ImageType = cv::Mat;

 public:
  // Utility for loading a single image object
  static ImageType loadImage(std::string const& filename) {
    ImageType image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
    image.convertTo(image, CV_32FC3);
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
    return image;
  }

  static void saveImage(std::string const& filename, float* buffer, int height,
                        int width) {
    ImageType image(height, width, CV_32FC3, buffer);
    // Make negative values zero.
    cv::threshold(image, image, 0, 0, cv::THRESH_TOZERO);
    cv::normalize(image, image, 0.0, 255.0, cv::NORM_MINMAX);
    image.convertTo(image, CV_8UC3);
    cv::imwrite(filename.c_str(), image);
  }

  static void displayImage() {
    // TODO ...
  }
};

// ...
template <typename DataType>
class SNNConvolutionManager {
 public:
  SNNConvolutionManager(const cl::sycl::device_selector& deviceSelector,
                        Eigen::SyclDevice& syclDevice)
      : m_syclQueue(std::unique_ptr<Eigen::QueueInterface>(
            new Eigen::QueueInterface{deviceSelector})) {
    // Create the SyclDevice (using the Eigen backend) from the QueueInterface
    m_syclDevice = Eigen::SyclDevice{m_syclQueue.get()};

    // Construct a SYCL-DNN Eigen backend
    m_snnBackend = sycldnn::backend::EigenBackend{m_syclDevice};
  }

  ~SNNConvolutionManager() {
    // ...
    if (m_tensorBuffers.size() > 0) {
      deallocateBuffers(m_tensorBuffers);
    }
    // ...
    if (m_filterBuffers.size() > 0) {
      deallocateBuffers(m_filterBuffers);

      // ...
      if (m_intermediateBuffers.size() > 0) {
        deallocateBuffers(m_intermediateBuffers);
      }
    }
  }

  void addTensorsAndFilters(std::initializer_list<DataType*> tensorBuffers,
                            std::initializer_list<DataType*> filterBuffers) {
    m_tensorBuffers = std::vector<DataType*>(tensorBuffers);
    m_filterBuffers = std::vector<DataType*>(filterBuffers);
  }

  // use either variadic templates or init_lists
  // initially: vectors with initalizer_list
  // then consider: tuples with variadic template
  // void addTensor() {}
  // void addFilter() {}

  template <int Rows, int Cols>
  void executeConvolution(float const* filter) {
    // TODO: ...
  }

 private:
  std::unique_ptr<Eigen::QueueInterface> m_syclQueue;
  Eigen::SyclDevice m_syclDevice;
  sycldnn::backend::EigenBackend m_snnBackend;

  std::vector<DataType*> m_tensorBuffers;
  std::vector<DataType*> m_filterBuffers;
  std::vector<DataType*> m_intermediateBuffers;

  // for each filter => allocate intermediate buf.
  // inline void allocateIntermediateBuffer() {}

  inline void deallocateBuffers(std::vector<DataType*>& buffers) {
    for (DataType* buffer : buffers) {
      m_syclDevice.deallocate(buffer);
    }
  }
};

namespace filters {
// sobel filter array (x and y axis)
static constexpr float sobelX[9] = {-1.0, 0.0,  1.0, -2.0, 0.0,
                                    2.0,  -1.0, 0.0, 1.0};
static constexpr float sobelY[9] = {-1.0, -2.0, -1.0, 0.0, 0.0,
                                    0.0,  1.0,  2.0,  1.0};
}  // namespace filters

int main() {
  using value_type = float;

  // Select OpenCL device
  auto device_selector = cl::sycl::default_selector{};

  // Create related Eigen objects (dispatch queue and associated device))
  auto queue = std::unique_ptr<Eigen::QueueInterface>(
      new Eigen::QueueInterface{device_selector});
  auto device = Eigen::SyclDevice{queue.get()};

  // Construct a SYCL-DNN Eigen backend
  auto backend = sycldnn::backend::EigenBackend{device};

  auto inputFile =
      std::string("/home/georgi/projects/SYCL-DNN/res/tensorflow.png");
  auto outputFile =
      std::string("/home/georgi/projects/SYCL-DNN/res/out.png");
  auto image = OpenCVManager::loadImage(inputFile);

  // input image read from OpenCV, filter: 3x3.
  sycldnn::conv2d::Conv2DParams convParams{};
  convParams.channels = image.channels();
  convParams.features = 3;
  convParams.batch = 1;
  convParams.in_rows = image.rows;
  convParams.in_cols = image.cols;
  convParams.window_rows = 3;
  convParams.window_cols = 3;
  convParams.stride_rows = 1;
  convParams.stride_cols = 1;
  convParams.out_rows = image.rows;
  convParams.out_cols = image.cols;
  convParams.pad_rows = 0;
  convParams.pad_cols = 0;
  convParams.dilation_rows = 1;
  convParams.dilation_cols = 1;

  // the tensor store the parameters and the sizes which are computed from the
  // paramaters inside the getTensor() function call
  auto convTensor =
      sycldnn::TensorDescriptor<sycldnn::TensorType::Conv2D>{convParams}
          .getTensor<sycldnn::conv2d::conv_type::Forward>();

  // input image read from OpenCV, filter: 3x3.
  sycldnn::conv2d::Conv2DParams conv2Params{};
  conv2Params.channels = image.channels();
  conv2Params.features = 3;
  conv2Params.batch = 1;
  conv2Params.in_rows = image.rows;
  conv2Params.in_cols = image.cols;
  conv2Params.window_rows = 3;
  conv2Params.window_cols = 3;
  conv2Params.stride_rows = 1;
  conv2Params.stride_cols = 1;
  conv2Params.out_rows = image.rows;
  conv2Params.out_cols = image.cols;
  conv2Params.pad_rows = 0;
  conv2Params.pad_cols = 0;
  conv2Params.dilation_rows = 1;
  conv2Params.dilation_cols = 1;

  // the tensor store the parameters and the sizes which are computed from the
  // paramaters inside the getTensor() function call
  auto conv2Tensor =
      sycldnn::TensorDescriptor<sycldnn::TensorType::Conv2D>{conv2Params}
          .getTensor<sycldnn::conv2d::conv_type::Forward>();

  // A 2D convolution requires an input tensor representing a batch of images,
  // a filter tensor containing a filter kernel for each feature, and an output
  // tensor to hold the generated feature maps.
  //
  // Here we calculate the storage requirements for these tensors, and then
  // allocate storage for them via Eigen's GPU device memory allocator.
  const auto inputNBytes = convTensor.sizes.input_size * sizeof(value_type);
  auto* inputBuffer = static_cast<value_type*>(device.allocate(inputNBytes));

  ///*
  auto intermediateNBytes = convTensor.sizes.output_size * sizeof(value_type);
  auto* intermediateBuffer =
      static_cast<value_type*>(device.allocate(intermediateNBytes));
  //*/

  const auto outputNBytes = conv2Tensor.sizes.input_size * sizeof(value_type);
  auto* outputBuffer = static_cast<value_type*>(device.allocate(outputNBytes));

  const auto filter1NBytes = convTensor.sizes.filter_size * sizeof(value_type);
  auto* filter1Buffer =
      static_cast<value_type*>(device.allocate(filter1NBytes));

  ///*
  const auto filter2NBytes = conv2Tensor.sizes.filter_size * sizeof(value_type);
  auto* filter2Buffer =
      static_cast<value_type*>(device.allocate(filter2NBytes));
  //*/

  // The GPU buffers are initially unpopulated. Here we fill the input and
  // filter tensors. The output tensors are left undefined.
  std::vector<value_type> input(
      reinterpret_cast<value_type*>(image.data),
      reinterpret_cast<value_type*>(image.data) + convTensor.sizes.input_size);
  {  // test the input image data loaded into the input vector
    std::cout << "Input Size:\t" << input.size() << std::endl;
  }

  device.memcpyHostToDevice(inputBuffer, input.data(), inputNBytes);
  device.memcpyHostToDevice(filter1Buffer, filters::sobelX, filter1NBytes);
  device.memcpyHostToDevice(filter2Buffer, filters::sobelY, filter2NBytes);

  int res = 0;

  // Now that all of our buffers are populated, and parameters configured, we
  // can execute the convolution itself. This happens asynchronously, so we
  // follow the launch of the convolution kernel with a blocking wait.
  auto algoSelctor = sycldnn::conv2d::DirectSelector{};
  auto status =
      sycldnn::conv2d::launch<value_type, sycldnn::conv2d::conv_type::Forward>(
          inputBuffer, filter1Buffer, intermediateBuffer, convTensor.params,
          algoSelctor, backend);
  if (sycldnn::StatusCode::OK != status.status) {
    // If the launch failed, then clean up our GPU buffers and return failure.
    device.deallocate(inputBuffer);
    device.deallocate(outputBuffer);
    device.deallocate(filter1Buffer);
    device.deallocate(filter2Buffer);
    std::cout << "messed up on first pass\n";
    return res = -1;
  }

  std::vector<value_type> intermediate;
  intermediate.resize(convTensor.sizes.output_size);
  // Wait for completion, then copy results to system memory.
  status.event.wait_and_throw();
  device.memcpyDeviceToHost(intermediate.data(), intermediateBuffer,
                            intermediateNBytes);
  OpenCVManager::saveImage(
      "/home/georgi/projects/SYCL-DNN/res/intermediate.png",
      intermediate.data(), image.rows, image.cols);

  algoSelctor = sycldnn::conv2d::DirectSelector{};
  status =
      sycldnn::conv2d::launch<value_type, sycldnn::conv2d::conv_type::Forward>(
          intermediateBuffer, filter2Buffer, outputBuffer, conv2Tensor.params,
          algoSelctor, backend);
  if (sycldnn::StatusCode::OK != status.status) {
    // If the launch failed, then clean up our GPU buffers and return failure.
    device.deallocate(inputBuffer);
    device.deallocate(outputBuffer);
    device.deallocate(filter1Buffer);
    device.deallocate(filter2Buffer);
    std::cout << "messed up on second pass\n";
    return res = -1;
  }
  ///*
  // Activation (tanh)
  status = sycldnn::pointwise::launch<value_type, sycldnn::pointwise::Tanh,
                                      sycldnn::pointwise::Forward>(
      outputBuffer, outputBuffer, conv2Tensor.sizes.output_size, backend);

  if (sycldnn::StatusCode::OK != status.status) {
    // If the launch failed, then clean up our GPU buffers and return failure.
    device.deallocate(inputBuffer);
    device.deallocate(outputBuffer);
    device.deallocate(filter1Buffer);
    device.deallocate(filter2Buffer);
    std::cout << "messed up on activation pass\n";
    return res = -1;
  }
  //*/

  // The convolutions are now executing. While they run, we can allocate a
  // host-accessible vector, then wait for completion and trigger a copy via
  // Eigen to return the results to system memory.
  std::vector<value_type> output;
  output.resize(conv2Tensor.sizes.output_size);

  // Wait for completion, then copy results to system memory.
  status.event.wait_and_throw();
  device.memcpyDeviceToHost(output.data(), outputBuffer, outputNBytes);

  // The convolution results are now available in host-accessible system memory.

  // We can now deallocate the Eigen GPU buffers.
  device.deallocate(inputBuffer);
  device.deallocate(outputBuffer);
  device.deallocate(filter1Buffer);
  device.deallocate(filter2Buffer);

  std::cout << "Success" << std::endl;

  OpenCVManager::saveImage(outputFile, output.data(), image.rows, image.cols);
  return res;
}
