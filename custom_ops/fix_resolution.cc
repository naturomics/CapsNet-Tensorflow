#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <math.h>

using namespace tensorflow;

REGISTER_OP("FixResolution")
.Input("to_fix: float") //input tensor
.Input("range_bits: int32") // range and precision bits (m, n)
.Input("precision_bits: int32") // range and precision bits (m, n)
.Output("fixed: float")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
});

class FixResolutionOp : public OpKernel {
  public:
    explicit FixResolutionOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& input_tensor = context->input(0);
      const Tensor& range_bits = context->input(1);
      const Tensor& precision_bits = context->input(2);

      auto input = input_tensor.flat<float>();
      auto m = range_bits.flat<int>(); //range - precision
      auto n = precision_bits.flat<int>(); //range - precision

      float range_min = -1 * pow(2, (m - 1));
      float range_max = pow(2, (m - 1)) - pow(2, -1 * n);
      float resolution = pow(2, -1 * n);

      //std::cout << "range: [" << range_min << ", " << range_max << "]"
      //  << " | resolution: " << resolution
      //  << std::endl;

      // Create an output tensor
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(
          context,
          context->allocate_output(0, input_tensor.shape(), &output_tensor));
      auto output = output_tensor->flat<float>();

      // convert input tensor to fixed point equivalent range
      // and precision with a 5% resolution tolerance
      // counts times when range is clipped, or tolerance exceeded
      const int input_count = input.size();
      for (int i = 0; i < input_count; i++) {

        // clip on max and min of allowed range
        if (input(i) > range_max || input(i) < range_min ) {
          if (input(i) > range_max) { output(i) = range_max; }
          if (input(i) < range_min) { output(i) = range_min; }

        // convert resolution to fixed point equivalent
        } else {
          float fix_equivalent = resolution * trunc(input(i) / resolution);
          float deviation_from_orig = abs(fix_equivalent - input(i)) / abs(input(i));
          output(i) = fix_equivalent;
          if(deviation_from_orig > 0.05){ // more than 5% deviation
          }
        }
      }
      //std::cout << "%over: " << accuracy(0)
      //  << " %under: " << accuracy(1) << std::endl;
    }
};
/*
REGISTER_OP("FixResolutionGrad")
  .Input("grad: float") //input tensor
  .Input("to_fix: float") //input tensor
  .Input("fixpt_definition: int32") // range and precision bits (m, n)
  .Input("accuracy: float") // overflow or unprecise percentage
  .Output("fixed_grad: float")
  .Output("fixpt_definition_grad: int32")
  .Output("accuracy_grad: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
});

class FixResolutionGradOp : public OpKernel {
  public:
    explicit FixResolutionGradOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& gradient = context->input(0);
      const Tensor& range_precision = context->input(2);
      // have to preform a const_cast to be able to pass by reference
      Tensor& overflow = const_cast<Tensor&>(context->input(3));

      auto m_n = range_precision.flat<int>(); //range - precision
      auto input = gradient.flat<float>();
      auto accuracy = overflow.flat<float>();

      //std::cout << "range: [" << range_min << ", " << range_max << "]"
      //  << " | resolution: " << resolution
      //  << std::endl;

      // Gradient output
      Tensor* fixed_grad = NULL;
      OP_REQUIRES_OK(
          context,
          context->allocate_output(0, gradient.shape(), &fixed_grad));
      auto output = fixed_grad->flat<float>();

      // Definition output
      Tensor* fixpt_definition_grad = NULL;
      OP_REQUIRES_OK(
          context,
          context->allocate_output(1, range_precision.shape(), &fixpt_definition_grad));

      // overflow/underflow output
      Tensor* accuracy_grad = NULL;
      OP_REQUIRES_OK(
          context,
          context->allocate_output(2, overflow.shape(), &accuracy_grad));

      // For our gradient we are simply seting input = output
      const int input_count = input.size();
      for (int i = 0; i < input_count; i++) {

        output(i) = input(i);

      }
      //accuracy(0) = 0.;
      //accuracy(1) = 0.;

      //std::cout << "%over: " << accuracy(0)
      //  << " %under: " << accuracy(1) << std::endl;
    }
};
*/
REGISTER_KERNEL_BUILDER(Name("FixResolution").Device(DEVICE_CPU), FixResolutionOp);

//REGISTER_KERNEL_BUILDER(Name("FixResolutionGrad").Device(DEVICE_CPU), FixResolutionGradOp);
