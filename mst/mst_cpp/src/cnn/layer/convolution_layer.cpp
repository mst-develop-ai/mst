/* include */
#include "../../../include/cnn/layer/convolution_layer.h"

#include <stdlib.h>
#include <memory.h>


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace layer
		{

			//	constructor
			ConvolutionLayer::ConvolutionLayer()
				: filter_(0)
				, kernel_size_(3)
				, stride_(1)
				, padding_(1)
				, padding_mode_(0)
				, use_bias_(true)
				, kernel_count_(0)
				, kernel_(nullptr)
				, bias_(nullptr)
				, inputs_()
				, outputs_()
				, outputs_data_size_(0)
				, outputs_batch_data_size_(0)
				, padding_workspace_cols_(0)
				, padding_workspace_rows_(0)
				, padding_workspace_size_(0)
				, padding_workspace_mem_size_(0)
				, padding_workspace_offset_rows_(0)
				, padding_workspace_copy_cols_mem_size_(0)
				, padding_workspace_(nullptr)
				, conv_target_cols_(0)
				, conv_target_rows_(0)
				, conv_elem_count_(0)
				, conv_data_count_(0)
				, conv_total_count_(0)
				, conv_workspace_mem_size_(0)
				, conv_workspace_(nullptr)
			{
			}


			//	destructor
			ConvolutionLayer::~ConvolutionLayer()
			{
				ReleaseMemory();
			}


			//	initialize
			bool ConvolutionLayer::Initialize(int _filter, int _kernel_size, int _strides, int _padding, int _padding_mode, bool _use_bias)
			{
				bool bret;


				//	input check
				if (_filter < 0)	return false;
				if (_kernel_size < 0)	return false;
				if (_strides < 0)	return false;
				if (_padding < 0)	return false;
				if ((_padding_mode < 0) || (_padding_mode > 2))	return false;


				//	set parameter
				filter_ = _filter;
				kernel_size_ = _kernel_size;
				stride_ = _strides;
				padding_ = _padding;
				padding_mode_ = _padding_mode;
				use_bias_ = _use_bias;


				//	allocate memory
				ReleaseMemory();

				bret = AllocateKernelMemory();
				if (!bret)	return false;

				bret = AllocateBiasMemory();
				if (!bret)	return false;

				return true;
			}


			//	reshape
			bool ConvolutionLayer::Reshape(const std::vector<mst::cnn::Blob*>& _input_blobs, const std::vector<mst::cnn::Blob*>& _output_blobs)
			{
				bool bret;


				//	input check
				bret = CheckInputBlobs(_input_blobs);
				if (!bret)	return false;

				bret = CheckOutputBlobs(_output_blobs);
				if (!bret)	return false;


				//	reset blob
				bret = ResetInputBlobs(_input_blobs);
				if (!bret)	return false;

				bret = ResetOutputBlobs(_output_blobs);
				if (!bret)	return false;


				//	allocate memory
				bret = AllocatePaddingWorkspaceMemory();
				if (!bret)	return false;

				bret = AllocateConvolutionWorkspaceMemory();
				if (!bret)	return false;

				return true;
			}


			//	forward
			void ConvolutionLayer::Forward()
			{
				int f;
				int b;
				int c;
				int n;
				int k;

				double* src;
				double* dst;
				double* tmp_dst;

				double* kernel_mem;
				double conv_dst;


				//	set workspace
				SetPaddingWorkspaceMemory();
				SetConvolutionWorkspaceMemory();


				//	conv
				dst = outputs_[0]->data_;
				kernel_mem = kernel_;
				for (f = 0; f < filter_; ++f)
				{
					tmp_dst = dst;
					src = conv_workspace_;
					for (b = 0; b < inputs_[0]->shape_[0]; ++b)
					{
						for (c = 0; c < inputs_[0]->shape_[1]; ++c)
						{

							for (n = 0; n < conv_elem_count_; ++n)
							{
								conv_dst = 0;
								for (k = 0; k < kernel_count_; ++k)
								{
									conv_dst += (*src) * (*(kernel_mem + k));
									++src;
								}

								tmp_dst[n] += conv_dst;
							}

						}

						tmp_dst += outputs_batch_data_size_;
					}

					dst += outputs_data_size_;
					kernel_mem += kernel_count_;
				}


				//	bias
				if (use_bias_)
				{

				}
			}


			//	backward
			void ConvolutionLayer::Backward()
			{

			}


			//	allocate kernel memory
			bool ConvolutionLayer::AllocateKernelMemory()
			{
				int size;


				//	check initialize
				if (kernel_ != nullptr)	return false;


				//	allocate
				kernel_count_ = kernel_size_ * kernel_size_;
				size = filter_ * kernel_count_ * sizeof(double);

				kernel_ = (double*)malloc(size);
				if (kernel_ == nullptr)	return false;


				//	initialize
				memset(kernel_, 0, size);

				kernel_[0] = -1;
				kernel_[1] = -2;
				kernel_[2] = -1;

				kernel_[3] = 0;
				kernel_[4] = 0;
				kernel_[5] = 0;

				kernel_[6] = 1;
				kernel_[7] = 2;
				kernel_[8] = 1;

				kernel_[9 + 0] = -1;
				kernel_[9 + 1] = 0;
				kernel_[9 + 2] = 1;

				kernel_[9 + 3] = -2;
				kernel_[9 + 4] = 0;
				kernel_[9 + 5] = 2;

				kernel_[9 + 6] = -1;
				kernel_[9 + 7] = 0;
				kernel_[9 + 8] = 1;

				return true;
			}


			//	allocate bias memory
			bool ConvolutionLayer::AllocateBiasMemory()
			{
				int size;


				//	check initialize
				if (bias_ != nullptr)	return false;

				if (!use_bias_)	return true;


				//	allocate
				size = filter_ * sizeof(double);

				bias_ = (double*)malloc(size);
				if (bias_ == nullptr)	return false;


				//	initialize
				memset(bias_, 0, size);


				return true;
			}


			//	allocate padding workspace memory
			bool ConvolutionLayer::AllocatePaddingWorkspaceMemory()
			{
				int size;


				//	allocate
				padding_workspace_rows_ = padding_ + inputs_[0]->shape_[2] + padding_;
				padding_workspace_cols_ = padding_ + inputs_[0]->shape_[3] + padding_;
				padding_workspace_size_ = padding_workspace_cols_ * padding_workspace_rows_;
				size = inputs_[0]->shape_[0] * inputs_[0]->shape_[1] * padding_workspace_size_ * sizeof(double);

				if (size > padding_workspace_mem_size_)
				{
					if (padding_workspace_ != nullptr)
					{
						free(padding_workspace_);
						padding_workspace_ = nullptr;

						padding_workspace_mem_size_ = 0;
					}

					padding_workspace_mem_size_ = size;
					padding_workspace_ = (double*)malloc(padding_workspace_mem_size_);
					if (padding_workspace_ == nullptr)	return false;
				}


				//	initialize
				memset(padding_workspace_, 0, padding_workspace_mem_size_);


				//	set variable
				padding_workspace_offset_rows_ = padding_ * padding_workspace_cols_;
				padding_workspace_copy_cols_mem_size_ = inputs_[0]->shape_[3] * sizeof(double);


				return true;
			}


			//	allocate convolution workspace memory
			bool ConvolutionLayer::AllocateConvolutionWorkspaceMemory()
			{
				int size;


				//	allocate
				conv_target_cols_ = padding_workspace_cols_ - (kernel_size_ - 1);
				conv_target_rows_ = padding_workspace_rows_ - (kernel_size_ - 1);

				conv_elem_count_ = conv_target_cols_ * conv_target_rows_;
				conv_data_count_ = inputs_[0]->shape_[1] * conv_elem_count_;
				conv_total_count_ = inputs_[0]->shape_[0] * conv_data_count_;

				size = conv_total_count_ * (kernel_size_ * kernel_size_) * sizeof(double);

				if (size > conv_workspace_mem_size_)
				{
					if (conv_workspace_ != nullptr)
					{
						free(conv_workspace_);
						conv_workspace_ = nullptr;

						conv_workspace_mem_size_ = 0;
					}

					conv_workspace_mem_size_ = size;
					conv_workspace_ = (double*)malloc(conv_workspace_mem_size_);
					if (conv_workspace_ == nullptr)	return false;
				}


				//	initialize
				memset(conv_workspace_, 0, conv_workspace_mem_size_);


				return true;
			}


			//	release memory
			void ConvolutionLayer::ReleaseMemory()
			{
				if (kernel_ != nullptr)
				{
					free(kernel_);
					kernel_ = nullptr;
				}

				if (bias_ != nullptr)
				{
					free(bias_);
					bias_ = nullptr;
				}

				if (padding_workspace_ != nullptr)
				{
					free(padding_workspace_);
					padding_workspace_ = nullptr;

					padding_workspace_mem_size_ = 0;
				}

				if (conv_workspace_ != nullptr)
				{
					free(conv_workspace_);
					conv_workspace_ = nullptr;

					conv_workspace_mem_size_ = 0;
				}
			}


			//	check input blobs
			bool ConvolutionLayer::CheckInputBlobs(const std::vector<mst::cnn::Blob*>& _blobs)
			{
				if (_blobs.size() != 1)	return false;

				if (_blobs[0] == nullptr)	return false;

				if (_blobs[0]->shape_.size() != 4)	return false;
				if (_blobs[0]->shape_[0] <= 0)	return false;
				if (_blobs[0]->shape_[1] <= 0)	return false;
				if (_blobs[0]->shape_[2] < (kernel_size_ - padding_ - padding_))	return false;
				if (_blobs[0]->shape_[3] < (kernel_size_ - padding_ - padding_))	return false;

				return true;
			}


			//	check output blobs
			bool ConvolutionLayer::CheckOutputBlobs(const std::vector<mst::cnn::Blob*>& _blobs)
			{
				if (_blobs.size() != 1)	return false;

				if (_blobs[0] == nullptr)	return false;

				return true;
			}


			//	reset input blobs
			bool ConvolutionLayer::ResetInputBlobs(const std::vector<mst::cnn::Blob*>& _blobs)
			{
				inputs_.clear();
				for each (mst::cnn::Blob* blob in _blobs)
				{
					inputs_.push_back(blob);
				}

				return true;
			}

			//	reset output blobs
			bool ConvolutionLayer::ResetOutputBlobs(const std::vector<mst::cnn::Blob*>& _blobs)
			{
				bool bret;
				std::vector<int> shape;


				//	reset output blobs
				outputs_.clear();
				for each (mst::cnn::Blob* blob in _blobs)
				{
					outputs_.push_back(blob);
				}

				shape.clear();
				shape.push_back(inputs_[0]->shape_[0]);
				shape.push_back(inputs_[0]->shape_[1]);
				shape.push_back((inputs_[0]->shape_[2] + padding_ + padding_ - (kernel_size_ - 1)) / stride_);
				shape.push_back((inputs_[0]->shape_[3] + padding_ + padding_ - (kernel_size_ - 1)) / stride_);

				bret = outputs_[0]->Reshape(shape);
				if (!bret)	return false;


				//	set output variable
				outputs_data_size_ = shape[2] * shape[3];
				outputs_batch_data_size_ = shape[1] * outputs_batch_data_size_;

				return true;
			}


			//	set workspace memory
			void ConvolutionLayer::SetPaddingWorkspaceMemory()
			{
				int b;
				int c;
				int y;
				double* src;
				double* dst;


				src = inputs_[0]->data_;
				dst = padding_workspace_ + padding_;
				for (b = 0; b < inputs_[0]->shape_[0]; ++b)
				{
					for (c = 0; c < inputs_[0]->shape_[1]; ++c)
					{
						dst += padding_workspace_offset_rows_;

						for (y = 0; y < inputs_[0]->shape_[2]; ++y)
						{
							memcpy_s(dst, padding_workspace_copy_cols_mem_size_, src, padding_workspace_copy_cols_mem_size_);

							src += inputs_[0]->shape_[3];
							dst += padding_workspace_cols_;
						}

						dst += padding_workspace_offset_rows_;
					}
				}
			}


			//	set conv workspace memory
			void ConvolutionLayer::SetConvolutionWorkspaceMemory()
			{
				int b;
				int c;
				int y;
				int x;

				int ky;
				int kx;

				double* src;
				double* dst;

				double* tmp_src;
				double* tmp_tmp_src;


				src = padding_workspace_;
				dst = conv_workspace_;
				for (b = 0; b < inputs_[0]->shape_[0]; ++b)
				{
					for (c = 0; c < inputs_[0]->shape_[1]; ++c)
					{

						tmp_src = src;
						for (y = 0; y < conv_target_rows_; ++y)
						{
							for (x = 0; x < conv_target_cols_; ++x)
							{

								tmp_tmp_src = tmp_src + x;
								for (ky = 0; ky < kernel_size_; ++ky)
								{
									for (kx = 0; kx < kernel_size_; ++kx)
									{
										*dst = *(tmp_tmp_src + kx);
										++dst;
									}
									tmp_tmp_src += padding_workspace_cols_;
								}

							}
							tmp_src += padding_workspace_cols_;
						}

						src += padding_workspace_size_;

					}
				}

			}

		}
	}
}
