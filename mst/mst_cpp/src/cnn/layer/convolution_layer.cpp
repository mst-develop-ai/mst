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
				: layer_param_()
				, kernel_count_(0)
				, kernel_mem_size_(0)
				, kernel_(nullptr)
				, bias_mem_size_(0)
				, bias_(nullptr)
				, padding_workspace_()
				, padding_workspace_offset_rows_(0)
				, padding_workspace_copy_cols_mem_size_(0)
				, conv_target_cols_(0)
				, conv_target_rows_(0)
				, conv_elem_count_(0)
				, conv_data_count_(0)
				, conv_total_count_(0)
				, conv_workspace_mem_size_(0)
				, conv_workspace_(nullptr)
			{
				type_ = "ConvolutionLayer";
			}


			//	destructor
			ConvolutionLayer::~ConvolutionLayer()
			{
				ReleaseMemory();
			}


			//	initialize
			bool ConvolutionLayer::Initialize(ConvolutionLayerParam& _param)
			{
				bool bret;


				//	input check
				bret = _param.CheckParam();
				if (!bret)	return false;


				//	set parameter
				layer_param_ = _param;


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
				kernel_mem = kernel_;
				dst = output_blobs_[0]->data_;
				for (f = 0; f < layer_param_.filter_; ++f)
				{
					tmp_dst = dst;
					src = conv_workspace_;
					for (b = 0; b < input_blobs_[0]->shape_[0]; ++b)
					{
						for (c = 0; c < input_blobs_[0]->shape_[1]; ++c)
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

						tmp_dst += output_blobs_[0]->count_[1];
					}

					dst += output_blobs_[0]->count_[2];
					kernel_mem += kernel_count_;
				}


				//	bias
				if (layer_param_.use_bias_)
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


				//	allocate
				kernel_count_ = layer_param_.kernel_size_ * layer_param_.kernel_size_;
				size = layer_param_.filter_ * kernel_count_ * sizeof(double);

				if (size > kernel_mem_size_)
				{
					if (kernel_ != nullptr)
					{
						free(kernel_);
						kernel_ = nullptr;

						kernel_mem_size_ = 0;
					}

					kernel_mem_size_ = size;

					kernel_ = (double*)malloc(kernel_mem_size_);
					if (kernel_ == nullptr)	return false;
				}


				//	initialize
				memset(kernel_, 0, kernel_mem_size_);

				return true;
			}


			//	allocate bias memory
			bool ConvolutionLayer::AllocateBiasMemory()
			{
				int size;


				//	check use
				if (!layer_param_.use_bias_)	return true;


				//	allocate
				size = layer_param_.filter_ * sizeof(double);
				if (size > bias_mem_size_)
				{
					if (bias_ != nullptr)
					{
						free(bias_);
						bias_ = nullptr;

						bias_mem_size_ = 0;
					}

					bias_mem_size_ = size;

					bias_ = (double*)malloc(bias_mem_size_);
					if (bias_ == nullptr)	return false;
				}


				//	initialize
				memset(bias_, 0, bias_mem_size_);

				return true;
			}


			//	allocate padding workspace memory
			bool ConvolutionLayer::AllocatePaddingWorkspaceMemory()
			{
				bool bret;
				std::vector<int> shape;


				//	reshape
				shape = {
					input_blobs_[0]->shape_[0],
					input_blobs_[0]->shape_[1],
					layer_param_.padding_ + input_blobs_[0]->shape_[2] + layer_param_.padding_,
					layer_param_.padding_ + input_blobs_[0]->shape_[3] + layer_param_.padding_
				};

				bret = padding_workspace_.Reshape(shape);
				if (!bret)	return false;


				//	set variable
				padding_workspace_offset_rows_ = layer_param_.padding_ * padding_workspace_.count_[3];
				padding_workspace_copy_cols_mem_size_ = input_blobs_[0]->shape_[3] * sizeof(double);
				

				return true;
			}


			//	allocate convolution workspace memory
			bool ConvolutionLayer::AllocateConvolutionWorkspaceMemory()
			{
				int size;


				//	allocate
				conv_target_rows_ = padding_workspace_.shape_[2] - (layer_param_.kernel_size_ - 1);
				conv_target_cols_ = padding_workspace_.shape_[3] - (layer_param_.kernel_size_ - 1);

				conv_elem_count_ = conv_target_cols_ * conv_target_rows_;
				conv_data_count_ = input_blobs_[0]->shape_[1] * conv_elem_count_;
				conv_total_count_ = input_blobs_[0]->shape_[0] * conv_data_count_;

				size = conv_total_count_ * (layer_param_.kernel_size_ * layer_param_.kernel_size_) * sizeof(double);

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

					kernel_mem_size_ = 0;
				}

				if (bias_ != nullptr)
				{
					free(bias_);
					bias_ = nullptr;

					bias_mem_size_ = 0;
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
				if (_blobs[0]->shape_[2] < (layer_param_.kernel_size_ - layer_param_.padding_ - layer_param_.padding_))	return false;
				if (_blobs[0]->shape_[3] < (layer_param_.kernel_size_ - layer_param_.padding_ - layer_param_.padding_))	return false;

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
				//	reset input blobls
				input_blobs_ = _blobs;

				return true;
			}

			//	reset output blobs
			bool ConvolutionLayer::ResetOutputBlobs(const std::vector<mst::cnn::Blob*>& _blobs)
			{
				bool bret;
				std::vector<int> shape;


				//	reset output blobs
				output_blobs_ = _blobs;

				shape = {
					input_blobs_[0]->shape_[0],
					input_blobs_[0]->shape_[1],
					(input_blobs_[0]->shape_[2] + layer_param_.padding_ + layer_param_.padding_ - (layer_param_.kernel_size_ - 1)) / layer_param_.stride_,
					(input_blobs_[0]->shape_[3] + layer_param_.padding_ + layer_param_.padding_ - (layer_param_.kernel_size_ - 1)) / layer_param_.stride_
				};

				bret = output_blobs_[0]->Reshape(shape);
				if (!bret)	return false;

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


				src = input_blobs_[0]->data_;
				dst = padding_workspace_.data_ + layer_param_.padding_;
				for (b = 0; b < input_blobs_[0]->shape_[0]; ++b)
				{
					for (c = 0; c < input_blobs_[0]->shape_[1]; ++c)
					{
						dst += padding_workspace_offset_rows_;

						for (y = 0; y < input_blobs_[0]->shape_[2]; ++y)
						{
							memcpy_s(dst, padding_workspace_copy_cols_mem_size_, src, padding_workspace_copy_cols_mem_size_);

							src += input_blobs_[0]->shape_[3];
							dst += padding_workspace_.shape_[3];
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


				src = padding_workspace_.data_;
				dst = conv_workspace_;
				for (b = 0; b < input_blobs_[0]->shape_[0]; ++b)
				{
					for (c = 0; c < input_blobs_[0]->shape_[1]; ++c)
					{

						tmp_src = src;
						for (y = 0; y < conv_target_rows_; ++y)
						{
							for (x = 0; x < conv_target_cols_; ++x)
							{

								tmp_tmp_src = tmp_src + x;
								for (ky = 0; ky < layer_param_.kernel_size_; ++ky)
								{
									for (kx = 0; kx < layer_param_.kernel_size_; ++kx)
									{
										*dst = *(tmp_tmp_src + kx);
										++dst;
									}
									tmp_tmp_src += padding_workspace_.shape_[3];
								}

							}
							tmp_src += padding_workspace_.shape_[3];
						}

						src += padding_workspace_.count_[2];

					}
				}

			}

		}
	}
}
