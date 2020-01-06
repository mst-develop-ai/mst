#pragma once

/* include */
#include "../../../include/cnn/blob.h"

#include <vector>


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace layer
		{

			/* Convolution Layer */
			class ConvolutionLayer
			{

			public:

				ConvolutionLayer();
				~ConvolutionLayer();

				ConvolutionLayer(const ConvolutionLayer& _obj) = delete;
				ConvolutionLayer& operator=(const ConvolutionLayer& _obj) = delete;

				bool Initialize(int _filter, int _kernel_size, int _strides, int _padding, int _padding_mode, bool _use_bias);
				bool Reshape(const std::vector<mst::cnn::Blob*>& _input_blobs, const std::vector<mst::cnn::Blob*>& _output_blobs);

				void Forward();
				void Backward();


			private:

				/* parameter */
				int filter_;
				int kernel_size_;
				int stride_;
				int padding_;
				int padding_mode_;
				bool use_bias_;


				/* weights */
				int kernel_count_;
				double* kernel_;
				double* bias_;


				/* blob */
				std::vector<mst::cnn::Blob*> inputs_;
				std::vector<mst::cnn::Blob*> outputs_;


				/* variable */
				int outputs_data_size_;
				int outputs_batch_data_size_;

				int padding_workspace_cols_;
				int padding_workspace_rows_;
				int padding_workspace_size_;
				int padding_workspace_mem_size_;
				int padding_workspace_offset_rows_;
				int padding_workspace_copy_cols_mem_size_;
				double* padding_workspace_;

				int conv_target_cols_;
				int conv_target_rows_;
				int conv_elem_count_;
				int conv_data_count_;
				int conv_total_count_;
				int conv_workspace_mem_size_;
				double* conv_workspace_;


				/* function */
				bool AllocateKernelMemory();
				bool AllocateBiasMemory();
				bool AllocatePaddingWorkspaceMemory();
				bool AllocateConvolutionWorkspaceMemory();
				void ReleaseMemory();

				bool CheckInputBlobs(const std::vector<mst::cnn::Blob*>& _blobs);
				bool CheckOutputBlobs(const std::vector<mst::cnn::Blob*>& _blobs);

				bool ResetInputBlobs(const std::vector<mst::cnn::Blob*>& _blobs);
				bool ResetOutputBlobs(const std::vector<mst::cnn::Blob*>& _blobs);

				void SetPaddingWorkspaceMemory();
				void SetConvolutionWorkspaceMemory();

			};

		}
	}
}
