#pragma once

/* include */
#include "../../../include/cnn/layer/base_layer.h"
#include "../../../include/cnn/layer_param/convolution_layer_param.h"

#include <vector>


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace layer
		{

			/* ConvolutionLayer */
			class ConvolutionLayer : public BaseLayer
			{

			public:

				/* function */
				ConvolutionLayer();
				~ConvolutionLayer();

				ConvolutionLayer(const ConvolutionLayer& _obj) = delete;
				ConvolutionLayer& operator=(const ConvolutionLayer& _obj) = delete;

				bool Initialize(ConvolutionLayerParam& _param);
				bool Reshape(const std::vector<mst::cnn::Blob*>& _input_blobs, const std::vector<mst::cnn::Blob*>& _output_blobs);

				void Forward();
				void Backward();


				/* parameter */
				ConvolutionLayerParam layer_param_;


				/* weights */
				int kernel_count_;
				int kernel_mem_size_;
				double* kernel_;

				int bias_mem_size_;
				double* bias_;


			private:

				/* variable */
				mst::cnn::Blob padding_workspace_;
				int padding_workspace_offset_rows_;
				int padding_workspace_copy_cols_mem_size_;

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
