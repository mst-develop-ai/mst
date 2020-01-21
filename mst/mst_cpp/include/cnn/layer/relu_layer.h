#pragma once

/* include */
#include "../../../include/cnn/layer/base_layer.h"
#include "../../../include/cnn/layer_param/relu_layer_param.h"

#include <vector>


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace layer
		{

			/* ReLULayer */
			class ReLULayer : public BaseLayer
			{

			public:

				/* function */
				ReLULayer();
				~ReLULayer();

				ReLULayer(const ReLULayer& _obj) = delete;
				ReLULayer& operator=(const ReLULayer& _obj) = delete;

				bool Initialize(ReLULayerParam& _param);
				bool Reshape(const std::vector<mst::cnn::Blob*>& _input_blobs, const std::vector<mst::cnn::Blob*>& _output_blobs);

				bool Forward();
				bool Backward();


				/* parameter */
				ReLULayerParam layer_param_;


			private:

				/* function */
				bool CheckInputBlobs(const std::vector<mst::cnn::Blob*>& _blobs);
				bool CheckOutputBlobs(const std::vector<mst::cnn::Blob*>& _blobs);

				bool ResetInputBlobs(const std::vector<mst::cnn::Blob*>& _blobs);
				bool ResetOutputBlobs(const std::vector<mst::cnn::Blob*>& _blobs);
			};

		}
	}
}
