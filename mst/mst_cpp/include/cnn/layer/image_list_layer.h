#pragma once

/* include */
#include "../../../include/cnn/layer/base_layer.h"
#include "../../../include/cnn/layer_param/image_list_layer_param.h"

#include <vector>


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace layer
		{

			/* ImageListLayer */
			class ImageListLayer : public BaseLayer
			{

			public:

				/* function */
				ImageListLayer();
				~ImageListLayer();

				ImageListLayer(const ImageListLayer& _obj) = delete;
				ImageListLayer& operator=(const ImageListLayer& _obj) = delete;

				bool Initialize(ImageListLayerParam& _param);
				bool Reshape(const std::vector<mst::cnn::Blob*>& _input_blobs, const std::vector<mst::cnn::Blob*>& _output_blobs);

				bool Forward();
				bool Backward();


				/* parameter */
				ImageListLayerParam layer_param_;


			private:

				/* function */
				bool CheckInputBlobs(const std::vector<mst::cnn::Blob*>& _blobs);
				bool CheckOutputBlobs(const std::vector<mst::cnn::Blob*>& _blobs);

				bool ResetInputBlobs(const std::vector<mst::cnn::Blob*>& _blobs);
				bool ResetOutputBlobs(const std::vector<mst::cnn::Blob*>& _blobs);


				/* read data */
				int data_pos_;
				std::vector<int> data_access_list_;

			};

		}
	}
}
