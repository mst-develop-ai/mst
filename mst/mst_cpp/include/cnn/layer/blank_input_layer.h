#pragma once

/* include */
#include "../../../include/cnn/layer/base_layer.h"
#include "../../../include/cnn/layer_param/blank_input_layer_param.h"


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace layer
		{

			/* BlankInputLayer */
			class BlankInputLayer : public BaseLayer
			{

			public:

				/* function */
				BlankInputLayer();
				~BlankInputLayer();

				BlankInputLayer(const BlankInputLayer& _obj) = delete;
				BlankInputLayer& operator=(const BlankInputLayer& _obj) = delete;

				bool Initialize(BlankInputLayerParam& _param);
				bool Reshape(const std::vector<mst::cnn::Blob*>& _input_blobs, const std::vector<mst::cnn::Blob*>& _output_blobs);

				void Forward();
				void Backward();

				bool SetImageDataLookupTable(double _scale, double _bias);
				bool SetOutputBlobImageData(int _cols, int _rows, int _channels, const unsigned char* _data);


				/* parameter */
				BlankInputLayerParam layer_param_;


			private:

				/* function */
				bool CheckInputBlobs(const std::vector<mst::cnn::Blob*>& _blobs);
				bool CheckOutputBlobs(const std::vector<mst::cnn::Blob*>& _blobs);

				bool ResetInputBlobs(const std::vector<mst::cnn::Blob*>& _blobs);
				bool ResetOutputBlobs(const std::vector<mst::cnn::Blob*>& _blobs);

				void InitializeImageDataLookupTable();


				/* lookup table */
				double image_lookup_table_[256];

			};

		}
	}
}
