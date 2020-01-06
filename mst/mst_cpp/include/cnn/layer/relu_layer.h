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

			/* ReLU Layer */
			class ReLULayer
			{

			public:

				ReLULayer();
				~ReLULayer();

				ReLULayer(const ReLULayer& _obj) = delete;
				ReLULayer& operator=(const ReLULayer& _obj) = delete;

				bool Initialize(double _negative_slope);
				bool Reshape(const std::vector<mst::cnn::Blob*>& _input_blobs, const std::vector<mst::cnn::Blob*>& _output_blobs);

				void Forward();
				void Backward();


			private:

				/* parameter */
				double negative_slope_;


				/* blob */
				std::vector<mst::cnn::Blob*> input_blobs_;
				std::vector<mst::cnn::Blob*> output_blobs_;


				/* variable */
				int total_count_;


				/* function */
				bool CheckInputBlobs(const std::vector<mst::cnn::Blob*>& _blobs);
				bool CheckOutputBlobs(const std::vector<mst::cnn::Blob*>& _blobs);

				bool ResetInputBlobs(const std::vector<mst::cnn::Blob*>& _blobs);
				bool ResetOutputBlobs(const std::vector<mst::cnn::Blob*>& _blobs);
			};

		}
	}
}
