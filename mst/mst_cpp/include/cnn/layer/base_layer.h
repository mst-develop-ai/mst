#pragma once

/* include */
#include "../../../include/cnn/blob.h"
#include "../../../include/cnn/layer_param/base_layer_param.h"

#include <string>
#include <vector>


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace layer
		{

			/* BaseLayer */
			class BaseLayer
			{

			public:

				BaseLayer();
				virtual ~BaseLayer();

				BaseLayer(const BaseLayer& _obj) = delete;
				BaseLayer& operator=(const BaseLayer& _obj) = delete;

				virtual void Forward() = 0;
				virtual void Backward() = 0;

				virtual bool Reshape(const std::vector<mst::cnn::Blob*>& _input_blobs, const std::vector<mst::cnn::Blob*>& _output_blobs) = 0;


				/* parameter */
				std::string type_;
				std::string name_;


				/* blob */
				std::vector<mst::cnn::Blob*> input_blobs_;
				std::vector<mst::cnn::Blob*> output_blobs_;


			private:


			};

		}
	}
}
