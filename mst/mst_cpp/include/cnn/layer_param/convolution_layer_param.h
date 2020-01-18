#pragma once

/* include */
#include "../../../include/cnn/layer_param/base_layer_param.h"

#include <vector>


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace layer
		{

			/* ConvolutionLayerParam */
			class ConvolutionLayerParam : public BaseLayerParam
			{

			public:

				/* function */
				ConvolutionLayerParam();
				~ConvolutionLayerParam();

				bool CheckParam();

				bool ParseConfigStrings(const std::vector<std::string> _config);


				/* parameter */
				int filter_;
				int kernel_size_;
				int stride_;
				int padding_;
				int padding_mode_;
				bool use_bias_;

			};

		}
	}
}
