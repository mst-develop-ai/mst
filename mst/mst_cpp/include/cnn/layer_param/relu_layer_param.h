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

			/* ReLULayerParam */
			class ReLULayerParam : public BaseLayerParam
			{

			public:

				/* function */
				ReLULayerParam();
				~ReLULayerParam();

				bool CheckParam();

				bool ParseConfigStrings(const std::vector<std::string> _config);


				/* parameter */
				double negative_slope_;

			};

		}
	}
}
