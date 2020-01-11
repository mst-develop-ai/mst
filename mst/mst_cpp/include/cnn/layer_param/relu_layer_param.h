#pragma once

/* include */
#include "../../../include/cnn/layer_param/base_layer_param.h"


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


				/* parameter */
				double negative_slope_;

			};

		}
	}
}
