#pragma once

/* include */
#include "../../../include/cnn/layer_param/base_layer_param.h"

#include <vector>
#include <string>


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace layer
		{

			/* BlankInputLayerParam */
			class BlankInputLayerParam : public BaseLayerParam
			{

			public:

				/* function */
				BlankInputLayerParam();
				~BlankInputLayerParam();

				bool CheckParam();

				bool ParseConfigStrings(const std::vector<std::string> _config);


				/* parameter */
				int batch_size_;
				int channels_;
				int rows_;
				int cols_;

			};

		}
	}
}
