#pragma once

/* include */
#include "../../../include/cnn/layer_param/base_layer_param.h"

#include <string>


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace layer
		{

			/* ImageListLayerParam */
			class ImageListLayerParam : public BaseLayerParam
			{

			public:

				/* function */
				ImageListLayerParam();
				~ImageListLayerParam();

				bool CheckParam();

				bool ParseConfigStrings(const std::vector<std::string> _config);


				/* parameter */
				std::string data_list_path_;
				int data_elem_count_;

				int target_data_dim_;
				bool single_target_data_;

				int batch_size_;
				int channels_;
				int rows_;
				int cols_;

				int target_rows_;
				int target_cols_;
			};

		}
	}
}
