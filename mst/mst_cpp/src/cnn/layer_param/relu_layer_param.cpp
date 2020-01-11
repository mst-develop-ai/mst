/* include */
#include "../../../include/cnn/layer_param/relu_layer_param.h"


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace layer
		{

			//	constructor
			ReLULayerParam::ReLULayerParam()
				: negative_slope_(0)
			{
			}


			//	destructor
			ReLULayerParam::~ReLULayerParam()
			{
			}


			//	check param
			bool ReLULayerParam::CheckParam()
			{
				if (negative_slope_ < 0)	return false;

				return true;
			}

		}
	}
}
