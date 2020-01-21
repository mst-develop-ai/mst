/* include */
#include "../../../include/cnn/layer/base_layer.h"


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace layer
		{

			//	constructor
			BaseLayer::BaseLayer()
				: type_("")
				, name_("")
				, input_blobs_()
				, output_blobs_()
			{
			}


			//	destructor
			BaseLayer::~BaseLayer()
			{
			}


			//	initialize weights
			bool BaseLayer::InitializeWeights()
			{
				return true;
			}

		}
	}
}
