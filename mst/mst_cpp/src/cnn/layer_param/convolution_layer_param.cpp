/* include */
#include "../../../include/cnn/layer_param/convolution_layer_param.h"


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace layer
		{

			//	constructor
			ConvolutionLayerParam::ConvolutionLayerParam()
				: filter_(0)
				, kernel_size_(3)
				, stride_(1)
				, padding_(1)
				, padding_mode_(0)
				, use_bias_(true)
			{
			}


			//	destructor
			ConvolutionLayerParam::~ConvolutionLayerParam()
			{
			}


			//	check param
			bool ConvolutionLayerParam::CheckParam()
			{
				if (filter_ < 0)	return false;

				if (kernel_size_ < 0)	return false;

				if (stride_ < 0)	return false;

				if (padding_ < 0)	return false;

				if ((padding_mode_ < 0) || (padding_mode_ > 2))	return false;

				return true;
			}

		}
	}
}
