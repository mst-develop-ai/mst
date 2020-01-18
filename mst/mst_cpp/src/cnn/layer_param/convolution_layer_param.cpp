/* include */
#include "../../../include/cnn/layer_param/convolution_layer_param.h"

#include "../../../include/cnn/utility.h"


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


			//	parse config stirng
			bool ConvolutionLayerParam::ParseConfigStrings(const std::vector<std::string> _config)
			{
				int n;
				bool bret;

				char* endptr;

				std::vector<std::string> config_keys;
				std::vector<std::vector<std::string>> config_values;


				//	parse config
				bret = mst::cnn::utility::ParseConfigStrings(_config, config_keys, config_values);
				if (!bret)	return false;

				for (n = 0; n < config_keys.size(); ++n)
				{

					std::string& key = config_keys[n];
					std::vector<std::string>& values = config_values[n];

					if (values.size() <= 0)	return false;

					if (_stricmp(key.c_str(), "filter") == 0)
					{
						filter_ = strtol(values[0].c_str(), &endptr, 10);
						if (*endptr != '\0')	return false;
					}
					else if (_stricmp(key.c_str(), "kernel") == 0)
					{
						kernel_size_ = strtol(values[0].c_str(), &endptr, 10);
						if (*endptr != '\0')	return false;
					}
					else if (_stricmp(key.c_str(), "stride") == 0)
					{
						stride_ = strtol(values[0].c_str(), &endptr, 10);
						if (*endptr != '\0')	return false;
					}
					else if (_stricmp(key.c_str(), "pad") == 0)
					{
						padding_ = strtol(values[0].c_str(), &endptr, 10);
						if (*endptr != '\0')	return false;
					}
					else if (_stricmp(key.c_str(), "pad_mode") == 0)
					{
						padding_mode_ = strtol(values[0].c_str(), &endptr, 10);
						if (*endptr != '\0')	return false;
					}
					else if (_stricmp(key.c_str(), "use_bias") == 0)
					{
						use_bias_ = ((_stricmp(values[0].c_str(), "true") == 0) || ((_stricmp(values[0].c_str(), "1") == 0)));
					}
				}

				return true;
			}

		}
	}
}
