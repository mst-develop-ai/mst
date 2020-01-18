/* include */
#include "../../../include/cnn/layer_param/relu_layer_param.h"

#include "../../../include/cnn/utility.h"


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


			//	parse config stirng
			bool ReLULayerParam::ParseConfigStrings(const std::vector<std::string> _config)
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

					if (_stricmp(key.c_str(), "negative_slope") == 0)
					{
						negative_slope_ = strtod(values[0].c_str(), &endptr);
						if (*endptr != '\0')	return false;
					}

				}

				return true;
			}

		}
	}
}
