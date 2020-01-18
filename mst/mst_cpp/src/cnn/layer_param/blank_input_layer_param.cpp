/* include */
#include "../../../include/cnn/layer_param/blank_input_layer_param.h"

#include "../../../include/cnn/utility.h"


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace layer
		{

			//	constructor
			BlankInputLayerParam::BlankInputLayerParam()
				: batch_size_(0)
				, channels_(0)
				, rows_(0)
				, cols_(0)
			{
			}


			//	destructor
			BlankInputLayerParam::~BlankInputLayerParam()
			{
			}


			//	check param
			bool BlankInputLayerParam::CheckParam()
			{
				if (batch_size_ <= 0)	return false;
				if (channels_ <= 0)	return false;
				if (rows_ <= 0)	return false;
				if (cols_ <= 0)	return false;

				return true;
			}


			//	parse config stirng
			bool BlankInputLayerParam::ParseConfigStrings(const std::vector<std::string> _config)
			{
				int n;
				bool bret;

				char* endptr;

				std::vector<std::string> config_keys;
				std::vector<std::vector<std::string>> config_values;


				//	parse config
				bret = mst::cnn::utility::ParseConfigStrings(_config, config_keys, config_values);
				if (!bret)	return false;

				for(n = 0; n < config_keys.size(); ++n)
				{

					std::string& key = config_keys[n];
					std::vector<std::string>& values = config_values[n];

					if (values.size() <= 0)	return false;

					if (_stricmp(key.c_str(), "batch") == 0)
					{
						batch_size_ = strtol(values[0].c_str(), &endptr, 10);
						if (*endptr != '\0')	return false;
					}
					else if (_stricmp(key.c_str(), "channels") == 0)
					{
						channels_ = strtol(values[0].c_str(), &endptr, 10);
						if (*endptr != '\0')	return false;
					}
					else if (_stricmp(key.c_str(), "rows") == 0)
					{
						rows_ = strtol(values[0].c_str(), &endptr, 10);
						if (*endptr != '\0')	return false;
					}
					else if (_stricmp(key.c_str(), "cols") == 0)
					{
						cols_ = strtol(values[0].c_str(), &endptr, 10);
						if (*endptr != '\0')	return false;
					}

				}

				return true;
			}

		}
	}
}
