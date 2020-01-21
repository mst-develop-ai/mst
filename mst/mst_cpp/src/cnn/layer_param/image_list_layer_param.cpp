/* include */
#include "../../../include/cnn/layer_param/image_list_layer_param.h"

#include "../../../include/cnn/utility.h"


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace layer
		{

			//	constructor
			ImageListLayerParam::ImageListLayerParam()
				: data_list_path_()
				, data_elem_count_(0)
				, target_data_dim_(0)
				, single_target_data_(false)
				, batch_size_(0)
				, channels_(0)
				, rows_(0)
				, cols_(0)
				, target_rows_(1)
				, target_cols_(1)
			{
			}


			//	destructor
			ImageListLayerParam::~ImageListLayerParam()
			{
			}

			//	check param
			bool ImageListLayerParam::CheckParam()
			{
				if (data_list_path_.length() == 0)	return false;
				if (data_elem_count_ <= 0)	return false;

				if ((target_data_dim_ <= 1) || (target_data_dim_ >= 5))	return false;

				if (single_target_data_)
				{
					if ((data_elem_count_ <= 0) || (data_elem_count_ >= 3))	return false;
				}

				if (batch_size_ <= 0)	return false;
				if (channels_ <= 0)	return false;
				if (rows_ <= 0)	return false;
				if (cols_ <= 0)	return false;

				if (target_rows_ <= 0)	return false;
				if (target_cols_ <= 0)	return false;

				return true;
			}


			//	parse config stirng
			bool ImageListLayerParam::ParseConfigStrings(const std::vector<std::string> _config)
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

					if (_stricmp(key.c_str(), "data_list_path") == 0)
					{
						data_list_path_ = values[0];
					}
					else if (_stricmp(key.c_str(), "data_elem_count") == 0)
					{
						data_elem_count_ = strtol(values[0].c_str(), &endptr, 10);
						if (*endptr != '\0')	return false;
					}
					else if (_stricmp(key.c_str(), "target_data_dim") == 0)
					{
						target_data_dim_ = strtol(values[0].c_str(), &endptr, 10);
						if (*endptr != '\0')	return false;
					}
					else if (_stricmp(key.c_str(), "single_target_data") == 0)
					{
						single_target_data_ = ((_stricmp(values[0].c_str(), "true") == 0) || (_stricmp(values[0].c_str(), "1") == 0));
					}
					else if (_stricmp(key.c_str(), "batch") == 0)
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
					else if (_stricmp(key.c_str(), "target_rows") == 0)
					{
						target_rows_ = strtol(values[0].c_str(), &endptr, 10);
						if (*endptr != '\0')	return false;
					}
					else if (_stricmp(key.c_str(), "target_cols") == 0)
					{
						target_cols_ = strtol(values[0].c_str(), &endptr, 10);
						if (*endptr != '\0')	return false;
					}

				}

				return true;
			}

		}
	}
}
