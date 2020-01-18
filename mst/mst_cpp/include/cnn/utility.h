#pragma once

/* include */
#include "../../include/cnn/blob.h"


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace utility
		{
			bool ParseConfigStrings(const std::vector<std::string>& _config, std::vector<std::string>& _config_keys, std::vector<std::vector<std::string>>& _config_values);

			bool SetBlobImageData(mst::cnn::Blob& _blob, int _rows, int _cols, int _channels, const unsigned char* _data);
			bool GetBlobImageData(const mst::cnn::Blob& _blob, int _batch_idx, int _channel_idx, int _dst_size, float* _dst);

		}
	}
}
