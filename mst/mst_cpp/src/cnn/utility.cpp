/* include */
#include "../../include/cnn/utility.h"

#include "../../include/string/string.h"


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace utility
		{

			//	parse config strings
			bool ParseConfigStrings(const std::vector<std::string>& _config, std::vector<std::string>& _config_keys, std::vector<std::vector<std::string>>& _config_values)
			{
				bool bret;

				const char* data;
				const char* prev_data;

				int parentheses;

				std::string tmp;
				std::vector<std::string> buff;

				std::string key;


				//	init
				_config_keys.clear();
				_config_values.clear();


				//	parse config lines
				parentheses = 0;

				for each (std::string line in _config)
				{
					if (line.length() == 0)	continue;

					prev_data = data = line.c_str();
					while (true)
					{
						if ((*data) == '\0')
						{
							//	set buffer
							tmp = std::string(prev_data, data);
							bret = mst::string::DeleteFrontLastSpaceCode(tmp);
							if (!bret)	return false;

							if (tmp.length() > 0)
							{
								buff.push_back(tmp);
							}

							break;
						}

						else if ((*data) == ':')
						{
							if (parentheses == 0)
							{
								key = std::string(prev_data, data);
								bret = mst::string::DeleteFrontLastSpaceCode(key);
								if (!bret)	return false;

								prev_data = data + 1;
								while (true)
								{
									++data;
									if ((*data) == '\0')
									{
										break;
									}
								}

								tmp = std::string(prev_data, data);
								bret = mst::string::DeleteFrontLastSpaceCode(key);
								if (!bret)	return false;

								_config_keys.push_back(key);
								_config_values.push_back({ tmp });

								key.clear();

								break;
							}
						}

						else if ((*data) == '{')
						{
							//	set buffer
							tmp = std::string(prev_data, data);
							bret = mst::string::DeleteFrontLastSpaceCode(tmp);
							if (!bret)	return false;

							if (tmp.length() > 0)
							{
								buff.push_back(tmp);
							}

							prev_data = data + 1;


							//	start parentheses
							++parentheses;
							if (parentheses == 1)
							{
								if (buff.size() != 1)	return false;
								if (key.length() != 0)	return false;

								key = buff[0];
								buff.clear();
							}
							else
							{
								buff.push_back("{");
							}
						}

						else if ((*data) == '}')
						{
							//	set buffer
							tmp = std::string(prev_data, data);
							bret = mst::string::DeleteFrontLastSpaceCode(tmp);
							if (!bret)	return false;

							if (tmp.length() > 0)
							{
								buff.push_back(tmp);
							}

							prev_data = data + 1;


							//	end parentheses
							--parentheses;
							if (parentheses < 0)	return false;
							else if (parentheses == 0)
							{
								if (key.length() == 0)	return false;

								_config_keys.push_back(key);
								_config_values.push_back(buff);

								key.clear();
								buff.clear();
							}
							else
							{
								buff.push_back("}");
							}
						}

						++data;
					}
				}

				if (parentheses != 0)	return false;

				return true;
			}


			//	set blob image data
			bool SetBlobImageData(mst::cnn::Blob& _blob, int _rows, int _cols, int _channels, const unsigned char* _data)
			{
				int y;
				int x;
				int xx;
				int c;

				bool bret;

				std::vector<int> shape;

				int step;

				double* dst;
				const unsigned char* src;


				//	check input
				if (_rows <= 0)	return false;
				if (_cols <= 0)	return false;
				if (_channels <= 0)	return false;
				if (_data == nullptr)	return false;


				//	reshape
				shape = { 1, _channels, _rows, _cols };

				bret = _blob.Reshape(shape);
				if (!bret)	return false;


				//	set data
				dst = _blob.data_;
				step = _cols * _channels;
				for (c = 0; c < _channels; ++c)
				{
					src = _data + c;
					for (y = 0; y < _rows; ++y)
					{
						for (x = 0, xx = 0; x < _cols; ++x, xx += _channels)
						{
							*dst = *(src + xx);
							++dst;
						}
						
						src += step;
					}
				}

				return true;
			}


			//	get blob image data
			bool GetBlobImageData(const mst::cnn::Blob& _blob, int _batch_idx, int _channel_idx, int _dst_size, float* _dst)
			{
				int n;

				double* src;
				float* dst;


				//	check input
				if (_blob.shape_.size() != 4)	return false;

				if (_batch_idx < 0)	return false;
				if (_batch_idx > _blob.shape_[0])	return false;

				if (_channel_idx < 0)	return false;
				if (_channel_idx > _blob.shape_[1])	return false;

				if (_dst_size < _blob.count_[2])	return false;
				if (_dst == nullptr)	return false;


				//	set data
				dst = _dst;
				src = _blob.data_ + (_batch_idx * _blob.count_[1]) + (_channel_idx * _blob.count_[2]);
				for (n = 0; n < _blob.count_[2]; ++n)
				{
					*dst = (float)*src;
					++dst;
					++src;
				}

				return true;
			}


		}
	}
}
