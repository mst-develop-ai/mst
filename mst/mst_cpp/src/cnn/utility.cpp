/* include */
#include "../../include/cnn/utility.h"


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace utility
		{

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
