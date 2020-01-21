/* include */
#include "../../include/cnn/blob.h"

#include <stdlib.h>
#include <memory.h>


/* namespace */
namespace mst
{
	namespace cnn
	{

		//	constructor
		Blob::Blob()
			: name_("")
			, dim_(0)
			, shape_()
			, count_()
			, no_diff_(false)
			, data_(nullptr)
			, data_mem_size_(0)
			, diff_(nullptr)
			, diff_mem_size_(0)
		{
		}


		//	destructor
		Blob::~Blob()
		{
			ReleaseMemory();
		}


		//	reshape
		bool Blob::Reshape(const std::vector<int>& _shape, bool _no_diff)
		{
			int n;
			bool bret;


			//	check input
			if (_shape.size() == 0)	return false;

			for each (int num in _shape)
			{
				if (num <= 0)	return false;
			}


			//	set parameter
			shape_ = _shape;
			dim_ = (int)shape_.size();

			no_diff_ = _no_diff;


			//	set variable
			count_ = shape_;
			for (n = dim_ - 2; n >= 0; --n)
			{
				count_[n] *= count_[n + 1];
			}


			//	allocate memory
			bret = AllocateMemory();
			if (!bret)	return false;

			return true;
		}


		//	allocate memory
		bool Blob::AllocateMemory()
		{
			int size;

			//	allocate
			size = count_[0] * sizeof(double);

			if (size > data_mem_size_)
			{
				if (data_ != nullptr)
				{
					free(data_);
					data_ = nullptr;

					data_mem_size_ = 0;
				}

				data_mem_size_ = size;
				data_ = (double*)malloc(data_mem_size_);
				if (data_ == nullptr)	return false;
			}

			//	initialize
			memset(data_, 0, data_mem_size_);


			//	allocate
			if (no_diff_)
			{
				if (diff_ != nullptr)
				{
					free(diff_);
					data_ = nullptr;

					diff_mem_size_ = 0;
				}
			}
			else
			{
				if (size > diff_mem_size_)
				{
					if (diff_ != nullptr)
					{
						free(diff_);
						data_ = nullptr;

						diff_mem_size_ = 0;
					}

					diff_mem_size_ = size;
					diff_ = (double*)malloc(diff_mem_size_);
					if (diff_ == nullptr)	return false;
				}

				//	initialize
				memset(diff_, 0, diff_mem_size_);
			}

			return true;
		}


		//	release memory
		void Blob::ReleaseMemory()
		{
			if (data_ != nullptr)
			{
				free(data_);
				data_ = nullptr;

				data_mem_size_ = 0;
			}

			if (diff_ != nullptr)
			{
				free(diff_);
				diff_ = nullptr;

				diff_mem_size_ = 0;
			}
		}

	}
}
