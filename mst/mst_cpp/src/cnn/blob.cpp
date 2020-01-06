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
			, data_(nullptr)
			, data_mem_size_(0)
		{
		}


		//	destructor
		Blob::~Blob()
		{
			ReleaseMemory();
		}


		//	reshape
		bool Blob::Reshape(const std::vector<int>& _shape)
		{
			bool bret;


			//	set parameter
			shape_ = _shape;
			dim_ = (int)shape_.size();


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
			size = sizeof(double);
			for each (int num in shape_)
			{
				size *= num;
			}

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
		}

	}
}
