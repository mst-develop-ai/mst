#pragma once

/* include */
#include <string>
#include <vector>


/* namespace */
namespace mst
{
	namespace cnn
	{

		/* blob */
		class Blob
		{

		public:

			/* variable */
			std::string name_;

			int dim_;
			std::vector<int> shape_;
			std::vector<int> count_;

			bool no_diff_;

			double* data_;
			int data_mem_size_;

			double* diff_;
			int diff_mem_size_;


			/* function */
			Blob();
			~Blob();

			Blob(const Blob& _obj) = delete;
			Blob& operator=(const Blob& _obj) = delete;

			bool Reshape(const std::vector<int>& _shape, bool _no_diff = false);


		private:

			bool AllocateMemory();
			void ReleaseMemory();

		};

	}
}
