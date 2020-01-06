/* include */
#include "../../../include/cnn/layer/relu_layer.h"


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace layer
		{

			//	constructor
			ReLULayer::ReLULayer()
				: negative_slope_(0)
				, input_blobs_()
				, output_blobs_()
				, total_count_(0)
			{
			}


			//	destructor
			ReLULayer::~ReLULayer()
			{
			}


			//	initialize
			bool ReLULayer::Initialize(double _negative_slope)
			{
				//	input check
				if (_negative_slope < 0)	return false;

				//	set parameter
				negative_slope_ = _negative_slope;

				return true;
			}


			//	reshape
			bool ReLULayer::Reshape(const std::vector<mst::cnn::Blob*>& _input_blobs, const std::vector<mst::cnn::Blob*>& _output_blobs)
			{
				bool bret;


				//	input check
				bret = CheckInputBlobs(_input_blobs);
				if (!bret)	return false;

				bret = CheckOutputBlobs(_output_blobs);
				if (!bret)	return false;


				//	reset blob
				bret = ResetInputBlobs(_input_blobs);
				if (!bret)	return false;

				bret = ResetOutputBlobs(_output_blobs);
				if (!bret)	return false;

				return true;
			}


			//	forward
			void ReLULayer::Forward()
			{
				int n;
				double* src;
				double* dst;

				src = input_blobs_[0]->data_;
				dst = output_blobs_[0]->data_;
				for (n = 0; n < total_count_; ++n)
				{
					if ((*src) < 0)
					{
						(*dst) = negative_slope_ * (*src);
					}
					else
					{
						(*dst) = (*src);
					}

					++src;
					++dst;
				}
			}


			//	backward
			void ReLULayer::Backward()
			{
			}


			//	check input blobs
			bool ReLULayer::CheckInputBlobs(const std::vector<mst::cnn::Blob*>& _blobs)
			{
				if (_blobs.size() != 1)	return false;

				if (_blobs[0] == nullptr)	return false;

				if (_blobs[0]->shape_.size() <= 0)	return false;

				for each (int size in _blobs[0]->shape_)
				{
					if (size <= 0)	return false;
				}

				return true;
			}


			//	check output blobs
			bool ReLULayer::CheckOutputBlobs(const std::vector<mst::cnn::Blob*>& _blobs)
			{
				if (_blobs.size() != 1)	return false;

				if (_blobs[0] == nullptr)	return false;

				return true;
			}


			//	reset input blobs
			bool ReLULayer::ResetInputBlobs(const std::vector<mst::cnn::Blob*>& _blobs)
			{
				//	reset input blobls
				input_blobs_.clear();
				for each (mst::cnn::Blob* blob in _blobs)
				{
					input_blobs_.push_back(blob);
				}


				//	set input blobs variable
				total_count_ = 1;
				for each (int size in input_blobs_[0]->shape_)
				{
					total_count_ *= size;
				}

				return true;
			}


			//	reset output blobs
			bool ReLULayer::ResetOutputBlobs(const std::vector<mst::cnn::Blob*>& _blobs)
			{
				int n;
				bool bret;
				std::vector<int> shape;


				//	reset output blobs
				output_blobs_.clear();
				for each (mst::cnn::Blob* blob in _blobs)
				{
					output_blobs_.push_back(blob);
				}


				//	check shape
				bret = false;
				if (input_blobs_[0]->shape_.size() != output_blobs_[0]->shape_.size())
				{
					bret = true;
				}
				else
				{
					for (n = 0; n < input_blobs_[0]->shape_.size(); ++n)
					{
						if (input_blobs_[0]->shape_[n] != output_blobs_[0]->shape_[n])
						{
							bret = true;
							break;
						}
					}
				}


				//	reshape output blobs
				if (bret)
				{
					shape.clear();
					for each (int size in output_blobs_[0]->shape_)
					{
						shape.push_back(size);
					}

					bret = output_blobs_[0]->Reshape(shape);
					if (!bret)	return false;
				}

				return true;
			}

		}
	}
}
