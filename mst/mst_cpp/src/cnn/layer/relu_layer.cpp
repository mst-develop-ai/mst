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
				: layer_param_()
			{
				type_ = "ReLULayer";
			}


			//	destructor
			ReLULayer::~ReLULayer()
			{
			}


			//	initialize
			bool ReLULayer::Initialize(ReLULayerParam& _param)
			{
				bool bret;

				//	input check
				bret = _param.CheckParam();
				if (!bret)	return false;

				//	set parameter
				layer_param_ = _param;

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
			bool ReLULayer::Forward()
			{
				int n;
				double* src;
				double* dst;

				src = input_blobs_[0]->data_;
				dst = output_blobs_[0]->data_;
				for (n = 0; n < input_blobs_[0]->count_[0]; ++n)
				{
					if ((*src) < 0)
					{
						(*dst) = layer_param_.negative_slope_ * (*src);
					}
					else
					{
						(*dst) = (*src);
					}

					++src;
					++dst;
				}

				return true;
			}


			//	backward
			bool ReLULayer::Backward()
			{
				int n;
				double* input_diff;
				double* output_data;
				double* output_diff;

				input_diff = input_blobs_[0]->diff_;
				output_data = output_blobs_[0]->data_;
				output_diff = output_blobs_[0]->diff_;
				for (n = 0; n < output_blobs_[0]->count_[0]; ++n)
				{
					if ((*output_data) < 0)
					{
						*input_diff = layer_param_.negative_slope_ * (*output_diff);
					}
					else
					{
						*input_diff = 1.0 * (*output_diff);
					}

					++input_diff;
					++output_data;
					++output_diff;
				}

				return true;
			}


			//	check input blobs
			bool ReLULayer::CheckInputBlobs(const std::vector<mst::cnn::Blob*>& _blobs)
			{
				if (_blobs.size() != 1)	return false;
				if (_blobs[0] == nullptr)	return false;
				if (_blobs[0]->shape_.size() < 2)	return false;

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
				input_blobs_ = _blobs;

				return true;
			}


			//	reset output blobs
			bool ReLULayer::ResetOutputBlobs(const std::vector<mst::cnn::Blob*>& _blobs)
			{
				bool bret;


				//	reset output blobs
				output_blobs_ = _blobs;

				//	reshape output blobs
				bret = output_blobs_[0]->Reshape(input_blobs_[0]->shape_);
				if (!bret)	return false;

				return true;
			}

		}
	}
}
