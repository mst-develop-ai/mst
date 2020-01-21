/* include */
#include "../../../include/cnn/layer/image_list_layer.h"


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace layer
		{

			//	constructor
			ImageListLayer::ImageListLayer()
				: layer_param_()
				, data_pos_(0)
				, data_access_list_()
			{
				type_ = "ImageListLayer";
			}


			//	destructor
			ImageListLayer::~ImageListLayer()
			{
			}


			//	initialize
			bool ImageListLayer::Initialize(ImageListLayerParam& _param)
			{
				bool bret;

				//	input check
				bret = _param.CheckParam();
				if (!bret)	return false;

				//	set parameter
				layer_param_ = _param;


				//	read data

				return true;
			}


			//	reshape
			bool ImageListLayer::Reshape(const std::vector<mst::cnn::Blob*>& _input_blobs, const std::vector<mst::cnn::Blob*>& _output_blobs)
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
			bool ImageListLayer::Forward()
			{

				return true;
			}


			//	backward
			bool ImageListLayer::Backward()
			{

				return true;
			}


			//	check input blobs
			bool ImageListLayer::CheckInputBlobs(const std::vector<mst::cnn::Blob*>& _blobs)
			{
				if (_blobs.size() != 0)	return false;

				return true;
			}


			//	check output blobs
			bool ImageListLayer::CheckOutputBlobs(const std::vector<mst::cnn::Blob*>& _blobs)
			{
				int n;

				if (_blobs.size() != layer_param_.data_elem_count_)	return false;
				for (n = 0; n < _blobs.size(); ++n)
				{
					if (_blobs[n] == nullptr)	return false;
				}

				return true;
			}


			//	reset input blobs
			bool ImageListLayer::ResetInputBlobs(const std::vector<mst::cnn::Blob*>& _blobs)
			{
				//	reset input blobls
				input_blobs_ = _blobs;

				return true;
			}


			//	reset output blobs
			bool ImageListLayer::ResetOutputBlobs(const std::vector<mst::cnn::Blob*>& _blobs)
			{
				int n;
				bool bret;
				std::vector<int> shape;


				//	reset output blobs
				output_blobs_ = _blobs;


				//	reshape output blobs
				shape = {
					layer_param_.batch_size_,
					layer_param_.channels_,
					layer_param_.rows_,
					layer_param_.cols_
				};

				bret = output_blobs_[0]->Reshape(shape);
				if (!bret)	return false;


				//	reshape output blobs
				if (layer_param_.single_target_data_)
				{
					if (layer_param_.target_data_dim_ == 2)
					{
						shape = {
							layer_param_.batch_size_,
							layer_param_.data_elem_count_ - 1
						};
					}
					else if (layer_param_.target_data_dim_ == 3)
					{
						shape = {
							layer_param_.batch_size_,
							layer_param_.data_elem_count_ - 1,
							layer_param_.target_rows_
						};
					}
					else if (layer_param_.target_data_dim_ == 4)
					{
						shape = {
							layer_param_.batch_size_,
							layer_param_.data_elem_count_ - 1,
							layer_param_.target_rows_,
							layer_param_.target_cols_
						};
					}
				}
				else
				{
					if (layer_param_.target_data_dim_ == 2)
					{
						shape = {
							layer_param_.batch_size_,
							1
						};
					}
					else if (layer_param_.target_data_dim_ == 3)
					{
						shape = {
							layer_param_.batch_size_,
							1,
							layer_param_.target_rows_
						};
					}
					else if (layer_param_.target_data_dim_ == 4)
					{
						shape = {
							layer_param_.batch_size_,
							1,
							layer_param_.target_rows_,
							layer_param_.target_cols_
						};
					}
				}

				for (n = 1; n < layer_param_.data_elem_count_; ++n)
				{
					bret = output_blobs_[n]->Reshape(shape);
					if (!bret)	return false;
				}

				return true;
			}

		}
	}
}
