/* include */
#include "../../../include/cnn/layer/blank_input_layer.h"


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace layer
		{

			//	constructor
			BlankInputLayer::BlankInputLayer()
				: layer_param_()
			{
				type_ = "BlankInput";

				InitializeImageDataLookupTable();
			}


			//	destructor
			BlankInputLayer::~BlankInputLayer()
			{
			}


			//	initialize
			bool BlankInputLayer::Initialize(BlankInputLayerParam& _param)
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
			bool BlankInputLayer::Reshape(const std::vector<mst::cnn::Blob*>& _input_blobs, const std::vector<mst::cnn::Blob*>& _output_blobs)
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
			void BlankInputLayer::Forward()
			{
			}


			//	backward
			void BlankInputLayer::Backward()
			{
			}


			//	set image data lookup table
			bool BlankInputLayer::SetImageDataLookupTable(double _scale, double _bias)
			{
				int n;

				for (n = 0; n < 256; ++n)
				{
					image_lookup_table_[n] = (_scale * n) + _bias;
				}

				return true;
			}


			//	set output blob image data
			bool BlankInputLayer::SetOutputBlobImageData(int _cols, int _rows, int _channels, const unsigned char* _data)
			{
				int n;

				bool bret;
				std::vector<int> shape;

				double* dst;


				//	check input
				if ((_cols < 0) || (_rows < 0) || (_channels < 0) || (_data == nullptr))	return false;


				//	reshape
				if ((_cols != layer_param_.cols_) || (_rows != layer_param_.rows_) || (_channels != layer_param_.channels_))
				{
					layer_param_.cols_ = _cols;
					layer_param_.rows_ = _rows;
					layer_param_.channels_ = _channels;

					shape =
					{
						layer_param_.batch_size_,
						layer_param_.channels_,
						layer_param_.rows_,
						layer_param_.cols_
					};

					bret = output_blobs_[0]->Reshape(shape);
					if (!bret)	return false;
				}


				//	set data
				dst = output_blobs_[0]->data_;
				for (n = 0; n < output_blobs_[0]->count_[0]; ++n)
				{
					*dst = *(image_lookup_table_ + (*_data));

					++dst;
					++_data;
				}

				return true;
			}


			//	check input blobs
			bool BlankInputLayer::CheckInputBlobs(const std::vector<mst::cnn::Blob*>& _blobs)
			{
				if (_blobs.size() != 0)	return false;

				return true;
			}


			//	check output blobs
			bool BlankInputLayer::CheckOutputBlobs(const std::vector<mst::cnn::Blob*>& _blobs)
			{
				if (_blobs.size() != 1)	return false;
				if (_blobs[0] == nullptr)	return false;

				return true;
			}


			//	reset input blobs
			bool BlankInputLayer::ResetInputBlobs(const std::vector<mst::cnn::Blob*>& _blobs)
			{				
				return true;
			}


			//	reset output blobs
			bool BlankInputLayer::ResetOutputBlobs(const std::vector<mst::cnn::Blob*>& _blobs)
			{
				bool bret;
				std::vector<int> shape;


				//	reset output blobs
				output_blobs_ = _blobs;

				//	reshape output blobs
				shape =
				{
					layer_param_.batch_size_,
					layer_param_.channels_,
					layer_param_.rows_,
					layer_param_.cols_
				};

				bret = output_blobs_[0]->Reshape(shape);
				if (!bret)	return false;


				return true;
			}


			//	initialize image data lookup table
			void BlankInputLayer::InitializeImageDataLookupTable()
			{
				int n;

				for (n = 0; n < 256; ++n)
				{
					image_lookup_table_[n] = n;
				}
			}

		}
	}
}
