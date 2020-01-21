#pragma once


/* include */
#include "../../include/cnn/blob.h"
#include "../../include/cnn/layer/base_layer.h"
#include "../../include/cnn/layer_param/base_layer_param.h"

#include <map>
#include <string>
#include <vector>


/* namespace */
namespace mst
{
	namespace cnn
	{

		class Network
		{

		public:

			/* function */
			Network();
			~Network();

			Network(const Network& _obj) = delete;
			Network& operator=(const Network& _obj) = delete;


			bool Initialize();
			void Release();

			bool ParseNetworkConfig(const char* _config_path);
			bool ParseLayerConfig(const std::vector<std::string>& _config_keys, const std::vector<std::vector<std::string>>& _config_values);

			bool Reshape();

			bool Forward();
			bool Backward();


			/* variable */
			std::vector<mst::cnn::layer::BaseLayer*> layers_;
			std::map<std::string, int> layer_name_dic_;

			std::vector<mst::cnn::Blob*> layer_blobs_;
			std::map<std::string, int> layer_blob_name_dic_;


			std::vector<std::vector<std::string>> layer_input_blob_name_;
			std::vector<std::vector<std::string>> layer_output_blob_name_;
			std::vector<std::vector<mst::cnn::Blob*>> layer_input_blobs_;
			std::vector<std::vector<mst::cnn::Blob*>> layer_output_blobs_;

			std::map<std::string, int> output_blob_name_;
			std::vector<mst::cnn::Blob*> output_blobs_;

		private:

			bool AppendLayer(const std::vector<std::string>& _config_keys, const std::vector<std::vector<std::string>>& _config_values, mst::cnn::layer::BaseLayer* _layer, mst::cnn::layer::BaseLayerParam* _param);
			bool AppendBlankInputLayer(const std::vector<std::string>& _config);
			bool AppendConvolutionLayer(const std::vector<std::string>& _config);
			bool AppendReLULayer(const std::vector<std::string>& _config);

			bool InitializeLayerBlobs();
			bool InitializeLayerWeights();

		};

	}
}
