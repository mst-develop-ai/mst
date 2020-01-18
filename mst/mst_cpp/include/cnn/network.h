#pragma once


/* include */
#include "../../include/cnn/blob.h"
#include "../../include/cnn/layer/base_layer.h"

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


			/* variable */
			std::vector<mst::cnn::layer::BaseLayer*> layers_;
			std::map<std::string, int> layer_name_dic_;

			std::vector<mst::cnn::Blob*> layer_blobs_;
			std::map<std::string, int> layer_blob_name_dic_;


		private:

			bool AppendBlankInputLayer(const std::vector<std::string>& _config);
			bool AppendConvolutionLayer(const std::vector<std::string>& _config);
			bool AppendReLULayer(const std::vector<std::string>& _config);

		};

	}
}
