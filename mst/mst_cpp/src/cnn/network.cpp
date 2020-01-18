/* include */
#include "../../include/cnn/network.h"
#include "../../include/cnn/utility.h"

#include "../../include/cnn/layer/blank_input_layer.h"
#include "../../include/cnn/layer/convolution_layer.h"
#include "../../include/cnn/layer/relu_layer.h"

#include "../../include/file/file.h"


/* namespace */
namespace mst
{
	namespace cnn
	{

		//	constructor
		Network::Network()
		{

		}


		//	destructor
		Network::~Network()
		{

		}


		//	initialize
		bool Network::Initialize()
		{

			return true;
		}


		//	release
		void Network::Release()
		{
			int n = 0;

			//	layer
			layer_name_dic_.clear();

			for (n = 0; n < layers_.size(); ++n)
			{
				delete layers_[n];
				layers_[n] = nullptr;
			}
			layers_.clear();


			//	blob
			layer_blob_name_dic_.clear();

			for (n = 0; n < layer_blobs_.size(); ++n)
			{
				delete layer_blobs_[n];
				layer_blobs_[n] = nullptr;
			}
			layer_blobs_.clear();
		}


		//	parse network config
		bool Network::ParseNetworkConfig(const char* _config_path)
		{
			bool bret;

			std::vector<std::string> lines;

			std::vector<std::string> config_keys;
			std::vector<std::vector<std::string>> config_values;


			//	read file
			bret = mst::file::ReadFileLines(_config_path, lines);
			if (!bret)	return false;

			//	parse config lines
			bret = mst::cnn::utility::ParseConfigStrings(lines, config_keys, config_values);
			if (!bret)	return false;

			//	parse layer config
			bret = ParseLayerConfig(config_keys, config_values);
			if (!bret)	return false;

			return true;
		}


		//	parse layer config
		bool Network::ParseLayerConfig(const std::vector<std::string>& _config_keys, const std::vector<std::vector<std::string>>& _config_values)
		{
			int n;
			bool bret;


			for (n = 0; n < _config_keys.size(); ++n)
			{
				const std::string& key = _config_keys[n];
				const std::vector<std::string>& values = _config_values[n];
				
				if (_stricmp(key.c_str(), "BlankInput") == 0)
				{
					bret = AppendBlankInputLayer(values);
					if (!bret)	return false;
				}
				else if (_stricmp(key.c_str(), "Convolution") == 0)
				{
					bret = AppendConvolutionLayer(values);
					if (!bret)	return false;
				}
				else if (_stricmp(key.c_str(), "DepthwiseConvolution") == 0)
				{

				}
				else if (_stricmp(key.c_str(), "MaxPooling") == 0)
				{

				}
				else if (_stricmp(key.c_str(), "AveragePooling") == 0)
				{

				}
				else if (_stricmp(key.c_str(), "GlobalAveragePooling") == 0)
				{

				}
				else if (_stricmp(key.c_str(), "Dense") == 0)
				{

				}
				else if (_stricmp(key.c_str(), "BatchNorm") == 0)
				{

				}
				else if (_stricmp(key.c_str(), "Concat") == 0)
				{

				}
				else if (_stricmp(key.c_str(), "Add") == 0)
				{

				}
				else if (_stricmp(key.c_str(), "Multiply") == 0)
				{

				}
				else if (_stricmp(key.c_str(), "ReLU") == 0)
				{
					bret = AppendReLULayer(values);
					if (!bret)	return false;
				}
				else if (_stricmp(key.c_str(), "Sigmoid") == 0)
				{

				}
				else
				{
					return false;
				}
			}

			return true;
		}


		//	append blank input layer
		bool Network::AppendBlankInputLayer(const std::vector<std::string>& _config)
		{
			int n;
			bool bret;

			std::vector<std::string> config_keys;
			std::vector<std::vector<std::string>> config_values;

			mst::cnn::layer::BlankInputLayer* layer;
			mst::cnn::layer::BlankInputLayerParam param;

			mst::cnn::Blob* blob;
			std::vector<mst::cnn::Blob*> output_blobs;

			bool flag_name;


			//	setup
			layer = new mst::cnn::layer::BlankInputLayer();
			layers_.push_back(layer);

			bret = mst::cnn::utility::ParseConfigStrings(_config, config_keys, config_values);
			if (!bret)	return false;

			flag_name = false;
			for (n = 0; n < config_keys.size(); ++n)
			{
				std::string& key = config_keys[n];
				std::vector<std::string>& values = config_values[n];

				if (values.size() <= 0)	return false;

				if (_stricmp(key.c_str(), "name") == 0)
				{
					if (flag_name)	return false;
					if (layer_name_dic_.count(values[0]) != 0)	return false;

					layer->layer_param_.name_ = values[0];
					layer_name_dic_[layer->layer_param_.name_] = (int)layers_.size() - 1;

					flag_name = true;
				}
				else if (_stricmp(key.c_str(), "output") == 0)
				{
					if (layer_blob_name_dic_.count(values[0]) == 0)
					{
						blob = new mst::cnn::Blob();
						layer_blobs_.push_back(blob);

						blob->name_ = values[0];
						layer_blob_name_dic_[blob->name_] = (int)layer_blobs_.size() - 1;
					}

					output_blobs.push_back(layer_blobs_[layer_blob_name_dic_[values[0]]]);
				}
				else if (_stricmp(key.c_str(), "param") == 0)
				{
					bret = param.ParseConfigStrings(values);
					if (!bret)	return false;
				}
			}

			if (!flag_name)	return false;


			//	initialize
			bret = layer->Initialize(param);
			if (!bret)	return false;

			bret = layer->Reshape({}, output_blobs);
			if (!bret)	return false;

			return true;
		}


		//	append convolution layer
		bool Network::AppendConvolutionLayer(const std::vector<std::string>& _config)
		{
			int n;
			bool bret;

			std::vector<std::string> config_keys;
			std::vector<std::vector<std::string>> config_values;

			mst::cnn::layer::ConvolutionLayer* layer;
			mst::cnn::layer::ConvolutionLayerParam param;

			mst::cnn::Blob* blob;
			std::vector<mst::cnn::Blob*> input_blobs;
			std::vector<mst::cnn::Blob*> output_blobs;

			bool flag_name;


			//	setup
			layer = new mst::cnn::layer::ConvolutionLayer();
			layers_.push_back(layer);

			bret = mst::cnn::utility::ParseConfigStrings(_config, config_keys, config_values);
			if (!bret)	return false;

			flag_name = false;
			for (n = 0; n < config_keys.size(); ++n)
			{
				std::string& key = config_keys[n];
				std::vector<std::string>& values = config_values[n];

				if (values.size() <= 0)	return false;

				if (_stricmp(key.c_str(), "name") == 0)
				{
					if (flag_name)	return false;
					if (layer_name_dic_.count(values[0]) != 0)	return false;

					layer->layer_param_.name_ = values[0];
					layer_name_dic_[layer->layer_param_.name_] = (int)layers_.size() - 1;

					flag_name = true;
				}
				else if (_stricmp(key.c_str(), "input") == 0)
				{
					if (layer_blob_name_dic_.count(values[0]) == 0)
					{
						blob = new mst::cnn::Blob();
						layer_blobs_.push_back(blob);

						blob->name_ = values[0];
						layer_blob_name_dic_[blob->name_] = (int)layer_blobs_.size() - 1;
					}

					input_blobs.push_back(layer_blobs_[layer_blob_name_dic_[values[0]]]);
				}
				else if (_stricmp(key.c_str(), "output") == 0)
				{
					if (layer_blob_name_dic_.count(values[0]) == 0)
					{
						blob = new mst::cnn::Blob();
						layer_blobs_.push_back(blob);

						blob->name_ = values[0];
						layer_blob_name_dic_[blob->name_] = (int)layer_blobs_.size() - 1;
					}

					output_blobs.push_back(layer_blobs_[layer_blob_name_dic_[values[0]]]);
				}
				else if (_stricmp(key.c_str(), "param") == 0)
				{
					bret = param.ParseConfigStrings(values);
					if (!bret)	return false;
				}
			}

			if (!flag_name)	return false;


			//	initialize
			bret = layer->Initialize(param);
			if (!bret)	return false;

			bret = layer->Reshape(input_blobs, output_blobs);
			if (!bret)	return false;

			return true;
		}


		//	append relu layer
		bool Network::AppendReLULayer(const std::vector<std::string>& _config)
		{
			int n;
			bool bret;

			std::vector<std::string> config_keys;
			std::vector<std::vector<std::string>> config_values;

			mst::cnn::layer::ReLULayer* layer;
			mst::cnn::layer::ReLULayerParam param;

			mst::cnn::Blob* blob;
			std::vector<mst::cnn::Blob*> input_blobs;
			std::vector<mst::cnn::Blob*> output_blobs;

			bool flag_name;


			//	setup
			layer = new mst::cnn::layer::ReLULayer();
			layers_.push_back(layer);

			bret = mst::cnn::utility::ParseConfigStrings(_config, config_keys, config_values);
			if (!bret)	return false;

			flag_name = false;
			for (n = 0; n < config_keys.size(); ++n)
			{
				std::string& key = config_keys[n];
				std::vector<std::string>& values = config_values[n];

				if (values.size() <= 0)	return false;

				if (_stricmp(key.c_str(), "name") == 0)
				{
					if (flag_name)	return false;
					if (layer_name_dic_.count(values[0]) != 0)	return false;

					layer->layer_param_.name_ = values[0];
					layer_name_dic_[layer->layer_param_.name_] = (int)layers_.size() - 1;

					flag_name = true;
				}
				else if (_stricmp(key.c_str(), "input") == 0)
				{
					if (layer_blob_name_dic_.count(values[0]) == 0)
					{
						blob = new mst::cnn::Blob();
						layer_blobs_.push_back(blob);

						blob->name_ = values[0];
						layer_blob_name_dic_[blob->name_] = (int)layer_blobs_.size() - 1;
					}

					input_blobs.push_back(layer_blobs_[layer_blob_name_dic_[values[0]]]);
				}
				else if (_stricmp(key.c_str(), "output") == 0)
				{
					if (layer_blob_name_dic_.count(values[0]) == 0)
					{
						blob = new mst::cnn::Blob();
						layer_blobs_.push_back(blob);

						blob->name_ = values[0];
						layer_blob_name_dic_[blob->name_] = (int)layer_blobs_.size() - 1;
					}

					output_blobs.push_back(layer_blobs_[layer_blob_name_dic_[values[0]]]);
				}
				else if (_stricmp(key.c_str(), "param") == 0)
				{
					bret = param.ParseConfigStrings(values);
					if (!bret)	return false;
				}
			}

			if (!flag_name)	return false;


			//	initialize
			bret = layer->Initialize(param);
			if (!bret)	return false;

			bret = layer->Reshape(input_blobs, output_blobs);
			if (!bret)	return false;

			return true;
		}
	}
}
