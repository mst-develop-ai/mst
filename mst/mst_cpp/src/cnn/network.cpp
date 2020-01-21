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


			//
			layer_input_blob_name_.clear();
			layer_output_blob_name_.clear();
			layer_input_blobs_.clear();
			layer_output_blobs_.clear();

			output_blob_name_.clear();
			output_blobs_.clear();
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

			//	set layer input/output blob
			bret = InitializeLayerBlobs();
			if (!bret)	return false;

			//	reshape network
			bret = Reshape();
			if (!bret)	return false;

			//	weights initialize
			bret = InitializeLayerWeights();
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
				else if (_stricmp(key.c_str(), "Clone") == 0)
				{

				}
				else if (_stricmp(key.c_str(), "Split") == 0)
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


		//	reshape
		bool Network::Reshape()
		{
			int n;
			bool bret;

			for (n = 0; n < layers_.size(); ++n)
			{
				bret = layers_[n]->Reshape(layer_input_blobs_[n], layer_output_blobs_[n]);
				if (!bret)	return false;
			}

			return true;
		}


		//	forward
		bool Network::Forward()
		{
			int n;
			bool bret;

			for (n = 0; n < layers_.size(); ++n)
			{
				bret = layers_[n]->Forward();
				if (!bret)	return false;
			}

			return true;
		}


		//	backward
		bool Network::Backward()
		{
			int n;
			bool bret;

			for (n = 0; n < layers_.size(); ++n)
			{
				bret = layers_[n]->Backward();
				if (!bret)	return false;
			}

			return true;
		}


		//	append layer
		bool Network::AppendLayer(const std::vector<std::string>& _config_keys, const std::vector<std::vector<std::string>>& _config_values, mst::cnn::layer::BaseLayer* _layer, mst::cnn::layer::BaseLayerParam* _param)
		{
			int n;
			bool bret;

			int layer_index;

			bool flag_name;
			mst::cnn::Blob* blob;


			//	append layer
			layer_index = (int)layers_.size();

			layers_.push_back(_layer);
			layer_input_blob_name_.push_back(std::vector<std::string>());
			layer_output_blob_name_.push_back(std::vector<std::string>());
			layer_input_blobs_.push_back(std::vector<mst::cnn::Blob*>());
			layer_output_blobs_.push_back(std::vector<mst::cnn::Blob*>());


			flag_name = false;
			for (n = 0; n < _config_keys.size(); ++n)
			{
				const std::string& key = _config_keys[n];
				const std::vector<std::string>& values = _config_values[n];

				if (values.size() <= 0)	return false;

				if (_stricmp(key.c_str(), "name") == 0)
				{
					//	check
					if (flag_name)	return false;
					if (layer_name_dic_.count(values[0]) != 0)	return false;

					//	set info
					_param->name_ = values[0];
					layer_name_dic_[_param->name_] = layer_index;

					//	set flag
					flag_name = true;
				}
				else if (_stricmp(key.c_str(), "input") == 0)
				{
					//	check exist
					if (layer_blob_name_dic_.count(values[0]) == 0)	return false;

					//	set layer input blob name
					layer_input_blob_name_[layer_index].push_back(values[0]);

					//	delete output blob
					if (output_blob_name_.count(values[0]) != 0)
					{
						output_blob_name_.erase(values[0]);
					}
				}
				else if (_stricmp(key.c_str(), "output") == 0)
				{
					//	add new blob info
					if (layer_blob_name_dic_.count(values[0]) == 0)
					{
						blob = new mst::cnn::Blob();
						layer_blobs_.push_back(blob);

						blob->name_ = values[0];
						layer_blob_name_dic_[blob->name_] = (int)layer_blobs_.size() - 1;
					}

					//	set layer output blob name
					layer_output_blob_name_[layer_index].push_back(values[0]);

					//	set output blob
					if (output_blob_name_.count(values[0]) != 0)	return false;
					output_blob_name_[values[0]] = layer_blob_name_dic_[values[0]];
				}
				else if (_stricmp(key.c_str(), "param") == 0)
				{
					//	parse param
					bret = _param->ParseConfigStrings(values);
					if (!bret)	return false;
				}
			}

			//	check name flag
			if (!flag_name)	return false;

			return true;
		}


		//	append blank input layer
		bool Network::AppendBlankInputLayer(const std::vector<std::string>& _config)
		{
			bool bret;

			std::vector<std::string> config_keys;
			std::vector<std::vector<std::string>> config_values;

			mst::cnn::layer::BlankInputLayer* layer;
			mst::cnn::layer::BlankInputLayerParam param;


			//	parse
			bret = mst::cnn::utility::ParseConfigStrings(_config, config_keys, config_values);
			if (!bret)	return false;


			//	append
			layer = new mst::cnn::layer::BlankInputLayer();

			bret = AppendLayer(config_keys, config_values, layer, &param);
			if (!bret)	return false;


			//	initialize
			bret = layer->Initialize(param);
			if (!bret)	return false;

			return true;
		}


		//	append convolution layer
		bool Network::AppendConvolutionLayer(const std::vector<std::string>& _config)
		{
			bool bret;

			std::vector<std::string> config_keys;
			std::vector<std::vector<std::string>> config_values;

			mst::cnn::layer::ConvolutionLayer* layer;
			mst::cnn::layer::ConvolutionLayerParam param;


			//	parse
			bret = mst::cnn::utility::ParseConfigStrings(_config, config_keys, config_values);
			if (!bret)	return false;


			//	append
			layer = new mst::cnn::layer::ConvolutionLayer();

			bret = AppendLayer(config_keys, config_values, layer, &param);
			if (!bret)	return false;


			//	initialize
			bret = layer->Initialize(param);
			if (!bret)	return false;

			return true;
		}


		//	append relu layer
		bool Network::AppendReLULayer(const std::vector<std::string>& _config)
		{
			bool bret;

			std::vector<std::string> config_keys;
			std::vector<std::vector<std::string>> config_values;

			mst::cnn::layer::ReLULayer* layer;
			mst::cnn::layer::ReLULayerParam param;


			//	parse
			bret = mst::cnn::utility::ParseConfigStrings(_config, config_keys, config_values);
			if (!bret)	return false;


			//	append
			layer = new mst::cnn::layer::ReLULayer();

			bret = AppendLayer(config_keys, config_values, layer, &param);
			if (!bret)	return false;


			//	initialize
			bret = layer->Initialize(param);
			if (!bret)	return false;

			return true;
		}


		//	initialize layer blobs
		bool Network::InitializeLayerBlobs()
		{
			int n;
			int nn;

			for (n = 0; n < layers_.size(); ++n)
			{
				//	layer input
				for (nn = 0; nn < layer_input_blob_name_[n].size(); ++nn)
				{
					layer_input_blobs_[n].push_back(layer_blobs_[layer_blob_name_dic_[layer_input_blob_name_[n][nn]]]);
				}

				//	layer output
				for (nn = 0; nn < layer_output_blob_name_[n].size(); ++nn)
				{
					layer_output_blobs_[n].push_back(layer_blobs_[layer_blob_name_dic_[layer_output_blob_name_[n][nn]]]);
				}
			}

			//	network output
			for each (std::pair<std::string, int> pair in output_blob_name_)
			{
				output_blobs_.push_back(layer_blobs_[layer_blob_name_dic_[pair.first]]);
			}

			return true;
		}


		//	initialize layer weights
		bool Network::InitializeLayerWeights()
		{
			int n;

			for (n = 0; n < layers_.size(); ++n)
			{
				layers_[n]->InitializeWeights();
			}

			return true;
		}

	}
}
