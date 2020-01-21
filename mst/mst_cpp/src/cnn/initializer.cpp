/* include */
#include "../../include/cnn/initializer.h"

#include "../../include/cnn/utility.h"

#include <cmath>
#include <random>


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace initializer
		{

			//	initialize weights
			bool InitializeWeights(std::vector<std::string> _config, int _unit_input, int _unit_output, double* _weights, int _weights_size)
			{
				int n;
				bool bret;

				std::vector<std::string> config_keys;
				std::vector<std::vector<std::string>> config_values;

				char* endptr;

				std::string type;
				double value;


				//	parse config
				bret = mst::cnn::utility::ParseConfigStrings(_config, config_keys, config_values);
				if (!bret)	return false;

				for (n = 0; n < config_keys.size(); ++n)
				{
					std::string& key = config_keys[n];
					std::vector<std::string>& values = config_values[n];

					if (values.size() <= 0)	return false;

					if (_stricmp(key.c_str(), "type") == 0)
					{
						type = values[0];
					}
					else if (_stricmp(key.c_str(), "value") == 0)
					{
						value = strtod(values[0].c_str(), &endptr);
						if (*endptr != '\0')	return false;
					}
				}


				//	initialize
				if (_stricmp(type.c_str(), "constant") == 0)
				{
					bret = Constant(_weights, _weights_size, value);
					if (!bret)	return false;
				}
				else if (_stricmp(type.c_str(), "zeros") == 0)
				{
					bret = Zeros(_weights, _weights_size);
					if (!bret)	return false;
				}
				else if (_stricmp(type.c_str(), "ones") == 0)
				{
					bret = Ones(_weights, _weights_size);
					if (!bret)	return false;
				}
				else if (_stricmp(type.c_str(), "he_normal") == 0)
				{
					bret = HeNormal(_weights, _weights_size, _unit_input);
					if (!bret)	return false;
				}
				else if (_stricmp(type.c_str(), "glorot_normal") == 0)
				{
					bret = GlorotNormal(_weights, _weights_size, _unit_input, _unit_output);
					if (!bret)	return false;
				}
				else
				{
					return false;
				}

				return true;
			}


			//	constant
			bool Constant(double* _weights, int _weights_size, double _value)
			{
				int n;

				for (n = 0; n < _weights_size; ++n)
				{
					*(_weights + n) = _value;
				}

				return true;
			}

			//	zeros
			bool Zeros(double* _weights, int _weights_size)
			{
				int n;

				for (n = 0; n < _weights_size; ++n)
				{
					*(_weights + n) = 0.0;
				}

				return true;
			}

			//	ones
			bool Ones(double* _weights, int _weights_size)
			{
				int n;

				for (n = 0; n < _weights_size; ++n)
				{
					*(_weights + n) = 1.0;
				}

				return true;
			}


			//	he normal
			bool HeNormal(double* _weights, int _weights_size, int unit_input)
			{
				int n;

				std::random_device rand;
				std::mt19937 mt;
				std::normal_distribution<double> dist;

				mt = std::mt19937(rand());
				dist = std::normal_distribution<double>(0.0, std::sqrt(2.0 / unit_input));
				for (n = 0; n < _weights_size; ++n)
				{
					*(_weights + n) = dist(mt);
				}

				return true;
			}


			//	glorot normal
			bool GlorotNormal(double* _weights, int _weights_size, int unit_input, int unit_output)
			{
				int n;

				std::random_device rand;
				std::mt19937 mt;
				std::normal_distribution<double> dist;

				mt = std::mt19937(rand());
				dist = std::normal_distribution<double>(0.0, std::sqrt(2.0 / (unit_input + unit_output)));
				for (n = 0; n < _weights_size; ++n)
				{
					*(_weights + n) = dist(mt);
				}

				return true;
			}

		}
	}
}
