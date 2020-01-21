#pragma once

/* include */
#include <vector>
#include <string>


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace initializer
		{

			bool InitializeWeights(std::vector<std::string> _config, int _unit_input, int _unit_output, double* _weights, int _weights_size);

			bool Constant(double* _weights, int _weights_size, double _value);
			bool Zeros(double* _weights, int _weights_size);
			bool Ones(double* _weights, int _weights_size);

			bool HeNormal(double* _weights, int _weights_size, int unit_input);
			bool GlorotNormal(double* _weights, int _weights_size, int unit_input, int unit_output);

		}
	}
}
