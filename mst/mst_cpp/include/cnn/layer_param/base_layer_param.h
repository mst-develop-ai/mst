#pragma once

/* include */
#include <vector>
#include <string>


/* namespace */
namespace mst
{
	namespace cnn
	{
		namespace layer
		{

			/* BaseLayerParam */
			class BaseLayerParam
			{

			public:

				/* function */
				BaseLayerParam();
				virtual ~BaseLayerParam();

				virtual bool CheckParam() = 0;
				virtual bool ParseConfigStrings(const std::vector<std::string> _config) = 0;


				/* parameter */
				std::string name_;

			};

		}
	}
}
