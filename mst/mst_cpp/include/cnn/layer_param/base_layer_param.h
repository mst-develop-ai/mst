#pragma once

/* include */
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


				/* parameter */
				std::string name_;

			};

		}
	}
}
