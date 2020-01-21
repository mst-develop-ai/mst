#pragma once

/* include */
#include <string>
#include <vector>


/* namespace */
namespace mst
{
	namespace string
	{

		bool DeleteLastNewLineCode(std::string& _str);
		bool DeleteFrontLastSpaceCode(std::string& _str);

		bool SplitDelimiter(const char* _str, char _delimiter, std::vector<std::string>& _dst);
		bool SplitDelimiter(const std::string& _str, char _delimiter, std::vector<std::string>& _dst);

	}
}
