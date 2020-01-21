#pragma once

/* pragma */
#pragma comment(lib, "shlwapi.lib")


/* include */
#include <string>


/* namespace */
namespace mst
{
	namespace windows
	{

		bool PathExists(const char* _path);
		bool PathExists(const std::string& _path);

	}
}
