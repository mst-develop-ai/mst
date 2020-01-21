/* include */
#include "../../include/windows/windows.h"

#include <windows.h>
#include <shlwapi.h>


/* namespace */
namespace mst
{
	namespace windows
	{

		//	path exists
		bool PathExists(const char* _path)
		{
			int iret;

			iret = ::PathFileExists(_path);
			if (iret != TRUE)	return false;

			return true;
		}


		//	path exists
		bool PathExists(const std::string& _path)
		{
			return PathExists(_path.c_str());
		}

	}
}
