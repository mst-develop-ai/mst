/* include */
#include "../../include/file/file.h"
#include "../../include/string/string.h"

#include <fstream>


/* namespace */
namespace mst
{
	namespace file
	{

		//	read file lines
		bool ReadFileLines(const char* _path, std::vector<std::string>& _lines)
		{
			bool bret;

			std::string line;
			std::ifstream ifs;


			//	init
			_lines.clear();


			//	open
			ifs.open(_path);

			bret = ifs.fail();
			if (bret)	return false;


			//	read
			while (std::getline(ifs, line))
			{
				mst::string::DeleteLastNewLineCode(line);

				_lines.push_back(line);
			}


			//	close
			ifs.close();

			return true;
		}



	}
}
