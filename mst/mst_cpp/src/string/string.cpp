/* include */
#include "../../include/string/string.h"


/* namespace */
namespace mst
{
	namespace string
	{

		//	delete last new line code
		bool DeleteLastNewLineCode(std::string& _str)
		{
			bool bret;
			size_t len;

			bret = false;
			len = _str.length();

			if ((len >= 2) && (_str[len - 2] == '\r') && (_str[len - 1] == '\n'))
			{
				bret = true;
				_str.pop_back();
				_str.pop_back();
			}
			else if ((len >= 1) && (_str[len - 1] == '\n'))
			{
				bret = true;
				_str.pop_back();
			}

			return bret;
		}


		//	delete front last space code
		bool DeleteFrontLastSpaceCode(std::string& _str)
		{
			const char* data;

			const char* front;
			const char* last;


			//	check blanck
			if (_str.length() == 0)	return true;


			//	set front
			data = _str.c_str();
			while (true)
			{
				if ((*data) == '\0')
				{
					front = data;
					break;
				}
				else if (((*data) != ' ') && ((*data) != 'Å@') && ((*data) != '\t'))
				{
					front = data;
					break;
				}

				++data;
			}


			//	set last
			data = _str.c_str() + _str.length();
			while (true)
			{
				if (data == front)
				{
					last = data;
					break;
				}
				else if (((*data) != ' ') && ((*data) != 'Å@') && ((*data) != '\t'))
				{
					last = data + 1;
					break;
				}

				--data;
			}


			//	set sub string
			_str = std::string(front, last);

			return true;
		}

	}
}
