#pragma once

/* include */
#include <map>
#include <vector>
#include <string>


/* namespace */
namespace mst
{
	namespace file
	{

		class CSVParser
		{

		public:


			/* function */
			CSVParser();
			~CSVParser();

			CSVParser(const CSVParser& _obj) = delete;
			CSVParser& operator=(const CSVParser& _obj) = delete;

			bool Create(int _cols_count, bool _exists_header, bool _unique_header = false, const std::vector<std::string>& _header = {});

			bool Read(const char* _path, int _cols_count, bool _exists_header, bool _unique_header);
			bool Write(const char* _path, bool _permit_overwrite);

			//bool SetHeader(int _cols_index, std::string& _header);

			//bool ResizeCols(int _cols_count);
			//bool InsertCols(int _cols_index);
			//bool InsertCols(int _cols_index, const std::string& _header);
			//bool DeleteCols(int _cols_index);

			//bool ResizeRows(int _rows_count);
			//bool InsertRows(int _rows_index, const std::vector<std::string>& _data);
			//bool DeleteRows(int _rows_index);

			//bool GetData(int _rows, int _cols, std::string& _dst);
			//bool GetData(int _rows, int _cols, bool& _dst);
			//bool GetData(int _rows, int _cols, int& _dst);
			//bool GetData(int _rows, int _cols, float& _dst);
			//bool GetData(int _rows, int _cols, double& _dst);

			//bool GetRowsData(int _rows_index, std::vector<std::string>& _dst);
			//bool GetColsData(int _cols_index, std::vector<std::string>& _dst);

			void Release();


		private:

			bool initialize_;

			bool exists_header_;
			bool unique_header_;
			std::vector<std::string> header_;
			std::map<std::string, int> header_dic_;

			int cols_count_;
			std::vector<std::vector<std::string>> data_;

		};

	}
}
