/* include */
#include "../../include/file/csv_parser.h"
#include "../../include/file/file.h"

#include "../../include/string/string.h"
#include "../../include/windows/windows.h"


/* namespace */
namespace mst
{
	namespace file
	{

		//	constructor
		CSVParser::CSVParser()
			: initialize_(false)
			, exists_header_(false)
			, unique_header_(false)
			, header_()
			, header_dic_()
			, cols_count_(0)
			, data_()
		{
		}


		//	destructor
		CSVParser::~CSVParser()
		{
		}


		//	create
		bool CSVParser::Create(int _cols_count, bool _exists_header, bool _unique_header, const std::vector<std::string>& _header)
		{
			int n;

			//	check initialize
			if (initialize_)	return false;

			//	check input
			if (_cols_count < 0)	return false;
			if (_header.size() > _cols_count)	return false;


			//	set param
			cols_count_ = _cols_count;
			exists_header_ = _exists_header;
			unique_header_ = _unique_header;


			//	set header
			if (exists_header_)
			{
				if (unique_header_)
				{
					for (n = 0; n < _header.size(); ++n)
					{
						if (_header[n].length() > 0)
						{
							if (header_dic_.count(_header[n]) != 0)	return false;

							header_dic_[_header[n]] = n;
						}
					}
				}

				header_ = _header;
				while (header_.size() < cols_count_)
				{
					header_.push_back("");
				}
			}


			//	set initialize
			initialize_ = true;

			return true;
		}


		//	read
		bool CSVParser::Read(const char* _path, int _cols_count, bool _exists_header, bool _unique_header)
		{
			int n;
			bool bret;

			std::vector<std::string> lines;
			std::vector<std::string> elems;

			int max_cols_count;


			//	check initialize
			if (initialize_)	return false;


			//	set param
			cols_count_ = _cols_count;
			exists_header_ = _exists_header;


			//	read file
			bret = mst::file::ReadFileLines(_path, lines);
			if (!bret)	return false;


			//	read data
			max_cols_count = 0;

			if (exists_header_)
			{
				if (lines.size() > 0)
				{
					bret = mst::string::SplitDelimiter(lines[0], ',', elems);
					if (!bret)	return false;

					if (unique_header_)
					{
						for (n = 0; n < elems.size(); ++n)
						{
							if(elems[n].length() > 0)
							{
								if (header_dic_.count(elems[n]) != 0)	return false;

								header_dic_[elems[n]] = n;
							}
						}
					}
					header_ = elems;

					if (elems.size() > max_cols_count)
					{
						max_cols_count = (int)elems.size();
					}
				}

				n = 1;
			}
			else
			{
				n = 0;
			}

			for (; n < lines.size(); ++n)
			{
				bret = mst::string::SplitDelimiter(lines[n], ',', elems);
				if (!bret)	return false;

				data_.push_back(elems);

				if (elems.size() > max_cols_count)
				{
					max_cols_count = (int)elems.size();
				}
			}

			if (cols_count_ < 0)
			{
				cols_count_ = max_cols_count;
			}
			else
			{
				if (max_cols_count > cols_count_)	return false;
			}


			//	organize cols
			if (exists_header_)
			{
				while (header_.size() < cols_count_)
				{
					header_.push_back("");
				}
			}

			for (n = 0; n < data_.size(); ++n)
			{
				while (data_[n].size() < cols_count_)
				{
					data_[n].push_back("");
				}
			}


			//	set initialize
			initialize_ = true;

			return true;
		}


		//	write
		bool CSVParser::Write(const char* _path, bool _permit_overwrite)
		{
			int r;
			int c;

			bool bret;

			FILE* fp;


			//	check initialize
			if (!initialize_)	return false;


			//	overwite
			if (!_permit_overwrite)
			{
				bret = mst::windows::PathExists(_path);
				if (bret)	return false;
			}


			//	open file
			fopen_s(&fp, _path, "w");
			if (fp == nullptr)	return false;


			//	write data
			if (exists_header_)
			{
				if (cols_count_ > 0)
				{
					fprintf_s(fp, "%s", header_[0].c_str());
					for (c = 1; c < cols_count_; ++c)
					{
						fprintf_s(fp, ",%s", header_[c].c_str());
					}
					fprintf_s(fp, "\n");
				}
			}

			if (cols_count_ > 0)
			{
				for (r = 0; r < data_.size(); ++r)
				{
					fprintf_s(fp, "%s", data_[r][0].c_str());
					for (c = 1; c < cols_count_; ++c)
					{
						fprintf_s(fp, ",%s", data_[r][c].c_str());
					}
					fprintf_s(fp, "\n");
				}
			}


			//	close file
			fclose(fp);

			return true;
		}


		//	release
		void CSVParser::Release()
		{
			cols_count_ = 0;
			data_.clear();

			exists_header_ = true;
			unique_header_ = false;
			header_.clear();
			header_dic_.clear();

			initialize_ = false;
		}

	}
}
