/* include */
#include "./file/csv_parser.h"


//	csv parser sample
void CSVParserSample()
{
	bool bret;
	mst::file::CSVParser csv_parser;


	bret = csv_parser.Read("../../data/sample_csv_data.csv", -1, false, true);
	if (!bret)	return;

}

