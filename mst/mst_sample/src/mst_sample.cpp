/* include */
#include "./opencv2/opencv.hpp"

#include "./cnn/blob.h"
#include "./cnn/layer/convolution_layer.h"


/* entry point */
int wmain(int _argc, wchar_t** _argv)
{
	int y;
	int x;
	int c;
	int n;

	cv::Mat image;
	cv::Mat result1;
	cv::Mat result2;

	std::vector<int> shape;
	mst::cnn::Blob input_blob;
	mst::cnn::Blob output_blob;
	std::vector<mst::cnn::Blob*> inputs_blob;
	std::vector<mst::cnn::Blob*> outputs_blob;


	mst::cnn::layer::ConvolutionLayer conv_layer;


	//	read image
	image = cv::imread("../../data/sample_image/sample01.jpg");

	
	//
	shape.clear();
	shape.push_back(1);
	shape.push_back(3);
	shape.push_back(image.rows);
	shape.push_back(image.cols);

	input_blob.Reshape(shape);

	n = 0;
	for (c = 0; c < image.channels(); ++c)
	{
		for (y = 0; y < image.rows; ++y)
		{
			for (x = 0; x < image.cols; ++x)
			{
				input_blob.data_[n] = image.data[(y * image.cols * image.channels()) + (x * image.channels()) + c];
				++n;
			}
		}
	}


	//
	conv_layer.Initialize(2, 3, 1, 1, 0, true);

	inputs_blob.clear();
	inputs_blob.push_back(&input_blob);

	outputs_blob.clear();
	outputs_blob.push_back(&output_blob);

	conv_layer.Reshape(inputs_blob, outputs_blob);

	conv_layer.Forward();


	//
	result1 = cv::Mat(image.rows, image.cols, CV_32FC1);
	result2 = cv::Mat(image.rows, image.cols, CV_32FC1);

	n = 0;
	for (y = 0; y < result1.rows; ++y)
	{
		for (x = 0; x < result1.cols; ++x)
		{
			((float*)result1.data)[(y * result1.cols) + x] = (float)output_blob.data_[n];
			++n;
		}
	}

	for (y = 0; y < result2.rows; ++y)
	{
		for (x = 0; x < result2.cols; ++x)
		{
			((float*)result2.data)[(y * result2.cols) + x] = (float)output_blob.data_[n];
			++n;
		}
	}

	cv::normalize(result1, result1, 0.0, 1.0, CV_MINMAX);
	cv::normalize(result2, result2, 0.0, 1.0, CV_MINMAX);


	//	show image
	cv::namedWindow("sample");
	cv::imshow("sample", image);
	cv::waitKey(0);

	cv::namedWindow("result1");
	cv::imshow("result1", result1);

	cv::namedWindow("result2");
	cv::imshow("result2", result2);

	cv::waitKey(0);

	return 0;
}
