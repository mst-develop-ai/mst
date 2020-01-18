/* include */
#include "./opencv2/opencv.hpp"

#include "./cnn/blob.h"
#include "./cnn/utility.h"
#include "./cnn/layer/convolution_layer.h"


//	convolution layer sample
void ConvolutionLayerSample()
{
	bool bret;

	int size;
	cv::Mat image;
	cv::Mat edge_y;
	cv::Mat edge_x;

	mst::cnn::Blob input_blob;
	mst::cnn::Blob output_blob;
	std::vector<mst::cnn::Blob*> inputs_blob;
	std::vector<mst::cnn::Blob*> outputs_blob;


	mst::cnn::layer::ConvolutionLayer conv_layer;
	mst::cnn::layer::ConvolutionLayerParam conv_layer_param;


	//	read image
	image = cv::imread("../../data/sample_image/sample01.jpg");


	//	set blob
	bret = mst::cnn::utility::SetBlobImageData(input_blob, image.rows, image.cols, image.channels(), image.data);


	//	initialize
	conv_layer_param.filter_ = 2;
	conv_layer_param.kernel_size_ = 3;
	conv_layer_param.stride_ = 1;
	conv_layer_param.padding_ = 1;
	conv_layer_param.padding_mode_ = 0;
	conv_layer_param.use_bias_ = true;

	bret = conv_layer.Initialize(conv_layer_param);


	//	set kernel
	conv_layer.kernel_[0] = -1;
	conv_layer.kernel_[1] = -2;
	conv_layer.kernel_[2] = -1;

	conv_layer.kernel_[3] = 0;
	conv_layer.kernel_[4] = 0;
	conv_layer.kernel_[5] = 0;

	conv_layer.kernel_[6] = 1;
	conv_layer.kernel_[7] = 2;
	conv_layer.kernel_[8] = 1;

	conv_layer.kernel_[9 + 0] = -1;
	conv_layer.kernel_[9 + 1] = 0;
	conv_layer.kernel_[9 + 2] = 1;

	conv_layer.kernel_[9 + 3] = -2;
	conv_layer.kernel_[9 + 4] = 0;
	conv_layer.kernel_[9 + 5] = 2;

	conv_layer.kernel_[9 + 6] = -1;
	conv_layer.kernel_[9 + 7] = 0;
	conv_layer.kernel_[9 + 8] = 1;


	//	reshape
	inputs_blob.clear();
	inputs_blob.push_back(&input_blob);

	outputs_blob.clear();
	outputs_blob.push_back(&output_blob);

	conv_layer.Reshape(inputs_blob, outputs_blob);


	//	forward
	conv_layer.Forward();


	//	get edge data
	size = image.cols * image.rows;
	edge_y = cv::Mat(image.rows, image.cols, CV_32FC1);
	edge_x = cv::Mat(image.rows, image.cols, CV_32FC1);

	bret = mst::cnn::utility::GetBlobImageData(output_blob, 0, 0, size, (float*)edge_y.data);

	bret = mst::cnn::utility::GetBlobImageData(output_blob, 0, 1, size, (float*)edge_x.data);

	cv::normalize(edge_y, edge_y, 0.0, 1.0, CV_MINMAX);
	cv::normalize(edge_x, edge_x, 0.0, 1.0, CV_MINMAX);


	//	show image
	cv::namedWindow("sample");
	cv::imshow("sample", image);
	cv::waitKey(0);

	cv::namedWindow("edge_y");
	cv::imshow("edge_y", edge_y);

	cv::namedWindow("edge_x");
	cv::imshow("edge_x", edge_x);

	cv::waitKey(0);
}
