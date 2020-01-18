/* include */
#include "./cnn/network.h"


//	network config parse sample
void NetworkConfigParseSample()
{
	bool bret;
	mst::cnn::Network network;

	bret = network.ParseNetworkConfig("../../data/neural_network/sample_network01.txt");
	if (!bret)
	{

	}

}
