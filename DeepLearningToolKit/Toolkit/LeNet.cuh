#include"list"
#include"Layers.cuh"
#pragma once
using namespace std;
#ifndef __LENET_H__
#define __LENET_H__
/**
* Computes the backpropagation results of the Softmax loss for each result in a batch.
* Uses the softmax values obtained from forward propagation to compute the difference.
*
* @param label The training batch label values.
* @param num_labels The number of possible labels.
* @param batch_size The size of the trained batch.
* @param diff The resulting gradient.
*/
class TrainingContext
{
private:
	int m_GPUID;
	cMemObj<float> m_WorkSpace;
public:
	DEVICE_HANDLE_PAIR m_HandlePair;
	vector<cNeuralLayer*> m_Network;
public:
	TrainingContext(int GPUID) 
		: m_GPUID(GPUID)
	{
		checkCudaErrors(cudaSetDevice(GPUID), LOCATION_STRING);

		checkCudaErrors(cublasCreate(&m_HandlePair.CUBLASHandle), LOCATION_STRING);
		checkCUDNN(cudnnCreate(&m_HandlePair.CUDNNHandle), LOCATION_STRING);
	};

	~TrainingContext()
	{
		checkCudaErrors(cublasDestroy(m_HandlePair.CUBLASHandle), LOCATION_STRING);
		checkCUDNN(cudnnDestroy(m_HandlePair.CUDNNHandle), LOCATION_STRING);
	};

	void iPushLayer(cNeuralLayer* pLayer)
	{
		this->m_Network.push_back(pLayer);
	}

	void iInitNetwork(default_random_engine &Engine)
	{
		size_t WorkSpaceSize = 0;

		const cNeuralLayer* pPrevLayer = nullptr;

		for (auto pLayer : this->m_Network)
		{
			pLayer->iSetDeviceHandle(this->m_HandlePair);
			pLayer->iResizeLayer(*pPrevLayer);

			pLayer->iRandomInit(Engine);
			pLayer->iSysnchornize();

			WorkSpaceSize = max(WorkSpaceSize, pLayer->iGetWorkSpaceSize());

			pPrevLayer = pLayer;
		}

		m_WorkSpace.iResize(cDimension(WorkSpaceSize));
	}

	void iReleaseCUDAEnvironment()
	{
	}

	// Disable copying
	TrainingContext& operator=(const TrainingContext&) = delete;
	TrainingContext(const TrainingContext&) = delete;

	void Forward(const cNeuralLayer &DataLayer)
	{
		checkCudaErrors(cudaSetDevice(m_GPUID), LOCATION_STRING);

		const cNeuralLayer* pPrevLayer = &DataLayer;

		for (auto pLayer : this->m_Network)
		{
			pLayer->iForward(*pPrevLayer, this->m_WorkSpace);
			pPrevLayer = pLayer;
		}
	}

	void Backward(const cNeuralLayer &DataLayer, cLabelLayer &LabelLayer, const cMemObj<float> &labels, const float LearningRate)
	{
		checkCudaErrors(cudaSetDevice(m_GPUID), LOCATION_STRING);

		const int LayerSize = this->m_Network.size();

		if (LayerSize < 2) { return; }

		LabelLayer.iBackward(*(this->m_Network[LayerSize - 2]), labels, m_WorkSpace);

		for (int seek = LayerSize - 2; seek > 0; seek = seek - 1)
		{
			this->m_Network[seek]->iBackward(*(this->m_Network[seek - 1]), *(this->m_Network[seek + 1]), this->m_WorkSpace);
		}

		for (auto pLayer : this->m_Network)
		{
			pLayer->iUpdateWeight(LearningRate);
		}
	}
};
#endif