#include"readubyte.h"
#include"cTensor.h"
#pragma once
using namespace std;
#ifndef __LAYERS_H__
#define __LAYERS_H__

// Block width for CUDA kernels
#define BW 128

/*Computes ceil(x / y) for integral nonnegative values.*/
static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator)
{
	return (nominator + denominator - 1) / denominator;
}

_SAN_PAIR_DEF(DEVICE_HANDLE_PAIR, cudnnHandle_t, CUDNNHandle, nullptr, cublasHandle_t, CUBLASHandle, nullptr, , );

class cNeuralLayer
{
private:
	cDimension m_In;
	cDimension m_Out;
	cTensor<float> m_Output;
	cTensor<float> m_Diff;
protected:
	DEVICE_HANDLE_PAIR m_HandlePair;
	size_t m_WorkSpaceSize;
protected:
	void _SetIn(const cDimension &In)
	{
		this->m_In = In;
		m_Diff.iResize(this->m_In + 1);
	};
	void _SetOut(const cDimension &Out)
	{
		this->m_Out = Out;
		m_Output.iResize(this->m_Out);
	};
public:
	cNeuralLayer(const cDimension &In = cDimension(), const cDimension &Out = cDimension())
		:m_In(In), m_Out(Out), m_Output(Out), m_Diff(In), m_WorkSpaceSize(0)
	{
		this->_SetOut(Out);
	};
	~cNeuralLayer()
	{
	};

	virtual void iReleaseLayer() {};

	virtual void iResizeLayer(const cNeuralLayer &InLayer)
	{
	};

	virtual bool iSetIn(const cDimension &In)
	{ 
		this->_SetIn(In);
		return true;
	};
	virtual bool iSetOut(const cDimension &Out)
	{ 
		this->_SetOut(Out);
		return true;
	};

	virtual void iRandomInit(default_random_engine &Engine) {};
	virtual void iSysnchornize(bool bHostToDevice = true) {};

	void iSetDeviceHandle(const DEVICE_HANDLE_PAIR &HandlePair)
	{
		this->m_HandlePair = HandlePair;
	}

	cDimension iIn() const { return this->m_In; }
	cDimension iOut() const { return this->m_Out; }

	const cTensor<float>& iOutput() const { return this->m_Output; }
	cTensor<float>& iOutput() { return this->m_Output; }

	const cTensor<float>& iDiff() const { return this->m_Diff; }
	cTensor<float>& iDiff() { return this->m_Diff; }

	size_t iGetWorkSpaceSize() const { return this->m_WorkSpaceSize; }

	virtual void iForward(const cNeuralLayer &PrevLayer, cMemObj<float> &WorkSpace) {};
	virtual void iBackward(const cNeuralLayer &PrevLayer, const cNeuralLayer &NextLayer, cMemObj<float> &WorkSpace) {};

	virtual float iUpdateWeight(const float LearningRate) { return 0.0; };
};
class cDataLayer : public cNeuralLayer
{
public:
public:
	cDataLayer(const cDimension &Shape)
		:cNeuralLayer(Shape, Shape)
	{
	};
	~cDataLayer(){};
};

__global__ void _DeviceBackward(const float *label, int num_labels, int batch_size, float *diff)
{
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	if (ID >= batch_size){ return; }

	const int Class = static_cast<int>(label[ID]);

	// For each item in the batch, decrease the result of the label's value by 1
	diff[ID * num_labels + Class] -= 1.0f;
}

class cLabelLayer : public cNeuralLayer
{
public:
	vector<float> m_Predict;
protected:
public:
	cLabelLayer(const cNeuralLayer &InLayer)
		:cNeuralLayer(InLayer.iOut(), cDimension())
	{
		this->iResizeLayer(InLayer);
	};
	~cLabelLayer(){};

	void iResizeLayer(const cNeuralLayer &InLayer) override final
	{
		this->_SetIn(InLayer.iOut());
		this->_SetOut(cDimension(1, 1, 1, InLayer.iOut().batches));

		m_Predict.resize(this->iOut().size);
	};

	void iForward(const cNeuralLayer &PrevLayer)
	{
		PrevLayer.iOutput().iRead(&m_Predict[0], 0, this->iIn().size);

		for (int seek = 0; seek < iIn().batches; seek = seek + 1)
		{
			const auto pPredict = &m_Predict[seek * iIn().volume];

			int MaxID = 0;

			for (int seek_label = 1; seek_label < iIn().volume; seek_label = seek_label + 1)
			{
				MaxID = pPredict[MaxID] < pPredict[seek_label] ? seek_label : MaxID;
			}

			iOutput().iGetPtr(HOST_MEM)[seek] = MaxID;
		}
	};
	void iBackward(const cNeuralLayer &PrevLayer, const cMemObj<float> &Output, cMemObj<float> &WorkSpace)
	{
		checkCudaErrors(cudaMemcpyAsync(this->iDiff().iGetPtr(), PrevLayer.iOutput().iGetPtr(), sizeof(float) * iIn().size, cudaMemcpyDeviceToDevice), LOCATION_STRING);
		_DeviceBackward<<<RoundUp(iIn().batches, BW), BW>>>(Output.iGetPtr(), iIn().volume, iIn().batches, iDiff().iGetPtr());
	};

};

class cConvLayer : public cNeuralLayer
{
public:
	vector<float> WeightSet;
	vector<float> BiasSet;

	size_t NodeSize;

	//Network
	cMemObj<float> Weight;
	cMemObj<float> WeightGrad;

	cMemObj<float> Bias;
	cMemObj<float> BiasGrad;

	cDimension KernelShape;

	/*CUDA*/
	cudnnFilterDescriptor_t FilterDesc;
	cudnnTensorDescriptor_t BiasTensorDesc;

	cudnnConvolutionDescriptor_t ConvDesc;

	cudnnConvolutionFwdAlgo_t ForwardAlgo;
	cudnnConvolutionBwdFilterAlgo_t BackwordFilterAlgo;
	cudnnConvolutionBwdDataAlgo_t BackwordDataAlgo;

private:
	cMemObj<float> ExtraSpace;
public:
	cConvLayer(const cDimension &KernelShape, const size_t NodeNumber)
		: cNeuralLayer(),
		KernelShape(KernelShape), NodeSize(NodeNumber)
	{
	};
	~cConvLayer(){};

	void iResizeLayer(const cNeuralLayer &InLayer) override final
	{
		this->WeightSet.resize(InLayer.iOut().channels * KernelShape.size * this->NodeSize); // ?
		this->BiasSet.resize(this->NodeSize);

		this->_SetIn(InLayer.iOut());
		this->_SetOut(cDimension(InLayer.iOut().width - KernelShape.width + 1, InLayer.iOut().height - KernelShape.height + 1, this->NodeSize, InLayer.iOut().batches));

		//Device weight, bias
		ExtraSpace.iResize(this->iOut());
		Weight.iResize(cDimension(WeightSet.size()));
		WeightGrad.iResize(Weight.iGetShape());

		Bias.iResize(cDimension(BiasSet.size()));
		BiasGrad.iResize(Bias.iGetShape());

		//Device parameter
		size_t Size = 0;
		this->m_WorkSpaceSize = 0;

		//Filiter
		checkCUDNN(cudnnCreateFilterDescriptor(&FilterDesc), LOCATION_STRING);
		checkCUDNN(cudnnSetFilter4dDescriptor(FilterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, iOut().channels, iIn().channels, KernelShape.height, KernelShape.width), LOCATION_STRING);

		//Bias
		checkCUDNN(cudnnCreateTensorDescriptor(&BiasTensorDesc), LOCATION_STRING);
		checkCUDNN(cudnnSetTensor4dDescriptor(BiasTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, iOut().channels, 1, 1), LOCATION_STRING);

		//Conv
		checkCUDNN(cudnnCreateConvolutionDescriptor(&ConvDesc), LOCATION_STRING);
		checkCUDNN(cudnnSetConvolution2dDescriptor(ConvDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION), LOCATION_STRING);

		//Forward algorithm
		checkCUDNN(cudnnGetConvolutionForwardAlgorithm(m_HandlePair.CUDNNHandle, InLayer.iOutput().iGetTensor(), FilterDesc, ConvDesc, iOutput().iGetTensor(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &ForwardAlgo), LOCATION_STRING);
		checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(m_HandlePair.CUDNNHandle, InLayer.iOutput().iGetTensor(), FilterDesc, ConvDesc, iOutput().iGetTensor(), ForwardAlgo, &Size), LOCATION_STRING);

		this->m_WorkSpaceSize = max(this->m_WorkSpaceSize, Size);

		//Backward algorithm
		checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(m_HandlePair.CUDNNHandle, InLayer.iOutput().iGetTensor(), iOutput().iGetTensor(), ConvDesc, FilterDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &BackwordFilterAlgo), LOCATION_STRING);
		checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(m_HandlePair.CUDNNHandle, InLayer.iOutput().iGetTensor(), iOutput().iGetTensor(), ConvDesc, FilterDesc, BackwordFilterAlgo, &Size), LOCATION_STRING);

		this->m_WorkSpaceSize = max(this->m_WorkSpaceSize, Size);

		//Backward data algorithm
		checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(m_HandlePair.CUDNNHandle, FilterDesc, iOutput().iGetTensor(), ConvDesc, InLayer.iOutput().iGetTensor(), CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &BackwordDataAlgo), LOCATION_STRING);
		checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(m_HandlePair.CUDNNHandle, FilterDesc, iOutput().iGetTensor(), ConvDesc, InLayer.iOutput().iGetTensor(), BackwordDataAlgo, &Size), LOCATION_STRING);

		this->m_WorkSpaceSize = max(this->m_WorkSpaceSize, Size);
	};

	void iRandomInit(default_random_engine &Engine) override final
	{
		// Xavier weight filling
		const float Boundary = sqrt(3.0 / (this->KernelShape.size * this->iIn().channels));

		uniform_real_distribution<> Dist(-Boundary, Boundary);

		for (auto& val : this->WeightSet)
		{
			val = static_cast<float>(Dist(Engine));
		}

		for (auto& val : this->BiasSet)
		{
			val = static_cast<float>(Dist(Engine));
		}
	};

	void iReleaseLayer()
	{
		checkCUDNN(cudnnDestroyFilterDescriptor(FilterDesc), LOCATION_STRING);
		checkCUDNN(cudnnDestroyTensorDescriptor(BiasTensorDesc), LOCATION_STRING);

		checkCUDNN(cudnnDestroyConvolutionDescriptor(ConvDesc), LOCATION_STRING);
	};

	void iSysnchornize(bool bHostToDevice = true) override final
	{
		if (bHostToDevice)
		{
			checkCudaErrors(cudaMemcpyAsync(Weight.iGetPtr(), &WeightSet[0], sizeof(float) * WeightSet.size(), cudaMemcpyHostToDevice), LOCATION_STRING);
			checkCudaErrors(cudaMemcpyAsync(Bias.iGetPtr(), &BiasSet[0], sizeof(float) * BiasSet.size(), cudaMemcpyHostToDevice), LOCATION_STRING);
		}
		else
		{
			checkCudaErrors(cudaMemcpyAsync(&WeightSet[0], Weight.iGetPtr(), sizeof(float) * WeightSet.size(), cudaMemcpyDeviceToHost), LOCATION_STRING);
			checkCudaErrors(cudaMemcpyAsync(&BiasSet[0], Bias.iGetPtr(), sizeof(float) * BiasSet.size(), cudaMemcpyDeviceToHost), LOCATION_STRING);
		}
	};

	void iForward(const cNeuralLayer &PrevLayer, cMemObj<float> &WorkSpace) override final
	{
		const float &Alpha = 1.0;
		const float &Beta = 0.0;

		const auto &Input = PrevLayer.iOutput();

		checkCUDNN(cudnnConvolutionForward(m_HandlePair.CUDNNHandle, &Alpha, Input.iGetTensor(), Input.iGetPtr(), FilterDesc, Weight.iGetPtr(), ConvDesc, ForwardAlgo, WorkSpace.iGetPtr(), WorkSpace.iGetShape().size, &Beta, iOutput().iGetTensor(), iOutput().iGetPtr()), LOCATION_STRING);
		checkCUDNN(cudnnAddTensor(m_HandlePair.CUDNNHandle, &Alpha, BiasTensorDesc, Bias.iGetPtr(), &Alpha, iOutput().iGetTensor(), iOutput().iGetPtr()), LOCATION_STRING);
	};
	void iBackward(const cNeuralLayer &PrevLayer, const cNeuralLayer &NextLayer, cMemObj<float> &WorkSpace) override final
	{
		const float &Alpha = 1.0;
		const float &Beta = 0.0;

		const auto &Input = PrevLayer.iOutput();
		const auto &Output = NextLayer.iOutput();
		const auto &OutputDiff = NextLayer.iDiff();

		checkCUDNN(cudnnConvolutionBackwardFilter(m_HandlePair.CUDNNHandle, &Alpha, Input.iGetTensor(), Input.iGetPtr(), iOutput().iGetTensor(), OutputDiff.iGetPtr(), ConvDesc, BackwordFilterAlgo, WorkSpace.iGetPtr(), m_WorkSpaceSize, &Beta, FilterDesc, WeightGrad.iGetPtr()), LOCATION_STRING);
		checkCUDNN(cudnnConvolutionBackwardBias(m_HandlePair.CUDNNHandle, &Alpha, iOutput().iGetTensor(), OutputDiff.iGetPtr(), &Beta, BiasTensorDesc, BiasGrad.iGetPtr()), LOCATION_STRING);

		checkCUDNN(cudnnConvolutionBackwardData(m_HandlePair.CUDNNHandle, &Alpha, FilterDesc, Weight.iGetPtr(), iOutput().iGetTensor(), OutputDiff.iGetPtr(), ConvDesc, BackwordDataAlgo, WorkSpace.iGetPtr(), m_WorkSpaceSize, &Beta, Input.iGetTensor(), iDiff().iGetPtr()), LOCATION_STRING);
	};

	float iUpdateWeight(const float LearningRate) override final
	{
		const float Alpha = -LearningRate;

		checkCudaErrors(cublasSaxpy(m_HandlePair.CUBLASHandle, static_cast<int>(WeightSet.size()), &Alpha, WeightGrad.iGetPtr(), 1, Weight.iGetPtr(), 1), LOCATION_STRING);
		checkCudaErrors(cublasSaxpy(m_HandlePair.CUBLASHandle, static_cast<int>(BiasSet.size()), &Alpha, BiasGrad.iGetPtr(), 1, Bias.iGetPtr(), 1), LOCATION_STRING);

		return 0.0;
	}
};

class cMaxPoolLayer : public cNeuralLayer
{
public:
	cDimension KernelShape;
	cudnnPoolingDescriptor_t PoolDesc;
private:
	cMemObj<float> Diff;
public:
	cMaxPoolLayer(cDimension &KernelShape)
		:cNeuralLayer(), KernelShape(KernelShape)
	{
	};
	~cMaxPoolLayer(){};

	void iResizeLayer(const cNeuralLayer &InLayer) override final
	{
		this->_SetIn(InLayer.iOut());
		this->_SetOut(cDimension(InLayer.iOut().width / this->KernelShape.width, InLayer.iOut().height / this->KernelShape.height, InLayer.iOut().channels, InLayer.iOut().batches));

		checkCUDNN(cudnnCreatePoolingDescriptor(&PoolDesc), LOCATION_STRING);
		checkCUDNN(cudnnSetPooling2dDescriptor(PoolDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, this->KernelShape.height, this->KernelShape.width, 0, 0, this->KernelShape.height, this->KernelShape.width), LOCATION_STRING);

		//Output
		Diff.iResize(this->iIn());
	};

	void iReleaseLayer()
	{
		checkCUDNN(cudnnDestroyPoolingDescriptor(PoolDesc), LOCATION_STRING);
	};

	void iForward(const cNeuralLayer &PrevLayer, cMemObj<float> &WorkSpace) override final
	{
		const float &Alpha = 1.0;
		const float &Beta = 0.0;

		const auto &Input = PrevLayer.iOutput();

		checkCUDNN(cudnnPoolingForward(m_HandlePair.CUDNNHandle, PoolDesc, &Alpha, Input.iGetTensor(), Input.iGetPtr(), &Beta, iOutput().iGetTensor(), iOutput().iGetPtr()), LOCATION_STRING);
	};
	void iBackward(const cNeuralLayer &PrevLayer, const cNeuralLayer &NextLayer, cMemObj<float> &WorkSpace) override final
	{
		const float &Alpha = 1.0;
		const float &Beta = 0.0;

		const auto &Input = PrevLayer.iOutput();
		const auto &Output = NextLayer.iOutput();
		const auto &OutputDiff = NextLayer.iDiff();

		checkCUDNN(cudnnPoolingBackward(m_HandlePair.CUDNNHandle, PoolDesc, &Alpha, iOutput().iGetTensor(), iOutput().iGetPtr(), iOutput().iGetTensor(), OutputDiff.iGetPtr(), Input.iGetTensor(), Input.iGetPtr(), &Beta, Input.iGetTensor(), iDiff().iGetPtr()), LOCATION_STRING);
	};
};

class cFullyConnectedLayer : public cNeuralLayer
{
public:
	enum eFUNCTYPE { FT_RELU, FT_SOFTMAX };
public:
	int NodeNumber;
	vector<float> WeightSet;
	vector<float> BiasSet;

	cMemObj<float> Diff;
	cMemObj<float> LinearOutput;
	cMemObj<float> FuncDiff;

	cMemObj<float> Weight;
	cMemObj<float> WeightGrad;

	cMemObj<float> Bias;
	cMemObj<float> BiasGrad;

	cMemObj<float> OneVec;

	cudnnActivationDescriptor_t ActivationDesc;

	eFUNCTYPE m_FuncType;
public:
	cFullyConnectedLayer(const size_t NodeNumber, const eFUNCTYPE FuncType)
		:cNeuralLayer(), NodeNumber(NodeNumber), m_FuncType(FuncType)
	{
	};
	~cFullyConnectedLayer(){};

	void iResizeLayer(const cNeuralLayer &InLayer) override final
	{
		this->_SetIn(cDimension(1, 1, InLayer.iOut().volume, InLayer.iOut().batches));
		this->_SetOut(cDimension(1, 1, this->NodeNumber, InLayer.iOut().batches));

		//this->BatchSize = BatchSize;

		this->WeightSet.resize(InLayer.iOut().volume * this->NodeNumber);
		this->BiasSet.resize(this->NodeNumber);

		// Create tensor descriptors
		checkCUDNN(cudnnCreateActivationDescriptor(&ActivationDesc), LOCATION_STRING);

		if (m_FuncType == FT_RELU)
		{
			checkCUDNN(cudnnSetActivationDescriptor(ActivationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0), LOCATION_STRING);
		}

		//Output
		LinearOutput.iResize(iOut());

		Diff.iResize(iIn());
		FuncDiff.iResize(iOut());

		Weight.iResize(cDimension(WeightSet.size()));
		WeightGrad.iResize(cDimension(WeightSet.size()));

		Bias.iResize(cDimension(BiasSet.size()));
		BiasGrad.iResize(cDimension(BiasSet.size()));

		//OneVec
		OneVec.iResize(iOut().batches);
		OneVec.iSet(1.0, 0, DEV_MEM);
	};

	void iRandomInit(default_random_engine &Engine) override final
	{
		// Xavier weight filling
		const float Boundary = sqrt(3.0 / static_cast<double>(this->WeightSet.size()));

		uniform_real_distribution<> Dist(-Boundary, Boundary);

		for (auto& val : this->WeightSet)
		{
			val = static_cast<float>(Dist(Engine));
		}

		for (auto& val : this->BiasSet)
		{
			val = static_cast<float>(Dist(Engine));
		}
	};

	void iReleaseLayer()
	{
		checkCUDNN(cudnnDestroyActivationDescriptor(ActivationDesc), LOCATION_STRING);
	};

	void iSysnchornize(bool bHostToDevice = true) override final
	{
		if (bHostToDevice)
		{
			checkCudaErrors(cudaMemcpyAsync(Weight.iGetPtr(), &WeightSet[0], sizeof(float) * WeightSet.size(), cudaMemcpyHostToDevice), LOCATION_STRING);
			checkCudaErrors(cudaMemcpyAsync(Bias.iGetPtr(), &BiasSet[0], sizeof(float) * BiasSet.size(), cudaMemcpyHostToDevice), LOCATION_STRING);
		}
		else
		{
			checkCudaErrors(cudaMemcpyAsync(&WeightSet[0], Weight.iGetPtr(), sizeof(float) * WeightSet.size(), cudaMemcpyDeviceToHost), LOCATION_STRING);
			checkCudaErrors(cudaMemcpyAsync(&BiasSet[0], Bias.iGetPtr(), sizeof(float) * BiasSet.size(), cudaMemcpyDeviceToHost), LOCATION_STRING);
		}
	};

	void iForward(const cNeuralLayer &PrevLayer, cMemObj<float> &WorkSpace) override final
	{
		const float &Alpha = 1.0;
		const float &Beta = 0.0;

		const auto &Input = PrevLayer.iOutput();

		checkCudaErrors(cublasSgemm(m_HandlePair.CUBLASHandle, CUBLAS_OP_T, CUBLAS_OP_N, iOut().volume, iOut().batches, iIn().volume, &Alpha, Weight.iGetPtr(), iIn().volume, Input.iGetPtr(), iIn().volume, &Beta, LinearOutput.iGetPtr(), iOut().volume), LOCATION_STRING);
		// Add bias using GEMM's "beta" (fc1 += pfc1bias*1_vec')
		checkCudaErrors(cublasSgemm(m_HandlePair.CUBLASHandle, CUBLAS_OP_N, CUBLAS_OP_N, iOut().volume, iOut().batches, 1, &Alpha, Bias.iGetPtr(), iOut().volume, OneVec.iGetPtr(), 1, &Alpha, LinearOutput.iGetPtr(), iOut().volume), LOCATION_STRING);

		if (this->m_FuncType == FT_RELU)
		{
			// ReLU activation
			checkCUDNN(cudnnActivationForward(m_HandlePair.CUDNNHandle, ActivationDesc, &Alpha, iOutput().iGetTensor(), LinearOutput.iGetPtr(), &Beta, iOutput().iGetTensor(), iOutput().iGetPtr()), LOCATION_STRING);
		}

		if (this->m_FuncType == FT_SOFTMAX)
		{
			checkCUDNN(cudnnSoftmaxForward(m_HandlePair.CUDNNHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &Alpha, iOutput().iGetTensor(), LinearOutput.iGetPtr(), &Beta, iOutput().iGetTensor(), LinearOutput.iGetPtr()), LOCATION_STRING);
		}
	};
	void iBackward(const cNeuralLayer &PrevLayer, const cNeuralLayer &NextLayer, cMemObj<float> &WorkSpace) override final
	{
		const float &Alpha = 1.0;
		const float &Beta = 0.0;

		const auto &Input = PrevLayer.iOutput();
		const auto &Output = NextLayer.iOutput();
		const auto &OutputDiff = NextLayer.iDiff();

		if (this->m_FuncType == FT_RELU)
		{
			// ReLU activation
			checkCUDNN(cudnnActivationBackward(m_HandlePair.CUDNNHandle, ActivationDesc, &Alpha, iOutput().iGetTensor(), iOutput().iGetPtr(), iOutput().iGetTensor(), OutputDiff.iGetPtr(), iOutput().iGetTensor(), LinearOutput.iGetPtr(), &Beta, iOutput().iGetTensor(), FuncDiff.iGetPtr()), LOCATION_STRING);
		}

		if (this->m_FuncType == FT_SOFTMAX)
		{
			// Accounting for batch size in SGD
			const float scalVal = 1.0f / static_cast<float>(this->iOut().batches);

			checkCudaErrors(cudaMemcpyAsync(FuncDiff.iGetPtr(), OutputDiff.iGetPtr(), sizeof(float) * iOut().size, cudaMemcpyDeviceToDevice), LOCATION_STRING);
			checkCudaErrors(cublasSscal(m_HandlePair.CUBLASHandle, iOut().size, &scalVal, FuncDiff.iGetPtr(), 1), LOCATION_STRING);
		}

		// FC1 layer
		// Compute derivative with respect to weights: gfc1 = (pool2 * dfc1relu')
		checkCudaErrors(cublasSgemm(m_HandlePair.CUBLASHandle, CUBLAS_OP_N, CUBLAS_OP_T, iIn().volume, iOut().volume, iOut().batches, &Alpha, Input.iGetPtr(), iIn().volume, FuncDiff.iGetPtr(), iOut().volume, &Beta, WeightGrad.iGetPtr(), iIn().volume), LOCATION_STRING);
		// Compute derivative with respect to bias: gfc1bias = dfc1relu * 1_vec
		checkCudaErrors(cublasSgemv(m_HandlePair.CUBLASHandle, CUBLAS_OP_N, iOut().volume, iOut().batches, &Alpha, FuncDiff.iGetPtr(), iOut().volume, OneVec.iGetPtr(), 1, &Beta, BiasGrad.iGetPtr(), 1), LOCATION_STRING);
		// Compute derivative with respect to data (for previous layer): pfc1*dfc1relu (800x500*500xN)
		checkCudaErrors(cublasSgemm(m_HandlePair.CUBLASHandle, CUBLAS_OP_N, CUBLAS_OP_N, iIn().volume, iOut().batches, iOut().volume, &Alpha, Weight.iGetPtr(), iIn().volume, FuncDiff.iGetPtr(), iOut().volume, &Beta, iDiff().iGetPtr(), iIn().volume), LOCATION_STRING);
	};

	float iUpdateWeight(const float LearningRate) override final
	{
		const float Alpha = -LearningRate;

		checkCudaErrors(cublasSaxpy(m_HandlePair.CUBLASHandle, static_cast<int>(WeightSet.size()), &Alpha, WeightGrad.iGetPtr(), 1, Weight.iGetPtr(), 1), LOCATION_STRING);
		checkCudaErrors(cublasSaxpy(m_HandlePair.CUBLASHandle, static_cast<int>(BiasSet.size()), &Alpha, BiasGrad.iGetPtr(), 1, Bias.iGetPtr(), 1), LOCATION_STRING);

		return 0;
	}
};
#endif