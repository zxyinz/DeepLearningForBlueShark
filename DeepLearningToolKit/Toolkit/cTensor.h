#include"cMemObj.h"
#include"cDimension.h"
#pragma once
using namespace std;
template<class _data>
class cTensor : public cMemObj<_data>
{
private:
	cudnnTensorDescriptor_t m_Tensor;
public:
	cTensor(const cDimension &Shape = cDimension(), const size_t MemStatus = DEV_MEM | HOST_MEM)
		:cMemObj(Shape, MemStatus), m_Tensor(nullptr)
	{
	};
	cTensor(const cTensor<_data> &Tensor) = delete;
	~cTensor()
	{
		this->iRelease();
	};

	cTensor<_data>& operator=(const cTensor<_data> &Tensor) = delete;

	cDimension iResize(const cDimension &Shape, const size_t MemStatus = 0)
	{
		cDimension TensorShape = static_cast<cMemObj*>(this)->iResize(Shape, MemStatus);

		this->iRelease();

		checkCUDNN(cudnnCreateTensorDescriptor(&(this->m_Tensor)), LOCATION_STRING);
		checkCUDNN(cudnnSetTensor4dDescriptor(this->m_Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, TensorShape.batches, TensorShape.channels, TensorShape.height, TensorShape.width), LOCATION_STRING);

		return TensorShape;
	};

	void iRelease()
	{
		if (this->m_Tensor != nullptr)
		{
			checkCUDNN(cudnnDestroyTensorDescriptor(this->m_Tensor), LOCATION_STRING);
			this->m_Tensor = nullptr;
		}
	};

	const cudnnTensorDescriptor_t& iGetTensor() const
	{
		return this->m_Tensor;
	};
	cudnnTensorDescriptor_t iGetTensor()
	{
		return this->m_Tensor;
	}
};