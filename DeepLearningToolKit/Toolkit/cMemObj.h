#include"DebugDef.h"
#include"cDimension.h"
#pragma once
using namespace std;
#ifndef __CMEMOBJ_H__
#define __CMEMOBJ_H__
enum eMEM_FLAG { DEV_MEM = 1, HOST_MEM = 2 };

template<class _data>
class cMemObj
{
private:
	_data* m_pDevMem;
	_data* m_pHostMem;
	size_t m_MemStatus;
	cDimension m_Shape;
protected:
	void _Create(_data* &pDevPtr, _data* &pHostPtr, const cDimension &Shape, size_t Flag)
	{
		if (Shape.size == 0) { return; }

		if (_CheckFlag(Flag, DEV_MEM))
		{
			checkCudaErrors(cudaMalloc(&pDevPtr, sizeof(_data) * Shape.size), LOCATION_STRING);
		}
		if (_CheckFlag(Flag, HOST_MEM))
		{
			pHostPtr = new _data[Shape.size]; //Size == 1
		}
	};
	void _Release(_data* &pDevPtr, _data* pHostPtr)
	{
		if (pDevPtr != nullptr)
		{
			checkCudaErrors(cudaFree(m_pDevMem), LOCATION_STRING);
			pDevPtr = nullptr;
		}
		if (pHostPtr != nullptr)
		{
			delete [] pHostPtr; // Size == 1
			pHostPtr = nullptr;
		}
	};
	bool _CheckFlag(const size_t Status, const eMEM_FLAG Flag) const
	{
		return (Status & Flag) == Flag;
	}
public:
	cMemObj(const cDimension &Shape = cDimension(), const size_t Flag = DEV_MEM | HOST_MEM)
		:m_pDevMem(nullptr), m_pHostMem(nullptr), m_Shape(Shape), m_MemStatus(Flag)
	{
	};
	cMemObj(const cMemObj &Obj) = delete;
	~cMemObj()
	{
		_Release(m_pDevMem, m_pHostMem);
	};

	//Copy constructor
	cMemObj& operator=(const cMemObj &Obj) = delete;

	cDimension iResize(const cDimension &Shape, size_t MemStatus = 0)
	{
		Shape.iUpdate();

		MemStatus = MemStatus == 0 ? m_MemStatus : MemStatus;

		if (MemStatus == 0){ return m_Shape; }

		_data* pDevBuffer = m_pDevMem;
		_data* pHostBuffer = m_pHostMem;

		_Create(this->m_pDevMem, this->m_pHostMem, Shape, MemStatus);

		if ((pDevBuffer != nullptr) && _CheckFlag(MemStatus, DEV_MEM))
		{
			checkCudaErrors(cudaMemcpyAsync(m_pDevMem, pDevBuffer, min(m_Shape.size, Shape.size) * sizeof(_data), cudaMemcpyDeviceToDevice), LOCATION_STRING);
		}

		if ((pHostBuffer != nullptr) && _CheckFlag(MemStatus, HOST_MEM))
		{
			memcpy(pHostBuffer, m_pHostMem, min(m_Shape.size, Shape.size) * sizeof(_data));
		}

		_Release(pDevBuffer, pHostBuffer);

		m_Shape = Shape;
		m_MemStatus = MemStatus;

		return m_Shape;
	};

	void iWrite(const _data* pBuffer, const size_t BufferSize, size_t Offset = 0, const eMEM_FLAG Target = DEV_MEM)
	{
		if ((pBuffer == nullptr) || (BufferSize == 0)){ return; }
		if (!_CheckFlag(m_MemStatus, Target)){ return; }

		const size_t Size = min(m_Shape.size - Offset, BufferSize) * sizeof(_data);

		if (Target == DEV_MEM)
		{
			checkCudaErrors(cudaMemcpyAsync(m_pDevMem + Offset, pBuffer, Size, cudaMemcpyHostToDevice), LOCATION_STRING);
		}
		else
		{
			::memcpy(m_pHostMem + Offset, pBuffer, Size);
		}
	};

	void iRead(_data* pBuffer, const size_t BufferSize, size_t Offset = 0, const eMEM_FLAG Target = DEV_MEM) const
	{
		if ((pBuffer == nullptr) || (BufferSize == 0)){ return; }
		if (!_CheckFlag(m_MemStatus, Target)){ return; }

		const size_t Size = min(m_Shape.size - Offset, BufferSize) * sizeof(_data);

		if (Target == DEV_MEM)
		{
			checkCudaErrors(cudaMemcpyAsync(pBuffer, m_pDevMem + Offset, Size, cudaMemcpyDeviceToHost), LOCATION_STRING);
		}
		else
		{
			::memcpy(pBuffer, m_pHostMem + Offset, Size);
		}
	};

	void iSet(const _data &Val, size_t Offset = 0, const eMEM_FLAG Target = HOST_MEM)
	{
		auto Func = [&Val](){ return Val; };
		this->iSet<decltype(Func)>(Func, Offset, Target);
	};

	template<class _Func>
	void iSet(_Func &Func, size_t Offset = 0, const eMEM_FLAG Target = HOST_MEM)
	{
		if (Offset >= m_Shape.size) { return; }
		if (!_CheckFlag(m_MemStatus, Target)){ return; }

		const size_t Size = m_Shape.size - Offset;

		if (Target == DEV_MEM)
		{
			_data* pPtr = new _data[Size + 1];
			for (int seek = 0; seek < Size; seek = seek + 1)
			{
				pPtr[seek] = Func();
			}

			checkCudaErrors(cudaMemcpyAsync(static_cast<_data*>(this->m_pDevMem) + Offset, pPtr, Size * sizeof(_data), cudaMemcpyHostToDevice), LOCATION_STRING);

			delete [] pPtr;
			pPtr = nullptr;
		}
		else
		{
			_data* pPtr = m_pHostMem + Offset;
			for (int seek = 0; seek < Size; seek = seek + 1)
			{
				pPtr[seek] = Func();
			}
		}
	};

	cDimension iGetShape() const
	{
		return m_Shape;
	};

	size_t iGetUnitSize() const
	{
		return this->m_UnitSize;
	};

	size_t iGetMemStatus() const
	{
		return this->m_MemStatus;
	};

	void iSynchronize(const bool bDeviceToHost = true)
	{
		if (m_MemStatus != 3) { return; }

		if (bDeviceToHost)
		{
			iRead(m_pHostMem, m_Shape.size, 0);
		}
		else
		{
			iWrite(m_pDevMem, m_Shape.size, 0);
		}
	};

	const _data* iGetPtr(eMEM_FLAG Flag = DEV_MEM) const
	{
		return Flag == DEV_MEM ? this->m_pDevMem : this->m_pHostMem;
	};
	_data* iGetPtr(eMEM_FLAG Flag = DEV_MEM)
	{
		return Flag == DEV_MEM ? this->m_pDevMem : this->m_pHostMem;
	};
};
#endif