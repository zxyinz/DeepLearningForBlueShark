#include"Vector.h"
using namespace std;
using namespace San;
using namespace San::Mathematics;
template struct _VECTOR_REF<float32, float32*, float32&>;
template struct _VECTOR_REF<float64, float64*, float64&>;

template struct _VECTOR<float32, float32*, float32&>;
template struct _VECTOR<float64, float64*, float64&>;

#define _M_TYPE typename _VECTOR_REF<_data, _data_ptr, _data_ref>
template<class _data, class _data_ptr, class _data_ref>
_VECTOR_REF<_data, _data_ptr, _data_ref>::_VECTOR_REF()
	:pData(nullptr), Dimension(0)
{
}
template<class _data, class _data_ptr, class _data_ref>
_VECTOR_REF<_data, _data_ptr, _data_ref>::_VECTOR_REF(const _M_TYPE::_dimension &Dimension, const _data_ptr pData)
	:pData(pData)
{
	this->iSetDimension(Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_VECTOR_REF<_data, _data_ptr, _data_ref>::_VECTOR_REF(const _sstream_ref<_data> &DataStream)
	:pData(DataStream.iGetPtr())
{
	this->iSetDimension(DataStream.iGetSize());
}
template<class _data, class _data_ptr, class _data_ref>
_VECTOR_REF<_data, _data_ptr, _data_ref>::_VECTOR_REF(const _M_TYPE::_dimension &Dimension, const _sstream_ref<_data> &DataStream)
	:pData(DataStream.iGetPtr())
{
	this->iSetDimension(Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_VECTOR_REF<_data, _data_ptr, _data_ref>::_VECTOR_REF(const _M_TYPE::_ref_type &Vec)
	:pData(Vec.pData), Dimension(Vec.Dimension)
{
}
template<class _data, class _data_ptr, class _data_ref>
_VECTOR_REF<_data, _data_ptr, _data_ref>::~_VECTOR_REF()
{
}
template<class _data, class _data_ptr, class _data_ref>
const _data_ref _VECTOR_REF<_data, _data_ptr, _data_ref>::operator[](const _M_TYPE::_dimension &Index) const
{
	return pData[Index];
}
template<class _data, class _data_ptr, class _data_ref>
_data_ref _VECTOR_REF<_data, _data_ptr, _data_ref>::operator[](const _M_TYPE::_dimension &Index)
{
	return pData[Index];
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_ref_type& _VECTOR_REF<_data, _data_ptr, _data_ref>::operator=(const _M_TYPE::_ref_type &Vec)
{
	this->pData = Vec.pData;
	this->Dimension = Vec.Dimension;

	return *this;
}
template<class _data, class _data_ptr, class _data_ref>
bool _VECTOR_REF<_data, _data_ptr, _data_ref>::operator==(const _M_TYPE::_ref_type &Vec) const
{
	if (this->Dimension != Vec.Dimension){ return false; }
	for (_M_TYPE::_size seek = 0; seek < this->Dimension; seek = seek + 1)
	{
		if (this->pData[seek] != Vec.pData[seek]){ return false; }
	}

	return true;
}
template<class _data, class _data_ptr, class _data_ref>
bool _VECTOR_REF<_data, _data_ptr, _data_ref>::operator!=(const _M_TYPE::_ref_type &Vec) const
{
	if (this->Dimension != Vec.Dimension){ return true; }
	for (_M_TYPE::_size seek = 0; seek < this->Dimension; seek = seek + 1)
	{
		if (this->pData[seek] != Vec.pData[seek]){ return true; }
	}

	return false;
}
template<class _data, class _data_ptr, class _data_ref>
void _VECTOR_REF<_data, _data_ptr, _data_ref>::iAdd(const _M_TYPE::_ref_type &Vec, const bool bRepeat)
{
	this->iAdd(0, this->Dimension, Vec, bRepeat);
}
template<class _data, class _data_ptr, class _data_ref>
void _VECTOR_REF<_data, _data_ptr, _data_ref>::iAdd(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_ref_type &Vec, const bool bRepeat)
{
	if (End < Start){ return; }

	const _M_TYPE::_size Size = bRepeat ? ::gloMin(this->Dimension, End) : ::gloMin(::gloMin(this->Dimension, Vec.Dimension + Start), End);
	const _M_TYPE::_size Base = Vec.Dimension;

	for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
	{
		this->pData[seek] = this->pData[seek] + Vec.pData[(seek - Start) % Base];
	}

	/*if (bRepeat)
	{
		const _M_TYPE::_size Size = ::gloMin(this->Dimension, End);
		const _M_TYPE::_size Base = Vec.Dimension;

		for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
		{
			this->pData[seek] = this->pData[seek] + Vec.pData[(seek - Start) % Base];
		}
	}
	else
	{
		_M_TYPE::_size Size = ::gloMin(this->Dimension, End);
		Size = (Size - Start) > Vec.Dimension ? (Vec.Dimension + Start) : Size;

		for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
		{
			this->pData[seek] = this->pData[seek] + Vec.pData[seek - Start];
		}
	}*/
}
template<class _data, class _data_ptr, class _data_ref>
void _VECTOR_REF<_data, _data_ptr, _data_ref>::iSub(const _M_TYPE::_ref_type &Vec, const bool bRepeat)
{
	this->iSub(0, this->Dimension, Vec, bRepeat);
}
template<class _data, class _data_ptr, class _data_ref>
void _VECTOR_REF<_data, _data_ptr, _data_ref>::iSub(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_ref_type &Vec, const bool bRepeat)
{
	if (End < Start){ return; }

	const _M_TYPE::_size Size = bRepeat ? ::gloMin(this->Dimension, End) : ::gloMin(::gloMin(this->Dimension, Vec.Dimension + Start), End);
	const _M_TYPE::_size Base = Vec.Dimension;

	for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
	{
		this->pData[seek] = this->pData[seek] - Vec.pData[(seek - Start) % Base];
	}

	/*if (bRepeat)
	{
		const _M_TYPE::_size Size = ::gloMin(this->Dimension, End);
		const _M_TYPE::_size Base = Vec.Dimension;

		for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
		{
			this->pData[seek] = this->pData[seek] - Vec.pData[(seek - Start) % Base];
		}
	}
	else
	{
		_M_TYPE::_size Size = ::gloMin(this->Dimension, End);
		Size = (Size - Start) > Vec.Dimension ? (Vec.Dimension + Start) : Size;

		for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
		{
			this->pData[seek] = this->pData[seek] - Vec.pData[seek - Start];
		}
	}*/
}
template<class _data, class _data_ptr, class _data_ref>
void _VECTOR_REF<_data, _data_ptr, _data_ref>::iMul(const _M_TYPE::_ref_type &Vec, const bool bRepeat)
{
	this->iMul(0, this->Dimension, Vec, bRepeat);
}
template<class _data, class _data_ptr, class _data_ref>
void _VECTOR_REF<_data, _data_ptr, _data_ref>::iMul(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_ref_type &Vec, const bool bRepeat)
{
	if (End < Start){ return; }

	const _M_TYPE::_size Size = bRepeat ? ::gloMin(this->Dimension, End) : ::gloMin(::gloMin(this->Dimension, Vec.Dimension + Start), End);
	const _M_TYPE::_size Base = Vec.Dimension;

	for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
	{
		this->pData[seek] = this->pData[seek] * Vec.pData[(seek - Start) % Base];
	}

	/*if (bRepeat)
	{
		const _M_TYPE::_size Size = ::gloMin(this->Dimension, End);
		const _M_TYPE::_size Base = Vec.Dimension;

		for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
		{
			this->pData[seek] = this->pData[seek] * Vec.pData[(seek - Start) % Base];
		}
	}
	else
	{
		_M_TYPE::_size Size = ::gloMin(this->Dimension, End);
		Size = (Size - Start) > Vec.Dimension ? (Vec.Dimension + Start) : Size;

		for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
		{
			this->pData[seek] = this->pData[seek] * Vec.pData[seek - Start];
		}
	}*/
}
template<class _data, class _data_ptr, class _data_ref>
void _VECTOR_REF<_data, _data_ptr, _data_ref>::iSet(const _M_TYPE::_ref_type &Vec, const bool bRepeat)
{
	this->iSet(0, this->Dimension, Vec, bRepeat);
}
template<class _data, class _data_ptr, class _data_ref>
void _VECTOR_REF<_data, _data_ptr, _data_ref>::iSet(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_ref_type &Vec, const bool bRepeat)
{
	if (End < Start){ return; }

	const _M_TYPE::_size Size = bRepeat ? ::gloMin(this->Dimension, End) : ::gloMin(::gloMin(this->Dimension, Vec.Dimension + Start), End);
	const _M_TYPE::_size Base = Vec.Dimension;

	for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
	{
		this->pData[seek] = Vec.pData[(seek - Start) % Base];
	}

	/*if (bRepeat)
	{
		const _M_TYPE::_size Size = ::gloMin(this->Dimension, End);
		const _M_TYPE::_size Base = Vec.Dimension;

		for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
		{
			this->pData[seek] = Vec.pData[(seek - Start) % Base];
		}
	}
	else
	{
		_M_TYPE::_size Size = ::gloMin(this->Dimension, End);
		Size = (Size - Start) > Vec.Dimension ? (Vec.Dimension + Start) : Size;

		for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
		{
			this->pData[seek] = Vec.pData[seek - Start];
		}
	}*/
}
template<class _data, class _data_ptr, class _data_ref>
_data _VECTOR_REF<_data, _data_ptr, _data_ref>::iDot(const _M_TYPE::_ref_type &Vec, const bool bRepeat) const
{
	return this->iDot(0, this->Dimension, Vec, bRepeat);
}
template<class _data, class _data_ptr, class _data_ref>
_data _VECTOR_REF<_data, _data_ptr, _data_ref>::iDot(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_ref_type &Vec, const bool bRepeat) const
{
	_data Val = 0.0;
	//const _M_TYPE::_size Size = ::gloMin(this->Dimension, End);

	if (End < Start){ return Val; }

	const _M_TYPE::_size Size = bRepeat ? ::gloMin(this->Dimension, End) : ::gloMin(::gloMin(this->Dimension, Vec.Dimension + Start), End);
	const _M_TYPE::_size Base = Vec.Dimension;

	for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
	{
		Val = Val + this->pData[seek] * Vec.pData[(seek - Start) % Base];
	}

	/*if (bRepeat)
	{
		const _M_TYPE::_size Base = Vec.Dimension;

		for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
		{
			Val = Val + this->pData[seek] * Vec.pData[(seek - Start) % Base];
		}
	}
	else
	{
		if ((End - Start) != Vec.Dimension){ return Val; }

		for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
		{
			Val = Val + this->pData[seek] * Vec.pData[seek - Start];
		}
	}*/

	return Val;
}
template<class _data, class _data_ptr, class _data_ref>
void _VECTOR_REF<_data, _data_ptr, _data_ref>::iConv(_M_TYPE::_ref_type &DestVec, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Vec, const bool bRepeat) const
{
	this->iConv(DestVec, 0, this->Dimension, BlockSize, Vec, bRepeat);
}
template<class _data, class _data_ptr, class _data_ref>
void _VECTOR_REF<_data, _data_ptr, _data_ref>::iConv(_M_TYPE::_ref_type &DestVec, const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Vec, const bool bRepeat) const
{
	if (End < Start){ return; }
	if ((End - Start) < BlockSize){ return; }
	if ((BlockSize % 2) != 1){ return; }
	if (DestVec.Dimension < ((End - Start) - BlockSize + 1)){ return; }

	const _M_TYPE::_dimension Offset = BlockSize / 2;
	const _M_TYPE::_size Size = (::gloMin(this->Dimension, End)) - (Offset << 1);

	_M_TYPE::_ref_type SrcVec(BlockSize, this->pData + Start);

	for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
	{
		SrcVec.pData = this->pData + seek;
		DestVec[seek] = SrcVec.iDot(Vec, bRepeat);
	}
}
template<class _data, class _data_ptr, class _data_ref>
void _VECTOR_REF<_data, _data_ptr, _data_ref>::iAdd(const _data &Data)
{
	this->iAdd(0, this->Dimension, Data);
}
template<class _data, class _data_ptr, class _data_ref>
void _VECTOR_REF<_data, _data_ptr, _data_ref>::iAdd(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _data &Data)
{
	const _M_TYPE::_size Size = ::gloMin(this->Dimension, End);

	for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
	{
		this->pData[seek] = this->pData[seek] + Data;
	}
}
template<class _data, class _data_ptr, class _data_ref>
void _VECTOR_REF<_data, _data_ptr, _data_ref>::iSub(const _data &Data)
{
	this->iSub(0, this->Dimension, Data);
}
template<class _data, class _data_ptr, class _data_ref>
void _VECTOR_REF<_data, _data_ptr, _data_ref>::iSub(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _data &Data)
{
	const _M_TYPE::_size Size = ::gloMin(this->Dimension, End);

	for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
	{
		this->pData[seek] = this->pData[seek] - Data;
	}
}
template<class _data, class _data_ptr, class _data_ref>
void _VECTOR_REF<_data, _data_ptr, _data_ref>::iMul(const _data &Data)
{
	this->iMul(0, this->Dimension, Data);
}
template<class _data, class _data_ptr, class _data_ref>
void _VECTOR_REF<_data, _data_ptr, _data_ref>::iMul(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _data &Data)
{
	const _M_TYPE::_size Size = ::gloMin(this->Dimension, End);

	for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
	{
		this->pData[seek] = this->pData[seek] * Data;
	}
}
template<class _data, class _data_ptr, class _data_ref>
void _VECTOR_REF<_data, _data_ptr, _data_ref>::iSet(const _data &Data)
{

	this->iSet(0, this->Dimension, Data);
}
template<class _data, class _data_ptr, class _data_ref>
void _VECTOR_REF<_data, _data_ptr, _data_ref>::iSet(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _data &Data)
{
	const _M_TYPE::_size Size = ::gloMin(this->Dimension, End);

	for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
	{
		this->pData[seek] = Data;
	}
}
template<class _data, class _data_ptr, class _data_ref>
_data _VECTOR_REF<_data, _data_ptr, _data_ref>::iMod() const
{
	return this->iMod(0, this->Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_data _VECTOR_REF<_data, _data_ptr, _data_ref>::iMod(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End) const
{
	_data Val = 0.0;

	const _M_TYPE::_size Size = ::gloMin(this->Dimension, End);
	for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
	{
		Val = Val + this->pData[seek] * this->pData[seek];
	}

	return sqrt(Val);
}
template<class _data, class _data_ptr, class _data_ref>
_data _VECTOR_REF<_data, _data_ptr, _data_ref>::iSum() const
{
	return this->iSum(0, this->Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_data _VECTOR_REF<_data, _data_ptr, _data_ref>::iSum(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End) const
{
	_data Sum = 0.0;

	const _M_TYPE::_size Size = ::gloMin(this->Dimension, End);
	for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
	{
		Sum = Sum + this->pData[seek];
	}

	return Sum;
}
template<class _data, class _data_ptr, class _data_ref>
_data _VECTOR_REF<_data, _data_ptr, _data_ref>::iProduct() const
{
	return this->iProduct(0, this->Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_data _VECTOR_REF<_data, _data_ptr, _data_ref>::iProduct(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End) const
{
	_data Product = 1.0;

	const _M_TYPE::_size Size = ::gloMin(this->Dimension, End);
	for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
	{
		Product = Product * this->pData[seek];
	}

	return Product;
}
template<class _data, class _data_ptr, class _data_ref>
_data _VECTOR_REF<_data, _data_ptr, _data_ref>::iMin() const
{
	return this->iMin(0, this->Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_data _VECTOR_REF<_data, _data_ptr, _data_ref>::iMin(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End) const
{
	if (Start >= this->Dimension)
	{ 
		_SDEBUG(::cout << "Error: size overflow, file: vector.cpp, func: iMin\r\n";)
		return 0.0; 
	}

	_data Min = this->pData[Start];

	const _M_TYPE::_size Size = ::gloMin(this->Dimension, End);
	for (_M_TYPE::_size seek = Start + 1; seek < Size; seek = seek + 1)
	{
		Min = ::gloMin(Min, this->pData[seek]);
	}

	return Min;
}
template<class _data, class _data_ptr, class _data_ref>
_data _VECTOR_REF<_data, _data_ptr, _data_ref>::iMax() const
{
	return this->iMax(0, this->Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_data _VECTOR_REF<_data, _data_ptr, _data_ref>::iMax(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End) const
{
	if (Start >= this->Dimension)
	{
		_SDEBUG(::cout << "Error: size overflow, file: vector.cpp, func: iMax\r\n";)
		return 0.0;
	}

	_data Max = this->pData[Start];

	const _M_TYPE::_size Size = ::gloMin(this->Dimension, End);
	for (_M_TYPE::_size seek = Start + 1; seek < Size; seek = seek + 1)
	{
		Max = ::gloMax(Max, this->pData[seek]);
	}

	return Max;
}
template<class _data, class _data_ptr, class _data_ref>
void _VECTOR_REF<_data, _data_ptr, _data_ref>::iNormalize()
{
	_data Val = this->iMod();

	if (!::gloIsFloatEqual(Val, 0.0))
	{
		const _M_TYPE::_size Size = this->Dimension;
		for (_M_TYPE::_size seek = 0; seek < Size; seek = seek + 1)
		{
			this->pData[seek] = this->pData[seek] / Val;
		}
	}
	else
	{
		Val = 0.0;
		this->iSet(Val);
	}
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _VECTOR_REF<_data, _data_ptr, _data_ref>::iSetVector(const _M_TYPE::_dimension &Dimension, const _data_ptr pData, const _M_TYPE::_dimension &Size)
{
	this->pData = pData;
	return this->iSetDimension(Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _VECTOR_REF<_data, _data_ptr, _data_ref>::iSetVector(const _sstream_ref<_data> &DataStream)
{
	return this->iSetVector(DataStream.iGetSize(), DataStream.iGetPtr());
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _VECTOR_REF<_data, _data_ptr, _data_ref>::iSetVector(const _M_TYPE::_dimension &Dimension, const _sstream_ref<_data> &DataStream)
{
	return this->iSetVector(Dimension, DataStream.iGetPtr());
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _VECTOR_REF<_data, _data_ptr, _data_ref>::iSetDimension(const _M_TYPE::_dimension &Dimension)
{
	this->Dimension = Dimension;
	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
const _data_ptr	_VECTOR_REF<_data, _data_ptr, _data_ref>::iGetContainer() const
{
	return this->pData;
}
template<class _data, class _data_ptr, class _data_ref>
_data_ptr _VECTOR_REF<_data, _data_ptr, _data_ref>::iGetContainer()
{
	return this->pData;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _VECTOR_REF<_data, _data_ptr, _data_ref>::iGetDimension() const
{
	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_size _VECTOR_REF<_data, _data_ptr, _data_ref>::iGetSize() const
{
	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
void _VECTOR_REF<_data, _data_ptr, _data_ref>::iHomogenzation()
{

}
#undef _M_TYPE
#define _M_TYPE typename _VECTOR<_data, _data_ptr, _data_ref>
template<class _data, class _data_ptr, class _data_ref>
_VECTOR<_data, _data_ptr, _data_ref>::_VECTOR()
	:DataContainer(), _VECTOR_REF<_data, _data_ptr, _data_ref>(0, nullptr)
{
}
template<class _data, class _data_ptr, class _data_ref>
_VECTOR<_data, _data_ptr, _data_ref>::_VECTOR(const _M_TYPE::_dimension &Dimension, const _data &Val)
	:DataContainer(Dimension, Val), _VECTOR_REF<_data, _data_ptr, _data_ref>(0, nullptr)
{
	this->_SetVector(this->DataContainer.iGetPtr(), this->DataContainer.iGetSize());
}
template<class _data, class _data_ptr, class _data_ref>
_VECTOR<_data, _data_ptr, _data_ref>::_VECTOR(const _data_ptr pData, const _M_TYPE::_dimension &Size)
	:DataContainer(Size, pData, Size), _VECTOR_REF<_data, _data_ptr, _data_ref>(0, nullptr)
{
	this->_SetVector(this->DataContainer.iGetPtr(), this->DataContainer.iGetSize());
}
template<class _data, class _data_ptr, class _data_ref>
_VECTOR<_data, _data_ptr, _data_ref>::_VECTOR(const _M_TYPE::_dimension &Dimension, const _data_ptr pData, const _M_TYPE::_dimension &Size)
	:DataContainer(Dimension, pData, Size), _VECTOR_REF<_data, _data_ptr, _data_ref>(0, nullptr)
{
	this->_SetVector(this->DataContainer.iGetPtr(), this->DataContainer.iGetSize());
}
template<class _data, class _data_ptr, class _data_ref>
_VECTOR<_data, _data_ptr, _data_ref>::_VECTOR(const _sstream_ref<_data> &DataStream)
	:DataContainer(DataStream), _VECTOR_REF<_data, _data_ptr, _data_ref>(0, nullptr)
{
	this->_SetVector(this->DataContainer.iGetPtr(), this->DataContainer.iGetSize());
}
template<class _data, class _data_ptr, class _data_ref>
_VECTOR<_data, _data_ptr, _data_ref>::_VECTOR(const _M_TYPE::_dimension &Dimension, const _sstream_ref<_data> &DataStream)
	:DataContainer(Dimension, DataStream), _VECTOR_REF<_data, _data_ptr, _data_ref>(0, nullptr)
{
	this->_SetVector(this->DataContainer.iGetPtr(), this->DataContainer.iGetSize());
}
template<class _data, class _data_ptr, class _data_ref>
_VECTOR<_data, _data_ptr, _data_ref>::_VECTOR(const _type &Vec)
	:DataContainer(Vec.DataContainer), _VECTOR_REF<_data, _data_ptr, _data_ref>(0, nullptr)
{
	this->_SetVector(this->DataContainer.iGetPtr(), this->DataContainer.iGetSize());
}
template<class _data, class _data_ptr, class _data_ref>
_VECTOR<_data, _data_ptr, _data_ref>::_VECTOR(const _M_TYPE::_ref_type &VecRef)
	:DataContainer(VecRef.iGetDimension(), VecRef.iGetContainer(), VecRef.iGetDimension()), _VECTOR_REF<_data, _data_ptr, _data_ref>(0, nullptr)
{
	this->_SetVector(this->DataContainer.iGetPtr(), this->DataContainer.iGetSize());
}
template<class _data, class _data_ptr, class _data_ref>
_VECTOR<_data, _data_ptr, _data_ref>::~_VECTOR()
{
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _VECTOR<_data, _data_ptr, _data_ref>::_SetVector(const _data_ptr &pData, const _M_TYPE::_dimension &Dimension)
{
	this->pData = pData;
	this->Dimension = Dimension;
	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type& _VECTOR<_data, _data_ptr, _data_ref>::operator=(const _M_TYPE::_type &Vec)
{
	this->DataContainer = Vec.DataContainer;
	this->_SetVector(this->DataContainer.iGetPtr(), this->DataContainer.iGetSize());

	return *this;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type& _VECTOR<_data, _data_ptr, _data_ref>::operator=(const _M_TYPE::_ref_type &Vec)
{
	this->DataContainer.iReSizeStream(Vec.iGetDimension());
	this->DataContainer.iSetStream(0, Vec.iGetContainer(), Vec.iGetDimension());
	this->_SetVector(this->DataContainer.iGetPtr(), this->DataContainer.iGetSize());

	return *this;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _VECTOR<_data, _data_ptr, _data_ref>::operator+(const _M_TYPE::_type &Vec) const
{
	return *this + Vec.iGetRef();
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _VECTOR<_data, _data_ptr, _data_ref>::operator-(const _M_TYPE::_type &Vec) const
{
	return *this - Vec.iGetRef();
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _VECTOR<_data, _data_ptr, _data_ref>::operator*(const _M_TYPE::_type &Vec) const
{
	return *this * Vec.iGetRef();
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _VECTOR<_data, _data_ptr, _data_ref>::operator+(const _M_TYPE::_ref_type &Vec) const
{
	_type DestVec(*this);
	DestVec.iAdd(Vec);

	return DestVec;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _VECTOR<_data, _data_ptr, _data_ref>::operator-(const _M_TYPE::_ref_type &Vec) const
{
	_type DestVec(*this);
	DestVec.iSub(Vec);

	return DestVec;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _VECTOR<_data, _data_ptr, _data_ref>::operator*(const _M_TYPE::_ref_type &Vec) const
{
	_type DestVec(*this);
	DestVec.iMul(Vec);

	return DestVec;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _VECTOR<_data, _data_ptr, _data_ref>::operator+(const _data &Data) const
{
	_type Vec(*this);
	Vec.iAdd(Data);

	return Vec;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _VECTOR<_data, _data_ptr, _data_ref>::operator-(const _data &Data) const
{
	_type Vec(*this);
	Vec.iSub(Data);

	return Vec;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _VECTOR<_data, _data_ptr, _data_ref>::operator*(const _data &Data) const
{
	_type Vec(*this);
	Vec.iMul(Data);

	return Vec;
}
template<class _data, class _data_ptr, class _data_ref>
bool _VECTOR<_data, _data_ptr, _data_ref>::operator==(const _type &Vec) const
{
	return this->iGetRef() == Vec.iGetRef();
}
template<class _data, class _data_ptr, class _data_ref>
bool _VECTOR<_data, _data_ptr, _data_ref>::operator!=(const _type &Vec) const
{
	return this->iGetRef() != Vec.iGetRef();
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _VECTOR<_data, _data_ptr, _data_ref>::iConv(const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Vec, const bool bRepeat) const
{
	return this->iConv(0, this->Dimension, BlockSize, Vec, bRepeat);
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _VECTOR<_data, _data_ptr, _data_ref>::iConv(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Vec, const bool bRepeat) const
{
	_type DestVec(this->Dimension, 0.0);
	this->iGetRef().iConv(DestVec.iGetRef(), Start, End, BlockSize, Vec, bRepeat);

	return DestVec;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _VECTOR<_data, _data_ptr, _data_ref>::iSetVector(const _data_ptr pData, const _M_TYPE::_dimension &Size)
{
	if (this->Dimension != Size)
	{
		this->DataContainer.iReSizeStream(Size);
		this->_SetVector(this->DataContainer.iGetPtr(), this->DataContainer.iGetSize());
	}

	this->DataContainer.iSetStream(0, pData, Size);

	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _VECTOR<_data, _data_ptr, _data_ref>::iSetVector(const _M_TYPE::_dimension &Dimension, const _data_ptr pData, const _M_TYPE::_dimension &Size)
{
	if (this->Dimension != Dimension)
	{
		this->DataContainer.iReSizeStream(Dimension);
		this->_SetVector(this->DataContainer.iGetPtr(), this->DataContainer.iGetSize());
	}

	this->DataContainer.iSetStream(0, pData, Size);

	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _VECTOR<_data, _data_ptr, _data_ref>::iSetVector(const _sstream_ref<_data> &DataStream)
{
	return this->iSetVector(DataStream.iGetSize(), DataStream.iGetPtr(), DataStream.iGetSize());
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _VECTOR<_data, _data_ptr, _data_ref>::iSetVector(const _M_TYPE::_dimension &Dimension, const _sstream_ref<_data> &DataStream)
{
	return this->iSetVector(Dimension, DataStream.iGetPtr(), DataStream.iGetSize());
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _VECTOR<_data, _data_ptr, _data_ref>::iSetDimension(const _M_TYPE::_dimension &Dimension)
{
	if (this->Dimension != Dimension)
	{ 
		this->DataContainer.iReSizeStream(Dimension);
		this->_SetVector(this->DataContainer.iGetPtr(), this->DataContainer.iGetSize());
	}

	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
const _data& _VECTOR<_data, _data_ptr, _data_ref>::iGetData(const _M_TYPE::_dimension &Index) const
{
	return this->DataContainer[Index];
}
template<class _data, class _data_ptr, class _data_ref>
const _data& _VECTOR<_data, _data_ptr, _data_ref>::iGetData(const _M_TYPE::_size Index) const
{
	return this->DataContainer[Index];
}
template<class _data, class _data_ptr, class _data_ref>
_data& _VECTOR<_data, _data_ptr, _data_ref>::iGetData(const _M_TYPE::_dimension &Index)
{
	return this->DataContainer[Index];
}
template<class _data, class _data_ptr, class _data_ref>
_data& _VECTOR<_data, _data_ptr, _data_ref>::iGetData(const _M_TYPE::_size Index)
{
	return this->DataContainer[Index];
}
template<class _data, class _data_ptr, class _data_ref>
const _M_TYPE::_ref_type& _VECTOR<_data, _data_ptr, _data_ref>::iGetRef() const
{
	return *this;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_ref_type& _VECTOR<_data, _data_ptr, _data_ref>::iGetRef()
{
	return *this;
}
#undef _M_TYPE