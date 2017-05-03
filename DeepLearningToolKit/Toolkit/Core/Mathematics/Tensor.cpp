#include"Tensor.h"
using namespace std;
using namespace San;
using namespace San::Mathematics;
template struct _TENSOR_REF<float32, float32*, float32&>;
template struct _TENSOR_REF<float64, float64*, float64&>;

template struct _TENSOR<float32, float32*, float32&>;
template struct _TENSOR<float64, float64*, float64&>;

#define _M_TYPE typename _TENSOR_REF<_data, _data_ptr, _data_ref>
template<class _data, class _data_ptr, class _data_ref>
_TENSOR_REF<_data, _data_ptr, _data_ref>::_TENSOR_REF()
	:MatRefSet(), Dimension(0.0, 0.0, 0.0)
{
}
template<class _data, class _data_ptr, class _data_ref>
_TENSOR_REF<_data, _data_ptr, _data_ref>::_TENSOR_REF(const _M_TYPE::_dimension &Dimension, const _data_ptr pData, const _M_TYPE::_size Size)
{
	this->iSetTensor(Dimension, pData, Size);
}
template<class _data, class _data_ptr, class _data_ref>
_TENSOR_REF<_data, _data_ptr, _data_ref>::_TENSOR_REF(const _M_TYPE::_dimension &Dimension, const _sstream<_data> &DataStream)
{
	this->iSetTensor(Dimension, DataStream);
}
template<class _data, class _data_ptr, class _data_ref>
_TENSOR_REF<_data, _data_ptr, _data_ref>::_TENSOR_REF(const _M_TYPE::_dimension &Dimension, const _data_ptr** pDataPtrArray, const _M_TYPE::_dimension Size)
{
	this->iSetTensor(Dimension, pDataPtrArray, Size);
}
template<class _data, class _data_ptr, class _data_ref>
_TENSOR_REF<_data, _data_ptr, _data_ref>::_TENSOR_REF(const _M_TYPE::_dimension &Dimension, const vector<vector<_data_ptr>> &DataPtrArray)
{
	this->iSetTensor(Dimension, DataPtrArray);
}
template<class _data, class _data_ptr, class _data_ref>
_TENSOR_REF<_data, _data_ptr, _data_ref>::_TENSOR_REF(const _M_TYPE::_dimension &Dimension, const _M_TYPE::_mat_ref* pMatrixArray, const _M_TYPE::_size Size)
{
	this->iSetTensor(Dimension, pMatrixArray, Size);
}
template<class _data, class _data_ptr, class _data_ref>
_TENSOR_REF<_data, _data_ptr, _data_ref>::_TENSOR_REF(const _M_TYPE::_dimension &Dimension, const vector<_M_TYPE::_mat_ref> &MatrixArray)
{
	this->iSetTensor(Dimension, MatrixArray);
}
template<class _data, class _data_ptr, class _data_ref>
_TENSOR_REF<_data, _data_ptr, _data_ref>::_TENSOR_REF(const _M_TYPE::_ref_type &Tensor)
	:MatRefSet(Tensor.MatRefSet), Dimension(Tensor.Dimension)
{
}
template<class _data, class _data_ptr, class _data_ref>
_TENSOR_REF<_data, _data_ptr, _data_ref>::~_TENSOR_REF()
{
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _TENSOR_REF<_data, _data_ptr, _data_ref>::_ConvCheck(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_dimension &DestTensorSize) const
{
	_TENSOR_REF<_data, _data_ptr, _data_ref>::_dimension Zero(0, 0, 0);

	if (End < Start){ return Zero; }
	if ((End - Start) < BlockSize){ return Zero; }
	if (((BlockSize.x % 2) != 1) || ((BlockSize.y % 2) != 1) || ((BlockSize.z % 2) != 1)){ return Zero; }

	const auto MinDimension = (End - Start) - BlockSize + 1;
	if ((DestTensorSize.x < MinDimension.x) || (DestTensorSize.y < MinDimension.y) || (DestTensorSize.z < MinDimension.z)){ return Zero; }

	const auto Offset(BlockSize * 0.5);
	const auto Size = _M_TYPE::_dimension(min(this->Dimension.x, End.x), min(this->Dimension.y, End.y), min(this->Dimension.z, End.z)) - (Offset * 2);

	return Size;
}
template<class _data, class _data_ptr, class _data_ref>
void _TENSOR_REF<_data, _data_ptr, _data_ref>::_CreateSubTensor(const _dimension &Start, const _dimension &BlockSize, _M_TYPE::_ref_type &Block) const
{
	for (_M_TYPE::_size seek_mat = 0; seek_mat < BlockSize.z; seek_mat = seek_mat + 1)
	{
		const auto &Matrix = this->MatRefSet[Start.z + seek_mat];
		auto &BlockMatrix = Block[seek_mat];

		for (_M_TYPE::_size seek_vec = 0; seek_vec < BlockSize.y; seek_vec = seek_vec + 1)
		{
			BlockMatrix[seek_vec].iSetVector(BlockSize.x, Matrix[Start.y + seek_vec].iGetContainer() + Start.x);
		}
	}
}
template<class _data, class _data_ptr, class _data_ref>
const _data_ref _TENSOR_REF<_data, _data_ptr, _data_ref>::operator[](const _M_TYPE::_dimension &Pos) const
{
	return this->MatRefSet[Pos.z][Pos.y][Pos.x];
}
template<class _data, class _data_ptr, class _data_ref>
_data_ref _TENSOR_REF<_data, _data_ptr, _data_ref>::operator[](const _M_TYPE::_dimension &Pos)
{
	return this->MatRefSet[Pos.z][Pos.y][Pos.x];
}
template<class _data, class _data_ptr, class _data_ref>
const _M_TYPE::_mat_ref& _TENSOR_REF<_data, _data_ptr, _data_ref>::operator[](const _M_TYPE::_size Index) const
{
	return this->MatRefSet[Index];
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_mat_ref& _TENSOR_REF<_data, _data_ptr, _data_ref>::operator[](const _M_TYPE::_size Index)
{
	return this->MatRefSet[Index];
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_ref_type& _TENSOR_REF<_data, _data_ptr, _data_ref>::operator=(const _M_TYPE::_ref_type &Tensor)
{
	this->MatRefSet = Tensor.MatRefSet;
	this->Dimension = Tensor.Dimension;

	return *this;
}
template<class _data, class _data_ptr, class _data_ref>
bool _TENSOR_REF<_data, _data_ptr, _data_ref>::operator==(const _M_TYPE::_ref_type &Tensor) const
{
	if (this->Dimension != Tensor.Dimension) { return false; }
	for (_M_TYPE::_size seek = 0; seek < this->Dimension.z; seek = seek + 1)
	{
		if (this->MatRefSet[seek] != Tensor.MatRefSet[seek]) { return false; }
	}

	return true;
}
template<class _data, class _data_ptr, class _data_ref>
bool _TENSOR_REF<_data, _data_ptr, _data_ref>::operator!=(const _M_TYPE::_ref_type &Tensor) const
{
	if (this->Dimension != Tensor.Dimension){ return true; }
	for (_M_TYPE::_size seek = 0; seek < this->Dimension.z; seek = seek + 1)
	{
		if (this->MatRefSet[seek] != Tensor.MatRefSet[seek]){ return true; }
	}

	return false;
}
template<class _data, class _data_ptr, class _data_ref>
void _TENSOR_REF<_data, _data_ptr, _data_ref>::iAdd(const _M_TYPE::_ref_type &Tensor, const bool bRepeat)
{
	this->iAdd(_M_TYPE::_dimension(0, 0, 0), this->Dimension, Tensor, bRepeat);
}
template<class _data, class _data_ptr, class _data_ref>
void _TENSOR_REF<_data, _data_ptr, _data_ref>::iAdd(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_ref_type &Tensor, const bool bRepeat)
{
	const _M_TYPE::_size Size = bRepeat ? ::gloMin(this->Dimension.z, End.z) : ::gloMin(::gloMin(this->Dimension.z, Tensor.Dimension.z + Start.z), End.z);
	const _M_TYPE::_size Base = Tensor.Dimension.z;

	for (_M_TYPE::_size seek = Start.z; seek < Size; seek = seek + 1)
	{
		this->MatRefSet[seek].iAdd(Start.x, End.x, Tensor[(seek - Start.z) % Base], bRepeat);
	}
}
template<class _data, class _data_ptr, class _data_ref>
void _TENSOR_REF<_data, _data_ptr, _data_ref>::iSub(const _M_TYPE::_ref_type &Tensor, const bool bRepeat)
{
	this->iSub(_M_TYPE::_dimension(0, 0, 0), this->Dimension, Tensor, bRepeat);
}
template<class _data, class _data_ptr, class _data_ref>
void _TENSOR_REF<_data, _data_ptr, _data_ref>::iSub(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_ref_type &Tensor, const bool bRepeat)
{
	const _M_TYPE::_size Size = bRepeat ? ::gloMin(this->Dimension.z, End.z) : ::gloMin(::gloMin(this->Dimension.z, Tensor.Dimension.z + Start.z), End.z);
	const _M_TYPE::_size Base = Tensor.Dimension.z;

	for (_M_TYPE::_size seek = Start.z; seek < Size; seek = seek + 1)
	{
		this->MatRefSet[seek].iSub(Start.x, End.x, Tensor[(seek - Start.z) % Base], bRepeat);
	}
}
template<class _data, class _data_ptr, class _data_ref>
void _TENSOR_REF<_data, _data_ptr, _data_ref>::iMul(const _M_TYPE::_ref_type &Tensor, const bool bRepeat)
{
	this->iMul(_M_TYPE::_dimension(0, 0, 0), this->Dimension, Tensor, bRepeat);
}
template<class _data, class _data_ptr, class _data_ref>
void _TENSOR_REF<_data, _data_ptr, _data_ref>::iMul(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_ref_type &Tensor, const bool bRepeat)
{
	const _M_TYPE::_size Size = bRepeat ? ::gloMin(this->Dimension.z, End.z) : ::gloMin(::gloMin(this->Dimension.z, Tensor.Dimension.z + Start.z), End.z);
	const _M_TYPE::_size Base = Tensor.Dimension.z;

	for (_M_TYPE::_size seek = Start.z; seek < Size; seek = seek + 1)
	{
		this->MatRefSet[seek].iMul(Start.x, End.x, Tensor[(seek - Start.z) % Base], bRepeat);
	}
}
template<class _data, class _data_ptr, class _data_ref>
void _TENSOR_REF<_data, _data_ptr, _data_ref>::iSet(const _M_TYPE::_ref_type &Tensor, const bool bRepeat)
{
	this->iSet(_M_TYPE::_dimension(0, 0, 0), this->Dimension, Tensor, bRepeat);
}
template<class _data, class _data_ptr, class _data_ref>
void _TENSOR_REF<_data, _data_ptr, _data_ref>::iSet(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_ref_type &Tensor, const bool bRepeat)
{
	const _M_TYPE::_size Size = bRepeat ? ::gloMin(this->Dimension.z, End.z) : ::gloMin(::gloMin(this->Dimension.z, Tensor.Dimension.z + Start.z), End.z);
	const _M_TYPE::_size Base = Tensor.Dimension.z;

	for (_M_TYPE::_size seek = Start.z; seek < Size; seek = seek + 1)
	{
		this->MatRefSet[seek].iSet(Start.x, End.x, Tensor[(seek - Start.z) % Base], bRepeat);
	}
}
template<class _data, class _data_ptr, class _data_ref>
void _TENSOR_REF<_data, _data_ptr, _data_ref>::iConv(_M_TYPE::_ref_type &DestTensor, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Tensor, const bool bRepeat) const
{
	this->iConv(DestTensor, _M_TYPE::_dimension(0, 0, 0), this->Dimension, BlockSize, Tensor, bRepeat);
}
template<class _data, class _data_ptr, class _data_ref>
void _TENSOR_REF<_data, _data_ptr, _data_ref>::iConv(_M_TYPE::_ref_type &DestTensor, const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Tensor, const bool bRepeat) const
{
	const auto Size = this->_ConvCheck(Start, End, BlockSize, DestTensor.Dimension);
	const _M_TYPE::_size Base = Tensor.iGetDimension().z;

	_M_TYPE::_size Pos = 0;

	const auto MinDimension = (End - Start) - BlockSize + 1;
	_matrix<_data> Mat(typename _matrix<_data>::_dimension(MinDimension), 0.0);

	for (_M_TYPE::_size seek_z = Start.z; seek_z < Size.z; seek_z = seek_z + 1, Pos = Pos + 1)
	{
		for (_M_TYPE::_size seek = 0; seek < BlockSize.z; seek = seek + 1)
		{
			_M_TYPE::_mat_ref CurrentMat(typename _matrix<_data>::_dimension(this->Dimension), this->MatRefSet[seek_z + seek].iGetContainer());
			CurrentMat.iConv(Mat, typename _matrix<_data>::_dimension(BlockSize), Tensor[seek % Base], bRepeat);

			DestTensor[Pos].iAdd(Mat);
		}
	}
}
template<class _data, class _data_ptr, class _data_ref>
void _TENSOR_REF<_data, _data_ptr, _data_ref>::iAdd(const _data &Data)
{
	this->iAdd(_M_TYPE::_dimension(0, 0, 0), this->Dimension, Data);
}
template<class _data, class _data_ptr, class _data_ref>
void _TENSOR_REF<_data, _data_ptr, _data_ref>::iAdd(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _data &Data)
{
	_M_TYPE::_size Size = ::gloMin(this->Dimension.z, End.z);

	for (_M_TYPE::_size seek = Start.z; seek < Size; seek = seek + 1)
	{
		this->MatRefSet[seek].iAdd(_M_TYPE::_mat_ref::_dimension(Start), _M_TYPE::_mat_ref::_dimension(End), Data);
	}
}
template<class _data, class _data_ptr, class _data_ref>
void _TENSOR_REF<_data, _data_ptr, _data_ref>::iSub(const _data &Data)
{
	this->iSub(_M_TYPE::_dimension(0, 0, 0), this->Dimension, Data);
}
template<class _data, class _data_ptr, class _data_ref>
void _TENSOR_REF<_data, _data_ptr, _data_ref>::iSub(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _data &Data)
{
	_M_TYPE::_size Size = ::gloMin(this->Dimension.z, End.z);

	for (_M_TYPE::_size seek = Start.z; seek < Size; seek = seek + 1)
	{
		this->MatRefSet[seek].iSub(_M_TYPE::_mat_ref::_dimension(Start), _M_TYPE::_mat_ref::_dimension(End), Data);
	}
}
template<class _data, class _data_ptr, class _data_ref>
void _TENSOR_REF<_data, _data_ptr, _data_ref>::iMul(const _data &Data)
{
	this->iMul(_M_TYPE::_dimension(0.0, 0.0, 0.0), this->Dimension, Data);
}
template<class _data, class _data_ptr, class _data_ref>
void _TENSOR_REF<_data, _data_ptr, _data_ref>::iMul(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _data &Data)
{
	_M_TYPE::_size Size = ::gloMin(this->Dimension.z, End.z);

	for (_M_TYPE::_size seek = Start.z; seek < Size; seek = seek + 1)
	{
		this->MatRefSet[seek].iMul(_M_TYPE::_mat_ref::_dimension(Start), _M_TYPE::_mat_ref::_dimension(End), Data);
	}
}
template<class _data, class _data_ptr, class _data_ref>
void _TENSOR_REF<_data, _data_ptr, _data_ref>::iSet(const _data &Data)
{
	this->iSet(_M_TYPE::_dimension(0.0, 0.0, 0.0), this->Dimension, Data);
}
template<class _data, class _data_ptr, class _data_ref>
void _TENSOR_REF<_data, _data_ptr, _data_ref>::iSet(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _data &Data)
{
	_M_TYPE::_size Size = ::gloMin(this->Dimension.z, End.z);

	for (_M_TYPE::_size seek = Start.z; seek < Size; seek = seek + 1)
	{
		this->MatRefSet[seek].iSet(_M_TYPE::_mat_ref::_dimension(Start), _M_TYPE::_mat_ref::_dimension(End), Data);
	}
}
template<class _data, class _data_ptr, class _data_ref>
_data _TENSOR_REF<_data, _data_ptr, _data_ref>::iSum() const
{
	return this->iSum(_M_TYPE::_dimension(0, 0, 0), this->Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_data _TENSOR_REF<_data, _data_ptr, _data_ref>::iSum(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End) const
{
	_data Sum = 0.0;

	const _M_TYPE::_size Size = ::gloMin(this->Dimension.z, End.z);
	for (_M_TYPE::_size seek = Start.z; seek < Size; seek = seek + 1)
	{
		Sum = Sum + this->MatRefSet[seek].iSum(_M_TYPE::_mat_ref::_dimension(Start), _M_TYPE::_mat_ref::_dimension(End));
	}

	return Sum;
}
template<class _data, class _data_ptr, class _data_ref>
_data _TENSOR_REF<_data, _data_ptr, _data_ref>::iProduct() const
{
	return this->iProduct(_M_TYPE::_dimension(0, 0, 0), this->Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_data _TENSOR_REF<_data, _data_ptr, _data_ref>::iProduct(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End) const
{
	_data Product = 1.0;

	const _M_TYPE::_size Size = ::gloMin(this->Dimension.z, End.z);
	for (_M_TYPE::_size seek = Start.z; seek < Size; seek = seek + 1)
	{
		Product = Product * this->MatRefSet[seek].iProduct(_M_TYPE::_mat_ref::_dimension(Start), _M_TYPE::_mat_ref::_dimension(End));
	}

	return Product;
}
template<class _data, class _data_ptr, class _data_ref>
_data _TENSOR_REF<_data, _data_ptr, _data_ref>::iMin() const
{
	return this->iMin(_M_TYPE::_dimension(0, 0, 0), this->Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_data _TENSOR_REF<_data, _data_ptr, _data_ref>::iMin(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End) const
{
	if (Start >= this->Dimension)
	{
		_SDEBUG(::cout << "Error: size overflow, file: tensor.cpp, func: iMin\r\n";)
		return 0.0;
	}

	_data Min = this->MatRefSet[Start.z][Start.y][Start.x];

	const _M_TYPE::_size Size = ::gloMin(this->Dimension.z, End.z);
	for (_M_TYPE::_size seek = Start.z; seek < Size; seek = seek + 1)
	{
		Min = ::gloMin(Min, this->MatRefSet[seek].iMin(_M_TYPE::_mat_ref::_dimension(Start), _M_TYPE::_mat_ref::_dimension(End)));
	}

	return Min;
}
template<class _data, class _data_ptr, class _data_ref>
_data _TENSOR_REF<_data, _data_ptr, _data_ref>::iMax() const
{
	return this->iMax(_M_TYPE::_dimension(0, 0, 0), this->Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_data _TENSOR_REF<_data, _data_ptr, _data_ref>::iMax(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End) const
{
	if (Start >= this->Dimension)
	{
		_SDEBUG(::cout << "Error: size overflow, file: tensor.cpp, func: iMax\r\n";)
		return 0.0;
	}

	_data Max = this->MatRefSet[Start.z][Start.y][Start.x];

	const _M_TYPE::_size Size = ::gloMin(this->Dimension.z, End.z);
	for (_M_TYPE::_size seek = Start.z; seek < Size; seek = seek + 1)
	{
		Max = ::gloMax(Max, this->MatRefSet[seek].iMax(_M_TYPE::_mat_ref::_dimension(Start), _M_TYPE::_mat_ref::_dimension(End)));
	}

	return Max;
}
template<class _data, class _data_ptr, class _data_ref>
void _TENSOR_REF<_data, _data_ptr, _data_ref>::iNormalize()
{

}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _TENSOR_REF<_data, _data_ptr, _data_ref>::iSetTensor(const _M_TYPE::_dimension &Dimension, const _data_ptr pData, const _M_TYPE::_size Size)
{
	if (Dimension == _M_TYPE::_dimension(0, 0, 0))
	{
		this->MatRefSet.iReSizeStream(0);
		this->Dimension = Dimension;

		return this->Dimension;
	}

	//if (pData == nullptr){ return this->Dimension; }
	if (Size < (Dimension.x * Dimension.y * Dimension.z)){ return this->Dimension; }
	if ((Dimension.x == 0) || (Dimension.y == 0) || (Dimension.z == 0)){ return this->Dimension; }

	this->MatRefSet.iReSizeStream(Dimension.z);

	_data_ptr pCurrentPtr = pData;
	_M_TYPE::_size Offset = pData == nullptr ? 0 : (Dimension.y * Dimension.x);

	for (_M_TYPE::_size seek = 0; seek < Dimension.z; seek = seek + 1, pCurrentPtr = pCurrentPtr + Offset)
	{
		this->MatRefSet[seek] = _M_TYPE::_mat_ref(_M_TYPE::_mat_ref::_dimension(Dimension), pCurrentPtr, Size - Offset);
	}

	this->Dimension = Dimension;

	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _TENSOR_REF<_data, _data_ptr, _data_ref>::iSetTensor(const _M_TYPE::_dimension &Dimension, const _sstream<_data> &DataStream)
{
	return this->iSetTensor(Dimension, DataStream.iGetPtr(), DataStream.iGetSize());
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _TENSOR_REF<_data, _data_ptr, _data_ref>::iSetTensor(const _M_TYPE::_dimension &Dimension, const _data_ptr** pDataPtrArray, const _M_TYPE::_dimension Size)
{
	if (Dimension == _M_TYPE::_dimension(0, 0, 0))
	{
		this->MatRefSet.iReSizeStream(0);
		this->Dimension = Dimension;

		return this->Dimension;
	}

	if (Size.z < Dimension.z){ return this->Dimension; }
	if ((Dimension.x == 0) || (Dimension.y == 0) || (Dimension.z == 0)){ return this->Dimension; }

	this->MatRefSet.iReSizeStream(Dimension.z);
	for (_M_TYPE::_size seek = 0; seek < Dimension.z; seek = seek + 1)
	{
		this->MatRefSet[seek] = _M_TYPE::_mat_ref(_M_TYPE::_mat_ref::_dimension(Dimension), pDataPtrArray[seek], _M_TYPE::_mat_ref::_dimension(Size));
	}

	this->Dimension = Dimension;

	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _TENSOR_REF<_data, _data_ptr, _data_ref>::iSetTensor(const _M_TYPE::_dimension &Dimension, const vector<vector<_data_ptr>> &DataPtrArray)
{
	if (Dimension == _M_TYPE::_dimension(0, 0, 0))
	{
		this->MatRefSet.iReSizeStream(0);
		this->Dimension = Dimension;

		return this->Dimension;
	}

	if (DataPtrArray.size() < Dimension.z){ return this->Dimension; }
	if ((Dimension.x == 0) || (Dimension.y == 0) || (Dimension.z == 0)){ return this->Dimension; }

	this->MatRefSet.iReSizeStream(Dimension.z);
	for (_M_TYPE::_size seek = 0; seek < Dimension.z; seek = seek + 1)
	{
		this->MatRefSet[seek] = _M_TYPE::_mat_ref(_M_TYPE::_mat_ref::_dimension(Dimension), DataPtrArray[seek]);
	}

	this->Dimension = Dimension;

	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _TENSOR_REF<_data, _data_ptr, _data_ref>::iSetTensor(const _M_TYPE::_dimension &Dimension, const _M_TYPE::_mat_ref* pMatrixArray, const _M_TYPE::_size Size)
{
	if (Dimension == _M_TYPE::_dimension(0, 0, 0))
	{
		this->MatRefSet.iReSizeStream(0);
		this->Dimension = Dimension;

		return this->Dimension;
	}

	if (pMatrixArray == nullptr){ return this->Dimension; }
	if (Size < Dimension.z){ return this->Dimension; }
	if ((Dimension.x == 0) || (Dimension.y == 0) || (Dimension.z == 0)){ return this->Dimension; }

	this->MatRefSet.iReSizeStream(Dimension.z);
	for (_M_TYPE::_size seek = 0; seek < Dimension.z; seek = seek + 1)
	{
		this->MatRefSet[seek] = _M_TYPE::_mat_ref(_M_TYPE::_mat_ref::_dimension(Dimension), pMatrixArray[seek].iGetContainer());
	}

	this->Dimension = Dimension;

	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _TENSOR_REF<_data, _data_ptr, _data_ref>::iSetTensor(const _M_TYPE::_dimension &Dimension, const vector<_M_TYPE::_mat_ref> &MatrixArray)
{
	if (Dimension == _M_TYPE::_dimension(0, 0, 0))
	{
		this->MatRefSet.iReSizeStream(0);
		this->Dimension = Dimension;

		return this->Dimension;
	}

	if (MatrixArray.size() < Dimension.z){ return this->Dimension; }
	if ((Dimension.x == 0) || (Dimension.y == 0) || (Dimension.z == 0)){ return this->Dimension; }

	this->MatRefSet.iReSizeStream(Dimension.z);
	for (_M_TYPE::_size seek = 0; seek < Dimension.z; seek = seek + 1)
	{
		this->MatRefSet[seek] = _M_TYPE::_mat_ref(_M_TYPE::_mat_ref::_dimension(Dimension), MatrixArray[seek].iGetContainer());
	}

	this->Dimension = Dimension;

	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _TENSOR_REF<_data, _data_ptr, _data_ref>::iSetDimension(const _M_TYPE::_dimension &Dimension)
{
	if (Dimension == _M_TYPE::_dimension(0, 0, 0))
	{
		this->MatRefSet.iReSizeStream(0);
		this->Dimension = Dimension;

		return this->Dimension;
	}

	if ((Dimension.x == 0) || (Dimension.y == 0) || (Dimension.z == 0)){ return this->Dimension; }

	this->MatRefSet.iReSizeStream(Dimension.z);
	this->Dimension = Dimension;

	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
vector<_M_TYPE::_mat_ref> _TENSOR_REF<_data, _data_ptr, _data_ref>::iGetContainer() const
{
	vector<_M_TYPE::_mat_ref> Container(this->MatRefSet.iGetSize());

	for (_M_TYPE::_size seek = 0; seek < this->MatRefSet.iGetSize(); seek = seek + 1)
	{
		Container[seek] = this->MatRefSet[seek];
	}

	return Container;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _TENSOR_REF<_data, _data_ptr, _data_ref>::iGetDimension() const
{
	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_size _TENSOR_REF<_data, _data_ptr, _data_ref>::iGetSize() const
{
	return this->Dimension.x * this->Dimension.y * this->Dimension.z;
}
#undef _M_TYPE
#define _M_TYPE typename _TENSOR<_data, _data_ptr, _data_ref>
template<class _data, class _data_ptr, class _data_ref>
_TENSOR<_data, _data_ptr, _data_ref>::_TENSOR()
	:DataContainer(0), _TENSOR_REF<_data, _data_ptr, _data_ref>()
{
}
template<class _data, class _data_ptr, class _data_ref>
_TENSOR<_data, _data_ptr, _data_ref>::_TENSOR(const _M_TYPE::_dimension &Dimension, const _data &Val)
	:DataContainer(Dimension.x * Dimension.y * Dimension.z, Val), _TENSOR_REF<_data, _data_ptr, _data_ref>(Dimension, (_data_ptr)nullptr, 0)
{
	this->_SetTensor(Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_TENSOR<_data, _data_ptr, _data_ref>::_TENSOR(const _M_TYPE::_dimension &Dimension, const _data_ptr pData, const _M_TYPE::_size Size)
	:DataContainer(Dimension.x * Dimension.y * Dimension.z), _TENSOR_REF<_data, _data_ptr, _data_ref>(Dimension, (_data_ptr)nullptr, 0)
{
	this->iSetTensor(Dimension, pData, Size);
}
template<class _data, class _data_ptr, class _data_ref>
_TENSOR<_data, _data_ptr, _data_ref>::_TENSOR(const _M_TYPE::_dimension &Dimension, const _sstream<_data> &DataStream)
	:DataContainer(Dimension.x * Dimension.y * Dimension.z), _TENSOR_REF<_data, _data_ptr, _data_ref>(Dimension, (_data_ptr)nullptr, 0)
{
	this->iSetTensor(Dimension, DataStream);
}
template<class _data, class _data_ptr, class _data_ref>
_TENSOR<_data, _data_ptr, _data_ref>::_TENSOR(const _M_TYPE::_dimension &Dimension, const _data_ptr** pDataPtrArray, const _M_TYPE::_dimension Size)
	:DataContainer(Dimension.x * Dimension.y * Dimension.z), _TENSOR_REF<_data, _data_ptr, _data_ref>(Dimension, (_data_ptr)nullptr, 0)
{
		this->iSetTensor(Dimension, pDataPtrArray, Size);
}
template<class _data, class _data_ptr, class _data_ref>
_TENSOR<_data, _data_ptr, _data_ref>::_TENSOR(const _M_TYPE::_dimension &Dimension, const vector<vector<_data_ptr>> &DataPtrArray)
	:DataContainer(Dimension.x * Dimension.y * Dimension.z), _TENSOR_REF<_data, _data_ptr, _data_ref>(Dimension, (_data_ptr)nullptr, 0)
{
	this->iSetTensor(Dimension, DataPtrArray);
}
template<class _data, class _data_ptr, class _data_ref>
_TENSOR<_data, _data_ptr, _data_ref>::_TENSOR(const _M_TYPE::_dimension &Dimension, const _M_TYPE::_mat_ref* pMatrixArray, const _M_TYPE::_size Size)
: DataContainer(Dimension.x * Dimension.y * Dimension.z), _TENSOR_REF<_data, _data_ptr, _data_ref>(Dimension, (_data_ptr)nullptr, 0)
{
	this->iSetTensor(Dimension, pMatrixArray, Size);
}
template<class _data, class _data_ptr, class _data_ref>
_TENSOR<_data, _data_ptr, _data_ref>::_TENSOR(const _M_TYPE::_dimension &Dimension, const vector<_M_TYPE::_mat_ref> &MatrixArray)
: DataContainer(Dimension.x * Dimension.y * Dimension.z), _TENSOR_REF<_data, _data_ptr, _data_ref>(Dimension, (_data_ptr)nullptr, 0)
{
	this->iSetTensor(Dimension, MatrixArray);
}
template<class _data, class _data_ptr, class _data_ref>
_TENSOR<_data, _data_ptr, _data_ref>::_TENSOR(const _type &Tensor)
	:DataContainer(Tensor.DataContainer), _TENSOR_REF<_data, _data_ptr, _data_ref>(Tensor.iGetDimension(), (_data_ptr)nullptr, 0)
{
	this->_SetTensor(Tensor.iGetDimension());
}
template<class _data, class _data_ptr, class _data_ref>
_TENSOR<_data, _data_ptr, _data_ref>::_TENSOR(const _M_TYPE::_ref_type &Tensor)
	:DataContainer(Tensor.iGetDimension().x * Tensor.iGetDimension().y * Tensor.iGetDimension().z), _TENSOR_REF<_data, _data_ptr, _data_ref>(Tensor.iGetDimension(), (_data_ptr)nullptr, 0)
{
	this->iSetTensor(Tensor.iGetDimension(), Tensor.iGetContainer());
}
template<class _data, class _data_ptr, class _data_ref>
_TENSOR<_data, _data_ptr, _data_ref>::~_TENSOR()
{
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _TENSOR<_data, _data_ptr, _data_ref>::_SetTensor(const _M_TYPE::_dimension &Dimension)
{
	this->MatRefSet.iReSizeStream(Dimension.z);

	_data_ptr pCurrentPtr = this->DataContainer.iGetPtr();
	_M_TYPE::_size Offset = pCurrentPtr == nullptr ? 0 : (Dimension.y * Dimension.x);

	for (_M_TYPE::_size seek = 0; seek < Dimension.z; seek = seek + 1, pCurrentPtr = pCurrentPtr + Offset)
	{
		this->MatRefSet[seek] = _M_TYPE::_mat_ref(_M_TYPE::_mat_ref::_dimension(Dimension), pCurrentPtr, this->DataContainer.iGetSize() - Offset);
	}

	this->Dimension = Dimension;

	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type& _TENSOR<_data, _data_ptr, _data_ref>::operator=(const _M_TYPE::_type &Tensor)
{
	this->DataContainer = Tensor.DataContainer;
	this->_SetTensor(Tensor.Dimension);

	return *this;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type& _TENSOR<_data, _data_ptr, _data_ref>::operator=(const _M_TYPE::_ref_type &Tensor)
{
	const _M_TYPE::_dimension Dimension = Tensor.iGetDimension();

	this->DataContainer.iReSizeStream(Dimension.x * Dimension.y * Dimension.z);
	
	for (_M_TYPE::_size seek_z = 0; seek_z < Dimension.z; seek_z = seek_z + 1)
	{
		const _M_TYPE::_mat_ref &Mat = Tensor[seek_z];
		const _M_TYPE::_size Offset = seek_z * Dimension.x * Dimension.y;

		for (_M_TYPE::_size seek_y = 0; seek_y < Dimension.y; seek_y = seek_y + 1)
		{
			this->DataContainer.iSetStream(Offset + seek_y * Dimension.x, Mat[seek_y].iGetContainer(), Dimension.x);
		}
	}

	this->_SetTensor(Tensor.iGetDimension());

	return *this;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _TENSOR<_data, _data_ptr, _data_ref>::operator+(const _type &Tensor) const
{
	return *this + Tensor.iGetRef();
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _TENSOR<_data, _data_ptr, _data_ref>::operator-(const _type &Tensor) const
{
	return *this - Tensor.iGetRef();
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _TENSOR<_data, _data_ptr, _data_ref>::operator*(const _type &Tensor) const
{
	return *this * Tensor.iGetRef();
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _TENSOR<_data, _data_ptr, _data_ref>::operator+(const _M_TYPE::_ref_type &Tensor) const
{
	_type DestTensor(*this);
	DestTensor.iAdd(Tensor);

	return DestTensor;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _TENSOR<_data, _data_ptr, _data_ref>::operator-(const _M_TYPE::_ref_type &Tensor) const
{
	_type DestTensor(*this);
	DestTensor.iSub(Tensor);

	return DestTensor;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _TENSOR<_data, _data_ptr, _data_ref>::operator*(const _M_TYPE::_ref_type &Tensor) const
{
	_type DestTensor(*this);
	DestTensor.iMul(Tensor);

	return DestTensor;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _TENSOR<_data, _data_ptr, _data_ref>::operator+(const _data &Data) const
{
	_type Tensor(*this);
	Tensor.iAdd(Data);

	return Tensor;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _TENSOR<_data, _data_ptr, _data_ref>::operator-(const _data &Data) const
{
	_type Tensor(*this);
	Tensor.iSub(Data);

	return Tensor;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _TENSOR<_data, _data_ptr, _data_ref>::operator*(const _data &Data) const
{
	_type Tensor(*this);
	Tensor.iMul(Data);

	return Tensor;
}
template<class _data, class _data_ptr, class _data_ref>
bool _TENSOR<_data, _data_ptr, _data_ref>::operator==(const _type &Tensor) const
{
	return this->iGetRef() == Tensor.iGetRef();
}
template<class _data, class _data_ptr, class _data_ref>
bool _TENSOR<_data, _data_ptr, _data_ref>::operator!=(const _type &Tensor) const
{
	return this->iGetRef() != Tensor.iGetRef();
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _TENSOR<_data, _data_ptr, _data_ref>::iConv(const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Tensor, const bool bRepeat) const
{
	return this->iConv(_M_TYPE::_dimension(0, 0, 0), this->Dimension, BlockSize, Tensor, bRepeat);
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _TENSOR<_data, _data_ptr, _data_ref>::iConv(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Tensor, const bool bRepeat) const
{
	_type DestTensor(this->Dimension, 0.0);
	this->iGetRef().iConv(DestTensor.iGetRef(), Start, End, BlockSize, Tensor, bRepeat);

	return DestTensor;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _TENSOR<_data, _data_ptr, _data_ref>::iSetTensor(const _M_TYPE::_dimension &Dimension, const _data_ptr pData, const _M_TYPE::_size Size)
{
	if (pData == nullptr){ return this->Dimension; }
	if (Size < (Dimension.x * Dimension.y * Dimension.z)){ return this->Dimension; }

	if (Dimension == _M_TYPE::_dimension(0, 0, 0)){ return this->_SetTensor(Dimension); }

	if ((Dimension.x == 0) || (Dimension.y == 0) || (Dimension.z)){ return this->Dimension; }

	this->DataContainer.iReSizeStream(Dimension.x * Dimension.y * Dimension.z);
	this->DataContainer.iSetStream(0, pData, this->DataContainer.iGetSize());

	return this->_SetTensor(Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _TENSOR<_data, _data_ptr, _data_ref>::iSetTensor(const _M_TYPE::_dimension &Dimension, const _sstream<_data> &DataStream)
{
	return this->iSetTensor(Dimension, DataStream.iGetPtr(), DataStream.iGetSize());
};
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _TENSOR<_data, _data_ptr, _data_ref>::iSetTensor(const _M_TYPE::_dimension &Dimension, const _data_ptr** pDataPtrArray, const _M_TYPE::_dimension ArraySize)
{
	if (pDataPtrArray == nullptr){ return this->Dimension; }
	if (ArraySize.z < Dimension.z){ return this->Dimension; }

	if (Dimension == _M_TYPE::_dimension(0, 0, 0)){ return this->_SetTensor(Dimension); }

	if ((Dimension.x == 0) || (Dimension.y == 0) || (Dimension.z)){ return this->Dimension; }

	this->DataContainer.iReSizeStream(Dimension.x * Dimension.y * Dimension.z);

	for (_M_TYPE::_size seek_z = 0; seek_z < Dimension.z; seek_z = seek_z + 1)
	{
		const _data_ptr* pMat = pDataPtrArray[seek_z];
		const _M_TYPE::_size Offset = seek_z * Dimension.x * Dimension.y;

		for (_M_TYPE::_size seek_y = 0; seek_y < Dimension.y; seek_y = seek_y + 1)
		{
			this->DataContainer.iSetStream(Offset + seek_y * Dimension.x, pMat[seek_y], Dimension.x);
		}
	}

	return this->_SetTensor(Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _TENSOR<_data, _data_ptr, _data_ref>::iSetTensor(const _M_TYPE::_dimension &Dimension, const vector<vector<_data_ptr>> &DataPtrArray)
{
	if (DataPtrArray.size() < Dimension.z){ return this->Dimension; }

	if (Dimension == _M_TYPE::_dimension(0, 0, 0)){ return this->_SetTensor(Dimension); }

	if ((Dimension.x == 0) || (Dimension.y == 0) || (Dimension.z)){ return this->Dimension; }

	this->DataContainer.iReSizeStream(Dimension.x * Dimension.y * Dimension.z);
	
	for (_M_TYPE::_size seek_z = 0; seek_z < Dimension.z; seek_z = seek_z + 1)
	{
		const vector<_data_ptr> &Mat = DataPtrArray[seek_z];
		const _M_TYPE::_size Offset = seek_z * Dimension.x * Dimension.y;

		for (_M_TYPE::_size seek_y = 0; seek_y < Dimension.y; seek_y = seek_y + 1)
		{
			this->DataContainer.iSetStream(Offset + seek_y * Dimension.x, Mat[seek_y], Dimension.x);
		}
	}

	return this->_SetTensor(Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _TENSOR<_data, _data_ptr, _data_ref>::iSetTensor(const _M_TYPE::_dimension &Dimension, const _M_TYPE::_mat_ref* pMatrixArray, const _M_TYPE::_size Size)
{
	if (pMatrixArray == nullptr){ return this->Dimension; }
	if (Size < Dimension.z){ return this->Dimension; }

	if (Dimension == _M_TYPE::_dimension(0, 0, 0)){ return this->_SetTensor(Dimension); }

	if ((Dimension.x == 0) || (Dimension.y == 0) || (Dimension.z)){ return this->Dimension; }

	this->DataContainer.iReSizeStream(Dimension.x * Dimension.y * Dimension.z);

	for (_M_TYPE::_size seek_z = 0; seek_z < Dimension.z; seek_z = seek_z + 1)
	{
		const _M_TYPE::_mat_ref &MatRef = pMatrixArray[seek_z];
		const _M_TYPE::_size Offset = seek_z * Dimension.x * Dimension.y;

		for (_M_TYPE::_size seek_y = 0; seek_y < Dimension.y; seek_y = seek_y + 1)
		{
			this->DataContainer.iSetStream(Offset + seek_y * Dimension.x, MatRef[seek_y].iGetContainer(), Dimension.x);
		}
	}

	return this->_SetTensor(Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _TENSOR<_data, _data_ptr, _data_ref>::iSetTensor(const _M_TYPE::_dimension &Dimension, const vector<_M_TYPE::_mat_ref> &MatrixArray)
{
	if (MatrixArray.size() < Dimension.z){ return this->Dimension; }

	if (Dimension == _M_TYPE::_dimension(0, 0, 0)){ return this->_SetTensor(Dimension); }

	if ((Dimension.x == 0) || (Dimension.y == 0) || (Dimension.z)){ return this->Dimension; }

	this->DataContainer.iReSizeStream(Dimension.x * Dimension.y * Dimension.z);
	
	for (_M_TYPE::_size seek_z = 0; seek_z < Dimension.z; seek_z = seek_z + 1)
	{
		const _M_TYPE::_mat_ref &MatRef = MatrixArray[seek_z];
		const _M_TYPE::_size Offset = seek_z * Dimension.x * Dimension.y;

		for (_M_TYPE::_size seek_y = 0; seek_y < Dimension.y; seek_y = seek_y + 1)
		{
			this->DataContainer.iSetStream(Offset + seek_y * Dimension.x, MatRef[seek_y].iGetContainer(), Dimension.x);
		}
	}

	return this->_SetTensor(Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _TENSOR<_data, _data_ptr, _data_ref>::iSetDimension(const _M_TYPE::_dimension &Dimension)
{
	if (this->Dimension != Dimension)
	{
		this->DataContainer.iReSizeStream(Dimension.x * Dimension.y * Dimension.z);
		this->_SetTensor(Dimension);
	}

	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
const _data& _TENSOR<_data, _data_ptr, _data_ref>::iGetData(const _M_TYPE::_dimension &Index) const
{
	return this->MatRefSet[Index.z][Index.y][Index.x];
}
template<class _data, class _data_ptr, class _data_ref>
const _data& _TENSOR<_data, _data_ptr, _data_ref>::iGetData(const _M_TYPE::_size Index) const
{
	return this->DataContainer[Index];
}
template<class _data, class _data_ptr, class _data_ref>
_data& _TENSOR<_data, _data_ptr, _data_ref>::iGetData(const _M_TYPE::_dimension &Index)
{
	return this->MatRefSet[Index.z][Index.y][Index.x];
}
template<class _data, class _data_ptr, class _data_ref>
_data& _TENSOR<_data, _data_ptr, _data_ref>::iGetData(const _M_TYPE::_size Index)
{
	return this->DataContainer[Index];
}
template<class _data, class _data_ptr, class _data_ref>
const _M_TYPE::_ref_type& _TENSOR<_data, _data_ptr, _data_ref>::iGetRef() const
{
	return *this;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_ref_type& _TENSOR<_data, _data_ptr, _data_ref>::iGetRef()
{
	return *this;
}
#undef _M_TYPE