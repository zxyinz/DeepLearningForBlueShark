#include"Matrix.h"
using namespace std;
using namespace San;
using namespace San::Mathematics;
template struct _MATRIX_REF<float32, float32*, float32&>;
template struct _MATRIX_REF<float64, float64*, float64&>;

template struct _MATRIX<float32, float32*, float32&>;
template struct _MATRIX<float64, float64*, float64&>;

#define _M_TYPE typename _MATRIX_REF<_data, _data_ptr, _data_ref>
template<class _data, class _data_ptr, class _data_ref>
_MATRIX_REF<_data, _data_ptr, _data_ref>::_MATRIX_REF()
	:VecRefSet(), Dimension(0.0, 0.0)
{
}
template<class _data, class _data_ptr, class _data_ref>
_MATRIX_REF<_data, _data_ptr, _data_ref>::_MATRIX_REF(const _M_TYPE::_dimension &Dimension, const _data_ptr pData, const _M_TYPE::_size Size)
{
	this->iSetMatrix(Dimension, pData, Size);
}
template<class _data, class _data_ptr, class _data_ref>
_MATRIX_REF<_data, _data_ptr, _data_ref>::_MATRIX_REF(const _M_TYPE::_dimension &Dimension, const _data_ptr* pDataPtrArray, const _M_TYPE::_dimension Size)
{
	this->iSetMatrix(Dimension, pDataPtrArray, Size);
}
template<class _data, class _data_ptr, class _data_ref>
_MATRIX_REF<_data, _data_ptr, _data_ref>::_MATRIX_REF(const _M_TYPE::_dimension &Dimension, const vector<_data_ptr> &DataPtrArray)
{
	this->iSetMatrix(Dimension, DataPtrArray);
}
template<class _data, class _data_ptr, class _data_ref>
_MATRIX_REF<_data, _data_ptr, _data_ref>::_MATRIX_REF(const _M_TYPE::_dimension &Dimension, const _M_TYPE::_vec_ref* pVectorArray, const _M_TYPE::_size Size)
{
	this->iSetMatrix(Dimension, pVectorArray, Size);
}
template<class _data, class _data_ptr, class _data_ref>
_MATRIX_REF<_data, _data_ptr, _data_ref>::_MATRIX_REF(const _M_TYPE::_dimension &Dimension, const vector<_M_TYPE::_vec_ref> &VectorArray)
{
	this->iSetMatrix(Dimension, VectorArray);
}
template<class _data, class _data_ptr, class _data_ref>
_MATRIX_REF<_data, _data_ptr, _data_ref>::_MATRIX_REF(const _M_TYPE::_ref_type &Matrix)
	:VecRefSet(Matrix.VecRefSet), Dimension(Matrix.Dimension)
{
}
template<class _data, class _data_ptr, class _data_ref>
_MATRIX_REF<_data, _data_ptr, _data_ref>::~_MATRIX_REF()
{
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _MATRIX_REF<_data, _data_ptr, _data_ref>::_ConvCheck(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_dimension &DestMatrixSize) const
{
	_MATRIX_REF<_data, _data_ptr, _data_ref>::_dimension Zero(0, 0);

	if (End < Start){ return Zero; }
	if ((End - Start) < BlockSize){ return Zero; }
	if (((BlockSize.x % 2) != 1) || ((BlockSize.y % 2) != 1)){ return Zero; }

	const auto MinDimension = (End - Start) - BlockSize + 1;
	if (DestMatrixSize < MinDimension){ return Zero; }

	const auto Offset(BlockSize * 0.5);
	const auto Size = _M_TYPE::_dimension(min(this->Dimension.x, End.x), min(this->Dimension.y, End.y)) - (Offset * 2);

	return Size;
}
template<class _data, class _data_ptr, class _data_ref>
const _data_ref _MATRIX_REF<_data, _data_ptr, _data_ref>::operator[](const _M_TYPE::_dimension &Pos) const
{
	return this->VecRefSet[Pos.y][Pos.x];
}
template<class _data, class _data_ptr, class _data_ref>
_data_ref _MATRIX_REF<_data, _data_ptr, _data_ref>::operator[](const _M_TYPE::_dimension &Pos)
{
	return this->VecRefSet[Pos.y][Pos.x];
}
template<class _data, class _data_ptr, class _data_ref>
const _M_TYPE::_vec_ref& _MATRIX_REF<_data, _data_ptr, _data_ref>::operator[](const _M_TYPE::_size Index) const
{
	return this->VecRefSet[Index];
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_vec_ref& _MATRIX_REF<_data, _data_ptr, _data_ref>::operator[](const _M_TYPE::_size Index)
{
	return this->VecRefSet[Index];
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_ref_type& _MATRIX_REF<_data, _data_ptr, _data_ref>::operator=(const _M_TYPE::_ref_type &Matrix)
{
	this->VecRefSet = Matrix.VecRefSet;
	this->Dimension = Matrix.Dimension;

	return *this;
}
template<class _data, class _data_ptr, class _data_ref>
bool _MATRIX_REF<_data, _data_ptr, _data_ref>::operator==(const _M_TYPE::_ref_type &Matrix) const
{
	if (this->Dimension != Matrix.Dimension) { return false; }
	for (_M_TYPE::_size seek = 0; seek < this->Dimension.y; seek = seek + 1)
	{
		if (this->VecRefSet[seek] != Matrix.VecRefSet[seek]) { return false; }
	}

	return true;
}
template<class _data, class _data_ptr, class _data_ref>
bool _MATRIX_REF<_data, _data_ptr, _data_ref>::operator!=(const _M_TYPE::_ref_type &Matrix) const
{
	if (this->Dimension != Matrix.Dimension){ return true; }
	for (_M_TYPE::_size seek = 0; seek < this->Dimension.y; seek = seek + 1)
	{
		if (this->VecRefSet[seek] != Matrix.VecRefSet[seek]){ return true; }
	}

	return false;
}
template<class _data, class _data_ptr, class _data_ref>
void _MATRIX_REF<_data, _data_ptr, _data_ref>::iAdd(const _M_TYPE::_ref_type &Matrix, const bool bRepeat)
{
	this->iAdd(_M_TYPE::_dimension(0, 0), this->Dimension, Matrix, bRepeat);
}
template<class _data, class _data_ptr, class _data_ref>
void _MATRIX_REF<_data, _data_ptr, _data_ref>::iAdd(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_ref_type &Matrix, const bool bRepeat)
{
	const _M_TYPE::_size Size = bRepeat ? ::gloMin(this->Dimension.y, End.y) : ::gloMin(::gloMin(this->Dimension.y, Matrix.Dimension.y + Start.y), End.y);
	const _M_TYPE::_size Base = Matrix.Dimension.y;

	for (_M_TYPE::_size seek = Start.y; seek < Size; seek = seek + 1)
	{
		this->VecRefSet[seek].iAdd(Start.x, End.x, Matrix[(seek - Start.y) % Base], bRepeat);
	}
}
template<class _data, class _data_ptr, class _data_ref>
void _MATRIX_REF<_data, _data_ptr, _data_ref>::iSub(const _M_TYPE::_ref_type &Matrix, const bool bRepeat)
{
	this->iSub(_M_TYPE::_dimension(0, 0), this->Dimension, Matrix, bRepeat);
}
template<class _data, class _data_ptr, class _data_ref>
void _MATRIX_REF<_data, _data_ptr, _data_ref>::iSub(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_ref_type &Matrix, const bool bRepeat)
{
	const _M_TYPE::_size Size = bRepeat ? ::gloMin(this->Dimension.y, End.y) : ::gloMin(::gloMin(this->Dimension.y, Matrix.Dimension.y + Start.y), End.y);
	const _M_TYPE::_size Base = Matrix.Dimension.y;

	for (_M_TYPE::_size seek = Start.y; seek < Size; seek = seek + 1)
	{
		this->VecRefSet[seek].iSub(Start.x, End.x, Matrix[(seek - Start.y) % Base], bRepeat);
	}
}
template<class _data, class _data_ptr, class _data_ref>
void _MATRIX_REF<_data, _data_ptr, _data_ref>::iMul(const _M_TYPE::_ref_type &Matrix, const bool bRepeat)
{
	this->iMul(_M_TYPE::_dimension(0, 0), this->Dimension, Matrix, bRepeat);
}
template<class _data, class _data_ptr, class _data_ref>
void _MATRIX_REF<_data, _data_ptr, _data_ref>::iMul(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_ref_type &Matrix, const bool bRepeat)
{
	const _M_TYPE::_size Size = bRepeat ? ::gloMin(this->Dimension.y, End.y) : ::gloMin(::gloMin(this->Dimension.y, Matrix.Dimension.y + Start.y), End.y);
	const _M_TYPE::_size Base = Matrix.Dimension.y;

	for (_M_TYPE::_size seek = Start.y; seek < Size; seek = seek + 1)
	{
		this->VecRefSet[seek].iMul(Start.x, End.x, Matrix[(seek - Start.y) % Base], bRepeat);
	}
}
template<class _data, class _data_ptr, class _data_ref>
void _MATRIX_REF<_data, _data_ptr, _data_ref>::iSet(const _M_TYPE::_ref_type &Matrix, const bool bRepeat)
{
	this->iSet(_M_TYPE::_dimension(0, 0), this->Dimension, Matrix, bRepeat);
}
template<class _data, class _data_ptr, class _data_ref>
void _MATRIX_REF<_data, _data_ptr, _data_ref>::iSet(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_ref_type &Matrix, const bool bRepeat)
{
	const _M_TYPE::_size Size = bRepeat ? ::gloMin(this->Dimension.y, End.y) : ::gloMin(::gloMin(this->Dimension.y, Matrix.Dimension.y + Start.y), End.y);
	const _M_TYPE::_size Base = Matrix.Dimension.y;

	for (_M_TYPE::_size seek = Start.y; seek < Size; seek = seek + 1)
	{
		this->VecRefSet[seek].iSet(Start.x, End.x, Matrix[(seek - Start.y) % Base], bRepeat);
	}
}
template<class _data, class _data_ptr, class _data_ref>
void _MATRIX_REF<_data, _data_ptr, _data_ref>::iConv(_M_TYPE::_ref_type &DestMatrix, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Matrix, const bool bRepeat) const
{
	this->iConv(DestMatrix, _M_TYPE::_dimension(0, 0), this->Dimension, BlockSize, Matrix, bRepeat);
}
template<class _data, class _data_ptr, class _data_ref>
void _MATRIX_REF<_data, _data_ptr, _data_ref>::iConv(_M_TYPE::_ref_type &DestMatrix, const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Matrix, const bool bRepeat) const
{
	const auto Size = this->_ConvCheck(Start, End, BlockSize, DestMatrix.Dimension);
	const _M_TYPE::_size Base = Matrix.iGetDimension().y;

	_M_TYPE::_size Pos = 0;

	const auto MinDimension = (End - Start) - BlockSize + 1;
	_vector<_data> Vec(MinDimension.x, 0.0);

	for (_M_TYPE::_size seek_y = Start.y; seek_y < Size.y; seek_y = seek_y + 1, Pos = Pos + 1)
	{
		/*for (_M_TYPE::_size seek_x = Start.x; seek_x < Size.x; seek_x = seek_x + 1, Pos.x = Pos.x + 1)
		{
			for (_M_TYPE::_size seek = 0; seek < BlockSize.y; seek = seek + 1)
			{
				_M_TYPE::_vec_ref CurrentVec(this->VecRefSet[seek_y + seek].iGetContainer() + seek_x, BlockSize.x);
				CurrentVec.iConv(Vec, BlockSize.x, Matrix[seek % Base], bRepeat);

				Vec.iOperationC<>([](sfloat Val, _M_TYPE::_size Index){::cout << Val << " "; });
				::cout << "\r\n";

				DestMatrix[Pos.y].iAdd(Vec);
			}
		}*/

		for (_M_TYPE::_size seek = 0; seek < BlockSize.y; seek = seek + 1)
		{
			_M_TYPE::_vec_ref CurrentVec(this->Dimension.x, this->VecRefSet[seek_y + seek].iGetContainer());
			CurrentVec.iConv(Vec, BlockSize.x, Matrix[seek % Base], bRepeat);

			DestMatrix[Pos].iAdd(Vec);
		}
	}
}
template<class _data, class _data_ptr, class _data_ref>
void _MATRIX_REF<_data, _data_ptr, _data_ref>::iAdd(const _data &Data)
{
	this->iAdd(_M_TYPE::_dimension(0, 0), this->Dimension, Data);
}
template<class _data, class _data_ptr, class _data_ref>
void _MATRIX_REF<_data, _data_ptr, _data_ref>::iAdd(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _data &Data)
{
	_M_TYPE::_size Size = ::gloMin(this->Dimension.y, End.y);

	for (_M_TYPE::_size seek = Start.y; seek < Size; seek = seek + 1)
	{
		this->VecRefSet[seek].iAdd(Start.x, End.x, Data);
	}
}
template<class _data, class _data_ptr, class _data_ref>
void _MATRIX_REF<_data, _data_ptr, _data_ref>::iSub(const _data &Data)
{
	this->iSub(_M_TYPE::_dimension(0, 0), this->Dimension, Data);
}
template<class _data, class _data_ptr, class _data_ref>
void _MATRIX_REF<_data, _data_ptr, _data_ref>::iSub(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _data &Data)
{
	_M_TYPE::_size Size = ::gloMin(this->Dimension.y, End.y);

	for (_M_TYPE::_size seek = Start.y; seek < Size; seek = seek + 1)
	{
		this->VecRefSet[seek].iSub(Start.x, End.x, Data);
	}
}
template<class _data, class _data_ptr, class _data_ref>
void _MATRIX_REF<_data, _data_ptr, _data_ref>::iMul(const _data &Data)
{
	this->iMul(_M_TYPE::_dimension(0.0, 0.0), this->Dimension, Data);
}
template<class _data, class _data_ptr, class _data_ref>
void _MATRIX_REF<_data, _data_ptr, _data_ref>::iMul(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _data &Data)
{
	_M_TYPE::_size Size = ::gloMin(this->Dimension.y, End.y);

	for (_M_TYPE::_size seek = Start.y; seek < Size; seek = seek + 1)
	{
		this->VecRefSet[seek].iMul(Start.x, End.x, Data);
	}
}
template<class _data, class _data_ptr, class _data_ref>
void _MATRIX_REF<_data, _data_ptr, _data_ref>::iSet(const _data &Data)
{
	this->iSet(_M_TYPE::_dimension(0.0, 0.0), this->Dimension, Data);
}
template<class _data, class _data_ptr, class _data_ref>
void _MATRIX_REF<_data, _data_ptr, _data_ref>::iSet(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _data &Data)
{
	_M_TYPE::_size Size = ::gloMin(this->Dimension.y, End.y);

	for (_M_TYPE::_size seek = Start.y; seek < Size; seek = seek + 1)
	{
		this->VecRefSet[seek].iSet(Start.x, End.x, Data);
	}
}
template<class _data, class _data_ptr, class _data_ref>
_data _MATRIX_REF<_data, _data_ptr, _data_ref>::iSum() const
{
	return this->iSum(_M_TYPE::_dimension(0, 0), this->Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_data _MATRIX_REF<_data, _data_ptr, _data_ref>::iSum(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End) const
{
	_data Sum = 0.0;

	const _M_TYPE::_size Size = ::gloMin(this->Dimension.y, End.y);
	for (_M_TYPE::_size seek = Start.y; seek < Size; seek = seek + 1)
	{
		Sum = Sum + this->VecRefSet[seek].iSum(Start.x, End.x);
	}

	return Sum;
}
template<class _data, class _data_ptr, class _data_ref>
_data _MATRIX_REF<_data, _data_ptr, _data_ref>::iProduct() const
{
	return this->iProduct(_M_TYPE::_dimension(0, 0), this->Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_data _MATRIX_REF<_data, _data_ptr, _data_ref>::iProduct(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End) const
{
	_data Product = 1.0;

	const _M_TYPE::_size Size = ::gloMin(this->Dimension.y, End.y);
	for (_M_TYPE::_size seek = Start.y; seek < Size; seek = seek + 1)
	{
		Product = Product * this->VecRefSet[seek].iProduct(Start.x, End.x);
	}

	return Product;
}
template<class _data, class _data_ptr, class _data_ref>
_data _MATRIX_REF<_data, _data_ptr, _data_ref>::iMin() const
{
	return this->iMin(_M_TYPE::_dimension(0, 0), this->Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_data _MATRIX_REF<_data, _data_ptr, _data_ref>::iMin(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End) const
{
	if (Start >= this->Dimension)
	{
		_SDEBUG(::cout << "Error: size overflow, file: matrix.cpp, func: iMin\r\n";)
		return 0.0;
	}

	_data Min = this->VecRefSet[Start.y][Start.x];

	const _M_TYPE::_size Size = ::gloMin(this->Dimension.y, End.y);
	for (_M_TYPE::_size seek = Start.y; seek < Size; seek = seek + 1)
	{
		Min = ::gloMin(Min, this->VecRefSet[seek].iMin(Start.x, End.x));
	}

	return Min;
}
template<class _data, class _data_ptr, class _data_ref>
_data _MATRIX_REF<_data, _data_ptr, _data_ref>::iMax() const
{
	return this->iMax(_M_TYPE::_dimension(0, 0), this->Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_data _MATRIX_REF<_data, _data_ptr, _data_ref>::iMax(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End) const
{
	if (Start >= this->Dimension)
	{
		_SDEBUG(::cout << "Error: size overflow, file: matrix.cpp, func: iMax\r\n";)
		return 0.0;
	}

	_data Max = this->VecRefSet[Start.y][Start.x];

	const _M_TYPE::_size Size = ::gloMin(this->Dimension.y, End.y);
	for (_M_TYPE::_size seek = Start.y; seek < Size; seek = seek + 1)
	{
		Max = ::gloMax(Max, this->VecRefSet[seek].iMax(Start.x, End.x));
	}

	return Max;
}
template<class _data, class _data_ptr, class _data_ref>
void _MATRIX_REF<_data, _data_ptr, _data_ref>::iNormalize()
{

}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _MATRIX_REF<_data, _data_ptr, _data_ref>::iSetMatrix(const _M_TYPE::_dimension &Dimension, const _data_ptr &pData, const _M_TYPE::_size Size)
{
	if (Dimension == _M_TYPE::_dimension(0, 0))
	{
		this->VecRefSet.iReSizeStream(0);
		this->Dimension = Dimension;

		return this->Dimension;
	}

	//if (pData == nullptr){ return this->Dimension; }
	if (Size < (Dimension.x * Dimension.y)){ return this->Dimension; }
	if ((Dimension.x == 0) || (Dimension.y == 0)){ return this->Dimension; }

	this->VecRefSet.iReSizeStream(Dimension.y);

	_data_ptr pCurrentPtr = pData;
	_M_TYPE::_size Offset = pData == nullptr ? 0 : Dimension.x;

	for (_M_TYPE::_size seek = 0; seek < Dimension.y; seek = seek + 1, pCurrentPtr = pCurrentPtr + Offset)
	{
		this->VecRefSet[seek] = _M_TYPE::_vec_ref(Dimension.x, pCurrentPtr);
	}

	this->Dimension = Dimension;

	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _MATRIX_REF<_data, _data_ptr, _data_ref>::iSetMatrix(const _M_TYPE::_dimension &Dimension, const _data_ptr* pDataPtrArray, const _M_TYPE::_dimension Size)
{
	if (Dimension == _M_TYPE::_dimension(0, 0))
	{
		this->VecRefSet.iReSizeStream(0);
		this->Dimension = Dimension;

		return this->Dimension;
	}

	if (Size.y < Dimension.y){ return this->Dimension; }
	if ((Dimension.x == 0) || (Dimension.y == 0)){ return this->Dimension; }

	this->VecRefSet.iReSizeStream(Dimension.y);
	for (_M_TYPE::_size seek = 0; seek < Dimension.y; seek = seek + 1)
	{
		this->VecRefSet[seek] = _M_TYPE::_vec_ref(Dimension.x, pDataPtrArray[seek]);
	}

	this->Dimension = Dimension;

	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _MATRIX_REF<_data, _data_ptr, _data_ref>::iSetMatrix(const _M_TYPE::_dimension &Dimension, const vector<_data_ptr> &DataPtrArray)
{
	if (Dimension == _M_TYPE::_dimension(0, 0))
	{
		this->VecRefSet.iReSizeStream(0);
		this->Dimension = Dimension;

		return this->Dimension;
	}

	if (DataPtrArray.size() < Dimension.y){ return this->Dimension; }
	if ((Dimension.x == 0) || (Dimension.y == 0)){ return this->Dimension; }

	this->VecRefSet.iReSizeStream(Dimension.y);
	for (_M_TYPE::_size seek = 0; seek < Dimension.y; seek = seek + 1)
	{
		this->VecRefSet[seek] = _M_TYPE::_vec_ref(Dimension.x, DataPtrArray[seek]);
	}

	this->Dimension = Dimension;

	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _MATRIX_REF<_data, _data_ptr, _data_ref>::iSetMatrix(const _M_TYPE::_dimension &Dimension, const _M_TYPE::_vec_ref* pVectorArray, const _M_TYPE::_size Size)
{
	if (Dimension == _M_TYPE::_dimension(0, 0))
	{
		this->VecRefSet.iReSizeStream(0);
		this->Dimension = Dimension;

		return this->Dimension;
	}

	if (pVectorArray == nullptr){ return this->Dimension; }
	if (Size < Dimension.y){ return this->Dimension; }
	if ((Dimension.x == 0) || (Dimension.y == 0)){ return this->Dimension; }

	this->VecRefSet.iReSizeStream(Dimension.y);
	for (_M_TYPE::_size seek = 0; seek < Dimension.y; seek = seek + 1)
	{
		this->VecRefSet[seek] = pVectorArray[seek];
	}

	this->Dimension = Dimension;

	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _MATRIX_REF<_data, _data_ptr, _data_ref>::iSetMatrix(const _M_TYPE::_dimension &Dimension, const vector<_M_TYPE::_vec_ref> &VectorArray)
{
	if (Dimension == _M_TYPE::_dimension(0, 0))
	{
		this->VecRefSet.iReSizeStream(0);
		this->Dimension = Dimension;

		return this->Dimension;
	}

	if (VectorArray.size() < Dimension.y){ return this->Dimension; }
	if ((Dimension.x == 0) || (Dimension.y == 0)){ return this->Dimension; }

	this->VecRefSet.iReSizeStream(Dimension.y);
	for (_M_TYPE::_size seek = 0; seek < Dimension.y; seek = seek + 1)
	{
		this->VecRefSet[seek] = VectorArray[seek];
	}

	this->Dimension = Dimension;

	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _MATRIX_REF<_data, _data_ptr, _data_ref>::iSetDimension(const _M_TYPE::_dimension &Dimension)
{
	if (Dimension == _M_TYPE::_dimension(0, 0))
	{
		this->VecRefSet.iReSizeStream(0);
		this->Dimension = Dimension;

		return this->Dimension;
	}

	if ((Dimension.x == 0) || (Dimension.y == 0)){ return this->Dimension; }

	this->VecRefSet.iReSizeStream(Dimension.y);
	this->Dimension = Dimension;

	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
vector<_M_TYPE::_vec_ref> _MATRIX_REF<_data, _data_ptr, _data_ref>::iGetContainer() const
{
	vector<_M_TYPE::_vec_ref> Container(this->VecRefSet.iGetSize());

	for (_M_TYPE::_size seek = 0; seek < this->VecRefSet.iGetSize(); seek = seek + 1)
	{
		Container[seek] = this->VecRefSet[seek];
	}

	return Container;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _MATRIX_REF<_data, _data_ptr, _data_ref>::iGetDimension() const
{
	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_size _MATRIX_REF<_data, _data_ptr, _data_ref>::iGetSize() const
{
	return this->Dimension.x * this->Dimension.y;
}
#undef _M_TYPE
#define _M_TYPE typename _MATRIX<_data, _data_ptr, _data_ref>
template<class _data, class _data_ptr, class _data_ref>
_MATRIX<_data, _data_ptr, _data_ref>::_MATRIX()
	:DataContainer(0), _MATRIX_REF<_data, _data_ptr, _data_ref>()
{
}
template<class _data, class _data_ptr, class _data_ref>
_MATRIX<_data, _data_ptr, _data_ref>::_MATRIX(const _M_TYPE::_dimension &Dimension, const _data &Val)
	:DataContainer(Dimension.x * Dimension.y, Val), _MATRIX_REF<_data, _data_ptr, _data_ref>(Dimension, (_data_ptr)nullptr, 0)
{
	this->_SetMatrix(Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_MATRIX<_data, _data_ptr, _data_ref>::_MATRIX(const _M_TYPE::_dimension &Dimension, const _data_ptr pData, const _M_TYPE::_size Size)
	:DataContainer(Dimension.x * Dimension.y), _MATRIX_REF<_data, _data_ptr, _data_ref>(Dimension, (_data_ptr)nullptr, 0)
{
		this->iSetMatrix(Dimension, pData, Size);
}
template<class _data, class _data_ptr, class _data_ref>
_MATRIX<_data, _data_ptr, _data_ref>::_MATRIX(const _M_TYPE::_dimension &Dimension, const _data_ptr* pDataPtrArray, const _M_TYPE::_dimension Size)
	:DataContainer(Dimension.x * Dimension.y), _MATRIX_REF<_data, _data_ptr, _data_ref>(Dimension, (_data_ptr)nullptr, 0)
{
		this->iSetMatrix(Dimension, pDataPtrArray, Size);
}
template<class _data, class _data_ptr, class _data_ref>
_MATRIX<_data, _data_ptr, _data_ref>::_MATRIX(const _M_TYPE::_dimension &Dimension, const vector<_data_ptr> &DataPtrArray)
	:DataContainer(Dimension.x * Dimension.y), _MATRIX_REF<_data, _data_ptr, _data_ref>(Dimension, (_data_ptr)nullptr, 0)
{
		this->iSetMatrix(Dimension, DataPtrArray);
}
template<class _data, class _data_ptr, class _data_ref>
_MATRIX<_data, _data_ptr, _data_ref>::_MATRIX(const _M_TYPE::_dimension &Dimension, const _M_TYPE::_vec_ref* pVectorArray, const _M_TYPE::_size Size)
	:DataContainer(Dimension.x * Dimension.y), _MATRIX_REF<_data, _data_ptr, _data_ref>(Dimension, (_data_ptr)nullptr, 0)
{
		this->iSetMatrix(Dimension, pVectorArray, Size);
}
template<class _data, class _data_ptr, class _data_ref>
_MATRIX<_data, _data_ptr, _data_ref>::_MATRIX(const _M_TYPE::_dimension &Dimension, const vector<_M_TYPE::_vec_ref> &VectorArray)
	:DataContainer(Dimension.x * Dimension.y), _MATRIX_REF<_data, _data_ptr, _data_ref>(Dimension, (_data_ptr)nullptr, 0)
{
		this->iSetMatrix(Dimension, VectorArray);
}
template<class _data, class _data_ptr, class _data_ref>
_MATRIX<_data, _data_ptr, _data_ref>::_MATRIX(const _M_TYPE::_type &Matrix)
: DataContainer(Matrix.DataContainer), _MATRIX_REF<_data, _data_ptr, _data_ref>(Matrix.iGetDimension(), (_data_ptr)nullptr, 0)
{
	this->_SetMatrix(Matrix.iGetDimension());
}
template<class _data, class _data_ptr, class _data_ref>
_MATRIX<_data, _data_ptr, _data_ref>::_MATRIX(const _M_TYPE::_ref_type &Matrix)
: DataContainer(Matrix.iGetDimension().x * Matrix.iGetDimension().y), _MATRIX_REF<_data, _data_ptr, _data_ref>(Matrix.iGetDimension(), (_data_ptr)nullptr, 0)
{
	const vector<_M_TYPE::_vec_ref> &VecSet = Matrix.iGetContainer();
	this->iSetMatrix(Matrix.iGetDimension(), VecSet);
}
template<class _data, class _data_ptr, class _data_ref>
_MATRIX<_data, _data_ptr, _data_ref>::~_MATRIX()
{
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _MATRIX<_data, _data_ptr, _data_ref>::_SetMatrix(const _M_TYPE::_dimension &Dimension)
{
	this->VecRefSet.iReSizeStream(Dimension.y);

	_data_ptr pCurrentPtr = this->DataContainer.iGetPtr();
	_M_TYPE::_size Offset = pCurrentPtr == nullptr ? 0 : Dimension.x;

	for (_M_TYPE::_size seek = 0; seek < Dimension.y; seek = seek + 1, pCurrentPtr = pCurrentPtr + Offset)
	{
		this->VecRefSet[seek] = _M_TYPE::_vec_ref(Dimension.x, pCurrentPtr);
	}

	this->Dimension = Dimension;

	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type& _MATRIX<_data, _data_ptr, _data_ref>::operator=(const _M_TYPE::_type &Matrix)
{
	this->DataContainer = Matrix.DataContainer;
	this->_SetMatrix(Matrix.Dimension);

	return *this;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type& _MATRIX<_data, _data_ptr, _data_ref>::operator=(const _M_TYPE::_ref_type &Matrix)
{
	const _M_TYPE::_dimension Dimension = Matrix.iGetDimension();

	this->DataContainer.iReSizeStream(Dimension.x * Dimension.y);
	for (_M_TYPE::_size seek = 0; seek < Dimension.y; seek = seek + 1)
	{
		this->DataContainer.iSetStream(seek * Dimension.x, Matrix[seek].iGetContainer(), Dimension.x);
	}

	this->_SetMatrix(Matrix.iGetDimension());

	return *this;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _MATRIX<_data, _data_ptr, _data_ref>::operator+(const _M_TYPE::_type &Matrix) const
{
	return *this + Matrix.iGetRef();
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _MATRIX<_data, _data_ptr, _data_ref>::operator-(const _M_TYPE::_type &Matrix) const
{
	return *this - Matrix.iGetRef();
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _MATRIX<_data, _data_ptr, _data_ref>::operator*(const _M_TYPE::_type &Matrix) const
{
	return *this * Matrix.iGetRef();
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _MATRIX<_data, _data_ptr, _data_ref>::operator+(const _M_TYPE::_ref_type &Matrix) const
{
	_M_TYPE::_type DestMatrix(*this);
	DestMatrix.iAdd(Matrix);

	return DestMatrix;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _MATRIX<_data, _data_ptr, _data_ref>::operator-(const _M_TYPE::_ref_type &Matrix) const
{
	_M_TYPE::_type DestMatrix(*this);
	DestMatrix.iSub(Matrix);

	return DestMatrix;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _MATRIX<_data, _data_ptr, _data_ref>::operator*(const _M_TYPE::_ref_type &Matrix) const
{
	_M_TYPE::_type DestMatrix(*this);
	DestMatrix.iMul(Matrix);

	return DestMatrix;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _MATRIX<_data, _data_ptr, _data_ref>::operator+(const _data &Data) const
{
	_M_TYPE::_type Matrix(*this);
	Matrix.iAdd(Data);

	return Matrix;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _MATRIX<_data, _data_ptr, _data_ref>::operator-(const _data &Data) const
{
	_M_TYPE::_type Matrix(*this);
	Matrix.iSub(Data);

	return Matrix;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _MATRIX<_data, _data_ptr, _data_ref>::operator*(const _data &Data) const
{
	_M_TYPE::_type Matrix(*this);
	Matrix.iMul(Data);

	return Matrix;
}
template<class _data, class _data_ptr, class _data_ref>
bool _MATRIX<_data, _data_ptr, _data_ref>::operator==(const _M_TYPE::_type &Matrix) const
{
	return this->iGetRef() == Matrix.iGetRef();
}
template<class _data, class _data_ptr, class _data_ref>
bool _MATRIX<_data, _data_ptr, _data_ref>::operator!=(const _M_TYPE::_type &Matrix) const
{
	return this->iGetRef() != Matrix.iGetRef();
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _MATRIX<_data, _data_ptr, _data_ref>::iConv(const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Matrix, const bool bRepeat) const
{
	return this->iConv(_M_TYPE::_dimension(0, 0), this->Dimension, BlockSize, Matrix, bRepeat);
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_type _MATRIX<_data, _data_ptr, _data_ref>::iConv(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Matrix, const bool bRepeat) const
{
	_M_TYPE::_type DestMatrix(this->Dimension, 0.0);
	this->iGetRef().iConv(DestMatrix.iGetRef(), Start, End, BlockSize, Matrix, bRepeat);

	return DestMatrix;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _MATRIX<_data, _data_ptr, _data_ref>::iSetMatrix(const _M_TYPE::_dimension &Dimension, const _data_ptr &pData, const _M_TYPE::_size Size)
{
	if (Dimension == _M_TYPE::_dimension(0, 0)) { return this->_SetMatrix(Dimension); }

	if (pData == nullptr){ return this->Dimension; }
	if (Size < (Dimension.x * Dimension.y)) { return this->Dimension; }

	this->DataContainer.iReSizeStream(Dimension.x * Dimension.y);
	this->DataContainer.iSetStream(0, pData, this->DataContainer.iGetSize());
	return this->_SetMatrix(Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _MATRIX<_data, _data_ptr, _data_ref>::iSetMatrix(const _M_TYPE::_dimension &Dimension, const _data_ptr* pDataPtrArray, const _M_TYPE::_dimension Size)
{
	if (Dimension == _M_TYPE::_dimension(0, 0)) { return this->_SetMatrix(Dimension); }

	if (pDataPtrArray == nullptr){ return this->Dimension; }
	if (Size.y < Dimension.y) { return this->Dimension; }

	this->DataContainer.iReSizeStream(Dimension.x * Dimension.y);
	for (_M_TYPE::_size seek = 0; seek < Size.y; seek = seek + 1)
	{
		this->DataContainer.iSetStream(seek * Dimension.x, pDataPtrArray[seek], Dimension.x);
	}

	return this->_SetMatrix(Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _MATRIX<_data, _data_ptr, _data_ref>::iSetMatrix(const _M_TYPE::_dimension &Dimension, const vector<_data_ptr> &DataPtrArray)
{
	if (Dimension == _M_TYPE::_dimension(0, 0)) { return this->_SetMatrix(Dimension); }

	const _M_TYPE::_size Size = DataPtrArray.size();
	if (Size < Dimension.y) { return this->Dimension; }

	this->DataContainer.iReSizeStream(Dimension.x * Dimension.y);
	for (_M_TYPE::_size seek = 0; seek < Size; seek = seek + 1)
	{
		this->DataContainer.iSetStream(seek * Dimension.x, DataPtrArray[seek], Dimension.x);
	}

	return this->_SetMatrix(Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _MATRIX<_data, _data_ptr, _data_ref>::iSetMatrix(const _M_TYPE::_dimension &Dimension, const _M_TYPE::_vec_ref* pVectorArray, const _M_TYPE::_size Size)
{
	if (Dimension == _M_TYPE::_dimension(0, 0)) { return this->_SetMatrix(Dimension); }

	if (pVectorArray == nullptr){ return this->Dimension; }
	if (Size < Dimension.y) { return this->Dimension; }

	this->DataContainer.iReSizeStream(Dimension.x * Dimension.y);
	for (_M_TYPE::_size seek = 0; seek < Size; seek = seek + 1)
	{
		this->DataContainer.iSetStream(seek * Dimension.x, pVectorArray[seek].iGetContainer(), Dimension.x);
	}

	return this->_SetMatrix(Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _MATRIX<_data, _data_ptr, _data_ref>::iSetMatrix(const _M_TYPE::_dimension &Dimension, const vector<_M_TYPE::_vec_ref> &VectorArray)
{
	if (Dimension == _M_TYPE::_dimension(0, 0)) { return this->_SetMatrix(Dimension); }

	const _M_TYPE::_size Size = VectorArray.size();
	if (Size < Dimension.y) { return this->Dimension; }

	this->DataContainer.iReSizeStream(Dimension.x * Dimension.y);
	for (_M_TYPE::_size seek = 0; seek < Size; seek = seek + 1)
	{
		this->DataContainer.iSetStream(seek * Dimension.x, VectorArray[seek].iGetContainer(), Dimension.x);
	}

	return this->_SetMatrix(Dimension);
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_dimension _MATRIX<_data, _data_ptr, _data_ref>::iSetDimension(const _M_TYPE::_dimension &Dimension)
{
	if (Dimension != _M_TYPE::_dimension(0, 0) || ((Dimension.x != 0) && (Dimension.y != 0)))
	{
		if (this->Dimension != Dimension)
		{
			this->DataContainer.iReSizeStream(Dimension.x * Dimension.y);
			this->_SetMatrix(Dimension);
		}
	}

	return this->Dimension;
}
template<class _data, class _data_ptr, class _data_ref>
const _data& _MATRIX<_data, _data_ptr, _data_ref>::iGetData(const _M_TYPE::_dimension &Index) const
{
	return this->VecRefSet[Index.y][Index.x];
}
template<class _data, class _data_ptr, class _data_ref>
const _data& _MATRIX<_data, _data_ptr, _data_ref>::iGetData(const _M_TYPE::_size Index) const
{
	return this->DataContainer[Index];
}
template<class _data, class _data_ptr, class _data_ref>
_data& _MATRIX<_data, _data_ptr, _data_ref>::iGetData(const _M_TYPE::_dimension &Index)
{
	return this->VecRefSet[Index.y][Index.x];
}
template<class _data, class _data_ptr, class _data_ref>
_data& _MATRIX<_data, _data_ptr, _data_ref>::iGetData(const _M_TYPE::_size Index)
{
	return this->DataContainer[Index];
}
template<class _data, class _data_ptr, class _data_ref>
const _M_TYPE::_ref_type& _MATRIX<_data, _data_ptr, _data_ref>::iGetRef() const
{
	return *this;
}
template<class _data, class _data_ptr, class _data_ref>
_M_TYPE::_ref_type& _MATRIX<_data, _data_ptr, _data_ref>::iGetRef()
{
	return *this;
}
#undef _M_TYPE