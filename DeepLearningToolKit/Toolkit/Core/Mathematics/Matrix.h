#include"SanMathematicsDef.h"
#include"SanMathematicsVectorT.h"
#include"Vector.h"
using namespace std;
#pragma once
namespace San
{
	namespace Mathematics
	{
#ifndef __MATRIX_H__
#define __MATRIX_H__
		template<class _data, class _data_ptr, class _data_ref>
		struct _MATRIX_REF : public _smemobj<>
		{
		public:
			typedef int32				_size;
			typedef _SVECTOR2<_size>	_dimension;

			typedef _VECTOR_REF<_data, _data_ptr, _data_ref> _vec_ref;
			typedef _MATRIX_REF<_data, _data_ptr, _data_ref> _ref_type;
		protected:
			_sstream<_vec_ref> VecRefSet;
			_dimension Dimension;
		protected:
			_dimension _ConvCheck(const _dimension &Start, const _dimension &End, const _dimension &BlockSize, const _dimension &DestMatrixSize) const;
		public:
			_MATRIX_REF();
			_MATRIX_REF(const _dimension &Dimension, const _data_ptr pData, const _size Size);
			_MATRIX_REF(const _dimension &Dimension, const _data_ptr* pDataPtrArray, const _dimension Size);
			_MATRIX_REF(const _dimension &Dimension, const vector<_data_ptr> &DataPtrArray);
			_MATRIX_REF(const _dimension &Dimension, const _vec_ref* pVectorArray, const _size Size);
			_MATRIX_REF(const _dimension &Dimension, const vector<_vec_ref> &VectorArray);
			_MATRIX_REF(const _ref_type &Matrix);
			~_MATRIX_REF();

			const _data_ref operator[](const _dimension &Index) const;
			_data_ref operator[](const _dimension &Index);

			const _vec_ref& operator[](const _size Index) const;
			_vec_ref& operator[](const _size Index);

			_ref_type& operator=(const _ref_type &Matrix);

			bool operator==(const _ref_type &Matrix) const;
			bool operator!=(const _ref_type &Matrix) const;

			/*May cause error when add itself*/
			void iAdd(const _ref_type &Matrix, const bool bRepeat = false);
			void iAdd(const _dimension &Start, const _dimension &End, const _ref_type &Matrix, const bool bRepeat = false);

			/*May cause error when sub itself*/
			void iSub(const _ref_type &Matrix, const bool bRepeat = false);
			void iSub(const _dimension &Start, const _dimension &End, const _ref_type &Matrix, const bool bRepeat = false);

			/*May cause error when multiply itself*/
			void iMul(const _ref_type &Matrix, const bool bRepeat = false);
			void iMul(const _dimension &Start, const _dimension &End, const _ref_type &Matrix, const bool bRepeat = false);

			void iSet(const _ref_type &Matrix, const bool bRepeat = false);
			void iSet(const _dimension &Start, const _dimension &End, const _ref_type &Matrix, const bool bRepeat = false);

			/*May cause error when convolution itself*/
			void iConv(_ref_type &DestMatrix, const _dimension &BlockSize, const _ref_type &Matrix, const bool bRepeat = false) const;
			void iConv(_ref_type &DestMatrix, const _dimension &Start, const _dimension &End, const _dimension &BlockSize, const _ref_type &Matrix, const bool bRepeat = false) const;

			void iAdd(const _data &Data);
			void iAdd(const _dimension &Start, const _dimension &End, const _data &Data);

			void iSub(const _data &Data);
			void iSub(const _dimension &Start, const _dimension &End, const _data &Data);

			void iMul(const _data &Data);
			void iMul(const _dimension &Start, const _dimension &End, const _data &Data);

			void iSet(const _data &Data);
			void iSet(const _dimension &Start, const _dimension &End, const _data &Data);

			_data iSum() const;
			_data iSum(const _dimension &Start, const _dimension &End) const;

			_data iProduct() const;
			_data iProduct(const _dimension &Start, const _dimension &End) const;

			_data iMin() const;
			_data iMin(const _dimension &Start, const _dimension &End) const;

			_data iMax() const;
			_data iMax(const _dimension &Start, const _dimension &End) const;

			void iNormalize();

			template<class _Func> void iConvolution(_ref_type &DestMatrix, const _dimension &BlockSize, _Func &Func) const;
			template<class _Func> void iConvolution(_ref_type &DestMatrix, const _dimension &Start, const _dimension &End, const _dimension &BlockSize, _Func &Func) const;
			template<class _Func> void iConvolution(_ref_type &DestMatrix, const _dimension &BlockSize, const _ref_type &Matrix, _Func &Func, const bool bRepeat = false) const;
			template<class _Func> void iConvolution(_ref_type &DestMatrix, const _dimension &Start, const _dimension &End, const _dimension &BlockSize, const _ref_type &Matrix, _Func &Func, const bool bRepeat = false) const;

			template<class _Func> void iConvolutionC(const _dimension &BlockSize, _Func &Func) const;
			template<class _Func> void iConvolutionC(const _dimension &Start, const _dimension &End, const _dimension &BlockSize, _Func &Func) const;
			template<class _Func> void iConvolutionC(const _dimension &BlockSize, const _ref_type &Matrix, _Func &Func, const bool bRepeat = false) const;
			template<class _Func> void iConvolutionC(const _dimension &Start, const _dimension &End, const _dimension &BlockSize, const _ref_type &Matrix, _Func &Func, const bool bRepeat = false) const;

			template<class _Func> void iOperation(_Func &Func);
			template<class _Func> void iOperation(const _dimension &Start, const _dimension &End, _Func &Func);
			template<class _Func> void iOperation(const _data &Data, _Func &Func);
			template<class _Func> void iOperation(const _dimension &Start, const _dimension &End, const _data &Data, _Func &Func);

			template<class _Func> void iOperationC(_Func &Func) const;
			template<class _Func> void iOperationC(const _dimension &Start, const _dimension &End, _Func &Func) const;
			template<class _Func> void iOperationC(const _data &Data, _Func &Func) const;
			template<class _Func> void iOperationC(const _dimension &Start, const _dimension &End, const _data &Data, _Func &Func) const;

			virtual _dimension iSetMatrix(const _dimension &Dimension, const _data_ptr &pData, const _size Size);
			virtual _dimension iSetMatrix(const _dimension &Dimension, const _data_ptr* pDataPtrArray, const _dimension Size);
			virtual _dimension iSetMatrix(const _dimension &Dimension, const vector<_data_ptr> &DataPtrArray);
			virtual _dimension iSetMatrix(const _dimension &Dimension, const _vec_ref* pVectorArray, const _size Size);
			virtual _dimension iSetMatrix(const _dimension &Dimension, const vector<_vec_ref> &VectorArray);
			virtual _dimension iSetDimension(const _dimension &Dimension);

			vector<_vec_ref> iGetContainer() const;

			_dimension iGetDimension() const;
			_size iGetSize() const;
		};

		template<class _data>
		using _matrix_ref = _MATRIX_REF<_data, _data*, _data&>;

#pragma region _MATRIX_REF CPP
#define _M_TYPE typename _MATRIX_REF<_data, _data_ptr, _data_ref>
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _MATRIX_REF<_data, _data_ptr, _data_ref>::iConvolution(_M_TYPE::_ref_type &DestMatrix, const _M_TYPE::_dimension &BlockSize, _Func &Func) const
		{
			this->iConvolution(DestMatrix, _M_TYPE::_dimension(0, 0), this->Dimension, BlockSize, Func);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _MATRIX_REF<_data, _data_ptr, _data_ref>::iConvolution(_M_TYPE::_ref_type &DestMatrix, const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, _Func &Func) const
		{
			const auto Size = this->_ConvCheck(Start, End, BlockSize, DestMatrix.Dimension);
			_M_TYPE::_dimension Pos(0, 0);

			_M_TYPE::_ref_type CurrentMatrix((_data_ptr) nullptr, BlockSize.y, BlockSize);

			for (int32 seek_y = Start.y; seek_y < Size.y; seek_y = seek_y + 1, Pos.y = Pos.y + 1)
			{
				for (int32 seek_x = Start.x; seek_x < Size.x; seek_x = seek_x + 1, Pos.x = Pos.x + 1)
				{
					for (int32 seek = 0; seek < BlockSize.y; seek = seek + 1)
					{
						CurrentMatrix[seek].iSetVector(this->VecRefSet[seek_y + seek].iGetContainer() + seek_x, BlockSize.x);
					}

					DestMatrix[Pos.y][Pos.x] = Func(CurrentMatrix, Pos, BlockSize);
				}
			}
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _MATRIX_REF<_data, _data_ptr, _data_ref>::iConvolution(_M_TYPE::_ref_type &DestMatrix, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Matrix, _Func &Func, const bool bRepeat) const
		{
			this->iConvolution(DestMatrix, _M_TYPE::_dimension(0, 0), this->Dimension, BlockSize, Matrix, Func, bRepeat);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _MATRIX_REF<_data, _data_ptr, _data_ref>::iConvolution(_M_TYPE::_ref_type &DestMatrix, const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Matrix, _Func &Func, const bool bRepeat) const
		{
			const auto Size = this->_ConvCheck(Start, End, BlockSize, DestMatrix.Dimension);
			_M_TYPE::_dimension Pos(0, 0);

			_M_TYPE::_ref_type CurrentMatrix((_data_ptr) nullptr, BlockSize.y, BlockSize);

			for (int32 seek_y = Start.y; seek_y < Size.y; seek_y = seek_y + 1, Pos.y = Pos.y + 1)
			{
				for (int32 seek_x = Start.x; seek_x < Size.x; seek_x = seek_x + 1, Pos.x = Pos.x + 1)
				{
					for (int32 seek = 0; seek < BlockSize.y; seek = seek + 1)
					{
						CurrentMatrix[seek].iSetVector(this->VecRefSet[seek_y + seek].iGetContainer() + seek_x, BlockSize.x);
					}

					DestMatrix[Pos.y][Pos.x] = Func(CurrentMatrix, Pos, BlockSize, Matrix, bRepeat);
				}
			}
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _MATRIX_REF<_data, _data_ptr, _data_ref>::iConvolutionC(const _M_TYPE::_dimension &BlockSize, _Func &Func) const
		{
			this->iConvolutionC(_M_TYPE::_dimension(0, 0), this->Dimension, BlockSize, Func);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _MATRIX_REF<_data, _data_ptr, _data_ref>::iConvolutionC(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, _Func &Func) const
		{
			const auto Size = this->_ConvCheck(Start, End, BlockSize, this->Dimension);
			_M_TYPE::_dimension Pos(0, 0);

			_M_TYPE::_ref_type CurrentMatrix((_data_ptr) nullptr, BlockSize.x * BlockSize.y, BlockSize);

			for (int32 seek_y = Start.y; seek_y < Size.y; seek_y = seek_y + 1, Pos.y = Pos.y + 1)
			{
				for (int32 seek_x = Start.x; seek_x < Size.x; seek_x = seek_x + 1, Pos.x = Pos.x + 1)
				{
					for (int32 seek = 0; seek < BlockSize.y; seek = seek + 1)
					{
						CurrentMatrix[seek].iSetVector(this->VecRefSet[seek_y + seek].iGetContainer() + seek_x, BlockSize.x);
					}

					Func(CurrentMatrix, Pos, BlockSize);
				}
			}
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _MATRIX_REF<_data, _data_ptr, _data_ref>::iConvolutionC(const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Matrix, _Func &Func, const bool bRepeat) const
		{
			this->iConvolutionC(_M_TYPE::_dimension(0, 0), this->Dimension, BlockSize, Matrix, Func, bRepeat);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _MATRIX_REF<_data, _data_ptr, _data_ref>::iConvolutionC(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Matrix, _Func &Func, const bool bRepeat) const
		{
			const auto Size = this->_ConvCheck(Start, End, BlockSize, this->Dimension);
			_M_TYPE::_dimension Pos(0, 0);

			_M_TYPE::_ref_type CurrentMatrix((_data_ptr) nullptr, BlockSize.y, BlockSize);

			for (int32 seek_y = Start.y; seek_y < Size.y; seek_y = seek_y + 1, Pos.y = Pos.y + 1)
			{
				for (int32 seek_x = Start.x; seek_x < Size.x; seek_x = seek_x + 1, Pos.x = Pos.x + 1)
				{
					for (int32 seek = 0; seek < BlockSize.y; seek = seek + 1)
					{
						CurrentMatrix[seek].iSetVector(this->VecRefSet[seek_y + seek].iGetContainer() + seek_x, BlockSize.x);
					}

					Func(CurrentMatrix, Pos, BlockSize, Matrix, bRepeat);
				}
			}
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _MATRIX_REF<_data, _data_ptr, _data_ref>::iOperation(_Func &Func)
		{
			this->iOperation(_M_TYPE::_dimension(0, 0), this->Dimension, Func);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _MATRIX_REF<_data, _data_ptr, _data_ref>::iOperation(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, _Func &Func)
		{
			const _M_TYPE::_dimension Size(min(this->Dimension.x, End.x), min(this->Dimension.y, End.y));
			_M_TYPE::_dimension Pos(0, 0);

			for (int32 seek_y = Start.y; seek_y < Size.y; seek_y = seek_y + 1, Pos.y = Pos.y + 1)
			{
				_vec_ref &CurrentVec = this->VecRefSet[seek_y];

				for (int32 seek_x = Start.x; seek_x < Size.x; seek_x = seek_x + 1, Pos.x = Pos.x + 1)
				{
					CurrentVec[seek_x] = Func(CurrentVec[seek_x], Pos);
				}
			}
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _MATRIX_REF<_data, _data_ptr, _data_ref>::iOperation(const _data &Data, _Func &Func)
		{
			this->iOperation(_M_TYPE::_dimension(0, 0), this->Dimension, Data, Func);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _MATRIX_REF<_data, _data_ptr, _data_ref>::iOperation(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _data &Data, _Func &Func)
		{
			const _M_TYPE::_dimension Size(min(this->Dimension.x, End.x), min(this->Dimension.y, End.y));
			_M_TYPE::_dimension Pos(0, 0);

			for (int32 seek_y = Start.y; seek_y < Size.y; seek_y = seek_y + 1, Pos.y = Pos.y + 1)
			{
				_vec_ref &CurrentVec = this->VecRefSet[seek_y];

				for (int32 seek_x = Start.x; seek_x < Size.x; seek_x = seek_x + 1, Pos.x = Pos.x + 1)
				{
					CurrentVec[seek_x] = Func(CurrentVec[seek_x], Pos, Data);
				}
			}
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _MATRIX_REF<_data, _data_ptr, _data_ref>::iOperationC(_Func &Func) const
		{
			this->iOperationC(_M_TYPE::_dimension(0, 0), this->Dimension, Func);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _MATRIX_REF<_data, _data_ptr, _data_ref>::iOperationC(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, _Func &Func) const
		{
			const _M_TYPE::_dimension Size(min(this->Dimension.x, End.x), min(this->Dimension.y, End.y));
			_M_TYPE::_dimension Pos(0, 0);

			for (int32 seek_y = Start.y; seek_y < Size.y; seek_y = seek_y + 1, Pos.y = Pos.y + 1)
			{
				const _vec_ref &CurrentVec = this->VecRefSet[seek_y];

				for (int32 seek_x = Start.x; seek_x < Size.x; seek_x = seek_x + 1, Pos.x = Pos.x + 1)
				{
					Func(CurrentVec[seek_x], Pos);
				}
			}
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _MATRIX_REF<_data, _data_ptr, _data_ref>::iOperationC(const _data &Data, _Func &Func) const
		{
			this->iOperation(_M_TYPE::_dimension(0, 0), this->Dimension, Data, Func);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _MATRIX_REF<_data, _data_ptr, _data_ref>::iOperationC(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _data &Data, _Func &Func) const
		{
			const _M_TYPE::_dimension Size(min(this->Dimension.x, End.x), min(this->Dimension.y, End.y));
			_M_TYPE::_dimension Pos(0, 0);

			for (int32 seek_y = Start.y; seek_y < Size.y; seek_y = seek_y + 1, Pos.y = Pos.y + 1)
			{
				const _vec_ref &CurrentVec = this->VecRefSet[seek_y];

				for (int32 seek_x = Start.x; seek_x < Size.x; seek_x = seek_x + 1, Pos.x = Pos.x + 1)
				{
					Func(CurrentVec[seek_x], Pos, Data);
				}
			}
		}
#undef _M_TYPE
#pragma endregion

		template<class _data, class _data_ptr, class _data_ref>
		struct _MATRIX : public _MATRIX_REF<_data, _data_ptr, _data_ref>, _smemobj<>
		{
		public:
			typedef _MATRIX<_data, _data_ptr, _data_ref> _type;
		protected:
			_sstream<_data> DataContainer;
		protected:
			typename _type::_dimension _SetMatrix(const typename _type::_dimension &Dimension); //Assume dimension value legal
		public:
			_MATRIX();
			_MATRIX(const typename _type::_dimension &Dimension, const _data &Val);
			_MATRIX(const typename _type::_dimension &Dimension, const _data_ptr pData, const typename _type::_size Size);
			_MATRIX(const typename _type::_dimension &Dimension, const _data_ptr* pDataPtrArray, const typename _type::_dimension Size);
			_MATRIX(const typename _type::_dimension &Dimension, const vector<_data_ptr> &DataPtrArray);
			_MATRIX(const typename _type::_dimension &Dimension, const typename _type::_vec_ref* pVectorArray, const typename _type::_size Size);
			_MATRIX(const typename _type::_dimension &Dimension, const vector<typename _type::_vec_ref> &VectorArray);
			_MATRIX(const _type &Matrix);
			_MATRIX(const typename _type::_ref_type &Matrix);
			~_MATRIX();

			/*const typename _type::_vec_ref& operator[](const _size Index) const;
			typename _type::_vec_ref& operator[](const _size Index);*/

			_type& operator=(const _type &Matrix);
			_type& operator=(const typename _type::_ref_type &Matrix);

			_type operator+(const _type &Matrix) const;
			_type operator-(const _type &Matrix) const;
			_type operator*(const _type &Matrix) const;

			_type operator+(const typename _type::_ref_type &Matrix) const;
			_type operator-(const typename _type::_ref_type &Matrix) const;
			_type operator*(const typename _type::_ref_type &Matrix) const;

			_type operator+(const _data &Data) const;
			_type operator-(const _data &Data) const;
			_type operator*(const _data &Data) const;

			bool operator==(const _type &Matrix) const;
			bool operator!=(const _type &Matrix) const;

			/*May cause error when convolution itself*/
			_type iConv(const typename _type::_dimension &BlockSize, const typename _type::_ref_type &Matrix, const bool bRepeat = false) const;
			_type iConv(const typename _type::_dimension &Start, const typename _type::_dimension &End, const typename _type::_dimension &BlockSize, const typename _type::_ref_type &Matrix, const bool bRepeat = false) const;

			template<class _Func> _type iConvolution(const typename _type::_dimension &BlockSize, _Func &Func) const;
			template<class _Func> _type iConvolution(const typename _type::_dimension &Start, const typename _type::_dimension &End, const typename _type::_dimension &BlockSize, _Func &Func) const;
			template<class _Func> _type iConvolution(const typename _type::_dimension &BlockSize, const typename _type::_ref_type &Matrix, _Func &Func, const bool bRepeat = false) const;
			template<class _Func> _type iConvolution(const typename _type::_dimension &Start, const typename _type::_dimension &End, const typename _type::_dimension &BlockSize, const typename _type::_ref_type &Matrix, _Func &Func, const bool bRepeat = false) const;

			//Should only set data, call iSetDmiension to set dimension, currently set data and dimension
			typename _type::_dimension iSetMatrix(const typename _type::_dimension &Dimension, const _data_ptr &pData, const typename _type::_size Size) override final;
			typename _type::_dimension iSetMatrix(const typename _type::_dimension &Dimension, const _data_ptr* pDataPtrArray, const typename _type::_dimension Size) override final;
			typename _type::_dimension iSetMatrix(const typename _type::_dimension &Dimension, const vector<_data_ptr> &DataPtrArray) override final;
			typename _type::_dimension iSetMatrix(const typename _type::_dimension &Dimension, const typename _type::_vec_ref* pVectorArray, const typename _type::_size Size) override final;
			typename _type::_dimension iSetMatrix(const typename _type::_dimension &Dimension, const vector<typename _type::_vec_ref> &VectorArray) override final;

			typename _type::_dimension iSetDimension(const typename _type::_dimension &Dimension) override final; // Set Dimension

			const _data& iGetData(const typename _type::_dimension &Index) const;
			const _data& iGetData(const typename _type::_size Index) const;

			_data& iGetData(const typename _type::_dimension &Index);
			_data& iGetData(const typename _type::_size Index);
			
			const typename _type::_ref_type& iGetRef() const;
			typename _type::_ref_type& iGetRef();
		};

		template<class _data>
		using _matrix = _MATRIX<_data, _data*, _data&>;

#pragma region _MATRIX CPP
#define _M_TYPE typename _MATRIX<_data, _data_ptr, _data_ref>
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		_M_TYPE::_type _MATRIX<_data, _data_ptr, _data_ref>::iConvolution(const _M_TYPE::_dimension &BlockSize, _Func &Func) const
		{
			return this->iConvolution(0, this->Dimension, BlockSize, Func);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		_M_TYPE::_type _MATRIX<_data, _data_ptr, _data_ref>::iConvolution(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, _Func &Func) const
		{
			_M_TYPE::_type DestMatrix(this->Dimension - BlockSize + 1, 0.0);
			this->iConvolution(DestMatrix, Start, End, BlockSize, Func);

			return DestMatrix;
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		_M_TYPE::_type _MATRIX<_data, _data_ptr, _data_ref>::iConvolution(const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Matrix, _Func &Func, const bool bRepeat) const
		{
			return this->iConvolution(0, this->Dimension, BlockSize, Matrix, Func, bRepeat);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		_M_TYPE::_type _MATRIX<_data, _data_ptr, _data_ref>::iConvolution(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Matrix, _Func &Func, const bool bRepeat) const
		{
			_M_TYPE::_type DestMatrix(this->Dimension - BlockSize + 1, 0.0);
			this->iConvolution(DestMatrix, Start, End, BlockSize, Matrix, Func, bRepeat);

			return DestMatrix;
		}
#undef _M_TYPE
#pragma endregion

#endif
	}
}