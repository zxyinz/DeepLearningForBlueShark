#include"SanMathematicsDef.h"
#include"Matrix.h"
#pragma once
using namespace std;
namespace San
{
	namespace Mathematics
	{
#ifndef __TENSOR_H__
#define __TENSOR_H__
		template<class _data, class _data_ptr, class _data_ref>
		struct _TENSOR_REF : public _smemobj<>
		{
		public:
			typedef int32				_size;
			typedef _SVECTOR3<_size>	_dimension;

			typedef _MATRIX_REF<_data, _data_ptr, _data_ref> _mat_ref;
			typedef _TENSOR_REF<_data, _data_ptr, _data_ref> _ref_type;
		protected:
			_sstream<_mat_ref> MatRefSet;
			_dimension Dimension;
		protected:
			_dimension _ConvCheck(const _dimension &Start, const _dimension &End, const _dimension &BlockSize, const _dimension &DestTensorSize) const;
			void _CreateSubTensor(const _dimension &Start, const _dimension &BlockSize, _ref_type &Block) const;
		public:
			_TENSOR_REF();
			_TENSOR_REF(const _dimension &Dimension, const _data_ptr pData, const _size Size);
			_TENSOR_REF(const _dimension &Dimension, const _sstream<_data> &DataStream);
			_TENSOR_REF(const _dimension &Dimension, const _data_ptr** pDataPtrArray, const _dimension Size);
			_TENSOR_REF(const _dimension &Dimension, const vector<vector<_data_ptr>> &DataPtrArray);
			_TENSOR_REF(const _dimension &Dimension, const _mat_ref* pMatrixArray, const _size Size);
			_TENSOR_REF(const _dimension &Dimension, const vector<_mat_ref> &MatrixArray);
			_TENSOR_REF(const _ref_type &Tensor);
			~_TENSOR_REF();

			const _data_ref operator[](const _dimension &Index) const;
			_data_ref operator[](const _dimension &Index);

			const _mat_ref& operator[](const _size Index) const;
			_mat_ref& operator[](const _size Index);

			_ref_type& operator=(const _ref_type &Tensor);

			bool operator==(const _ref_type &Tensor) const;
			bool operator!=(const _ref_type &Tensor) const;

			/*May cause error when add itself*/
			void iAdd(const _ref_type &Tensor, const bool bRepeat = false);
			void iAdd(const _dimension &Start, const _dimension &End, const _ref_type &Tensor, const bool bRepeat = false);

			/*May cause error when sub itself*/
			void iSub(const _ref_type &Tensor, const bool bRepeat = false);
			void iSub(const _dimension &Start, const _dimension &End, const _ref_type &Tensor, const bool bRepeat = false);

			/*May cause error when multiply itself*/
			void iMul(const _ref_type &Tensor, const bool bRepeat = false);
			void iMul(const _dimension &Start, const _dimension &End, const _ref_type &Tensor, const bool bRepeat = false);

			void iSet(const _ref_type &Tensor, const bool bRepeat = false);
			void iSet(const _dimension &Start, const _dimension &End, const _ref_type &Tensor, const bool bRepeat = false);

			/*May cause error when convolution itself*/
			void iConv(_ref_type &DestTensor, const _dimension &BlockSize, const _ref_type &Tensor, const bool bRepeat = false) const;
			void iConv(_ref_type &DestTensor, const _dimension &Start, const _dimension &End, const _dimension &BlockSize, const _ref_type &Tensor, const bool bRepeat = false) const;

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

			template<class _Func> void iConvolution(_ref_type &DestTensor, const _dimension &BlockSize, _Func &Func) const;
			template<class _Func> void iConvolution(_ref_type &DestTensor, const _dimension &Start, const _dimension &End, const _dimension &BlockSize, _Func &Func) const;
			template<class _Func> void iConvolution(_ref_type &DestTensor, const _dimension &BlockSize, const _ref_type &Tensor, _Func &Func, const bool bRepeat = false) const;
			template<class _Func> void iConvolution(_ref_type &DestTensor, const _dimension &Start, const _dimension &End, const _dimension &BlockSize, const _ref_type &Tensor, _Func &Func, const bool bRepeat = false) const;

			template<class _Func> void iConvolutionC(const _dimension &BlockSize, _Func &Func) const;
			template<class _Func> void iConvolutionC(const _dimension &Start, const _dimension &End, const _dimension &BlockSize, _Func &Func) const;
			template<class _Func> void iConvolutionC(const _dimension &BlockSize, const _ref_type &Tensor, _Func &Func, const bool bRepeat = false) const;
			template<class _Func> void iConvolutionC(const _dimension &Start, const _dimension &End, const _dimension &BlockSize, const _ref_type &Tensor, _Func &Func, const bool bRepeat = false) const;

			template<class _Func> void iOperation(_Func &Func);
			template<class _Func> void iOperation(const _dimension &Start, const _dimension &End, _Func &Func);
			template<class _Func> void iOperation(const _data &Data, _Func &Func);
			template<class _Func> void iOperation(const _dimension &Start, const _dimension &End, const _data &Data, _Func &Func);

			template<class _Func> void iOperationC(_Func &Func) const;
			template<class _Func> void iOperationC(const _dimension &Start, const _dimension &End, _Func &Func) const;
			template<class _Func> void iOperationC(const _data &Data, _Func &Func) const;
			template<class _Func> void iOperationC(const _dimension &Start, const _dimension &End, const _data &Data, _Func &Func) const;

			virtual _dimension iSetTensor(const _dimension &Dimension, const _data_ptr pData, const _size Size);
			virtual _dimension iSetTensor(const _dimension &Dimension, const _sstream<_data> &DataStream);
			virtual _dimension iSetTensor(const _dimension &Dimension, const _data_ptr** pDataPtrArray, const _dimension Size);
			virtual _dimension iSetTensor(const _dimension &Dimension, const vector<vector<_data_ptr>> &DataPtrArray);
			virtual _dimension iSetTensor(const _dimension &Dimension, const _mat_ref* pMatrixArray, const _size Size);
			virtual _dimension iSetTensor(const _dimension &Dimension, const vector<_mat_ref> &MatrixArray);

			virtual _dimension iSetDimension(const _dimension &Dimension);

			vector<_mat_ref> iGetContainer() const;

			_dimension iGetDimension() const;
			_size iGetSize() const;
		};

		template<class _data>
		using _tensor_ref = _TENSOR_REF<_data, _data*, _data&>;

#pragma region _TENSOR_REF CPP
#define _M_TYPE typename _TENSOR_REF<_data, _data_ptr, _data_ref>
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _TENSOR_REF<_data, _data_ptr, _data_ref>::iConvolution(_M_TYPE::_ref_type &DestTensor, const _M_TYPE::_dimension &BlockSize, _Func &Func) const
		{
			this->iConvolution(DestTensor, _M_TYPE::_dimension(0, 0, 0), this->Dimension, BlockSize, Func);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _TENSOR_REF<_data, _data_ptr, _data_ref>::iConvolution(_M_TYPE::_ref_type &DestTensor, const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, _Func &Func) const
		{
			const auto Size = this->_ConvCheck(Start, End, BlockSize, DestTensor.Dimension);
			auto seek = Size;
			_M_TYPE::_dimension Pos(0, 0, 0);
			_M_TYPE::_ref_type Block(BlockSize, (_data_ptr) nullptr, BlockSize.y);

			for (seek.z = Start.z; seek.z < Size.z; seek.z = seek.z + 1, Pos.z = Pos.z + 1)
			{
				auto &Mat = DestTensor[seek.z];
				for (seek.y = Start.y; seek.y < Size.y; seek.y = seek.y + 1, Pos.y = Pos.y + 1)
				{
					auto &Vec = Mat[seek.y];
					for (seek.x = Start.x; seek.x < Size.x; seek.x = seek.x + 1, Pos.x = Pos.x + 1)
					{
						this->_CreateSubTensor(seek, BlockSize, Block);
						Vec[Pos.x] = Func(Block, Pos);
					}
				}
			}
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _TENSOR_REF<_data, _data_ptr, _data_ref>::iConvolution(_M_TYPE::_ref_type &DestTensor, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Tensor, _Func &Func, const bool bRepeat) const
		{
			this->iConvolution(DestTensor, _M_TYPE::_dimension(0, 0, 0), this->Dimension, BlockSize, Tensor, Func, bRepeat);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _TENSOR_REF<_data, _data_ptr, _data_ref>::iConvolution(_M_TYPE::_ref_type &DestTensor, const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Tensor, _Func &Func, const bool bRepeat) const
		{
			const auto Size = this->_ConvCheck(Start, End, BlockSize, DestTensor.Dimension);
			auto seek = Size;
			_M_TYPE::_dimension Pos(0, 0, 0);
			_M_TYPE::_ref_type Block(BlockSize, (_data_ptr) nullptr, BlockSize.y);

			for (seek.z = Start.z; seek.z < Size.z; seek.z = seek.z + 1, Pos.z = Pos.z + 1)
			{
				auto &Mat = DestTensor[seek.z];
				for (seek.y = Start.y; seek.y < Size.y; seek.y = seek.y + 1, Pos.y = Pos.y + 1)
				{
					auto &Vec = Mat[seek.y];
					for (seek.x = Start.x; seek.x < Size.x; seek.x = seek.x + 1, Pos.x = Pos.x + 1)
					{
						this->_CreateSubTensor(seek, BlockSize, Block);
						Vec[Pos.x] = Func(Block, Pos, Tensor, bRepeat);
					}
				}
			}
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _TENSOR_REF<_data, _data_ptr, _data_ref>::iConvolutionC(const _M_TYPE::_dimension &BlockSize, _Func &Func) const
		{
			this->iConvolutionC(_M_TYPE::_dimension(0, 0, 0), this->Dimension, BlockSize, Func);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _TENSOR_REF<_data, _data_ptr, _data_ref>::iConvolutionC(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, _Func &Func) const
		{
			const auto Size = this->_ConvCheck(Start, End, BlockSize, this->Dimension);
			auto seek = Size;
			_M_TYPE::_dimension Pos(0, 0, 0);
			_M_TYPE::_ref_type Block(BlockSize, (_data_ptr) nullptr, BlockSize.y);

			for (seek.z = Start.z; seek.z < Size.z; seek.z = seek.z + 1, Pos.z = Pos.z + 1)
			{
				for (seek.y = Start.y; seek.y < Size.y; seek.y = seek.y + 1, Pos.y = Pos.y + 1)
				{
					for (seek.x = Start.x; seek.x < Size.x; seek.x = seek.x + 1, Pos.x = Pos.x + 1)
					{
						this->_CreateSubTensor(seek, BlockSize, Block);
						Func(Block, Pos);
					}
				}
			}
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _TENSOR_REF<_data, _data_ptr, _data_ref>::iConvolutionC(const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Tensor, _Func &Func, const bool bRepeat) const
		{
			this->iConvolutionC(_M_TYPE::_dimension(0, 0, 0), this->Dimension, BlockSize, Tensor, Func, bRepeat);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _TENSOR_REF<_data, _data_ptr, _data_ref>::iConvolutionC(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Tensor, _Func &Func, const bool bRepeat) const
		{
			const auto Size = this->_ConvCheck(Start, End, BlockSize, this->Dimension);
			auto seek = Size;
			_M_TYPE::_dimension Pos(0, 0, 0);
			_M_TYPE::_ref_type Block(BlockSize, (_data_ptr) nullptr, BlockSize.y);

			for (seek.z = Start.z; seek.z < Size.z; seek.z = seek.z + 1, Pos.z = Pos.z + 1)
			{
				for (seek.y = Start.y; seek.y < Size.y; seek.y = seek.y + 1, Pos.y = Pos.y + 1)
				{
					for (seek.x = Start.x; seek.x < Size.x; seek.x = seek.x + 1, Pos.x = Pos.x + 1)
					{
						this->_CreateSubTensor(seek, BlockSize, Block);
						Func(Block, Pos, Tensor, bRepeat);
					}
				}
			}
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _TENSOR_REF<_data, _data_ptr, _data_ref>::iOperation(_Func &Func)
		{
			this->iOperation(_M_TYPE::_dimension(0, 0, 0), this->Dimension, Func);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _TENSOR_REF<_data, _data_ptr, _data_ref>::iOperation(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, _Func &Func)
		{
			const _M_TYPE::_dimension Size(min(this->Dimension.x, End.x), min(this->Dimension.y, End.y), min(this->Dimension.z, End.z));
			_M_TYPE::_dimension Pos(0, 0, 0);

			for (int32 seek_z = Start.z; seek_z < Size.z; seek_z = seek_z + 1, Pos.z = Pos.z + 1)
			{
				_M_TYPE::_mat_ref &CurrentMat = this->MatRefSet[seek_z];

				for (int32 seek_y = Start.y; seek_y < Size.y; seek_y = seek_y + 1, Pos.y = Pos.y + 1)
				{
					_vector_ref<_data> &CurrentVec = CurrentMat[seek_y];

					for (int32 seek_x = Start.x; seek_x < Size.x; seek_x = seek_x + 1, Pos.x = Pos.x + 1)
					{
						CurrentVec[seek_x] = Func(CurrentVec[seek_x], Pos);
					}
				}
			}
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _TENSOR_REF<_data, _data_ptr, _data_ref>::iOperation(const _data &Data, _Func &Func)
		{
			this->iOperation(_M_TYPE::_dimension(0, 0, 0), this->Dimension, Data, Func);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _TENSOR_REF<_data, _data_ptr, _data_ref>::iOperation(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _data &Data, _Func &Func)
		{
			const _M_TYPE::_dimension Size(min(this->Dimension.x, End.x), min(this->Dimension.y, End.y), min(this->Dimension.z, End.z));
			_M_TYPE::_dimension Pos(0, 0, 0);

			for (int32 seek_z = Start.z; seek_z < Size.z; seek_z = seek_z + 1, Pos.z = Pos.z + 1)
			{
				_M_TYPE::_mat_ref &CurrentMat = this->MatRefSet[seek_z];

				for (int32 seek_y = Start.y; seek_y < Size.y; seek_y = seek_y + 1, Pos.y = Pos.y + 1)
				{
					_vector_ref<_data> &CurrentVec = CurrentMat[seek_y];

					for (int32 seek_x = Start.x; seek_x < Size.x; seek_x = seek_x + 1, Pos.x = Pos.x + 1)
					{
						CurrentVec[seek_x] = Func(CurrentVec[seek_x], Pos, Data);
					}
				}
			}
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _TENSOR_REF<_data, _data_ptr, _data_ref>::iOperationC(_Func &Func) const
		{
			this->iOperationC(_M_TYPE::_dimension(0, 0, 0), this->Dimension, Func);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _TENSOR_REF<_data, _data_ptr, _data_ref>::iOperationC(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, _Func &Func) const
		{
			const _M_TYPE::_dimension Size(min(this->Dimension.x, End.x), min(this->Dimension.y, End.y), min(this->Dimension.z, End.z));
			_M_TYPE::_dimension Pos(0, 0, 0);

			for (int32 seek_z = Start.z; seek_z < Size.z; seek_z = seek_z + 1, Pos.z = Pos.z + 1)
			{
				const _M_TYPE::_mat_ref &CurrentMat = this->MatRefSet[seek_z];

				for (int32 seek_y = Start.y; seek_y < Size.y; seek_y = seek_y + 1, Pos.y = Pos.y + 1)
				{
					const _vector_ref<_data> &CurrentVec = CurrentMat[seek_y];

					for (int32 seek_x = Start.x; seek_x < Size.x; seek_x = seek_x + 1, Pos.x = Pos.x + 1)
					{
						Func(CurrentVec[seek_x], Pos);
					}
				}
			}
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _TENSOR_REF<_data, _data_ptr, _data_ref>::iOperationC(const _data &Data, _Func &Func) const
		{
			this->iOperation(_M_TYPE::_dimension(0, 0, 0), this->Dimension, Data, Func);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _TENSOR_REF<_data, _data_ptr, _data_ref>::iOperationC(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _data &Data, _Func &Func) const
		{
			const _M_TYPE::_dimension Size(min(this->Dimension.x, End.x), min(this->Dimension.y, End.y), min(this->Dimension.z, End.z));
			_M_TYPE::_dimension Pos(0, 0, 0);

			for (int32 seek_z = Start.z; seek_z < Size.z; seek_z = seek_z + 1, Pos.z = Pos.z + 1)
			{
				const _M_TYPE::_mat_ref &CurrentMat = this->MatRefSet[seek_z];

				for (int32 seek_y = Start.y; seek_y < Size.y; seek_y = seek_y + 1, Pos.y = Pos.y + 1)
				{
					const _vector_ref<_data> &CurrentVec = CurrentMat[seek_y];

					for (int32 seek_x = Start.x; seek_x < Size.x; seek_x = seek_x + 1, Pos.x = Pos.x + 1)
					{
						Func(CurrentVec[seek_x], Pos, Data);
					}
				}
			}
		}
#undef _M_TYPE
#pragma endregion

		template<class _data, class _data_ptr, class _data_ref>
		struct _TENSOR : public _TENSOR_REF<_data, _data_ptr, _data_ref>, _smemobj<>
		{
		public:
			typedef _TENSOR<_data, _data_ptr, _data_ref> _type;
		protected:
			_sstream<_data> DataContainer;
		protected:
			typename _type::_dimension _SetTensor(const typename _type::_dimension &Dimension); //Assume dimension value legal
		public:
			_TENSOR();
			_TENSOR(const typename _type::_dimension &Dimension, const _data &Val);
			_TENSOR(const typename _type::_dimension &Dimension, const _data_ptr pData, const typename _type::_size Size);
			_TENSOR(const typename _type::_dimension &Dimension, const _sstream<_data> &DataStream);
			_TENSOR(const typename _type::_dimension &Dimension, const _data_ptr** pDataPtrArray, const typename _type::_dimension Size);
			_TENSOR(const typename _type::_dimension &Dimension, const vector<vector<_data_ptr>> &DataPtrArray);
			_TENSOR(const typename _type::_dimension &Dimension, const typename _type::_mat_ref* pMatrixArray, const typename _type::_size Size);
			_TENSOR(const typename _type::_dimension &Dimension, const vector<typename _type::_mat_ref> &MatrixArray);
			_TENSOR(const _type &Tensor);
			_TENSOR(const typename _type::_ref_type &Tensor);
			~_TENSOR();

			/*const typename _type::_mat_ref& operator[](const _size Index) const;
			typename _type::_mat_ref& operator[](const _size Index);*/

			_type& operator=(const _type &Tensor);
			_type& operator=(const typename _type::_ref_type &Tensor);

			_type operator+(const _type &Tensor) const;
			_type operator-(const _type &Tensor) const;
			_type operator*(const _type &Tensor) const;

			_type operator+(const typename _type::_ref_type &Tensor) const;
			_type operator-(const typename _type::_ref_type &Tensor) const;
			_type operator*(const typename _type::_ref_type &Tensor) const;

			_type operator+(const _data &Data) const;
			_type operator-(const _data &Data) const;
			_type operator*(const _data &Data) const;

			bool operator==(const _type &Tensor) const;
			bool operator!=(const _type &Tensor) const;

			/*May cause error when convolution itself*/
			_type iConv(const typename _type::_dimension &BlockSize, const typename _type::_ref_type &Tensor, const bool bRepeat = false) const;
			_type iConv(const typename _type::_dimension &Start, const typename _type::_dimension &End, const typename _type::_dimension &BlockSize, const typename _type::_ref_type &Tensor, const bool bRepeat = false) const;

			template<class _Func> _type iConvolution(const typename _type::_dimension &BlockSize, _Func &Func) const;
			template<class _Func> _type iConvolution(const typename _type::_dimension &Start, const typename _type::_dimension &End, const typename _type::_dimension &BlockSize, _Func &Func) const;
			template<class _Func> _type iConvolution(const typename _type::_dimension &BlockSize, const typename _type::_ref_type &Tensor, _Func &Func, const bool bRepeat = false) const;
			template<class _Func> _type iConvolution(const typename _type::_dimension &Start, const typename _type::_dimension &End, const typename _type::_dimension &BlockSize, const typename _type::_ref_type &Tensor, _Func &Func, const bool bRepeat = false) const;

			typename _type::_dimension iSetTensor(const typename _type::_dimension &Dimension, const _data_ptr pData, const typename _type::_size Size) override final;
			typename _type::_dimension iSetTensor(const typename _type::_dimension &Dimension, const _sstream<_data> &DataStream) override final;
			typename _type::_dimension iSetTensor(const typename _type::_dimension &Dimension, const _data_ptr** pDataPtrArray, const typename _type::_dimension ArraySize) override final;
			typename _type::_dimension iSetTensor(const typename _type::_dimension &Dimension, const vector<vector<_data_ptr>> &DataPtrArray) override final;
			typename _type::_dimension iSetTensor(const typename _type::_dimension &Dimension, const typename _type::_mat_ref* pMatrixArray, const typename _type::_size Size) override final;
			typename _type::_dimension iSetTensor(const typename _type::_dimension &Dimension, const vector<typename _type::_mat_ref> &MatrixArray) override final;

			typename _type::_dimension iSetDimension(const typename _type::_dimension &Dimension) override final;

			const _data& iGetData(const typename _type::_dimension &Index) const;
			const _data& iGetData(const typename _type::_size Index) const;

			_data& iGetData(const typename _type::_dimension &Index);
			_data& iGetData(const typename _type::_size Index);

			const typename _type::_ref_type& iGetRef() const;
			typename _type::_ref_type& iGetRef();
		};

		template<class _data>
		using _tensor = _TENSOR<_data, _data*, _data&>;

#pragma region _TENSOR CPP
#define _M_TYPE typename _TENSOR<_data, _data_ptr, _data_ref>
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		_M_TYPE::_type _TENSOR<_data, _data_ptr, _data_ref>::iConvolution(const _M_TYPE::_dimension &BlockSize, _Func &Func) const
		{
			return this->iConvolution(0, this->Dimension, BlockSize, Func);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		_M_TYPE::_type _TENSOR<_data, _data_ptr, _data_ref>::iConvolution(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, _Func &Func) const
		{
			_type DestTensor(this->Dimension - BlockSize + 1, 0.0);
			this->iConvolution(DestTensor, Start, End, BlockSize, Func);

			return DestTensor;
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		_M_TYPE::_type _TENSOR<_data, _data_ptr, _data_ref>::iConvolution(const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Tensor, _Func &Func, const bool bRepeat) const
		{
			return this->iConvolution(0, this->Dimension, BlockSize, Tensor, Func, bRepeat);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		_M_TYPE::_type _TENSOR<_data, _data_ptr, _data_ref>::iConvolution(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Tensor, _Func &Func, const bool bRepeat) const
		{
			_type DestTensor(this->Dimension - BlockSize + 1, 0.0);
			this->iConvolution(DestTensor, Start, End, BlockSize, Tensor, Func, bRepeat);

			return DestTensor;
		}
#undef _M_TYPE
#pragma endregion

#endif
	}
}