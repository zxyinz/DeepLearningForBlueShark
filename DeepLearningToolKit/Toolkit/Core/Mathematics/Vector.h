#include"SanMathematicsDef.h"
using namespace std;
#pragma once
namespace San
{
	namespace Mathematics
	{
#ifndef __VECTOR_H__
#define __VECTOR_H__
		template<class _data, class _data_ptr, class _data_ref>
		struct _VECTOR_REF : public _smemobj<>
		{
		public:
			typedef int32 _size;
			typedef _size _dimension;

			typedef _VECTOR_REF<_data, _data_ptr, _data_ref> _ref_type;
		protected:
			_data_ptr pData;
			_dimension Dimension;
		public:
			_VECTOR_REF();
			_VECTOR_REF(const _dimension &Dimension, const _data_ptr pData);
			_VECTOR_REF(const _sstream_ref<_data> &DataStream);
			_VECTOR_REF(const _dimension &Dimension, const _sstream_ref<_data> &DataStream);
			_VECTOR_REF(const _ref_type &Vec);
			~_VECTOR_REF();

			const _data_ref operator[](const _dimension &Index) const;
			_data_ref operator[](const _dimension &Index);

			_ref_type& operator=(const _ref_type &Vec);

			bool operator==(const _ref_type &Vec) const;
			bool operator!=(const _ref_type &Vec) const;

			/*May cause error when add itself*/
			void iAdd(const _ref_type &Vec, const bool bRepeat = false);
			void iAdd(const _dimension &Start, const _dimension &End, const _ref_type &Vec, const bool bRepeat = false);

			/*May cause error when sub itself*/
			void iSub(const _ref_type &Vec, const bool bRepeat = false);
			void iSub(const _dimension &Start, const _dimension &End, const _ref_type &Vec, const bool bRepeat = false);

			/*May cause error when multiply itself*/
			void iMul(const _ref_type &Vec, const bool bRepeat = false);
			void iMul(const _dimension &Start, const _dimension &End, const _ref_type &Vec, const bool bRepeat = false);

			void iSet(const _ref_type &Vec, const bool bRepeat = false);
			void iSet(const _dimension &Start, const _dimension &End, const _ref_type &Vec, const bool bRepeat = false);

			_data iDot(const _ref_type &Vec, const bool bRepeat = false) const;
			_data iDot(const _dimension &Start, const _dimension &End, const _ref_type &Vec, const bool bRepeat = false) const;

			/*May cause error when convolution itself*/
			void iConv(_ref_type &DestVec, const _dimension &BlockSize, const _ref_type &Vec, const bool bRepeat = false) const;
			void iConv(_ref_type &DestVec, const _dimension &Start, const _dimension &End, const _dimension &BlockSize, const _ref_type &Vec, const bool bRepeat = false) const;

			void iAdd(const _data &Data);
			void iAdd(const _dimension &Start, const _dimension &End, const _data &Data);

			void iSub(const _data &Data);
			void iSub(const _dimension &Start, const _dimension &End, const _data &Data);

			void iMul(const _data &Data);
			void iMul(const _dimension &Start, const _dimension &End, const _data &Data);

			void iSet(const _data &Data);
			void iSet(const _dimension &Start, const _dimension &End, const _data &Data);

			_data iMod() const;
			_data iMod(const _dimension &Start, const _dimension &End) const;

			_data iSum() const;
			_data iSum(const _dimension &Start, const _dimension &End) const;

			_data iProduct() const;
			_data iProduct(const _dimension &Start, const _dimension &End) const;

			_data iMin() const;
			_data iMin(const _dimension &Start, const _dimension &End) const;

			_data iMax() const;
			_data iMax(const _dimension &Start, const _dimension &End) const;

			void iNormalize();

			template<class _Func> void iConvolution(_ref_type &DestVec, const _dimension &BlockSize, _Func &Func) const;
			template<class _Func> void iConvolution(_ref_type &DestVec, const _dimension &Start, const _dimension &End, const _dimension &BlockSize, _Func &Func) const;
			template<class _Func> void iConvolution(_ref_type &DestVec, const _dimension &BlockSize, const _ref_type &Vec, _Func &Func, const bool bRepeat = false) const;
			template<class _Func> void iConvolution(_ref_type &DestVec, const _dimension &Start, const _dimension &End, const _dimension &BlockSize, const _ref_type &Vec, _Func &Func, const bool bRepeat = false) const;

			template<class _Func> void iConvolutionC(const _dimension &BlockSize, _Func &Func) const;
			template<class _Func> void iConvolutionC(const _dimension &Start, const _dimension &End, const _dimension &BlockSize, _Func &Func) const;
			template<class _Func> void iConvolutionC(const _dimension &BlockSize, const _ref_type &Vec, _Func &Func, const bool bRepeat = false) const;
			template<class _Func> void iConvolutionC(const _dimension &Start, const _dimension &End, const _dimension &BlockSize, const _ref_type &Vec, _Func &Func, const bool bRepeat = false) const;

			template<class _Func> void iOperation(_Func &Func);
			template<class _Func> void iOperation(const _dimension &Start, const _dimension &End, _Func &Func);
			template<class _Func> void iOperation(const _data &Data, _Func &Func);
			template<class _Func> void iOperation(const _dimension &Start, const _dimension &End, const _data &Data, _Func &Func);

			template<class _Func> void iOperationC(_Func &Func) const;
			template<class _Func> void iOperationC(const _dimension &Start, const _dimension &End, _Func &Func) const;
			template<class _Func> void iOperationC(const _data &Data, _Func &Func) const;
			template<class _Func> void iOperationC(const _dimension &Start, const _dimension &End, const _data &Data, _Func &Func) const;

			virtual _dimension iSetVector(const _dimension &Dimension, const _data_ptr pData, const _dimension &Size = 0);
			virtual _dimension iSetVector(const _sstream_ref<_data> &DataStream);
			virtual _dimension iSetVector(const _dimension &Dimension, const _sstream_ref<_data> &DataStream);

			virtual _dimension iSetDimension(const _dimension &Dimension);

			const _data_ptr	iGetContainer() const;
			_data_ptr		iGetContainer();

			_dimension iGetDimension() const;
			_size iGetSize() const;

			void iHomogenzation();
		};

		template<class _data>
		using _vector_ref = _VECTOR_REF<_data, _data*, _data&>;

#pragma region _VECTOR_REF CPP
#define _M_TYPE typename _VECTOR_REF<_data, _data_ptr, _data_ref>
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _VECTOR_REF<_data, _data_ptr, _data_ref>::iConvolution(_M_TYPE::_ref_type &DestVec, const _M_TYPE::_dimension &BlockSize, _Func &Func) const
		{
			this->iConvolution(DestVec, 0, this->Dimension, BlockSize, Func);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _VECTOR_REF<_data, _data_ptr, _data_ref>::iConvolution(_M_TYPE::_ref_type &DestVec, const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, _Func &Func) const
		{
			if (End < Start){ return; }
			if ((End - Start) < BlockSize){ return; }
			if ((BlockSize % 2) != 1){ return; }
			if (DestVec.Dimension < ((End - Start) - BlockSize + 1)){ return; }

			const _M_TYPE::_dimension Offset = BlockSize / 2;
			const _M_TYPE::_size Size = (this->Dimension > End ? End : this->Dimension) - (Offset << 1);

			_M_TYPE::_size Index = 0;
			for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1, Index = Index + 1)
			{
				DestVec[Index] = Func(_M_TYPE::_ref_type((_data_ptr) (this->pData + seek), BlockSize), Index, BlockSize);
			}
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _VECTOR_REF<_data, _data_ptr, _data_ref>::iConvolution(_M_TYPE::_ref_type &DestVec, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Vec, _Func &Func, const bool bRepeat) const
		{
			this->iConvolution(DestVec, 0, this->Dimension, BlockSize, Vec, Func, bRepeat);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _VECTOR_REF<_data, _data_ptr, _data_ref>::iConvolution(_M_TYPE::_ref_type &DestVec, const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Vec, _Func &Func, const bool bRepeat) const
		{
			if (End < Start){ return; }
			if ((End - Start) < BlockSize){ return; }
			if ((BlockSize % 2) != 1){ return; }
			if (DestVec.Dimension < ((End - Start) - BlockSize + 1)){ return; }

			const _M_TYPE::_dimension Offset = BlockSize / 2;
			const _M_TYPE::_size Size = (this->Dimension > End ? End : this->Dimension) - (Offset << 1);

			_M_TYPE::_size Index = 0;
			for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1, Index = Index + 1)
			{
				DestVec[Index] = Func(_M_TYPE::_ref_type((_data_ptr) (this->pData + seek), BlockSize), Index, BlockSize, Vec, bRepeat);
			}
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _VECTOR_REF<_data, _data_ptr, _data_ref>::iConvolutionC(const _M_TYPE::_dimension &BlockSize, _Func &Func) const
		{
			this->iConvolutionC(0, this->Dimension, BlockSize, Func);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _VECTOR_REF<_data, _data_ptr, _data_ref>::iConvolutionC(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, _Func &Func) const
		{
			if (End < Start){ return; }
			if ((End - Start) < BlockSize){ return; }
			if ((BlockSize % 2) != 1){ return; }

			const _M_TYPE::_dimension Offset = BlockSize / 2;
			const _M_TYPE::_size Size = (this->Dimension > End ? End : this->Dimension) - (Offset << 1);

			_M_TYPE::_size Index = 0;
			for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1, Index = Index + 1)
			{
				Func(_M_TYPE::_ref_type((_data_ptr) (this->pData + seek), BlockSize), Index, BlockSize);
			}
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _VECTOR_REF<_data, _data_ptr, _data_ref>::iConvolutionC(const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Vec, _Func &Func, const bool bRepeat) const
		{
			this->iConvolutionC(0, this->Dimension, BlockSize, Vec, Func, bRepeat);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _VECTOR_REF<_data, _data_ptr, _data_ref>::iConvolutionC(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Vec, _Func &Func, const bool bRepeat) const
		{
			if (End < Start){ return; }
			if ((End - Start) < BlockSize){ return; }
			if ((BlockSize % 2) != 1){ return; }

			const _M_TYPE::_dimension Offset = BlockSize / 2;
			const _M_TYPE::_size Size = (this->Dimension > End ? End : this->Dimension) - (Offset << 1);

			_M_TYPE::_size Index = 0;
			for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1, Index = Index + 1)
			{
				Func(_M_TYPE::_ref_type((_data_ptr) (this->pData + seek), BlockSize), Index, BlockSize, Vec, bRepeat);
			}
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _VECTOR_REF<_data, _data_ptr, _data_ref>::iOperation(_Func &Func)
		{
			this->iOperation(0, this->Dimension, Func);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _VECTOR_REF<_data, _data_ptr, _data_ref>::iOperation(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, _Func &Func)
		{
			const _M_TYPE::_size Size = this->Dimension > End ? this->Dimension : End;

			for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
			{
				this->pData[seek] = Func(this->pData[seek], seek - Start);
			}
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _VECTOR_REF<_data, _data_ptr, _data_ref>::iOperation(const _data &Data, _Func &Func)
		{
			this->iOperation(0, this->Dimension, Data, Func);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _VECTOR_REF<_data, _data_ptr, _data_ref>::iOperation(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _data &Data, _Func &Func)
		{
			const _M_TYPE::_size Size = this->Dimension > End ? this->Dimension : End;

			for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
			{
				this->pData[seek] = Func(this->pData[seek], seek - Start, Data);
			}
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _VECTOR_REF<_data, _data_ptr, _data_ref>::iOperationC(_Func &Func) const
		{
			this->iOperationC(0, this->Dimension, Func);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _VECTOR_REF<_data, _data_ptr, _data_ref>::iOperationC(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, _Func &Func) const
		{
			const _M_TYPE::_size Size = this->Dimension > End ? this->Dimension : End;
			_M_TYPE::_dimension Pos = 0;

			for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1, Pos = Pos + 1)
			{
				Func(this->pData[seek], Pos);
			}
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _VECTOR_REF<_data, _data_ptr, _data_ref>::iOperationC(const _data &Data, _Func &Func) const
		{
			this->iOperation(0, this->Dimension, Data, Func);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		void _VECTOR_REF<_data, _data_ptr, _data_ref>::iOperationC(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _data &Data, _Func &Func) const
		{
			const _M_TYPE::_size Size = this->Dimension > End ? this->Dimension : End;

			for (_M_TYPE::_size seek = Start; seek < Size; seek = seek + 1)
			{
				Func(this->pData[seek], seek - Start, Data);
			}
		}
#undef _M_TYPE
#pragma endregion

		template<class _data, class _data_ptr, class _data_ref>
		struct _VECTOR : public _VECTOR_REF<_data, _data_ptr, _data_ref>, public _smemobj<>
		{
		public:
			typedef _VECTOR<_data, _data_ptr, _data_ref> _type;
		private:
			_sstream<_data> DataContainer;
		protected:
			typename _type::_dimension _SetVector(const _data_ptr &pData, const typename _type::_dimension &Dimension);
		public:
			_VECTOR();
			_VECTOR(const typename _type::_dimension &Dimension, const _data &Val);
			_VECTOR(const _data_ptr pData, const typename _type::_dimension &Size);
			_VECTOR(const typename _type::_dimension &Dimension, const _data_ptr pData, const typename _type::_dimension &Size);
			_VECTOR(const _sstream_ref<_data> &DataStream);
			_VECTOR(const typename _type::_dimension &Dimension, const _sstream_ref<_data> &DataStream);
			_VECTOR(const _type &Vec);
			_VECTOR(const typename _type::_ref_type &VecRef);
			~_VECTOR();

			_type& operator=(const _type &Vec);
			_type& operator=(const typename _type::_ref_type &VecRef);

			_type operator+(const _type &Vec) const;
			_type operator-(const _type &Vec) const;
			_type operator*(const _type &Vec) const;

			_type operator+(const typename _type::_ref_type &Vec) const;
			_type operator-(const typename _type::_ref_type &Vec) const;
			_type operator*(const typename _type::_ref_type &Vec) const;

			_type operator+(const _data &Data) const;
			_type operator-(const _data &Data) const;
			_type operator*(const _data &Data) const;

			bool operator==(const _type &Vec) const;
			bool operator!=(const _type &Vec) const;

			_type iConv(const typename _type::_dimension &BlockSize, const typename _type::_ref_type &Vec, const bool bRepeat = false) const;
			_type iConv(const typename _type::_dimension &Start, const typename _type::_dimension &End, const typename _type::_dimension &BlockSize, const typename _type::_ref_type &Vec, const bool bRepeat = false) const;

			template<class _Func> _type iConvolution(const typename _type::_dimension &BlockSize, _Func &Func) const;
			template<class _Func> _type iConvolution(const typename _type::_dimension &Start, const typename _type::_dimension &End, const typename _type::_dimension &BlockSize, _Func &Func) const;
			template<class _Func> _type iConvolution(const typename _type::_dimension &BlockSize, const typename _type::_ref_type &Vec, _Func &Func, const bool bRepeat = false) const;
			template<class _Func> _type iConvolution(const typename _type::_dimension &Start, const typename _type::_dimension &End, const typename _type::_dimension &BlockSize, const typename _type::_ref_type &Vec, _Func &Func, const bool bRepeat = false) const;

			typename _type::_dimension iSetVector(const _data_ptr pData, const typename _type::_dimension &Size);
			typename _type::_dimension iSetVector(const typename _type::_dimension &Dimension, const _data_ptr pData, const typename _type::_dimension &Size) override final;
			typename _type::_dimension iSetVector(const _sstream_ref<_data> &DataStream) override final;
			typename _type::_dimension iSetVector(const typename _type::_dimension &Dimension, const _sstream_ref<_data> &DataStream) override final;

			typename _type::_dimension iSetDimension(const typename _type::_dimension &Dimension) override final;

			const _data& iGetData(const typename _type::_dimension &Index) const;
			const _data& iGetData(const typename _type::_size Index) const;

			_data& iGetData(const typename _type::_dimension &Index);
			_data& iGetData(const typename _type::_size Index);
			
			const typename _type::_ref_type& iGetRef() const;
			typename _type::_ref_type& iGetRef();
		};

		template<class _data>
		using _vector = _VECTOR<_data, _data*, _data&>;

#pragma region _VECTOR CPP
#define _M_TYPE typename _VECTOR<_data, _data_ptr, _data_ref>
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		_M_TYPE::_type _VECTOR<_data, _data_ptr, _data_ref>::iConvolution(const _M_TYPE::_dimension &BlockSize, _Func &Func) const
		{
			return this->iConvolution(0, this->Dimension, BlockSize, Func);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		_M_TYPE::_type _VECTOR<_data, _data_ptr, _data_ref>::iConvolution(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, _Func &Func) const
		{
			_type DestVec(this->Dimension - BlockSize + 1, 0.0);
			this->iConvolution(DestVec, Start, End, BlockSize, Func);

			return DestVec;
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		_M_TYPE::_type _VECTOR<_data, _data_ptr, _data_ref>::iConvolution(const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Vec, _Func &Func, const bool bRepeat) const
		{
			return this->iConvolution(0, this->Dimension, BlockSize, Vec, Func, bRepeat);
		}
		template<class _data, class _data_ptr, class _data_ref>
		template<class _Func>
		_M_TYPE::_type _VECTOR<_data, _data_ptr, _data_ref>::iConvolution(const _M_TYPE::_dimension &Start, const _M_TYPE::_dimension &End, const _M_TYPE::_dimension &BlockSize, const _M_TYPE::_ref_type &Vec, _Func &Func, const bool bRepeat) const
		{
			_type DestVec(this->Dimension - BlockSize + 1, 0.0);
			this->iConvolution(DestVec, Start, End, BlockSize, Vec, Func, bRepeat);

			return DestVec;
		}
#undef _M_TYPE
#pragma endregion

#endif
	}
}