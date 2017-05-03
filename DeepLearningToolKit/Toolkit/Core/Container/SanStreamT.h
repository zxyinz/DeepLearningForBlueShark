//Project: San Lab Game Engine
//Version: 2.1.1
//Debug State: Add stream operator [<< and >>], begin, end veriable, opt the pop and push speed for _sstream<>
#include"SanContainerDef.h"
#include"SanStreamIterator.h"
#pragma once
using namespace std;
namespace San
{
#ifndef __STDCONTAINERSTREAMT_H__
#define __STDCONTAINERSTREAMT_H__
	template<class _data, class _ptr, class _ref>
	struct _SANSTREAM_REF
	{
	public:
		typedef _SANSTREAM_REF<_data, _ptr, _ref> _ref_type;
		typedef int32 _size;

		typedef _SANSTREAMEX_CONST_ITERATOR<_data, _ptr, _ref>	const_iterator;
		typedef _SANSTREAMEX_ITERATOR<_data, _ptr, _ref>		iterator;

		typedef _SANSTREAMEX_ITERATOR_S<_data, _ptr, _ref>		safe_iterator;
		typedef _SANSTREAMEX_ITERATOR_B<_data, _ptr, _ref>		byte_iterator;
	private:
		_ptr pStream;
		_size Size;
		_size SizeInByte;
	protected:
		void _UpdateStream(const _ptr pStream, const _size &Size)
		{
			this->pStream = pStream;
			this->Size = Size;
			this->SizeInByte = sizeof(_data) * Size;
		};
	public:
		_SANSTREAM_REF(const _ptr pStream, const _size &BufferSize);
		_SANSTREAM_REF(const _ref_type &Stream);
		~_SANSTREAM_REF();

		//operators overload
		_ref_type& operator=(const _ref_type &Stream);

		bool operator==(const _ref_type &Stream) const;
		bool operator!=(const _ref_type &Stream) const;

		const _ref operator[](const _size &Position) const;
		_ref operator[](const _size &Position);

		//Stream IO Function
		_size iSetStream(const _data &Data);
		_size iSetStream(const _size &Offset, const _size &Size, const _data &Data);

		_size iSetStream(const _size &Offset, const _ref_type &Stream);
		_size iSetStream(const _size &Offset, const _ptr pBuffer, const _size &BufferSize);

		_size iGetStream(const _size &Offset, _data &Data) const;
		_size iGetStream(const _size &Offset, _ref_type &Stream) const; //Error, size
		_size iGetStream(const _size &Offset, _ptr pBuffer, const _size &BufferSize) const;

		_size iGetSize() const;
		_size iGetSizeInByte() const;

		const _ptr iGetPtr() const;
		_ptr iGetPtr();

		bool iEmpty() const;

		//C++ STL Style Function
		_size set(const _data &Data);
		_size set(const _size &Offset, const _size &Size, const _data &Data);
		_size set(const _size &Offset, const _ref_type &Stream);
		_size set(const _size &Offset, const _ptr pBuffer, const uint32 &BufferSize);

		_size get(const _size &Offset, _data &Data) const;
		_size get(const _size &Offset, _ref_type &Stream) const;
		_size get(const _size &Offset, _ptr pBuffer, const uint32 &BufferSize) const;

		const_iterator cbegin() const;
		const_iterator cend() const;

		const_iterator begin() const;
		iterator begin();
		const_iterator end() const;
		iterator end();

		const_iterator find(const _data &Target) const;
		iterator find(const _data &Target);

		safe_iterator begin_s() const;
		safe_iterator end_s() const;

		safe_iterator find_s(const _data &Target) const;

		byte_iterator begin_b() const;
		byte_iterator end_b() const;

		byte_iterator find_b(const uint8 &TargetByte) const;

		_size size() const;
		_size size_in_byte() const;

		const _ptr ptr() const;
		_ptr ptr();

		bool empty() const;
	};

	template<class _data>
	using _sstream_ref = _SANSTREAM_REF<_data, _data*, _data&>;

#pragma region _SANSTREAM_REF CPP
	template<class _data, class _ptr, class _ref>
	_SANSTREAM_REF<_data, _ptr, _ref>::_SANSTREAM_REF(const _ptr pStream, const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size &BufferSize)
		:pStream(pStream), Size(BufferSize), SizeInByte(sizeof(_data) * BufferSize)
	{
	}
	template<class _data, class _ptr, class _ref>
	_SANSTREAM_REF<_data, _ptr, _ref>::_SANSTREAM_REF(const typename _SANSTREAM_REF<_data, _ptr, _ref>::_ref_type &Stream)
		:pStream(Stream.pStream), Size(Stream.Size), SizeInByte(Stream.SizeInByte)
	{
	}
	template<class _data, class _ptr, class _ref>
	_SANSTREAM_REF<_data, _ptr, _ref>::~_SANSTREAM_REF()
	{
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::_ref_type& _SANSTREAM_REF<_data, _ptr, _ref>::operator=(const typename _SANSTREAM_REF<_data, _ptr, _ref>::_ref_type &Stream)
	{
		this->_UpdateStream(Stream.pStream, Stream.Size);
	}
	template<class _data, class _ptr, class _ref>
	bool _SANSTREAM_REF<_data, _ptr, _ref>::operator==(const typename _SANSTREAM_REF<_data, _ptr, _ref>::_ref_type &Stream) const
	{
		if (this->Size != Stream.Size){ return false; }

		const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size Size = this->Size;
		_ptr pS1 = this->pStream;
		_ptr pS2 = Stream.pStream;

		for (typename _SANSTREAM_REF<_data, _ptr, _ref>::_size seek = 0; seek < this->Size; seek = seek + 1, pS1++, pS2++)
		{
			if ((*pS1) != (*pS2)) { return false; }
		}

		return true;
	}
	template<class _data, class _ptr, class _ref>
	bool _SANSTREAM_REF<_data, _ptr, _ref>::operator!=(const typename _SANSTREAM_REF<_data, _ptr, _ref>::_ref_type &Stream) const
	{
		if (this->Size != Stream.Size){ return true; }

		const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size Size = this->Size;
		_ptr pS1 = this->pStream;
		_ptr pS2 = Stream.pStream;

		for (typename _SANSTREAM_REF<_data, _ptr, _ref>::_size seek = 0; seek < this->Size; seek = seek + 1, pS1++, pS2++)
		{
			if ((*pS1) == (*pS2)) { return false; }
		}

		return true;
	}
	template<class _data, class _ptr, class _ref>
	const _ref _SANSTREAM_REF<_data, _ptr, _ref>::operator[](const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size &Position) const
	{
		return this->pStream[Position];
	}
	template<class _data, class _ptr, class _ref>
	_ref _SANSTREAM_REF<_data, _ptr, _ref>::operator[](const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size &Position)
	{
		return this->pStream[Position];
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::_size _SANSTREAM_REF<_data, _ptr, _ref>::iSetStream(const _data &Data)
	{
		return this->iSetStream(0, this->Size, Data);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::_size _SANSTREAM_REF<_data, _ptr, _ref>::iSetStream(const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size &Offset, const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size &Size, const _data &Data)
	{
		if (Offset >= this->Size){ return 0; }

		const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size EndPos = min(Offset + Size, this->Size);

		for (typename _SANSTREAM_REF<_data, _ptr, _ref>::_size seek = Offset; seek < EndPos; seek = seek + 1)
		{
			this->pStream[seek] = Data;
		}

		return EndPos - Offset;
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::_size _SANSTREAM_REF<_data, _ptr, _ref>::iSetStream(const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size &Offset, const typename _SANSTREAM_REF<_data, _ptr, _ref>::_ref_type &Stream)
	{
		return this->iSetStream(Offset, Stream.pStream, Stream.Size);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::_size _SANSTREAM_REF<_data, _ptr, _ref>::iSetStream(const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size &Offset, const _ptr pBuffer, const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size &BufferSize)
	{
		if ((pBuffer == nullptr) || (BufferSize == 0)){ return 0; }
		if (Offset >= this->Size){ return 0; }

		const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size EndPos = min(Offset + BufferSize, this->Size);

		for (typename _SANSTREAM_REF<_data, _ptr, _ref>::_size seek = Offset; seek < EndPos; seek = seek + 1)
		{
			this->pStream[seek] = pBuffer[seek - Offset];
		}

		return EndPos - Offset;
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::_size _SANSTREAM_REF<_data, _ptr, _ref>::iGetStream(const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size &Offset, _data &Data) const
	{
		return this->iGetStream(Offset, (_ptr) &Data, sizeof(Data));
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::_size _SANSTREAM_REF<_data, _ptr, _ref>::iGetStream(const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size &Offset, typename _SANSTREAM_REF<_data, _ptr, _ref>::_ref_type &Stream) const //Error, size
	{
		return this->iGetStream(Offset, Stream.pStream, Stream.Size);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::_size _SANSTREAM_REF<_data, _ptr, _ref>::iGetStream(const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size &Offset, _ptr pBuffer, const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size &BufferSize) const
	{
		if ((pBuffer == nullptr) || (BufferSize == 0)){ return 0; }
		if (Offset >= this->Size){ return 0; }

		const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size EndPos = min(Offset + BufferSize, this->Size);

		for (typename _SANSTREAM_REF<_data, _ptr, _ref>::_size seek = Offset; seek < EndPos; seek = seek + 1)
		{
			pBuffer[seek - Offset] = this->pStream[seek];
		}

		return EndPos - Offset;
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::_size _SANSTREAM_REF<_data, _ptr, _ref>::iGetSize() const
	{
		return this->Size;
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::_size _SANSTREAM_REF<_data, _ptr, _ref>::iGetSizeInByte() const
	{
		return this->SizeInByte;
	}
	template<class _data, class _ptr, class _ref>
	const _ptr _SANSTREAM_REF<_data, _ptr, _ref>::iGetPtr() const
	{
		return this->pStream;
	}
	template<class _data, class _ptr, class _ref>
	_ptr _SANSTREAM_REF<_data, _ptr, _ref>::iGetPtr()
	{
		return this->pStream;
	}
	template<class _data, class _ptr, class _ref>
	bool _SANSTREAM_REF<_data, _ptr, _ref>::iEmpty() const
	{
		return this->Size == 0;
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::_size _SANSTREAM_REF<_data, _ptr, _ref>::set(const _data &Data)
	{
		return this->iSetStream(Data);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::_size _SANSTREAM_REF<_data, _ptr, _ref>::set(const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size &Offset, const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size &Size, const _data &Data)
	{
		return this->iSetStream(Offset, Size, Data);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::_size _SANSTREAM_REF<_data, _ptr, _ref>::set(const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size &Offset, const typename _SANSTREAM_REF<_data, _ptr, _ref>::_ref_type &Stream)
	{
		return this->iSetStream(Offset, Stream);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::_size _SANSTREAM_REF<_data, _ptr, _ref>::set(const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size &Offset, const _ptr pBuffer, const uint32 &BufferSize)
	{
		return this->iSetStream(Offset, pBuffer, BufferSize);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::_size _SANSTREAM_REF<_data, _ptr, _ref>::get(const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size &Offset, _data &Data) const
	{
		return this->iGetStream(Offset, Data);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::_size _SANSTREAM_REF<_data, _ptr, _ref>::get(const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size &Offset, typename _SANSTREAM_REF<_data, _ptr, _ref>::_ref_type &Stream) const
	{
		return this->iGetStream(Offset, Stream);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::_size _SANSTREAM_REF<_data, _ptr, _ref>::get(const typename _SANSTREAM_REF<_data, _ptr, _ref>::_size &Offset, _ptr pBuffer, const uint32 &BufferSize) const
	{
		return this->iGetStream(Offset, pBuffer, BufferSize);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::const_iterator _SANSTREAM_REF<_data, _ptr, _ref>::cbegin() const
	{
		return ((const _ref_type*)this)->begin();
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::const_iterator _SANSTREAM_REF<_data, _ptr, _ref>::cend() const
	{
		return ((const _ref_type*)this)->end();
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::const_iterator _SANSTREAM_REF<_data, _ptr, _ref>::begin() const
	{
		return typename _SANSTREAM_REF<_data, _ptr, _ref>::const_iterator(this->pStream, this->pStream + this->Size, this->pStream);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::iterator _SANSTREAM_REF<_data, _ptr, _ref>::begin()
	{
		return typename _SANSTREAM_REF<_data, _ptr, _ref>::iterator(this->pStream, this->pStream + this->Size, this->pStream);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::const_iterator _SANSTREAM_REF<_data, _ptr, _ref>::end() const
	{
		return typename _SANSTREAM_REF<_data, _ptr, _ref>::const_iterator(this->pStream, this->pStream + this->Size, nullptr);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::iterator _SANSTREAM_REF<_data, _ptr, _ref>::end()
	{
		return typename _SANSTREAM_REF<_data, _ptr, _ref>::iterator(this->pStream, this->pStream + this->Size, nullptr);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::const_iterator _SANSTREAM_REF<_data, _ptr, _ref>::find(const _data &Target) const
	{
		for (typename _SANSTREAM_REF<_data, _ptr, _ref>::_size seek = 0; seek < this->Size; seek = seek + 1)
		{
			if (this->pStream[seek] == Target)
			{
				return typename _SANSTREAM_REF<_data, _ptr, _ref>::const_iterator(this->pStream, this->pStream + this->Size, this->pStream + seek);
			}
		}
		return typename _SANSTREAM_REF<_data, _ptr, _ref>::const_iterator(this->pStream, this->pStream + this->Size, nullptr);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::iterator _SANSTREAM_REF<_data, _ptr, _ref>::find(const _data &Target)
	{
		for (typename _SANSTREAM_REF<_data, _ptr, _ref>::_size seek = 0; seek < this->Size; seek = seek + 1)
		{
			if (this->pStream[seek] == Target)
			{
				return typename _SANSTREAM_REF<_data, _ptr, _ref>::iterator(this->pStream, this->pStream + this->Size, this->pStream + seek);
			}
		}
		return typename _SANSTREAM_REF<_data, _ptr, _ref>::iterator(this->pStream, this->pStream + this->Size, nullptr);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::safe_iterator _SANSTREAM_REF<_data, _ptr, _ref>::begin_s() const
	{
		return typename _SANSTREAM_REF<_data, _ptr, _ref>::safe_iterator(this->pStream, this->Size, this->pStream);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::safe_iterator _SANSTREAM_REF<_data, _ptr, _ref>::end_s() const
	{
		return typename _SANSTREAM_REF<_data, _ptr, _ref>::safe_iterator(this->pStream, this->Size, nullptr);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::safe_iterator _SANSTREAM_REF<_data, _ptr, _ref>::find_s(const _data &Target) const
	{
		for (typename _SANSTREAM_REF<_data, _ptr, _ref>::_size seek = 0; seek < this->Size; seek = seek + 1)
		{
			if (this->pStream[seek] == Target)
			{
				return typename _SANSTREAM_REF<_data, _ptr, _ref>::safe_iterator(this->pStream, this->Size, this->pStream + seek);
			}
		}
		return typename _SANSTREAM_REF<_data, _ptr, _ref>::safe_iterator(this->pStream, this->Size, nullptr);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::byte_iterator _SANSTREAM_REF<_data, _ptr, _ref>::begin_b() const
	{
		return typename _SANSTREAM_REF<_data, _ptr, _ref>::byte_iterator(this->pStream, this->Size, this->pStream);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::byte_iterator _SANSTREAM_REF<_data, _ptr, _ref>::end_b() const
	{
		return typename _SANSTREAM_REF<_data, _ptr, _ref>::byte_iterator(this->pStream, this->Size, nullptr);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::byte_iterator _SANSTREAM_REF<_data, _ptr, _ref>::find_b(const uint8 &TargetByte) const
	{
		const uint8* pByteStream = this->pStream;

		for (typename _SANSTREAM_REF<_data, _ptr, _ref>::_size seek = 0; seek < this->SizeInByte; seek = seek + 1)
		{
			if (pByteStream[seek] == TargetByte)
			{
				return typename _SANSTREAM_REF<_data, _ptr, _ref>::byte_iterator(this->pStream, this->Size, pByteStream + seek);
			}
		}
		return typename _SANSTREAM_REF<_data, _ptr, _ref>::byte_iterator(this->pStream, this->Size, nullptr);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::_size _SANSTREAM_REF<_data, _ptr, _ref>::size() const
	{
		return this->iGetSize();
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM_REF<_data, _ptr, _ref>::_size _SANSTREAM_REF<_data, _ptr, _ref>::size_in_byte() const
	{
		return this->iGetSizeInByte();
	}
	template<class _data, class _ptr, class _ref>
	const _ptr _SANSTREAM_REF<_data, _ptr, _ref>::ptr() const
	{
		return this->iGetPtr();
	}
	template<class _data, class _ptr, class _ref>
	_ptr _SANSTREAM_REF<_data, _ptr, _ref>::ptr()
	{
		return this->iGetPtr();
	}
	template<class _data, class _ptr, class _ref>
	bool _SANSTREAM_REF<_data, _ptr, _ref>::empty() const
	{
		return this->iEmpty();
	}
#pragma endregion

	template<class _data, class _ptr, class _ref>
	struct _SANSTREAM : public _SANSTREAM_REF<_data, _ptr,_ref>
	{
	public:
		typedef _SANSTREAM<_data, _ptr, _ref> _type;
	private:
		_ptr pDataContainer;
		typename _type::_size ContainerSize;
	protected:
		void _ReleaseStream();
	public:
		_SANSTREAM(const typename _type::_size &Size = 0);
		_SANSTREAM(const typename _type::_size &Size, const _data &InitializeValue);
		_SANSTREAM(const typename _type::_size &Size, const _ptr pStream, const typename _type::_size &BufferSize);
		_SANSTREAM(const _type &Stream);
		_SANSTREAM(const typename _type::_size &Size, const _type &Stream);
		_SANSTREAM(const typename _type::_ref_type &Stream);
		_SANSTREAM(const typename _type::_size &Size, const typename _type::_ref_type &Stream);
		~_SANSTREAM();

		//operators overload
		_type& operator=(const _type &Stream);
		_type& operator=(const typename _type::_ref_type &Stream);

		_type operator+(const _type &Stream) const;
		_type operator+(const typename _type::_ref_type &Stream) const;

		_type operator-(const typename _type::_size &Size) const;

		bool operator==(const _type &Stream) const;
		bool operator!=(const _type &Stream) const;

		//Stream IO Function
		typename _type::_size iPopFront(const typename _type::_size &Size = 1);
		typename _type::_size iPopBack(const typename _type::_size &Size = 1);

		typename _type::_size iReSizeStream(const typename _type::_size &Size);
		typename _type::_size iReSizeStream(const typename _type::_size &Size, const _data &Data);

		void iClear();

		const typename _type::_ref_type& iGetRef() const;
		typename _type::_ref_type& iGetRef();

		//C++ STL Style Function
		typename _type::_size pop_front(const typename _type::_size &Size = 1);
		typename _type::_size pop_back(const typename _type::_size &Size = 1);

		typename _type::_size resize(const typename _type::_size &Size);
		typename _type::_size resize(const typename _type::_size &Size, const _data &Data);

		void clear();
	};

#pragma region _SANSTREAMEX CPP file
	template<class _data, class _ptr, class _ref>
	_SANSTREAM<_data, _ptr, _ref>::_SANSTREAM(const typename _SANSTREAM<_data, _ptr, _ref>::_size &Size)
		:_SANSTREAM_REF<_data, _ptr, _ref>(nullptr, 0), pDataContainer(nullptr), ContainerSize(0)
	{
		if (Size != 0)
		{
			this->ContainerSize = Size;
			this->pDataContainer = new _data[this->ContainerSize];

			this->_UpdateStream(this->pDataContainer, this->ContainerSize);
		}
	}
	template<class _data, class _ptr, class _ref>
	_SANSTREAM<_data, _ptr, _ref>::_SANSTREAM(const typename _SANSTREAM<_data, _ptr, _ref>::_size &Size, const _data &InitializeValue)
		:_SANSTREAM_REF<_data, _ptr, _ref>(nullptr, 0), pDataContainer(nullptr), ContainerSize(0)
	{
		if (Size != 0)
		{
			this->ContainerSize = Size;
			this->pDataContainer = new _data[this->ContainerSize];

			this->_UpdateStream(this->pDataContainer, this->ContainerSize);
			this->iSetStream(InitializeValue);
		}
	}
	template<class _data, class _ptr, class _ref>
	_SANSTREAM<_data, _ptr, _ref>::_SANSTREAM(const typename _SANSTREAM<_data, _ptr, _ref>::_size &Size, const _ptr pStream, const typename _SANSTREAM<_data, _ptr, _ref>::_size &BufferSize)
		:_SANSTREAM_REF<_data, _ptr, _ref>(nullptr, 0), pDataContainer(nullptr), ContainerSize(0)
	{
		if (BufferSize != 0)
		{
			this->ContainerSize = Size;
			this->pDataContainer = new _data[this->ContainerSize];

			this->_UpdateStream(this->pDataContainer, this->ContainerSize);
			this->iSetStream(0, pStream, BufferSize);
		}
	}
	template<class _data, class _ptr, class _ref>
	_SANSTREAM<_data, _ptr, _ref>::_SANSTREAM(const typename _SANSTREAM<_data, _ptr, _ref>::_type &Stream)
		:_SANSTREAM_REF<_data, _ptr, _ref>(nullptr, 0), pDataContainer(nullptr), ContainerSize(0)
	{
		if (!Stream.iEmpty())
		{
			this->ContainerSize = Stream.iGetSize();
			this->pDataContainer = new _data[this->ContainerSize];

			this->_UpdateStream(this->pDataContainer, this->ContainerSize);
			this->iSetStream(0, Stream);
		}
	}
	template<class _data, class _ptr, class _ref>
	_SANSTREAM<_data, _ptr, _ref>::_SANSTREAM(const typename _SANSTREAM<_data, _ptr, _ref>::_size &Size, const typename _SANSTREAM<_data, _ptr, _ref>::_type &Stream)
		:_SANSTREAM_REF<_data, _ptr, _ref>(nullptr, 0), pDataContainer(nullptr), ContainerSize(0)
	{
		if (Size != 0)
		{
			this->ContainerSize = Stream.iGetSize();
			this->pDataContainer = new _data[this->ContainerSize];

			this->_UpdateStream(this->pDataContainer, this->ContainerSize);
			this->iSetStream(0, Stream);
		}
	}
	template<class _data, class _ptr, class _ref>
	_SANSTREAM<_data, _ptr, _ref>::_SANSTREAM(const typename _SANSTREAM<_data, _ptr, _ref>::_ref_type &Stream)
		:_SANSTREAM_REF<_data, _ptr, _ref>(nullptr, 0), pDataContainer(nullptr), ContainerSize(0)
	{
		if (!Stream.iEmpty())
		{
			this->ContainerSize = Stream.iGetSize();
			this->pDataContainer = new _data[this->ContainerSize];

			this->_UpdateStream(this->pDataContainer, this->ContainerSize);
			this->iSetStream(0, Stream);
		}
	}
	template<class _data, class _ptr, class _ref>
	_SANSTREAM<_data, _ptr, _ref>::_SANSTREAM(const typename _SANSTREAM<_data, _ptr, _ref>::_size &Size, const typename _SANSTREAM<_data, _ptr, _ref>::_ref_type &Stream)
		:_SANSTREAM_REF<_data, _ptr, _ref>(nullptr, 0), pDataContainer(nullptr), ContainerSize(0)
	{
		if (Size != 0)
		{
			this->ContainerSize = Stream.iGetSize();
			this->pDataContainer = new _data[this->ContainerSize];

			this->_UpdateStream(this->pDataContainer, this->ContainerSize);
			this->iSetStream(0, Stream);
		}
	}
	template<class _data, class _ptr, class _ref>
	_SANSTREAM<_data, _ptr, _ref>::~_SANSTREAM()
	{
		this->_ReleaseStream();
	}
	template<class _data, class _ptr, class _ref>
	void _SANSTREAM<_data, _ptr, _ref>::_ReleaseStream()
	{
		if (this->pDataContainer != nullptr)
		{
			this->ContainerSize == 1 ? (delete this->pDataContainer) : (delete[] this->pDataContainer);
		}

		this->pDataContainer = nullptr;
		this->ContainerSize = 0;
		this->_UpdateStream(this->pDataContainer, this->ContainerSize);
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM<_data, _ptr, _ref>::_type& _SANSTREAM<_data, _ptr, _ref>::operator=(const typename _SANSTREAM<_data, _ptr, _ref>::_type &Stream)
	{
		return this->operator=(Stream.iGetRef());
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM<_data, _ptr, _ref>::_type& _SANSTREAM<_data, _ptr, _ref>::operator=(const typename _SANSTREAM<_data, _ptr, _ref>::_ref_type &Stream)
	{
		if (this == &(Stream)){ return *this; }

		const typename _SANSTREAM<_data, _ptr, _ref>::_size StreamSize = Stream.iGetSize();

		if ((Stream.iGetPtr() == nullptr) || (StreamSize == 0))
		{
			this->_ReleaseStream();
			return *this;
		}

		_ptr pStream = new _data[StreamSize];
		Stream.iGetStream(0, pStream, StreamSize);

		this->_ReleaseStream();

		this->pDataContainer = pStream;
		this->ContainerSize = Stream.iGetSize();
		this->_UpdateStream(this->pDataContainer, this->ContainerSize);

		return *this;
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM<_data, _ptr, _ref>::_type _SANSTREAM<_data, _ptr, _ref>::operator+(const typename _SANSTREAM<_data, _ptr, _ref>::_type &Stream) const
	{
		return this->operator+(Stream.iGetRef());
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM<_data, _ptr, _ref>::_type _SANSTREAM<_data, _ptr, _ref>::operator+(const typename _SANSTREAM<_data, _ptr, _ref>::_ref_type &Stream) const
	{
		if ((Stream.iGetPtr() == nullptr) || (Stream.iGetSize() == 0)){ return *this; }

		uint32 Size = this->iGetSize() + Stream.iGetSize();
		typename _SANSTREAM<_data, _ptr, _ref>::_type OutputStream(Size);

		OutputStream.iSetStream(0, *this);
		OutputStream.iSetStream(this->iGetSize(), Stream);

		return OutputStream;
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM<_data, _ptr, _ref>::_type _SANSTREAM<_data, _ptr, _ref>::operator-(const typename _SANSTREAM<_data, _ptr, _ref>::_size &Size) const
	{
		if (Size >= this->iGetSize()) { return typename _SANSTREAM<_data, _ptr, _ref>::_type(); }

		typename _SANSTREAM<_data, _ptr, _ref>::_type OutputStream(this->iGetSize() - Size);

		OutputStream.iSetStream(0, *this);

		return OutputStream;
	}
	template<class _data, class _ptr, class _ref>
	bool _SANSTREAM<_data, _ptr, _ref>::operator==(const typename _SANSTREAM<_data, _ptr, _ref>::_type &Stream) const
	{
		return this->iGetRef() == Stream.iGetRef();
	}
	template<class _data, class _ptr, class _ref>
	bool _SANSTREAM<_data, _ptr, _ref>::operator!=(const typename _SANSTREAM<_data, _ptr, _ref>::_type &Stream) const
	{
		return this->iGetRef() != Stream.iGetRef();
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM<_data, _ptr, _ref>::_size _SANSTREAM<_data, _ptr, _ref>::iPopFront(const typename _SANSTREAM<_data, _ptr, _ref>::_size &Size)
	{
		if (Size >= this->ContainerSize)
		{
			this->_ReleaseStream();
			this->_UpdateStream(this->pDataContainer, this->ContainerSize);
			return this->iGetSize();
		}

		const typename _SANSTREAM<_data, _ptr, _ref>::_size StreamSize = this->ContainerSize - Size;

		_ptr pStream = new _data[StreamSize];
		for (typename _SANSTREAM<_data, _ptr, _ref>::_size seek = 0; seek < StreamSize; seek = seek + 1)
		{
			pStream[seek] = this->pDataContainer[seek + Size];
		}

		this->_ReleaseStream();

		this->pDataContainer = pStream;
		this->ContainerSize = StreamSize;

		this->_UpdateStream(this->pDataContainer, this->ContainerSize);
		return this->iGetSize();
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM<_data, _ptr, _ref>::_size _SANSTREAM<_data, _ptr, _ref>::iPopBack(const typename _SANSTREAM<_data, _ptr, _ref>::_size &Size)
	{
		if (Size >= this->ContainerSize)
		{
			this->_ReleaseStream();
			this->_UpdateStream(this->pDataContainer, this->ContainerSize);
			return this->iGetSize();
		}

		const typename _SANSTREAM<_data, _ptr, _ref>::_size StreamSize = this->ContainerSize - Size;

		_ptr pStream = new _data[StreamSize];
		for (typename _SANSTREAM<_data, _ptr, _ref>::_size seek = 0; seek < StreamSize; seek = seek + 1)
		{
			pStream[seek] = this->pDataContainer[seek + Size];
		}

		this->_ReleaseStream();

		this->pDataContainer = pStream;
		this->ContainerSize = StreamSize;

		this->_UpdateStream(this->pDataContainer, this->ContainerSize);
		return this->iGetSize();
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM<_data, _ptr, _ref>::_size _SANSTREAM<_data, _ptr, _ref>::iReSizeStream(const typename _SANSTREAM<_data, _ptr, _ref>::_size &Size)
	{
		if (Size == 0)
		{
			if (this->pDataContainer != nullptr)
			{
				this->_ReleaseStream();
				this->_UpdateStream(this->pDataContainer, this->ContainerSize);
				return this->iGetSize();
			}
		}

		_ptr p_stream = new _data[Size];

		if (this->pDataContainer != nullptr)
		{
			const typename _SANSTREAM<_data, _ptr, _ref>::_size DataSize = min(Size, this->ContainerSize);

			for (typename _SANSTREAM<_data, _ptr, _ref>::_size seek = 0; seek < DataSize; seek = seek + 1)
			{
				p_stream[seek] = this->pDataContainer[seek];
			}

			this->_ReleaseStream();
		}

		this->pDataContainer = p_stream;
		this->ContainerSize = Size;

		this->_UpdateStream(this->pDataContainer, this->ContainerSize);

		return this->iGetSize();
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM<_data, _ptr, _ref>::_size _SANSTREAM<_data, _ptr, _ref>::iReSizeStream(const typename _SANSTREAM<_data, _ptr, _ref>::_size &Size, const _data &Data)
	{
		if (Size == 0)
		{
			if (this->pDataContainer != nullptr)
			{
				this->_ReleaseStream();
				this->_UpdateStream(this->pDataContainer, this->ContainerSize);
				return this->iGetSize();
			}
		}

		_ptr p_stream = new _data[Size];

		if (this->pDataContainer != nullptr)
		{
			const typename _SANSTREAM<_data, _ptr, _ref>::_size DataSize = min(Size, this->ContainerSize);

			for (typename _SANSTREAM<_data, _ptr, _ref>::_size seek = 0; seek < DataSize; seek = seek + 1)
			{
				p_stream[seek] = Data;
			}

			this->_ReleaseStream();
		}

		this->pDataContainer = p_stream;
		this->ContainerSize = Size;

		this->_UpdateStream(this->pDataContainer, this->ContainerSize);

		return this->iGetSize();
	}
	template<class _data, class _ptr, class _ref>
	void _SANSTREAM<_data, _ptr, _ref>::iClear()
	{
		this->_ReleaseStream();
		this->_UpdateStream(this->pDataContainer, this->ContainerSize);
	}
	template<class _data, class _ptr, class _ref>
	const typename _SANSTREAM<_data, _ptr, _ref>::_ref_type& _SANSTREAM<_data, _ptr, _ref>::iGetRef() const
	{ 
		return *this; 
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM<_data, _ptr, _ref>::_ref_type& _SANSTREAM<_data, _ptr, _ref>::iGetRef()
	{ 
		return *this; 
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM<_data, _ptr, _ref>::_size _SANSTREAM<_data, _ptr, _ref>::pop_front(const typename _SANSTREAM<_data, _ptr, _ref>::_size &Size)
	{ 
		return this->iPopFront(Size); 
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM<_data, _ptr, _ref>::_size _SANSTREAM<_data, _ptr, _ref>::pop_back(const typename _SANSTREAM<_data, _ptr, _ref>::_size &Size)
	{ 
		return this->iPopBack(Size); 
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM<_data, _ptr, _ref>::_size _SANSTREAM<_data, _ptr, _ref>::resize(const typename _SANSTREAM<_data, _ptr, _ref>::_size &Size)
	{ 
		return this->iReSizeStream(Size); 
	}
	template<class _data, class _ptr, class _ref>
	typename _SANSTREAM<_data, _ptr, _ref>::_size _SANSTREAM<_data, _ptr, _ref>::resize(const typename _SANSTREAM<_data, _ptr, _ref>::_size &Size, const _data &Data)
	{ 
		return this->iReSizeStream(Size, Data); 
	}
	template<class _data, class _ptr, class _ref>
	void _SANSTREAM<_data, _ptr, _ref>::clear()
	{ 
		this->iClear(); 
	}
#pragma endregion

	template<class _data>
	using _sstream = _SANSTREAM<_data, _data*, _data&>;
#endif
}