#include"SanContainerDef.h"
#pragma once
using namespace std;
namespace San
{
#ifndef __SANSTREAMITERATOR_H__
#define __SANSTREAMITERATOR_H__
	template<class _type, class _ptr, class _ref>
	struct _SANSTREAMEX_CONST_ITERATOR
	{
	private:
		const _ptr pBegin;
		const _ptr pEnd;
		_ptr pCurrent;
	public:
		_SANSTREAMEX_CONST_ITERATOR(const _ptr pBegin = nullptr, const _ptr pEnd = nullptr, const _ptr pCurrent = nullptr);
		_SANSTREAMEX_CONST_ITERATOR(const _SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref> &It);
		~_SANSTREAMEX_CONST_ITERATOR();

		_SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref>& operator=(const _SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref> &It);

		bool operator==(const _SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref> &It) const;
		bool operator!=(const _SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref> &It) const;

		const _ref operator*() const;
		const _ptr operator->() const;

		_SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref>& operator++();
		_SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref> operator++(int Val);
		_SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref>& operator--();
		_SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref> operator--(int Val);
	};

#pragma region _SANSTREAMEX_CONST_ITERATOR CPP file
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref>::_SANSTREAMEX_CONST_ITERATOR(const _ptr pBegin, const _ptr pEnd, const _ptr pCurrent)
		:pBegin(pBegin),
		pEnd(pEnd),
		pCurrent(pCurrent)
	{
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref>::_SANSTREAMEX_CONST_ITERATOR(const _SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref> &It)
		:pBegin(It.pBegin),
		pEnd(It.pEnd),
		pCurrent(It.pCurrent)
	{
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref>::~_SANSTREAMEX_CONST_ITERATOR()
	{
		/*this->pBegin = nullptr;
		this->pEnd = nullptr;*/
		this->pCurrent = nullptr;
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref>& _SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref>::operator=(const _SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref> &It)
	{
		this->pBegin = It.pBegin;
		this->pEnd = It.pEnd;
		this->pCurrent = It.pCurrent;

		return *this;
	}
	template<class _type, class _ptr, class _ref>
	bool _SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref>::operator==(const _SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref> &It) const
	{
		if (this->pCurrent != It.pCurrent){ return false; }

		if (this->pBegin != It.pBegin){ return false; }
		if (this->pEnd != It.pEnd){ return false; }

		return true;
	}
	template<class _type, class _ptr, class _ref>
	bool _SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref>::operator!=(const _SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref> &It) const
	{
		if (this->pCurrent != It.pCurrent){ return true; }

		if (this->pBegin != It.pBegin){ return true; }
		if (this->pEnd != It.pEnd){ return true; }

		return false;
	}
	template<class _type, class _ptr, class _ref>
	const _ref _SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref>::operator*() const
	{
		return *this->pCurrent;
	}
	template<class _type, class _ptr, class _ref>
	const _ptr _SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref>::operator->() const
	{
		return this->pCurrent;
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref>& _SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref>::operator++()
	{
		if ((this->pCurrent == nullptr) || (this->pBegin == nullptr) || (this->pEnd == nullptr))
		{
			return *this;
		}

		this->pCurrent = this->pCurrent + 1;
		this->pCurrent = this->pCurrent >= this->pEnd ? nullptr : this->pCurrent;

		return *this;
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref> _SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref>::operator++(int Val)
	{
		_SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref> OrigionalIterator(*this);

		this->operator++();

		return OrigionalIterator;
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref>& _SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref>::operator--()
	{
		if ((this->pCurrent == nullptr) || (this->pBegin == nullptr) || (this->pEnd == nullptr))
		{
			return *this;
		}

		if (this->pCurrent < sizeof(_type))
		{
			this->pCurrent = nullptr;
			return *this;
		}

		this->pCurrent = this->pCurrent - 1;
		this->pCurrent = this->pCurrent < this->pBegin ? nullptr : this->pCurrent;

		return *this;
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref> _SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref>::operator--(int Val)
	{
		_SANSTREAMEX_CONST_ITERATOR<_type, _ptr, _ref> OrigionalIterator(*this);

		this->operator--();

		return OrigionalIterator;
	}
#pragma endregion

	template<class _type, class _ptr, class _ref>
	struct _SANSTREAMEX_ITERATOR
	{
	private:
		const _ptr pBegin;
		const _ptr pEnd;
		_ptr pCurrent;
	public:
		_SANSTREAMEX_ITERATOR(const _ptr pBegin = nullptr, const _ptr pEnd = nullptr, const _ptr pCurrent = nullptr);
		_SANSTREAMEX_ITERATOR(const _SANSTREAMEX_ITERATOR<_type, _ptr, _ref> &It);
		~_SANSTREAMEX_ITERATOR();

		_SANSTREAMEX_ITERATOR<_type, _ptr, _ref>& operator=(const _SANSTREAMEX_ITERATOR<_type, _ptr, _ref> &It);

		bool operator==(const _SANSTREAMEX_ITERATOR<_type, _ptr, _ref> &It) const;
		bool operator!=(const _SANSTREAMEX_ITERATOR<_type, _ptr, _ref> &It) const;

		_ref operator*();
		_ptr operator->();

		_SANSTREAMEX_ITERATOR<_type, _ptr, _ref>& operator++();
		_SANSTREAMEX_ITERATOR<_type, _ptr, _ref> operator++(int Val);
		_SANSTREAMEX_ITERATOR<_type, _ptr, _ref>& operator--();
		_SANSTREAMEX_ITERATOR<_type, _ptr, _ref> operator--(int Val);
	};

#pragma region _SANSTREAMEX_ITERATOR CPP file
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR<_type, _ptr, _ref>::_SANSTREAMEX_ITERATOR(const _ptr pBegin, const _ptr pEnd, const _ptr pCurrent)
		:pBegin(pBegin),
		pEnd(pEnd),
		pCurrent(pCurrent)
	{
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR<_type, _ptr, _ref>::_SANSTREAMEX_ITERATOR(const _SANSTREAMEX_ITERATOR<_type, _ptr, _ref> &It)
		:pBegin(It.pBegin),
		pEnd(It.pEnd),
		pCurrent(It.pCurrent)
	{
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR<_type, _ptr, _ref>::~_SANSTREAMEX_ITERATOR()
	{
		/*this->pBegin = nullptr;
		this->pEnd = nullptr;*/
		this->pCurrent = nullptr;
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR<_type, _ptr, _ref>& _SANSTREAMEX_ITERATOR<_type, _ptr, _ref>::operator=(const _SANSTREAMEX_ITERATOR<_type, _ptr, _ref> &It)
	{
		this->pBegin = It.pBegin;
		this->pEnd = It.pEnd;
		this->pCurrent = It.pCurrent;

		return *this;
	}
	template<class _type, class _ptr, class _ref>
	bool _SANSTREAMEX_ITERATOR<_type, _ptr, _ref>::operator==(const _SANSTREAMEX_ITERATOR<_type, _ptr, _ref> &It) const
	{
		if (this->pCurrent != It.pCurrent){ return false; }

		if (this->pBegin != It.pBegin){ return false; }
		if (this->pEnd != It.pEnd){ return false; }

		return true;
	}
	template<class _type, class _ptr, class _ref>
	bool _SANSTREAMEX_ITERATOR<_type, _ptr, _ref>::operator!=(const _SANSTREAMEX_ITERATOR<_type, _ptr, _ref> &It) const
	{
		if (this->pCurrent != It.pCurrent){ return true; }

		if (this->pBegin != It.pBegin){ return true; }
		if (this->pEnd != It.pEnd){ return true; }

		return false;
	}
	template<class _type, class _ptr, class _ref>
	_ref _SANSTREAMEX_ITERATOR<_type, _ptr, _ref>::operator*()
	{
		return *this->pCurrent;
	}
	template<class _type, class _ptr, class _ref>
	_ptr _SANSTREAMEX_ITERATOR<_type, _ptr, _ref>::operator->()
	{
		return this->pCurrent;
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR<_type, _ptr, _ref>& _SANSTREAMEX_ITERATOR<_type, _ptr, _ref>::operator++()
	{
		if ((this->pCurrent == nullptr) || (this->pBegin == nullptr) || (this->pEnd == nullptr))
		{
			return *this;
		}

		this->pCurrent = this->pCurrent + 1;
		this->pCurrent = this->pCurrent >= this->pEnd ? nullptr : this->pCurrent;

		return *this;
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR<_type, _ptr, _ref> _SANSTREAMEX_ITERATOR<_type, _ptr, _ref>::operator++(int Val)
	{
		_SANSTREAMEX_ITERATOR<_type, _ptr, _ref> OrigionalIterator(*this);

		this->operator++();

		return OrigionalIterator;
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR<_type, _ptr, _ref>& _SANSTREAMEX_ITERATOR<_type, _ptr, _ref>::operator--()
	{
		if ((this->pCurrent == nullptr) || (this->pBegin == nullptr) || (this->pEnd == nullptr))
		{
			return *this;
		}

		if (this->pCurrent < sizeof(_type))
		{
			this->pCurrent = nullptr;
			return *this;
		}

		this->pCurrent = this->pCurrent - 1;
		this->pCurrent = this->pCurrent < this->pBegin ? nullptr : this->pCurrent;

		return *this;
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR<_type, _ptr, _ref> _SANSTREAMEX_ITERATOR<_type, _ptr, _ref>::operator--(int Val)
	{
		_SANSTREAMEX_ITERATOR<_type, _ptr, _ref> OrigionalIterator(*this);

		this->operator--();

		return OrigionalIterator;
	}
#pragma endregion

	template<class _type, class _ptr, class _ref>
	struct _SANSTREAMEX_ITERATOR_S
	{
	private:
		const _ptr* pBeginPtr;
		const uint32* pSize;
		_ptr pCurrent;
	public:
		_SANSTREAMEX_ITERATOR_S(const _ptr* pBeginPtr = nullptr, const uint32* pSize = nullptr, const _ptr pCurrent = nullptr);
		_SANSTREAMEX_ITERATOR_S(const _SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref> &It);
		~_SANSTREAMEX_ITERATOR_S();

		_SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref>& operator=(const _SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref> &It);

		bool operator==(const _SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref> &It) const;
		bool operator!=(const _SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref> &It) const;

		_ref operator*();
		_ptr operator->();

		_SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref>& operator++();
		_SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref> operator++(int Val);
		_SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref>& operator--();
		_SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref> operator--(int Val);
	};

#pragma region _SANSTREAMEX_ITERATOR CPP file
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref>::_SANSTREAMEX_ITERATOR_S(const _ptr* pBeginPtr, const uint32* pSize, const _ptr pCurrent)
		:pBeginPtr(pBeginPtr),
		pCurrent(pCurrent)
	{
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref>::_SANSTREAMEX_ITERATOR_S(const _SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref> &It)
		:pBeginPtr(It.pBeginPtr),
		pSize(It.pSize),
		pCurrent(It.pCurrent)
	{
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref>::~_SANSTREAMEX_ITERATOR_S()
	{
		this->pBeginPtr = nullptr;
		this->pSize = nullptr;
		this->pCurrent = nullptr;
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref>& _SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref>::operator=(const _SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref> &It)
	{
		this->pBeginPtr = It.pBeginPtr;
		this->pSize = It.pSize;
		this->pCurrent = It.pCurrent;

		return *this;
	}
	template<class _type, class _ptr, class _ref>
	bool _SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref>::operator==(const _SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref> &It) const
	{
		if (this->pCurrent != It.pCurrent){ return false; }

		if (this->pBeginPtr != It.pBeginPtr){ return false; }
		if (this->pSize != It.pSize){ return false; }

		return true;
	}
	template<class _type, class _ptr, class _ref>
	bool _SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref>::operator!=(const _SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref> &It) const
	{
		if (this->pCurrent != It.pCurrent){ return true; }

		if (this->pBeginPtr != It.pBeginPtr){ return true; }
		if (this->pSize != It.pSize){ return true; }

		return false;
	}
	template<class _type, class _ptr, class _ref>
	_ref _SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref>::operator*()
	{
		return *this->pCurrent;
	}
	template<class _type, class _ptr, class _ref>
	_ptr _SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref>::operator->()
	{
		return this->pCurrent;
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref>& _SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref>::operator++()
	{
		if ((this->pCurrent == nullptr) || (this->pBeginPtr == nullptr) || (this->pSize == nullptr))
		{
			return *this;
		}

		_ptr pEnd = (*this->pBeginPtr) + (*this->pSize);

		this->pCurrent = this->pCurrent + 1;
		this->pCurrent = this->pCurrent >= pEnd ? nullptr : this->pCurrent;

		return *this;
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref> _SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref>::operator++(int Val)
	{
		_SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref> OrigionalIterator(*this);

		this->operator++();

		return OrigionalIterator;
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref>& _SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref>::operator--()
	{
		if ((this->pCurrent == nullptr) || (this->pBeginPtr == nullptr) || (this->pSize == nullptr))
		{
			return *this;
		}

		if (this->pCurrent < sizeof(_type))
		{
			this->pCurrent = nullptr;
			return *this;
		}

		this->pCurrent = this->pCurrent - 1;
		this->pCurrent = this->pCurrent < this->pBeginPtr ? nullptr : this->pCurrent;

		return *this;
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref> _SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref>::operator--(int Val)
	{
		_SANSTREAMEX_ITERATOR_S<_type, _ptr, _ref> OrigionalIterator(*this);

		this->operator--();

		return OrigionalIterator;
	}
#pragma endregion

	template<class _type, class _ptr, class _ref>
	struct _SANSTREAMEX_ITERATOR_B
	{
	private:
		const uint8* pByteBegin;
		const uint8* pByteEnd;
		uint8* pCurrentByte;
	public:
		_SANSTREAMEX_ITERATOR_B(const _ptr pBegin = nullptr, const uint32 Size = 0, const uint8* pCurrentByte = nullptr);
		_SANSTREAMEX_ITERATOR_B(const _SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref> &It);
		~_SANSTREAMEX_ITERATOR_B();

		_SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref>& operator=(const _SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref> &It);

		bool operator==(const _SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref> &It) const;
		bool operator!=(const _SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref> &It) const;

		uint8& operator*();
		uint8* operator->();

		_SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref>& operator++();
		_SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref> operator++(int Val);
		_SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref>& operator--();
		_SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref> operator--(int Val);
	};

#pragma region _SANSTREAMEX_ITERATOR_B CPP file
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref>::_SANSTREAMEX_ITERATOR_B(const _ptr pBegin, const uint32 Size, const uint8* pCurrentByte)
		:pByteBegin((uint8*) pBegin),
		pByteEnd(((uint8*) pBegin) + (Size*sizeof(_type))),
		pCurrentByte(pCurrentByte)
	{
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref>::_SANSTREAMEX_ITERATOR_B(const _SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref> &It)
		:pByteBegin(It.pByteBegin),
		pByteEnd(It.pByteEnd),
		pCurrentByte(It.pCurrentByte)
	{
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref>::~_SANSTREAMEX_ITERATOR_B()
	{
		this->pByteBegin = nullptr;
		this->pByteEnd = nullptr;
		this->pCurrentByte = nullptr;
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref>& _SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref>::operator=(const _SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref> &It)
	{
		this->pByteBegin = It.pByteBegin;
		this->pByteEnd = It.pByteEnd;
		this->pCurrentByte = It.pCurrentByte;

		return *this;
	}
	template<class _type, class _ptr, class _ref>
	bool _SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref>::operator==(const _SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref> &It) const
	{
		if (this->pCurrentByte != It.pCurrentByte){ return false; }

		if (this->pByteBegin != It.pByteBegin){ return false; }
		if (this->pByteEnd != It.pByteEnd){ return false; }

		return true;
	}
	template<class _type, class _ptr, class _ref>
	bool _SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref>::operator!=(const _SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref> &It) const
	{
		if (this->pCurrentByte != It.pCurrentByte){ return true; }

		if (this->pByteBegin != It.pByteBegin){ return true; }
		if (this->pByteEnd != It.pByteEnd){ return true; }

		return false;
	}
	template<class _type, class _ptr, class _ref>
	uint8& _SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref>::operator*()
	{
		return *this->pCurrentByte;
	}
	template<class _type, class _ptr, class _ref>
	uint8* _SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref>::operator->()
	{
		return this->pCurrentByte;
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref>& _SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref>::operator++()
	{
		if ((this->pCurrentByte == nullptr) || (this->pByteBegin == nullptr) || (this->pByteEnd == nullptr))
		{
			return *this;
		}

		this->pCurrentByte = this->pCurrentByte + 1;
		this->pCurrentByte = this->pCurrentByte >= this->pByteEnd ? nullptr : this->pCurrentByte;

		return *this;
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref> _SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref>::operator++(int Val)
	{
		_SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref> OrigionalIterator(*this);

		this->operator++();

		return OrigionalIterator;
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref>& _SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref>::operator--()
	{
		if ((this->pCurrentByte == nullptr) || (this->pByteBegin == nullptr) || (this->pByteEnd == nullptr))
		{
			return *this;
		}

		this->pCurrentByte = this->pCurrentByte - 1;
		this->pCurrentByte = this->pCurrentByte < this->pByteBegin ? nullptr : this->pCurrentByte;

		return *this;
	}
	template<class _type, class _ptr, class _ref>
	_SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref> _SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref>::operator--(int Val)
	{
		_SANSTREAMEX_ITERATOR_B<_type, _ptr, _ref> OrigionalIterator(*this);

		this->operator--();

		return OrigionalIterator;
	}
#pragma endregion

#endif
}