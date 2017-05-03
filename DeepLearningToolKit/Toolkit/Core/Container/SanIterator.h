//Project: San Lab Game Engine
//Version: 2.1.1
//Debug State: Need test
#include"SanContainerDef.h"
#pragma once
using namespace std;
namespace San
{
#ifndef __STDSANITERATOR_H__
#define __STDSANITERATOR_H__
	template<class _ptr, class _it>
	struct _INDIRECT_ITERATOR
	{
	protected:
		_it Iterator;
	public:
		_INDIRECT_ITERATOR();
		_INDIRECT_ITERATOR(const _it &It);
		_INDIRECT_ITERATOR(const _INDIRECT_ITERATOR<_ptr, _it> &It);
		~_INDIRECT_ITERATOR();

		_INDIRECT_ITERATOR<_ptr, _it>& operator=(const _INDIRECT_ITERATOR<_ptr, _it> &It);
		_INDIRECT_ITERATOR<_ptr, _it>& operator=(const _it &It);

		bool operator==(const _INDIRECT_ITERATOR<_ptr, _it> &It) const;
		bool operator!=(const _INDIRECT_ITERATOR<_ptr, _it> &It) const;

		bool operator==(const _it &It) const;
		bool operator!=(const _it &It) const;

		const _ptr operator->() const;
		_ptr operator->();

		const _ptr operator*() const;
		_ptr operator*();

		const _it& raw_iterator() const;
		_it& raw_iterator();

		_INDIRECT_ITERATOR<_ptr, _it>& operator++();
		_INDIRECT_ITERATOR<_ptr, _it> operator++(const int32 Val);

		_INDIRECT_ITERATOR<_ptr, _it>& operator--();
		_INDIRECT_ITERATOR<_ptr, _it> operator--(const int32 Val);
	};

#pragma region _INDIRECT_ITERATOR CPP
	template<class _ptr, class _it>
	_INDIRECT_ITERATOR<_ptr, _it>::_INDIRECT_ITERATOR()
	{
	}
	template<class _ptr, class _it>
	_INDIRECT_ITERATOR<_ptr, _it>::_INDIRECT_ITERATOR(const _it &It)
		:Iterator(It)
	{
	}
	template<class _ptr, class _it>
	_INDIRECT_ITERATOR<_ptr, _it>::_INDIRECT_ITERATOR(const _INDIRECT_ITERATOR<_ptr, _it> &It)
		: Iterator(It.Iterator)
	{
	}
	template<class _ptr, class _it>
	_INDIRECT_ITERATOR<_ptr, _it>::~_INDIRECT_ITERATOR()
	{
	}
	template<class _ptr, class _it>
	_INDIRECT_ITERATOR<_ptr, _it>& _INDIRECT_ITERATOR<_ptr, _it>::operator=(const _INDIRECT_ITERATOR<_ptr, _it> &It)
	{
		this->Iterator = It.Iterator;
		return *this;
	}
	template<class _ptr, class _it>
	_INDIRECT_ITERATOR<_ptr, _it>& _INDIRECT_ITERATOR<_ptr, _it>::operator=(const _it &It)
	{
		this->Iterator = It;
		return *this;
	}
	template<class _ptr, class _it>
	bool _INDIRECT_ITERATOR<_ptr, _it>::operator==(const _INDIRECT_ITERATOR<_ptr, _it> &It) const
	{
		return this->Iterator == It.Iterator;
	}
	template<class _ptr, class _it>
	bool _INDIRECT_ITERATOR<_ptr, _it>::operator!=(const _INDIRECT_ITERATOR<_ptr, _it> &It) const
	{
		return this->Iterator == It.Iterator;
	}
	template<class _ptr, class _it>
	bool _INDIRECT_ITERATOR<_ptr, _it>::operator==(const _it &It) const
	{
		return this->Iterator == It;
	}
	template<class _ptr, class _it>
	bool _INDIRECT_ITERATOR<_ptr, _it>::operator!=(const _it &It) const
	{
		return this->Iterator == It;
	}
	template<class _ptr, class _it>
	const _ptr _INDIRECT_ITERATOR<_ptr, _it>::operator->() const
	{
		return this->Iterator->operator->();
	}
	template<class _ptr, class _it>
	_ptr _INDIRECT_ITERATOR<_ptr, _it>::operator->()
	{
		return this->Iterator->operator->();
	}
	template<class _ptr, class _it>
	const _ptr _INDIRECT_ITERATOR<_ptr, _it>::operator*() const
	{
		return (*(*this->Iterator));
	}
	template<class _ptr, class _it>
	_ptr _INDIRECT_ITERATOR<_ptr, _it>::operator*()
	{
		return (*(*this->Iterator));
	}
	template<class _ptr, class _it>
	const _it& _INDIRECT_ITERATOR<_ptr, _it>::raw_iterator() const
	{
		return this->Iterator;
	}
	template<class _ptr, class _it>
	_it& _INDIRECT_ITERATOR<_ptr, _it>::raw_iterator()
	{
		return this->Iterator;
	}
	template<class _ptr, class _it>
	_INDIRECT_ITERATOR<_ptr, _it>& _INDIRECT_ITERATOR<_ptr, _it>::operator++()
	{
		this->Iterator++;
		return *this;
	}
	template<class _ptr, class _it>
	_INDIRECT_ITERATOR<_ptr, _it> _INDIRECT_ITERATOR<_ptr, _it>::operator++(const int32 Val)
	{
		_INDIRECT_ITERATOR<_ptr, _it> CurrentIt(*this);
		this->Iterator++;
		return CurrentIt;
	};
	template<class _ptr, class _it>
	_INDIRECT_ITERATOR<_ptr, _it>& _INDIRECT_ITERATOR<_ptr, _it>::operator--()
	{
		this->Iterator--;
		return *this;
	}
	template<class _ptr, class _it>
	_INDIRECT_ITERATOR<_ptr, _it> _INDIRECT_ITERATOR<_ptr, _it>::operator--(const int32 Val)
	{
		_INDIRECT_ITERATOR<_ptr, _it> CurrentIt(*this);
		this->Iterator--;
		return CurrentIt;
	}
#pragma endregion

	template<class _ptr, class _it>
	struct _INDIRECT_CONST_ITERATOR
	{
	protected:
		_it Iterator;
	public:
		_INDIRECT_CONST_ITERATOR();
		_INDIRECT_CONST_ITERATOR(const _it &It);
		_INDIRECT_CONST_ITERATOR(const _INDIRECT_CONST_ITERATOR<_ptr, _it> &It);
		_INDIRECT_CONST_ITERATOR(const _INDIRECT_ITERATOR<_ptr, _it> &It);
		~_INDIRECT_CONST_ITERATOR();

		_INDIRECT_CONST_ITERATOR<_ptr, _it>& operator=(const _INDIRECT_CONST_ITERATOR<_ptr, _it> &It);
		_INDIRECT_CONST_ITERATOR<_ptr, _it>& operator=(const _INDIRECT_ITERATOR<_ptr, _it> &It);

		_INDIRECT_CONST_ITERATOR<_ptr, _it>& operator=(const _it &It);

		bool operator==(const _INDIRECT_CONST_ITERATOR<_ptr, _it> &It) const;
		bool operator!=(const _INDIRECT_CONST_ITERATOR<_ptr, _it> &It) const;

		bool operator==(const _it &It) const;
		bool operator!=(const _it &It) const;

		const _ptr operator->() const;
		const _ptr operator*() const;

		const _it& raw_iterator() const;

		_INDIRECT_CONST_ITERATOR<_ptr, _it>& operator++();
		_INDIRECT_CONST_ITERATOR<_ptr, _it> operator++(const int32 Val);

		_INDIRECT_CONST_ITERATOR<_ptr, _it>& operator--();
		_INDIRECT_CONST_ITERATOR<_ptr, _it> operator--(const int32 Val);
	};

#pragma region _INDIRECT_CONST_ITERATOR CPP
	template<class _ptr, class _it>
	_INDIRECT_CONST_ITERATOR<_ptr, _it>::_INDIRECT_CONST_ITERATOR()
	{
	}
	template<class _ptr, class _it>
	_INDIRECT_CONST_ITERATOR<_ptr, _it>::_INDIRECT_CONST_ITERATOR(const _it &It)
		:Iterator(It)
	{
	}
	template<class _ptr, class _it>
	_INDIRECT_CONST_ITERATOR<_ptr, _it>::_INDIRECT_CONST_ITERATOR(const _INDIRECT_CONST_ITERATOR<_ptr, _it> &It)
		: Iterator(It.Iterator)
	{
	}
	template<class _ptr, class _it>
	_INDIRECT_CONST_ITERATOR<_ptr, _it>::~_INDIRECT_CONST_ITERATOR()
	{
	}
	template<class _ptr, class _it>
	_INDIRECT_CONST_ITERATOR<_ptr, _it>& _INDIRECT_CONST_ITERATOR<_ptr, _it>::operator=(const _INDIRECT_CONST_ITERATOR<_ptr, _it> &It)
	{
		this->Iterator = It.Iterator;
		return *this;
	}
	template<class _ptr, class _it>
	_INDIRECT_CONST_ITERATOR<_ptr, _it>& _INDIRECT_CONST_ITERATOR<_ptr, _it>::operator=(const _it &It)
	{
		this->Iterator = It;
		return *this;
	}
	template<class _ptr, class _it>
	bool _INDIRECT_CONST_ITERATOR<_ptr, _it>::operator==(const _INDIRECT_CONST_ITERATOR<_ptr, _it> &It) const
	{
		return this->Iterator == It.Iterator;
	}
	template<class _ptr, class _it>
	bool _INDIRECT_CONST_ITERATOR<_ptr, _it>::operator!=(const _INDIRECT_CONST_ITERATOR<_ptr, _it> &It) const
	{
		return this->Iterator == It.Iterator;
	}
	template<class _ptr, class _it>
	bool _INDIRECT_CONST_ITERATOR<_ptr, _it>::operator==(const _it &It) const
	{
		return this->Iterator == It;
	}
	template<class _ptr, class _it>
	bool _INDIRECT_CONST_ITERATOR<_ptr, _it>::operator!=(const _it &It) const
	{
		return this->Iterator == It;
	}
	template<class _ptr, class _it>
	const _ptr _INDIRECT_CONST_ITERATOR<_ptr, _it>::operator->() const
	{
		return this->Iterator->operator->();
	}
	template<class _ptr, class _it>
	const _ptr _INDIRECT_CONST_ITERATOR<_ptr, _it>::operator*() const
	{
		return (*(*this->Iterator));
	}
	template<class _ptr, class _it>
	const _it& _INDIRECT_CONST_ITERATOR<_ptr, _it>::raw_iterator() const
	{
		return this->Iterator;
	}
	template<class _ptr, class _it>
	_INDIRECT_CONST_ITERATOR<_ptr, _it>& _INDIRECT_CONST_ITERATOR<_ptr, _it>::operator++()
	{
		this->Iterator++;
		return *this;
	}
	template<class _ptr, class _it>
	_INDIRECT_CONST_ITERATOR<_ptr, _it> _INDIRECT_CONST_ITERATOR<_ptr, _it>::operator++(const int32 Val)
	{
		_INDIRECT_CONST_ITERATOR<_ptr, _it> CurrentIt(*this);
		this->Iterator++;
		return CurrentIt;
	};
	template<class _ptr, class _it>
	_INDIRECT_CONST_ITERATOR<_ptr, _it>& _INDIRECT_CONST_ITERATOR<_ptr, _it>::operator--()
	{
		this->Iterator--;
		return *this;
	}
	template<class _ptr, class _it>
	_INDIRECT_CONST_ITERATOR<_ptr, _it> _INDIRECT_CONST_ITERATOR<_ptr, _it>::operator--(const int32 Val)
	{
		_INDIRECT_CONST_ITERATOR<_ptr, _it> CurrentIt(*this);
		this->Iterator--;
		return CurrentIt;
	}
#pragma endregion
#endif
}