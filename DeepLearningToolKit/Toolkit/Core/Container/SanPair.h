//Project: San Lab Game Engine
//Version: 2.1.1
//Debug State: Need test
#include"SanContainerDef.h"
#pragma once
using namespace std;
namespace San
{
#ifndef __STDSANPAIR_H__
#define __STDSANPAIR_H__

#define _SAN_PAIR_DEF(__NAME__, __KEY_TYPE__, __KEY__, __KEY_INIT__, __VAL_TYPE__, __VAL__, __VAL_INIT__, __COMPARATOR__, __ALLOCATOR__) \
	struct __NAME__ : public _smemobj<__ALLOCATOR__>\
	{\
	public:\
	__KEY_TYPE__ __KEY__; \
	__VAL_TYPE__ __VAL__; \
	public:\
	__NAME__(const __KEY_TYPE__ &__KEY__ = __KEY_INIT__, const __VAL_TYPE__ &__VAL__ = __VAL_INIT__) :__KEY__(__KEY__), __VAL__(__VAL__){}; \
	__NAME__(const __NAME__ &Var) :__KEY__(Var.__KEY__), __VAL__(Var.__VAL__){}; \
	~__NAME__(){}; \
	\
	__NAME__& operator=(const __NAME__ &Var){ this->__KEY__ = Var.__KEY__; this->__VAL__ = Var.__VAL__; return *this; }\
	\
	bool operator==(const __NAME__ &Var) const { return (this->__KEY__ == Var.__KEY__) && (this->__VAL__ == Var.__VAL__); }; \
	bool operator!=(const __NAME__ &Var) const { return (this->__KEY__ != Var.__KEY__) || (this->__VAL__ != Var.__VAL__); }; \
	\
	__COMPARATOR__; \
}\

	template<class _first, class _second, class _Allocator = cSanSystemAllocator>
	struct _spair : public _smemobj<_Allocator>
	{
	public:
		_first first;
		_second second;
	public:
		_spair(){};
		_spair(const _first &First, const _second &Second) :first(First), second(Second){};
		_spair(const _spair<_first, _second, _Allocator> &Pair) :first(Pair.first), second(Pair.second){};
		~_spair(){};

		_spair<_first, _second, _Allocator>& operator=(const _spair<_first, _second, _Allocator> &Pair){ this->first = Pair.first; this->second = Pair.second; return *this; }

		bool operator==(const _spair<_first, _second, _Allocator> &Pair) const { return (this->first == Pair.first) && (this->second == Pair.second); };
		bool operator!=(const _spair<_first, _second, _Allocator> &Pair) const { return (this->first != Pair.first) || (this->second != Pair.second); };
	};
#endif
}