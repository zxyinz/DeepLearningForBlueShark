#include"iostream"
#include"vector"
#include"algorithm"
#include"chrono"
#include"iomanip"
#include"map"
#include"memory"
#include"random"
#include"sstream"
#include"fstream"
#include"string"
#include"vector"
#include"cstdio"
#include"cstdlib"
#include"cmath"
#include"ctime"
#include"cfloat"
#include"cstdint"
#include"cstddef"
#pragma once
using namespace std;
#ifndef __CDIMENSION_H__
#define __CDIMENSION_H__
#define _SAN_PAIR_DEF(__NAME__, __KEY_TYPE__, __KEY__, __KEY_INIT__, __VAL_TYPE__, __VAL__, __VAL_INIT__, __COMPARATOR__, __ALLOCATOR__) \
struct __NAME__\
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

class cDimension
{
public:
	size_t width, height, channels, batches;
	mutable size_t resolution;
	mutable size_t volume;
	mutable size_t size;
public:
	cDimension(const size_t width = 1, const size_t height = 1, const size_t channels = 1, const size_t batches = 1)
		:width(width), height(height), channels(channels), batches(batches),
		resolution(width * height),
		volume(resolution * channels),
		size(volume * batches)
	{
	};
	cDimension(const cDimension &Dimension)
		:width(Dimension.width), height(Dimension.height), channels(Dimension.channels), batches(Dimension.batches),
		resolution(width * height),
		volume(resolution * channels),
		size(volume * batches)
	{
	};
	~cDimension(){};

	cDimension& operator=(const cDimension &Dimension)
	{
		this->iResize(Dimension.width, Dimension.height, Dimension.channels, Dimension.batches);
		return *this;
	};

	void iUpdate() const
	{
		this->resolution = this->width * this->height;
		this->volume = this->resolution * this->channels;
		this->size = this->volume * this->batches;
	};

	void iResize(const size_t width = 1, const size_t height = 1, const size_t channels = 1, const size_t batches = 1)
	{
		this->width = width;
		this->height = height;
		this->channels = channels;
		this->batches = batches;

		iUpdate();
	};

	//Batches not involved in computing
	cDimension operator+(const cDimension &Dimension) const
	{
		return cDimension(this->width + Dimension.width, this->height + Dimension.height, this->channels + Dimension.channels, this->batches);
	};
	cDimension operator+(const size_t &Value) const
	{
		return cDimension(this->width + Value, this->height + Value, this->channels + Value, this->batches);
	};

	cDimension operator-(const cDimension &Dimension) const
	{
		return cDimension(this->width - Dimension.width, this->height - Dimension.height, this->channels - Dimension.channels, this->batches);
	};
	cDimension operator-(const size_t &Value) const
	{
		return cDimension(this->width - Value, this->height - Value, this->channels - Value, this->batches);
	};

	cDimension operator*(const cDimension &Dimension) const
	{
		return cDimension(this->width * Dimension.width, this->height * Dimension.height, this->channels * Dimension.channels, this->batches);
	};
	cDimension operator*(const size_t &Value) const
	{
		return cDimension(this->width * Value, this->height * Value, this->channels * Value, this->batches);
	};

	cDimension operator/(const cDimension &Dimension) const
	{
		return cDimension(this->width / Dimension.width, this->height / Dimension.height, this->channels / Dimension.channels,this->batches);
	};
	cDimension operator/(const size_t &Value) const
	{
		return cDimension(this->width / Value, this->height / Value, this->channels / Value, this->batches);
	};
};
#endif