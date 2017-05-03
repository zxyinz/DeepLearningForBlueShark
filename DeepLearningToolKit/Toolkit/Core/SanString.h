//Project: San Lab Game Engine
//Version: 2.1.1
//Debug State: Add functions and need test [utf8, optimize]
#include"list"
#include"vector"
#include"string"
#include"sstream"
#include"../RenderSystem.h"
#include"SanMemory.h"
#pragma once
using namespace std;
namespace San
{
#ifndef __STDSANSTRING_H__
#define __STDSANSTRING_H__

	typedef char	sachar;
	typedef wchar_t	swchar;
	typedef string	SStringA;
	typedef wstring	SStringW;

	//typedef basic_string<sachar, char_traits<sachar>, cSanSTLAllocator<sachar>>	SStringA;
	//typedef basic_string<swchar, char_traits<swchar>, cSanSTLAllocator<swchar>>	SStringW;

	//template<template<class ST> class _Allocator> using _SStringA = basic_string<sachar, char_traits<sachar>, _Allocator<sachar>>;
	//template<template<class ST> class _Allocator> using _SStringW = basic_string<swchar, char_traits<swchar>, _Allocator<swchar>>;

#ifndef _UNICODE
	typedef sachar		schar;
	typedef SStringA	SString;
#else
	typedef swchar		schar;
	typedef SStringW	SString;
#endif

#ifdef _UNICODE
#define _SSTR(__String__) L##__String__
#else
#define _SSTR(__String__) __String__
#endif

	swchar		gloAToW(const sachar Data);
	SStringW	gloAToW(const SStringA &strString);
	SStringW	gloAToW(const sachar* pString, int StringLength = -1);
	//swchar	gloUToW(const swchar Data, const size_t UnicodeID = CP_UTF8);
	//SStringW	gloAToW(const SStringW &strString, const size_t UnicodeID = CP_UTF8);
	//SStringW	gloUToW(const swchar* pString, int StringLength = -1, const size_t UnicodeID = CP_UTF8);
	sachar		gloWToA(const swchar Data, const sachar DefaultChar = '*');
	SStringA	gloWToA(const SStringW &strString, const sachar DefaultChar = '*');
	SStringA	gloWToA(const swchar* pString, int StringLength = -1, const sachar DefaultChar = '*');
	//sachar	gloWToU(const swchar Data, const sachar DefaultChar = '*', const size_t UnicodeID = CP_UTF8);
	//SStringA	gloWToU(const SStringW &strString, const sachar DefaultChar = '*', const size_t UnicodeID = CP_UTF8);
	//SStringA	gloWToU(const swchar* pString, int StringLength = -1, const sachar DefaultChar = '*', const size_t UnicodeID = CP_UTF8);
	schar		gloAToT(const sachar Data);
	SString		gloAToT(const SStringA &strString);
	SString		gloAToT(const sachar* pString, int StringLength = -1);
	schar		gloWToT(const swchar Data, const sachar DefaultChar = '*');
	SString		gloWToT(const SStringW &strString, const sachar DefaultChar = '*');
	SString		gloWToT(const swchar* pString, int StringLength = -1, const sachar DefaultChar = '*');
	//schar		gloUToT(const swchar Data, const sachar DefaultChar = '*', const size_t UnicodeID = CP_UTF8);
	//SString	gloUToT(const SStringW &strString, const sachar DefaultChar = '*', const size_t UnicodeID = CP_UTF8);
	//SString	gloUToT(const swchar* pString, int StringLength = -1, const sachar DefaultChar = '*', const size_t UnicodeID = CP_UTF8);
	sachar		gloTToA(const schar Data, const sachar DefaultChar = '*');
	SStringA	gloTToA(const SString &strString, const sachar DefaultChar = '*');
	SStringA	gloTToA(const schar* pString, int StringLength = -1, const sachar DefaultChar = '*');
	swchar		gloTToW(schar Data);
	SStringW	gloTToW(const SString &strString);
	SStringW	gloTToW(const schar* pString, int StringLength = -1);
	//sachar	gloTToU(const schar Data, const sachar DefaultChar = '*', const size_t UnicodeID = CP_UTF8);
	//SStringA	gloTToU(const SString &strString, const sachar DefaultChar = '*', const size_t UnicodeID = CP_UTF8);
	//SStringA	gloTToU(const schar* pString, int StringLength = -1, const sachar DefaultChar = '*', const size_t UnicodeID = CP_UTF8);

	SString		gloIToS(const long long &Data, const unsigned int Radix = 10);
	long long	gloSToI(const SString &strString, const unsigned int Radix = 10);
	SString		gloFToS(const double &Data, SString strFormat = _SSTR(""));
	double		gloSToF(const SString &strString);

	bool	operator==(const SStringA &strStringA, const SStringA &strStringB);
	bool	operator!=(const SStringA &strStringA, const SStringA &strStringB);
	bool	operator==(const SStringW &strStringA, const SStringW &strStringB);
	bool	operator!=(const SStringW &strStringA, const SStringW &strStringB);

	SStringA operator+(const SStringA &strStringA, const SStringA &strStringB);
	SStringW operator+(const SStringW &strStringA, const SStringW &strStringB);

	istream& operator>>(istream &InputStream, SStringA &strString);
	istream& operator>>(istream &InputStream, SStringW &strString);

	ostream& operator<<(ostream &OutputStream, const SStringA &strString);
	ostream& operator<<(ostream &OutputStream, const SStringW &strString);

	SStringA& operator>>(SStringA &strInputString, long long &Data);
	SStringW& operator>>(SStringW &strInputString, long long &Data);
	SStringA& operator>>(SStringA &strInputString, unsigned long long &Data);
	SStringW& operator>>(SStringW &strInputString, unsigned long long &Data);
	SStringA& operator>>(SStringA &strInputString, double &Data);
	SStringW& operator>>(SStringW &strInputString, double &Data);

	SStringA& operator<<(SStringA &strOutputString, const long long &Data);
	SStringW& operator<<(SStringW &strOutputString, const long long &Data);
	SStringA& operator<<(SStringA &strOutputString, const unsigned long long &Data);
	SStringW& operator<<(SStringW &strOutputString, const unsigned long long &Data);
	SStringA& operator<<(SStringA &strOutputString, const double &Data);
	SStringW& operator<<(SStringW &strOutputString, const double &Data);

	SStringA& operator<<(SStringA &strOutputString, const SStringA &strString);
	SStringW& operator<<(SStringW &strOutputString, const SStringW &strString);

	//SStringA gloStringFormat(SStringA Format,...);
	//SStringW gloStringFormat(SStringW Format,...);
#ifndef _UNICODE
	//#define gloStringFormat std::sscanf
#else
	//#define gloStringFormat std::wscanf
#endif
	SString	gloStringFormat(SString Format, ...);

	vector<SStringA> gloGetStringItemsA(const SStringA &strString, SStringA strStringMark = "\0");
	vector<SStringW> gloGetStringItemsW(const SStringW &strString, SStringW strStringMark = L"\0");

#ifndef _UNICODE
#define gloGetStringItems gloGetStringItemsA
#else
#define gloGetStringItems gloGetStringItemsW
#endif

	inline void gloPrintWideString(const SStringW &strWString){ std::wprintf(L"%s", strWString.c_str());/*std::wcout<<strWString.c_str();*/ };

	template<class ST>
	inline void gloPrintArray(const ST* pArray, size_t Size, size_t RowSize = 0, schar Block = _SSTR('\t'), schar RowBlock = _SSTR('\n'), SString strAdditionString = _SSTR("\n"))
	{
		if (RowSize == 0){ RowSize = Size; }
		for (size_t seek = 0; seek<Size; seek = seek + 1)
		{
#ifndef _UNICODE
			::cout << pArray[seek] << Block;
#else
			::wcout << pArray[seek] << Block;
#endif
			if (((seek + 1) % RowSize) == 0)
			{
#ifndef _UNICODE
				::cout << RowBlock;
#else
				::wcout << RowBlock;
#endif
			}
		}
#ifndef _UNICODE
		::cout << strAdditionString.c_str();
#else
		::wcout << strAdditionString.c_str();
#endif
	};
#endif
}
