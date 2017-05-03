#include"iostream"
#include"cfloat"
#include"../RenderSystem.h"
#include"SanString.h"
//#include"SanMemory.h"
#pragma once
using namespace std;
#include"WinDataDef.h"
namespace San
{
#ifndef __STDSANTYPES_H__
#define __STDSANTYPES_H__

#ifdef _UNIX
	typedef char				int8;
	typedef unsigned char		uint8;
	typedef short				int16;
	typedef unsigned short		uint16;
	typedef int					int32;
	typedef unsigned int		uint32;
	typedef long long			int64;
	typedef unsigned long long	uint64;
#else
	typedef __int8				int8;
	typedef unsigned __int8		uint8;
	typedef __int16				int16;
	typedef unsigned __int16	uint16;
	typedef __int32				int32;
	typedef unsigned __int32	uint32;
	typedef __int64				int64;
	typedef unsigned __int64	uint64;
#endif

#ifdef _X32
	typedef int32	sint;
	typedef uint32	suint;
#else
	typedef int64	sint;
	typedef uint64	suint;
#endif

	typedef float		float32;
	typedef double		float64;
#ifdef _FLOAT_TYPE_DOUBLE
	typedef float64	sfloat;
#else
	typedef float32	sfloat;
#endif

	static const int8	MIN_INT8_VALUE				= 0x80;
	static const int8	MAX_INT8_VALUE				= 0x7f;
	static const uint8	MIN_UNSIGNED_INT8_VALUE		= 0x00;
	static const uint8	MAX_UNSIGNED_INT8_VALUE		= 0xff;
	static const int16	MIN_INT16_VALUE				= 0x8000;
	static const int16	MAX_INT16_VALUE				= 0x7fff;
	static const uint16	MIN_UNSIGNED_INT16_VALUE	= 0x0000;
	static const uint16	MAX_UNSIGNED_INT16_VALUE	= 0xffff;
	static const int32	MIN_INT32_VALUE				= 0x80000000;
	static const int32	MAX_INT32_VALUE				= 0x7fffffff;
	static const uint32	MIN_UNSIGNED_INT32_VALUE	= 0x00000000;
	static const uint32	MAX_UNSIGNED_INT32_VALUE	= 0xffffffff;
	static const int64	MIN_INT64_VALUE				= 0x8000000000000000LL;
	static const int64	MAX_INT64_VALUE				= 0x7fffffffffffffffLL;
	static const uint64	MIN_UNSIGNED_INT64_VALUE	= 0x0000000000000000ULL;
	static const uint64	MAX_UNSIGNED_INT64_VALUE	= 0xffffffffffffffffULL;

	static const float32	MIN_FLOAT32_VALUE	= FLT_MIN;
	static const float32	MAX_FLOAT32_VALUE	= FLT_MAX;
	static const float64	MIN_FLOAT64_VALUE	= DBL_MIN;
	static const float64	MAX_FLOAT64_VALUE	= DBL_MAX;

	typedef uint32			SHANDLE32;
	typedef uint64			SHANDLE64;
	typedef uint32			SVALUE32;
	typedef uint64			SVALUE64;
	typedef SVALUE32		SRESULT32;
	typedef SVALUE64		SRESULT64;

	typedef void*			SHANDLE;
#ifdef _X32
	typedef SVALUE32		SVALUE;
#else
	typedef SVALUE64		SVALUE;
#endif
	typedef SVALUE			SRESULT;

#ifndef _CPP_VERSION_CPP0x
#define nullptr NULL
#endif

	enum eSANVARIABLETYPE
	{
		SVT_INT8		= 0x0000,
		SVT_UINT8		= 0x0001,
		SVT_INT16		= 0x0002,
		SVT_UINT16		= 0x0003,
		SVT_INT32		= 0x0004,
		SVT_UINT32		= 0x0005,
		SVT_INT64		= 0x0006,
		SVT_UINT64		= 0x0007,
		SVT_CHAR		= 0x0008,
		SVT_WCHAR		= 0x0009,
		SVT_FLOAT		= 0x000a,
		SVT_DOUBLE		= 0x000b,
		SVT_IPv4		= 0x000c,//Input save as int32
		SVT_IPv6		= 0x000d,//Input save as int32[4]
		SVT_MAC			= 0x000e,//Input save as int64
		SVT_TIME		= 0x000f,//Input save as int32
		SVT_STRING		= 0x0010,
		SVT_BOOL		= 0x0011,
		SVT_TCOUNT		= 0x0012,
		SVT_TYPEMASK	= 0x00ff,
		SVT_DYNAMIC		= 0x0100,
		SVT_CONST		= 0x0200,
	};

//#pragma pack(8)
	struct stSANVARIABLE
	{
	protected:
		SString	strName;
		eSANVARIABLETYPE	Type;
		SHANDLE	pBuffer;
		uint32	BufferSize;
		uint32	ArraySize;
		uint32 count;
	public:
		stSANVARIABLE();
		stSANVARIABLE(const stSANVARIABLE&);
		~stSANVARIABLE();
		stSANVARIABLE& operator=(const stSANVARIABLE&);
	};

	struct SANVARIABLE :public stSANVARIABLE
	{
		SANVARIABLE();
		SANVARIABLE(eSANVARIABLETYPE type,SVALUE value);

		template<typename ST>
		SANVARIABLE& operator = (ST value);

		template<typename ST>
		bool operator == (ST value);
		template<typename ST>
		bool operator != (ST value);
	};
	typedef SANVARIABLE*	lpSANVARIABLE;
//#pragma pack()

	typedef SString	SANSYSTEMID;

	struct stSANRESPTR;

#ifndef _UNICODE
#define SAN_CODE_LOCATION __FILE__, __LINE__
#else
#define SAN_CODE_LOCATION __FILEW__, __LINE__
#endif

#ifdef _DEBUG
#define _SDEBUG(__CODE__) __CODE__
#define _CHECK_POINT(__BOOL__, __STR__, __CODE_FILE__, __CODE_LINE__) \
	if (__BOOL__) \
	{ \
		::cout << "Check point: " << #__BOOL__ << " - " << __STR__ << "\r\n"; \
		::cout << "\tFile: " << string(__CODE_FILE__) << " : " << int(__CODE_LINE__) << "\r\n"; \
		::system("pause"); \
	}
#else
#define _SDEBUG(__CODE__) 
#define _CHECK_POINT(__BOOL__, __STR__, __CODE_FILE__, __CODE_LINE__) 
#endif

#endif
}