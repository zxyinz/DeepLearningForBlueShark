//Project: San Lab Game Engine
//Version: 2.1.1
//Debug State: Add local memory manager [stack, pool]
#include"../RenderSystem.h"
#include"Memory/cSanMemoryManagerDef.h"
using namespace std;
#pragma once
namespace San
{
#ifndef __STDSANMEMORY_H__
#define __STDSANMEMORY_H__

	class cSanMemoryFuncSet
	{
	public:
#ifdef _WINDOWS
		typedef __int16 _m16;
		typedef __int32 _m32;
		typedef __int64 _m64;
#endif
#ifdef _UNIX
		typedef unsigned short _m16;
		typedef unsigned long _m32;
		typedef unsigned long long _m64;
#endif
	public:
		static _m16 iBtoL16(_m16 data){ return (data << 8) | (data >> 8); };
		static _m16 iLtoB16(_m16 data){ return (data << 8) | (data >> 8); };
		static _m32 iBtoL32(_m32 data){ return (data << 24) | ((data & 0x00ff0000) >> 8) | ((data & 0x0000ff00) << 8) | ((data & 0x000000ff) << 24); };
		static _m32 iLtoB32(_m32 data){ return (data << 24) | ((data & 0x00ff0000) >> 8) | ((data & 0x0000ff00) << 8) | ((data & 0x000000ff) << 24); };
		static _m64 iBtoL64(_m64 data){ return (((_m64) (iBtoL32((_m32) data))) << 32) | ((_m64) (iBtoL32((_m32) (data >> 32)))); };
		static _m64 iLtoB64(_m64 data){ return (((_m64) (iLtoB32((_m32) data))) << 32) | ((_m64) (iLtoB32((_m32) (data >> 32)))); };

		static void* iMemCopy(void* _Dest, const void* _Src, const size_t _Size)
		{
#ifndef _WINDOWS
			if ((_Dest == nullptr) || (_Src == nullptr) || (_Size == 0))
			{
				return 0;
			}
			unsigned char* pDest = (unsigned char*) _Dest;
			const unsigned char* pSrc = (const unsigned char*) _Src;

			for (size_t seek = 0; seek<_Size; seek = seek + 1, pDest = pDest + 1)
			{
				*pDest = *(pSrc + seek);
			}
			return (void*) _Size;
#else
			return ::memcpy(_Dest, _Src, _Size);
#endif
		};
		static void* iMemSet(void* _Dest, unsigned int _Set, const size_t _Size)
		{
#ifndef _WINDOWS
			if ((_Dest == nullptr) || (_Size == 0))
			{
				return 0;
			}
			unsigned char* pDest = (unsigned char*) _Dest;
			for (size_t seek = 0; seek<_Size; seek = seek + 1, pDest = pDest + 1)
			{
				*pDest = _Set;
			}
			return (void*) _Size;
#else
			return ::memset(_Dest, _Set, _Size);
#endif
		};
	};

#endif
}