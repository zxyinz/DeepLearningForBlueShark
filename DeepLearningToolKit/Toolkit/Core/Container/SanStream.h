//Project: San Lab Game Engine
//Version: 2.1.1
//Debug State: Need test
#include"SanContainerDef.h"
#include"SanStreamT.h"
#pragma once
using namespace std;
namespace San
{
#ifndef __STDCONTAINERSTREAM_H__
#define __STDCONTAINERSTREAM_H__
	struct SANSTREAM : public _sstream<uint8>
	{
	public:
		SANSTREAM(const _size Size = 0);
		SANSTREAM(const SANSTREAM& Stream);
		SANSTREAM(const uint8* pStream, const _size BufSize);
		~SANSTREAM();

		//Stream IO Function
		_size iSetStream(const _size Offset, const SANSTREAM& Stream);
		_size iSetStream(const _size Offset, const uint8* pBuffer, const _size BufSize);
		_size iSetStream(const _size Offset, const uint8 Data);
		_size iSetStream(const _size Offset, const uint16 Data);
		_size iSetStream(const _size Offset, const uint32 Data);
		_size iSetStream(const _size Offset, const uint64 Data);

		//May increase bin file size and reduce the effeciency
		template<class _data> _size iSetStreamT(const _size Offset, const _data &Data)
		{
			this->iSetStream(Offset, static_cast<uint8*>(&Data), sizeof(_data));
		};

		_size iGetStream(const _size Offset, uint8* pBuffer, const _size BufSize) const;
		_size iGetStream(const _size Offset, uint8 &Data) const;
		_size iGetStream(const _size Offset, uint16 &Data) const;
		_size iGetStream(const _size Offset, uint32 &Data) const;
		_size iGetStream(const _size Offset, uint64 &Data) const;

		//May increase bin file size and reduce the effeciency
		template<class _data> _size iGetStreamT(const _size Offset, _data &Data) const
		{
			return this->iGetStream(Offset, static_cast<uint8*>(&Data), sizeof(Data));
		}
	};
	typedef SANSTREAM* lpSANSTREAM;

	struct SANBITSTREAM : public _sstream<bool>
	{
	public:
		SANBITSTREAM(const _size BitStreamSize = 8);
		SANBITSTREAM(const SANBITSTREAM& BitStream);
		SANBITSTREAM(const SANSTREAM& Stream, const _size Begin, const _size BitStreamSize);
		SANBITSTREAM(const uint8* pStream, const _size BufSize, const _size Begin, const _size BitStreamSize);
		~SANBITSTREAM();

		//operators overload
		//SANBITSTREAM& operator=(const SANBITSTREAM& BitStream);
		//SANBITSTREAM operator+(const SANBITSTREAM& BitStream);
		//SANBITSTREAM operator-(const _size SizeInBit);

		//Stream IO Function
		_size iSetBitStream(const _size Offset, const SANBITSTREAM& Stream, _size Begin = 0, _size Length = 0);
		_size iSetBitStream(const _size Offset, const uint8* pBuffer, const _size BufSize, _size Begin = 0, _size Length = 0);
		_size iSetBitStream(const _size Offset, const uint8 Data, _size Begin = 0, _size Length = 0);
		_size iSetBitStream(const _size Offset, const uint16 Data, _size Begin = 0, _size Length = 0);
		_size iSetBitStream(const _size Offset, const uint32 Data, _size Begin = 0, _size Length = 0);
		_size iSetBitStream(const _size Offset, const uint64 Data, _size Begin = 0, _size Length = 0);
		_size iGetBitStream(const _size Offset, uint8* pBuffer, const _size BufSize, _size Begin = 0, _size Length = 0) const;
		_size iGetBitStream(const _size Offset, uint8 &Data, _size Begin = 0, _size Length = 0) const;
		_size iGetBitStream(const _size Offset, uint16 &Data, _size Begin = 0, _size Length = 0) const;
		_size iGetBitStream(const _size Offset, _size &Data, _size Begin = 0, _size Length = 0) const;
		_size iGetBitStream(const _size Offset, uint64 &Data, _size Begin = 0, _size Length = 0) const;
	};
	typedef SANBITSTREAM* lpSANBITSTREAM;
#endif
}