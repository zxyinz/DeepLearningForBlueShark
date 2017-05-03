#include"SanStream.h"
using namespace std;
using namespace San;
SANSTREAM::SANSTREAM(const _size Size)
	:_sstream<uint8>(Size, 0)
{
}
SANSTREAM::SANSTREAM(const SANSTREAM& Stream)
	:_sstream<uint8>(Stream)
{
}
SANSTREAM::SANSTREAM(const uint8* pStream, const _size BufSize)
	//:_sstream<uint8>(BufSize, pStream, BufSize)
{
}
SANSTREAM::~SANSTREAM()
{
}
SANSTREAM::_size SANSTREAM::iSetStream(const _size Offset, const SANSTREAM& Stream)
{
	if (Stream.iGetSize() == 0) { return 0; }
	this->iSetStream(Offset, Stream.iGetPtr(), Stream.iGetSize());
}
SANSTREAM::_size SANSTREAM::iSetStream(const _size Offset, const uint8* pBuffer, const _size BufSize)
{
	if ((pBuffer == nullptr) || (BufSize == 0)) { return 0; }
	if (Offset >= this->iGetSize()) { return 0; }

	const _size EndPos = (Offset + BufSize) < this->iGetSize() ? (Offset + BufSize) : this->iGetSize();
	_size Index = 0;

	for (_size seek = Offset; seek < EndPos; seek = seek + 1)
	{
		this->iGetPtr()[seek] = pBuffer[Index];
		Index = Index + 1;
	}

	return Index;
}
SANSTREAM::_size SANSTREAM::iSetStream(const _size Offset, const uint8 Data)
{
	return this->iSetStream(Offset, (uint8*) &Data, sizeof(Data));
}
SANSTREAM::_size SANSTREAM::iSetStream(const _size Offset, const uint16 Data)
{
	return this->iSetStream(Offset, (uint8*) &Data, sizeof(Data));
}
SANSTREAM::_size SANSTREAM::iSetStream(const _size Offset, const uint32 Data)
{
	return this->iSetStream(Offset, (uint8*) &Data, sizeof(Data));
}
SANSTREAM::_size SANSTREAM::iSetStream(const _size Offset, const uint64 Data)
{
	return this->iSetStream(Offset, (uint8*) &Data, sizeof(Data));
}
SANSTREAM::_size SANSTREAM::iGetStream(const _size Offset, uint8* pBuffer, const _size BufSize) const
{
	if ((pBuffer == nullptr) || (BufSize == 0)) { return 0; }
	if (Offset >= this->iGetSize()) { return 0; }

	const _size EndPos = (Offset + BufSize) < this->iGetSize() ? (Offset + BufSize) : this->iGetSize();
	_size Index = 0;

	for (_size seek = Offset; seek < EndPos; seek = seek + 1)
	{
		pBuffer[Index] = this->iGetPtr()[seek];
		Index = Index + 1;
	}

	return Index;
}
SANSTREAM::_size SANSTREAM::iGetStream(const _size Offset, uint8 &Data) const
{
	return this->iGetStream(Offset, (uint8*) &Data, sizeof(Data));
}
SANSTREAM::_size SANSTREAM::iGetStream(const _size Offset, uint16 &Data) const
{
	return this->iGetStream(Offset, (uint8*) &Data, sizeof(Data));
}
SANSTREAM::_size SANSTREAM::iGetStream(const _size Offset, uint32 &Data) const
{
	return this->iGetStream(Offset, (uint8*) &Data, sizeof(Data));
}
SANSTREAM::_size SANSTREAM::iGetStream(const _size Offset, uint64 &Data) const
{
	return this->iGetStream(Offset, (uint8*) &Data, sizeof(Data));
}
SANBITSTREAM::SANBITSTREAM(const _size BitStreamSize)
	:_sstream<bool>(BitStreamSize)
{
	this->iSetStream(false);
}
SANBITSTREAM::SANBITSTREAM(const SANBITSTREAM& BitStream)
	:_sstream<bool>(BitStream)
{
	this->iSetStream(false);
}
SANBITSTREAM::SANBITSTREAM(const SANSTREAM& Stream, const _size Begin, const _size BitStreamSize)
	:_sstream<bool>(BitStreamSize)
{
	this->iSetStream(false);
	_size Index = 0;
	_size Offset = 0;
	_size Length = Begin + BitStreamSize;

	for (_size seek = 0; seek < BitStreamSize; seek = seek + 1)
	{
		Index = (Begin + seek) >> 3;
		if (Index >= Stream.iGetSize())
		{
			break;
		}
		Offset = (Begin + seek) % 8;

		(*this)[seek] = (bool) ((Stream.iGetPtr()[Index] >> Offset) & 0x01);
	}
}
SANBITSTREAM::SANBITSTREAM(const uint8* pStream, const _size BufSize, const _size Begin, const _size BitStreamSize)
	:_sstream<bool>(BitStreamSize)
{
	this->iSetStream(false);
	if (pStream != nullptr)
	{
		_size Index = 0;
		_size Offset = 0;
		_size Length = Begin + BitStreamSize;
		for (_size seek = 0; seek < BitStreamSize; seek = seek + 1)
		{
			Index = (Begin + seek) >> 3;
			if (Index >= BufSize)
			{
				break;
			}
			Offset = (Begin + seek) % 8;
			(*this)[seek] = (bool) ((pStream[Index] >> Offset) & 0x01);
		}
	}
}
SANBITSTREAM::~SANBITSTREAM()
{
}
SANSTREAM::_size SANBITSTREAM::iSetBitStream(const _size Offset, const SANBITSTREAM& Stream, _size Begin, _size Length)
{
	if ((Begin >> 3) >= Stream.iGetSize())
	{
		return 0;
	}
	_size ByteIndex = 0;
	_size BitOffset = 0;
	Length = Length == 0 ? (Stream.iGetSize() << 3) - Begin : Length;
	_size End = Offset + Length;
	End = End >= this->iGetSize() ? this->iGetSize() : End;
	for (_size seek = Offset; seek < End; seek = seek + 1)
	{
		ByteIndex = (Begin + seek - Offset) >> 3;
		if (ByteIndex >= Stream.iGetSize())
		{
			return seek - Offset;
		}
		BitOffset = (Begin + seek - Offset) % 8;
		(*this)[seek] = (bool) ((Stream[ByteIndex] >> BitOffset) & 0x01);
	}
	return Length;
}
SANSTREAM::_size SANBITSTREAM::iSetBitStream(const _size Offset, const uint8* pBuffer, const _size BufSize, _size Begin, _size Length)
{
	if (pBuffer == nullptr)
	{
		return 0;
	}
	if ((Begin >> 3) >= BufSize)
	{
		return 0;
	}
	_size ByteIndex = 0;
	_size BitOffset = 0;
	Length = Length == 0 ? (BufSize << 3) - Begin : Length;
	_size End = Offset + Length;
	End = End >= this->iGetSize() ? this->iGetSize() : End;
	for (_size seek = Offset; seek < End; seek = seek + 1)
	{
		ByteIndex = (Begin + seek - Offset) >> 3;
		if (ByteIndex >= BufSize)
		{
			return seek - Offset;
		}
		BitOffset = (Begin + seek - Offset) % 8;
		(*this)[seek] = (bool) ((pBuffer[ByteIndex] >> BitOffset) & 0x01);
	}
	return Length;
}
SANSTREAM::_size SANBITSTREAM::iSetBitStream(const _size Offset, const uint8 Data, _size Begin, _size Length)
{
	if (Begin >= 8)
	{
		return 0;
	}
	Length = (Length + Begin) >= 8 ? 8 - Begin : Length;
	for (uint8 seek = 0; seek < Length; seek = seek + 1)
	{
		if ((Offset + seek) >= this->iGetSize())
		{
			return seek;
		}
		(*this)[Offset + seek] = (bool) ((Data >> (Begin + seek)) & 0x01);
	}
	return Length;
}
SANSTREAM::_size SANBITSTREAM::iSetBitStream(const _size Offset, const uint16 Data, _size Begin, _size Length)
{
	if (Begin >= 16)
	{
		return 0;
	}
	Length = (Begin + Length) >= 16 ? 16 - Begin : Length;
	for (uint8 seek = 0; seek < Length; seek = seek + 1)
	{
		if ((Offset + seek)>this->iGetSize())
		{
			return seek;
		}
		(*this)[Offset + seek] = (bool) ((Data >> (Begin + seek)) & 0x01);
	}
	return Length;
}
SANSTREAM::_size SANBITSTREAM::iSetBitStream(const _size Offset, const uint32 Data, _size Begin, _size Length)
{
	if (Begin >= 32)
	{
		return 0;
	}
	Length = (Begin + Length) >= 32 ? 32 - Begin : Length;
	for (uint8 seek = 0; seek < Length; seek = seek + 1)
	{
		if ((Offset + seek)>this->iGetSize())
		{
			return seek;
		}
		(*this)[Offset + seek] = (bool) ((Data >> (Begin + Offset)) & 0x01);
	}
	return Length;
}
SANSTREAM::_size SANBITSTREAM::iSetBitStream(const _size Offset, const uint64 Data, _size Begin, _size Length)
{
	if (Begin >= 64)
	{
		return 0;
	}
	Length = (Begin + Length) > 64 ? 64 - Begin : Length;
	for (uint8 seek = 0; seek < Length; seek = seek + 1)
	{
		if ((Offset + seek) >= this->iGetSize())
		{
			return seek;
		}
		(*this)[Offset + seek] = (bool) ((Data >> (Begin >> seek)) & 0x01);
	}
}
SANSTREAM::_size SANBITSTREAM::iGetBitStream(const _size Offset, uint8* pBuffer, const _size BufSize, _size Begin, _size Length) const
{
	if (Begin >= (BufSize >> 3))
	{
		return 0;
	}
	Length = (Begin + Length) >= (BufSize >> 3) ? (BufSize >> 3) - Begin : Length;
	_size ByteIndex = 0;
	_size BitOffset = 0;
	for (_size seek = 0; seek < Length; seek = seek + 1)
	{
		if ((Offset + seek)>this->iGetSize())
		{
			return seek;
		}
		ByteIndex = (Begin + seek) >> 3;
		BitOffset = (Begin + seek) % 8;
		if ((*this)[Offset + seek])
		{
			pBuffer[ByteIndex] = pBuffer[ByteIndex] | (0x01 << BitOffset);
		}
		else
		{
			pBuffer[ByteIndex] = pBuffer[ByteIndex] & (~(0x01 << BitOffset));
		}
	}
	return Length;
}
SANSTREAM::_size SANBITSTREAM::iGetBitStream(const _size Offset, uint8 &Data, _size Begin, _size Length) const
{
	if (Begin >= 8)
	{
		return 0;
	}
	Length = (Begin + Length) >= 8 ? 8 - Begin : Length;
	uint8 Base = 0x01;
	for (uint8 seek = 0; seek < Length; seek = seek + 1)
	{
		if ((Offset + seek)>this->iGetSize())
		{
			return seek;
		}
		Data = (*this)[Offset + seek] ? Data | (Base << (Begin + seek)) : Data & (~(Base << (Begin + seek)));
	}
	return Length;
}
SANSTREAM::_size SANBITSTREAM::iGetBitStream(const _size Offset, uint16 &Data, _size Begin, _size Length) const
{
	if (Begin >= 16)
	{
		return 0;
	}
	Length = (Begin + Length) >= 16 ? 16 - Begin : Length;
	uint16 Base = 0x0001;
	for (uint8 seek = 0; seek < Length; seek = seek + 1)
	{
		if ((Offset + seek)>this->iGetSize())
		{
			return seek;
		}
		Data = (*this)[Offset + seek] ? Data | (Base << (Begin + seek)) : Data & (~(Base << (Begin + seek)));
	}
	return Length;
}
SANSTREAM::_size SANBITSTREAM::iGetBitStream(const _size Offset, _size &Data, _size Begin, _size Length) const
{
	if (Begin >= 32)
	{
		return 0;
	}
	Length = (Begin + Length) >= 32 ? 32 - Begin : Length;
	_size Base = 0x00000001;
	for (uint8 seek = 0; seek < Length; seek = seek + 1)
	{
		if ((Offset + seek)>this->iGetSize())
		{
			return seek;
		}
		Data = (*this)[Offset + seek] ? Data | (Base << (Begin + seek)) : Data & (~(Base << (Begin + seek)));
	}
	return Length;
}
SANSTREAM::_size SANBITSTREAM::iGetBitStream(const _size Offset, uint64 &Data, _size Begin, _size Length) const
{
	if (Begin >= 64)
	{
		return 0;
	}
	Length = (Begin + Length) >= 64 ? 64 - Begin : Length;
	uint64 Base = 0x00000001;
	for (uint8 seek = 0; seek < Length; seek = seek + 1)
	{
		if ((Offset + seek)>this->iGetSize())
		{
			return seek;
		}
		Data = (*this)[Offset + seek] ? Data | (Base << (Begin + seek)) : Data & (~(Base << (Begin + seek)));
	}
	return Length;
}