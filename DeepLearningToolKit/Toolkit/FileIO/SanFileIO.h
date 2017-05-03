#include"fstream"
#include"../RenderSystem.h"
#include"../Core/SanTypes.h"
#include"../Core/SanContainer.h"
using namespace std;
#pragma once
namespace San
{
	namespace FileIOStream
	{
#ifndef __STDSANFILEIOSTREAM_H__
#define __STDSANFILEIOSTREAM_H__

#include"FileLoader.h"

		bool	gloLoadFile(const SString &strFilePath, SStringA &strData);
		bool	gloLoadFile(const SString &strFilePath, SStringW &strData);
		bool	gloLoadFile(const SString &strFilePath, SANSTREAM &Stream);

		bool	gloLoadFile(const SString &strFilePath, uint32 Offset, uint32 Size, SStringA &strData);
		bool	gloLoadFile(const SString &strFilePath, uint32 Offset, uint32 Size, SStringW &strData);
		bool	gloLoadFile(const SString &strFilePath, uint32 Offset, uint32 Size, SANSTREAM &Stream);

		bool	gloWriteFile(const SString &strFilePath, const SStringA &strData, const bool bCover = false);
		bool	gloWriteFile(const SString &strFilePath, const SStringW &strData, const bool bCover = false);
		bool	gloWriteFile(const SString &strFilePath, const SANSTREAM &Stream, const bool bCover = false);

#endif
	}
}