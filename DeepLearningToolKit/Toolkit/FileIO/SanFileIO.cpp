#include"SanFileIO.h"
using namespace std;
using namespace San;
using namespace San::FileIOStream;
bool San::FileIOStream::gloLoadFile(const SString &strFilePath, SStringA &strData)
{
	return ::gloLoadFile(strFilePath, 0, 0, strData);
}
bool San::FileIOStream::gloLoadFile(const SString &strFilePath, SStringW &strData)
{
	return ::gloLoadFile(strFilePath, 0, 0, strData);
}
bool San::FileIOStream::gloLoadFile(const SString &strFilePath, SANSTREAM &Stream)
{
	return ::gloLoadFile(strFilePath, 0, 0, Stream);
}
bool San::FileIOStream::gloLoadFile(const SString &strFilePath, uint32 Offset, uint32 Size, SStringA &strData)
{
	if (strFilePath.empty()) { return false; }

	fstream* pFile = new fstream(strFilePath, ios::binary | ios::in, 0x40);
	if (pFile == nullptr) { return false; }

	pFile->seekg(0, ios::end);
	
	int32 FileSize = pFile->tellg();
	if (FileSize <= 0) { return false; }

	pFile->clear();
	
	Offset = Offset * sizeof(sachar);
	Size = Size * sizeof(sachar);
	
	if (Offset >= FileSize) { return false; }
	
	FileSize = FileSize - Offset;
	Size = (Size == 0) ? FileSize : (FileSize > Size ? Size : FileSize);
	
	pFile->seekg(Offset, ios::beg);
	
	SANSTREAM Stream(Size + sizeof(sachar));
	pFile->read((char*)Stream.iGetPtr(), Size);
	pFile->close();
	
	strData = (sachar*)Stream.iGetPtr();
	
	delete pFile;
	pFile = nullptr;
	
	return true;
}
bool San::FileIOStream::gloLoadFile(const SString &strFilePath, uint32 Offset, uint32 Size, SStringW &strData)
{
	if (strFilePath.empty()) { return false; }
	
	fstream* pFile = new fstream(strFilePath, ios::binary | ios::in, 0x40);
	if (pFile == nullptr) { return false; }
	
	pFile->seekg(0, ios::end);
	
	int32 FileSize = pFile->tellg();
	if (FileSize <= 0) { return false; }
	
	pFile->clear();
	
	Offset = Offset * sizeof(swchar);
	Size = Size * sizeof(swchar);
	
	if (Offset >= FileSize) { return false; }
	
	FileSize = FileSize - Offset;
	Size = (Size == 0) ? FileSize : (FileSize > Size ? Size : FileSize);
	
	pFile->seekg(Offset, ios::beg);
	
	SANSTREAM Stream(Size + sizeof(swchar));
	pFile->read((char*)Stream.iGetPtr(), Size);
	pFile->close();
	
	strData = (swchar*)Stream.iGetPtr();
	
	delete pFile;
	pFile = nullptr;
	
	return true;
}
bool San::FileIOStream::gloLoadFile(const SString &strFilePath, uint32 Offset, uint32 Size, SANSTREAM &Stream)
{
	if (strFilePath.empty()) { return false; }
	
	fstream* pFile = new fstream(strFilePath, ios::binary | ios::in, 0x40);
	if (pFile == nullptr) { return false; }
	
	pFile->seekg(0, ios::end);
	int32 FileSize = pFile->tellg();
	
	if (FileSize <= 0) { return false; }
	pFile->clear();

	if (Offset >= FileSize) { return false; }
	
	FileSize = FileSize - Offset;
	Size = (Size == 0) ? FileSize : (FileSize > Size ? Size : FileSize);
	
	pFile->seekg(0, ios::beg);
	
	Stream.iClear();
	
	Stream.iReSizeStream(Size);
	pFile->read((char*)Stream.iGetPtr(), Size);
	pFile->close();
	
	delete pFile;
	pFile = nullptr;
	
	return true;
}
bool San::FileIOStream::gloWriteFile(const SString &strFilePath, const SStringA &strData, const bool bCover)
{
	if (strFilePath.empty()) { return false; }
	
	ios::openmode FileFlag = ios::binary | ios::out | (bCover ? ios::trunc : ios::ate);
	fstream* pFile = new fstream(strFilePath, FileFlag, 0x40);
	
	if (pFile == nullptr) { return false; }
	
	(*pFile) << strData.c_str();
	
	pFile->close();
	delete pFile;
	pFile = nullptr;
	
	return true;
}
bool San::FileIOStream::gloWriteFile(const SString &strFilePath, const SStringW &strData, const bool bCover)
{
	if (strFilePath.empty()) { return false; }

	ios::openmode FileFlag = ios::binary | ios::out | (bCover ? ios::trunc : ios::ate);
	fstream* pFile = new fstream(strFilePath, FileFlag, 0x40);
	
	if (pFile == nullptr) { return false; }
	
	pFile->write((const char*)strData.c_str(), sizeof(swchar) *strData.size());
	pFile->close();
	
	delete pFile;
	pFile = nullptr;
	
	return true;
}
bool San::FileIOStream::gloWriteFile(const SString &strFilePath, const SANSTREAM &Stream, const bool bCover)
{
	if (strFilePath.empty()) { return false; }

	ios::openmode FileFlag = ios::binary | ios::out | (bCover ? ios::trunc : ios::ate);
	fstream* pFile = new fstream(strFilePath, FileFlag, 0x40);

	if (pFile == nullptr) { return false; }

	pFile->write((const char*)Stream.iGetPtr(), Stream.iGetSize());
	pFile->close();
	
	delete pFile;
	pFile = nullptr;
	
	return true;
}