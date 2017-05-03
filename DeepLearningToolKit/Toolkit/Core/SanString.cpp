#include"SanString.h"
using namespace std;
using namespace San;
swchar San::gloAToW(const sachar Data)
{
	return gloAToW((const sachar*)&Data,1)[0];
}
SStringW San::gloAToW(const SStringA &strString)
{
	return ::gloAToW(strString.c_str(),strString.length());
}
SStringW San::gloAToW(const sachar* pString,int StringLength)
{
	SStringW strDestString;
	if (StringLength == -1)
	{
		StringLength = ::MultiByteToWideChar(CP_ACP, 0, pString, -1, nullptr, 0);
	}
	if(StringLength==0)
	{
		return strDestString;
	}
	swchar *pBuffer = new swchar[StringLength + 1];
	cSanMemoryFuncSet::iMemSet(pBuffer, 0, sizeof(pBuffer));
	::MultiByteToWideChar(CP_ACP, 0, pString, -1, pBuffer, StringLength + 1);
	strDestString=pBuffer;
	delete[] pBuffer;
	pBuffer=nullptr;
	return strDestString;
}
sachar San::gloWToA(const swchar Data,const sachar DefaultChar)
{
	return gloWToA((const swchar*)&Data,1,DefaultChar)[0];
}
SStringA San::gloWToA(const SStringW &strString,const sachar DefaultChar)
{
	return ::gloWToA(strString.c_str(),strString.length(),DefaultChar);
}
SStringA San::gloWToA(const swchar* pString,int StringLength,const sachar DefaultChar)
{
	SStringA strDestString;
	int bUseDefaultChar=true;
	if (StringLength == -1)
	{
		StringLength = ::WideCharToMultiByte(CP_ACP, 0, pString, -1, nullptr, 0, &DefaultChar, &bUseDefaultChar);
	}
	if(StringLength==0)
	{
		return strDestString;
	}
	sachar *pBuffer = new sachar[StringLength + 1];
	cSanMemoryFuncSet::iMemSet(pBuffer,0,sizeof(pBuffer));
	::WideCharToMultiByte(CP_ACP, 0, pString, -1, pBuffer, StringLength + 1, &DefaultChar, &bUseDefaultChar);
	strDestString=pBuffer;
	delete[] pBuffer;
	pBuffer=nullptr;
	return strDestString;
}
schar San::gloAToT(const sachar Data)
{
#ifndef _UNICODE
	return Data;
#else
	return gloAToW((const sachar*)&Data,1)[0];
#endif
}
SString San::gloAToT(const SStringA &strString)
{
#ifndef _UNICODE
	return strString;
#else
	return gloAToW(strString.c_str(),strString.length());
#endif
}
SString San::gloAToT(const sachar* pString,int StringLength)
{
#ifndef _UNICODE
	return pString;
#else
	return gloAToW(pString,StringLength);
#endif
}
schar San::gloWToT(const swchar Data,const sachar DefaultChar)
{
#ifndef _UNICODE
	return gloWToA((const swchar*)&Data,1,DefaultChar)[0];
#else
	return Data;
#endif
}
SString San::gloWToT(const SStringW &strString,const sachar DefaultChar)
{
#ifndef _UNICODE
	return gloWToA(strString.c_str(),strString.length(),DefaultChar);
#else
	return strString;
#endif
}
SString San::gloWToT(const swchar* pString,int StringLength,const sachar DefaultChar)
{
#ifndef _UNICODE
	return gloWToA(pString,StringLength,DefaultChar);
#else
	return pString;
#endif
}
sachar San::gloTToA(const schar Data,const sachar DefaultChar)
{
#ifndef _UNICODE
	return Data;
#else
	return gloWToA((const schar*)Data,1,DefaultChar)[0];
#endif
}
SStringA San::gloTToA(const SString &strString,const sachar DefaultChar)
{
#ifndef _UNICODE
	return strString;
#else
	return ::gloWToA(strString.c_str(),strString.length(),DefaultChar);
#endif
}
SStringA San::gloTToA(const schar* pString,int StringLength,const sachar DefaultChar)
{
#ifndef _UNICODE
	return pString;
#else
	return ::gloWToA(pString,StringLength,DefaultChar);
#endif
}
swchar San::gloTToW(const schar Data)
{
#ifndef _UNICODE
	return gloAToW((const schar*)&Data,1)[0];
#else
	return Data;
#endif
}
SStringW San::gloTToW(const SString &strString)
{
#ifndef _UNICODE
	return gloAToW(strString.c_str(),strString.length());
#else
	return strString;
#endif
}
SStringW San::gloTToW(const schar* pString,int StringLength)
{
#ifndef _UNICODE
	return gloAToW(pString,StringLength);
#else
	return pString;
#endif
}
SString San::gloIToS(const long long &Data, const unsigned int Radix)
{
	schar String[512];
#ifndef _UNICODE
	::_itoa(Data, String, Radix);
#else
	::_itow(Data, String, Radix);
#endif
	return String;
}
long long San::gloSToI(const SString &strString, const unsigned int Radix)
{
	switch (Radix)
	{
	case 2:
		{
			long long Data = 0;
			for (unsigned int seek = 0; seek < strString.size(); seek = seek + 1)
			{
				if (strString[seek] == _SSTR('0'))
				{
					Data = Data << 1;
					continue;
				}
				if (strString[seek] == _SSTR('1'))
				{
					Data = Data << 1 + 1;
					continue;
				}
				Data = 0;
				return Data;
			}
			return Data;
		}
		break;
	case 10:
#ifndef _UNICODE
		return ::atoi(strString.c_str());
#else
		return ::_wtoi(strString.c_str());
#endif
		break;
	case 16:
		{
			long long Data;
			std::stringstream strStream;
			strStream << std::hex << strString.c_str();
			strStream >> Data;
			return Data;
		}
		break;
	default:
		return 0;
		break;
	}
	return 0;
}
SString San::gloFToS(const double &Data, SString strFormat)
{
	schar SStringA[512];
	strFormat = _SSTR("%") + strFormat + _SSTR("f");
#ifndef _UNICODE
	::sprintf_s(SStringA, 128, strFormat.c_str(), Data);
#else
	::swprintf_s(SStringA, 128, strFormat.c_str(), Data);
#endif
	return SStringA;
}
double San::gloSToF(const SString &strString)
{
#ifndef _UNICODE
	return ::atof(strString.c_str());
#else
	return ::_wtof(strString.c_str());
#endif
}
bool San::operator==(const SStringA &strStringA, const SStringA &strStringB)
{
	return (((int) strStringA.find(strStringB.c_str())) == 0 && (strStringA.size() == strStringB.size()));
}
bool San::operator!=(const SStringA &strStringA, const SStringA &strStringB)
{
	return !(((int) strStringA.find(strStringB.c_str()) == 0) && (strStringA.size() == strStringB.size()));
}
bool San::operator==(const SStringW &strStringA, const SStringW &strStringB)
{
	return (((int) strStringA.find(strStringB.c_str()) == 0) && (strStringA.size() == strStringB.size()));
}
bool San::operator!=(const SStringW &strStringA, const SStringW &strStringB)
{
	int Index = strStringA.find(strStringB.c_str());
	return !(((int) strStringA.find(strStringB.c_str()) == 0) && (strStringA.size() == strStringB.size()));
}
SStringA San::operator+(const SStringA &strStringA, const SStringA &strStringB)
{
	sachar *pschar = nullptr;
	size_t Size = strStringA.size() + strStringB.size();
	if (Size == 0)
	{
		pschar = new sachar[1];
		pschar[0] = '\0';
	}
	else
	{
		pschar = new sachar[Size + 1];
		unsigned int StrASize = strStringA.size();
		for (size_t seeka = 0; seeka<StrASize; seeka = seeka + 1)
		{
			pschar[seeka] = strStringA[seeka];
		}
		for (size_t seekb = 0; seekb<strStringB.size(); seekb = seekb + 1)
		{
			pschar[seekb + StrASize] = strStringB[seekb];
		}
		pschar[Size] = '\0';
	}
	SStringA strDestString = pschar;
	delete[] pschar;
	return strDestString;
}
SStringW San::operator+(const SStringW &strStringA, const SStringW &strStringB)
{
	swchar *pschar = nullptr;
	size_t Size = strStringA.size() + strStringB.size();
	if (Size == 0)
	{
		pschar = new swchar[1];
		pschar[0] = L'\0';
	}
	else
	{
		pschar = new swchar[Size + 1];
		unsigned int StrASize = strStringA.size();
		for (size_t seeka = 0; seeka<StrASize; seeka = seeka + 1)
		{
			pschar[seeka] = strStringA[seeka];
		}
		for (size_t seekb = 0; seekb<strStringB.size(); seekb = seekb + 1)
		{
			pschar[seekb + StrASize] = strStringB[seekb];
		}
		pschar[Size] = L'\0';
	}
	SStringW strDestString = pschar;
	delete[] pschar;
	return strDestString;
}
istream& San::operator>>(istream &InputStream, SStringA &strString)
{
	size_t Size = InputStream.tellg();

	sachar* pBuffer= new (cSanSystemAllocator::alloc_mem((Size + 1) * sizeof(sachar))) sachar[Size + 1];
	pBuffer[Size] = '\0';

	InputStream.read(pBuffer, Size * sizeof(sachar));
	strString = pBuffer;

	cSanSystemAllocator::dealloc_mem(pBuffer);
	return InputStream;
}
istream& San::operator>>(istream &InputStream, SStringW &strString)
{
	size_t Size = (InputStream.tellg() / sizeof(swchar)) + 1;

	swchar* pBuffer = new (cSanSystemAllocator::alloc_mem((Size + 1) * sizeof(sachar))) swchar[Size + 1];
	pBuffer[Size] = L'\0';

	InputStream.read((char*) pBuffer, Size * sizeof(swchar));
	strString = pBuffer;

	cSanSystemAllocator::dealloc_mem(pBuffer);
	return InputStream;
}
ostream& San::operator<<(ostream &OutputStream, const SStringA &strString)
{
	return OutputStream << strString.c_str();
}
ostream& San::operator<<(ostream &OutputStream, const SStringW &strString)
{
	return OutputStream << strString.c_str();
}
SStringA& San::operator>>(SStringA &strInputString, long long &Data)
{
	Data = ::gloSToI(::gloAToT(strInputString));

	return strInputString;
}
SStringW& San::operator>>(SStringW &strInputString, long long &Data)
{
	Data = ::gloSToI(::gloWToT(strInputString));

	return strInputString;
}
SStringA& San::operator>>(SStringA &strInputString, unsigned long long &Data)
{
	Data = ::gloSToI(::gloAToT(strInputString));

	return strInputString;
}
SStringW& San::operator>>(SStringW &strInputString, unsigned long long &Data)
{
	Data = ::gloSToI(::gloWToT(strInputString));

	return strInputString;
}
SStringA& San::operator>>(SStringA &strInputString, double &Data)
{
	Data = ::gloSToF(::gloAToT(strInputString));

	return strInputString;
}
SStringW& San::operator>>(SStringW &strInputString, double &Data)
{
	Data = ::gloSToF(::gloWToT(strInputString));

	return strInputString;
}
SStringA& San::operator<<(SStringA &strOutputString, const long long &Data)
{
	return strOutputString + ::gloTToA(::gloIToS(Data));
}
SStringW& San::operator<<(SStringW &strOutputString, const long long &Data)
{
	return strOutputString + ::gloTToW(::gloIToS(Data));
}
SStringA& San::operator<<(SStringA &strOutputString, const unsigned long long &Data)
{
	return strOutputString + ::gloTToA(::gloIToS(Data));
}
SStringW& San::operator<<(SStringW &strOutputString, const unsigned long long &Data)
{
	return strOutputString + ::gloTToW(::gloIToS(Data));
}
SStringA& San::operator<<(SStringA &strOutputString, const double &Data)
{
	return strOutputString + ::gloTToA(::gloFToS(Data));
}
SStringW& San::operator<<(SStringW &strOutputString, const double &Data)
{
	return strOutputString + ::gloTToW(::gloFToS(Data));
}
SStringA& San::operator<<(SStringA &strOutputString, const SStringA &strString)
{
	return strOutputString + strString;
}
SStringW& San::operator<<(SStringW &strOutputString, const SStringW &strString)
{
	return strOutputString + strString;
}
vector<SStringA> San::gloGetStringItemsA(const SStringA &strString, SStringA strStringMark)
{
	vector<SStringA> SubStringList;

	if (strString.empty()){ return SubStringList; }
	if (strStringMark.empty()){ strStringMark = " \n\t"; }

	SStringA strTarget = strString + strStringMark[0];
	size_t StrLength = strTarget.length();
	size_t MarkSize = strStringMark.length();
	size_t SubStringBegin = 0;
	for (size_t seek = 0; seek < StrLength; seek = seek + 1)
	{
		for (size_t seek_mark = 0; seek_mark < MarkSize; seek_mark = seek_mark + 1)
		{
			if (strTarget[seek] == strStringMark[seek_mark])
			{
				if (seek != SubStringBegin)
				{
					SubStringList.insert(SubStringList.end(), strTarget.substr(SubStringBegin, seek - SubStringBegin));
				}
				SubStringBegin = seek + 1;
				continue;
			}
		}
	}
	return SubStringList;
}
vector<SStringW> San::gloGetStringItemsW(const SStringW &strString, SStringW strStringMark)
{
	vector<SStringW> SubStringList;

	if (strString.empty()){ return SubStringList; }
	if (strStringMark.empty()){ strStringMark = L" \n\t"; }

	SStringW strTarget = strString + strStringMark[0];
	size_t StrLength = strTarget.length();
	size_t MarkSize = strStringMark.length();
	size_t SubStringBegin = 0;
	for (size_t seek = 0; seek < StrLength; seek = seek + 1)
	{
		for (size_t seek_mark = 0; seek_mark < MarkSize; seek_mark = seek_mark + 1)
		{
			if (strTarget[seek] == strStringMark[seek_mark])
			{
				if (seek != SubStringBegin)
				{
					SubStringList.insert(SubStringList.end(), strTarget.substr(SubStringBegin, seek - SubStringBegin));
				}
				SubStringBegin = seek + 1;
				continue;
			}
		}
	}
	return SubStringList;
}