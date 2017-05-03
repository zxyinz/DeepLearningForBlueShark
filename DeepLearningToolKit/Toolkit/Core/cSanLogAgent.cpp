#include"cSanLogAgent.h"
using namespace std;
using namespace std::chrono;
using namespace San;
SANLOG::SANLOG()
	:Type(SLT_LOG), Level(0), Code(0),
	strLog(_SSTR("")),
	TimeStamp(system_clock::now()),
	ClockStamp(clock()),
	strFile(_SSTR("")), Line(0), MemAddr(0),
	UserData()
{
}
SANLOG::SANLOG(const eSANLOGTYPE Type, const SString &strLog, const uint32 Level, const uint32 Code, const SString &strFile, const uint32 Line, const SHANDLE MemAddr, const _sstream<uint8> &UserData)
	:Type(Type), Level(Level), Code(Code),
	strLog(strLog),
	TimeStamp(system_clock::now()),
	ClockStamp(clock()),
	strFile(strFile), Line(Line), MemAddr(MemAddr),
	UserData(UserData)
{
}
SANLOG::~SANLOG()
{
}
SANMSG::SANMSG(const SString &strLog, const uint32 Level, const uint32 Code, const SString &strFile, const uint32 Line, const SHANDLE MemAddr, const _sstream<uint8> &UserData)
	:SANLOG(SLT_MSG, strLog, Level, Code, strFile, Line, MemAddr, UserData)
{
}
SANMSG::~SANMSG()
{
}
SANWARN::SANWARN(const SString &strLog, const uint32 Level, const uint32 Code, const SString &strFile, const uint32 Line, const SHANDLE MemAddr, const _sstream<uint8> &UserData)
	:SANLOG(SLT_WARN, strLog, Level, Code, strFile, Line, MemAddr, UserData)
{
}
SANWARN::~SANWARN()
{
}
SANERR::SANERR(const SString &strLog, const uint32 Level, const uint32 Code, const SString &strFile, const uint32 Line, const SHANDLE MemAddr, const _sstream<uint8> &UserData)
	:SANLOG(SLT_ERR, strLog, Level, Code, strFile, Line, MemAddr, UserData)
{
}
SANERR::~SANERR()
{
}
SANDBG::SANDBG(const SString &strLog, const uint32 Level, const uint32 Code, const SString &strFile, const uint32 Line, const SHANDLE MemAddr, const _sstream<uint8> &UserData)
	:SANLOG(SLT_DBG, strLog, Level, Code, strFile, Line, MemAddr, UserData)
{
}
SANDBG::~SANDBG()
{
}
SANSYS::SANSYS(const SString &strLog, const uint32 Level, const uint32 Code, const SString &strFile, const uint32 Line, const SHANDLE MemAddr, const _sstream<uint8> &UserData)
	:SANLOG(SLT_SYS, strLog, Level, Code, strFile, Line, MemAddr, UserData)
{
}
SANSYS::~SANSYS()
{
}
cSanLogAgent::FILITER::FILITER(const uint32 Type, const _srange<uint32> LevelRange, const _srange<uint32> CodeRange, const _srange<time_t> TimeRange, const _srange<clock_t> ClockRange, const bool bAccept)
	:Type(Type), LevelRange(LevelRange), CodeRange(CodeRange), TimeRange(TimeRange), ClockRange(ClockRange), bAccept(bAccept)
{
}
cSanLogAgent::FILITER::~FILITER()
{
}
bool cSanLogAgent::FILITER::operator==(const FILITER &Filiter) const
{
	if (this->Type != Filiter.Type){ return false; }
	if (this->LevelRange != Filiter.LevelRange){ return false; }
	if (this->CodeRange != Filiter.CodeRange){ return false; }
	if (this->TimeRange != Filiter.TimeRange){ return false; }
	if (this->ClockRange != Filiter.ClockRange){ return false; }

	return true;
}
bool cSanLogAgent::FILITER::operator!=(const FILITER &Filiter) const
{
	if (this->Type != Filiter.Type){ return true; }
	if (this->LevelRange != Filiter.LevelRange){ return true; }
	if (this->CodeRange != Filiter.CodeRange){ return true; }
	if (this->TimeRange != Filiter.TimeRange){ return true; }
	if (this->ClockRange != Filiter.ClockRange){ return true; }

	return false;
}
cSanLogAgent::cSanLogAgent(const SString &strLogFormat, const SString &strTimeFormat, const SString &strClockFormat)
	:m_strLogFormat(strLogFormat), m_strTimeFormat(strTimeFormat), m_strClockFormat(strClockFormat)
{
	this->iResetTagString();
}
cSanLogAgent::cSanLogAgent(const vector<FILITER> &FiliterArray, const SString &strLogFormat, const SString &strTimeFormat, const SString &strClockFormat)
	:m_FiliterArray(FiliterArray), m_strLogFormat(strLogFormat), m_strTimeFormat(strTimeFormat), m_strClockFormat(strClockFormat)
{
	this->iResetTagString();
}
cSanLogAgent::~cSanLogAgent()
{
	this->iReleaseAgent();
}
bool cSanLogAgent::_LogAcceptable(const SANLOG &Log) const
{
	if (this->m_FiliterArray.empty()) { return true; }

	for (const auto &Filiter : this->m_FiliterArray)
	{
		if ((Log.Type & Filiter.Type) != Log.Type){ continue; }

		if (!Filiter.LevelRange.empty()) { if (!Filiter.LevelRange.iInRange(Log.Level)){ continue; } }
		if (!Filiter.CodeRange.empty()) { if (!Filiter.CodeRange.iInRange(Log.Code)){ continue; } }
		if (!Filiter.TimeRange.empty()) { if (!Filiter.TimeRange.iInRange(system_clock::to_time_t(Log.TimeStamp))){ continue; } }
		if (!Filiter.ClockRange.empty()) { if (!Filiter.ClockRange.iInRange(Log.ClockStamp)){ continue; } }

		return Filiter.bAccept;
	}

	return false;
}
bool cSanLogAgent::iCreateAgent()
{
	return true;
}
void cSanLogAgent::iReleaseAgent()
{
}
bool cSanLogAgent::iAddFilter(const FILITER &Filiter, const bool bToFront)
{
	for (const auto &F : this->m_FiliterArray)
	{
		if (F == Filiter){ return true; }
	}

	bToFront ? this->m_FiliterArray.insert(this->m_FiliterArray.begin(), Filiter) : this->m_FiliterArray.push_back(Filiter);

	return true;
}
bool cSanLogAgent::iDeleteFilter(const FILITER &Filiter)
{
	for (auto It = this->m_FiliterArray.begin(); It != this->m_FiliterArray.end(); It++)
	{
		if ((*It) == Filiter){ this->m_FiliterArray.erase(It); return true; }
	}

	return false;
}
void cSanLogAgent::iSetLogFormat(const SString &strLogFormat)
{
	this->m_strLogFormat = strLogFormat;
}
void cSanLogAgent::iSetTimeFormat(const SString &strTimeFormat)
{
	this->m_strTimeFormat = strTimeFormat;
}
void cSanLogAgent::iSetClockFormat(const SString &strClockFormat)
{
	this->m_strClockFormat = strClockFormat;
}
void cSanLogAgent::iUpdateTagString(const eSANLOGTYPE Type, const SString &strString)
{
	auto It = this->m_TagStringSet.find(Type);
	if ( It == this->m_TagStringSet.end()){ return; }

	It->second = strString;
}
void cSanLogAgent::iResetTagString()
{
	this->m_TagStringSet.clear();

	this->iResetTagString(SLT_LOG);
	this->iResetTagString(SLT_MSG);
	this->iResetTagString(SLT_WARN);
	this->iResetTagString(SLT_ERR);
	this->iResetTagString(SLT_DBG);
	this->iResetTagString(SLT_SYS);
}
void cSanLogAgent::iResetTagString(const eSANLOGTYPE Type)
{
	switch (Type)
	{
	case SLT_LOG: this->m_TagStringSet[SLT_LOG] = _SSTR("LOG"); break;
	case SLT_MSG: this->m_TagStringSet[SLT_MSG] = _SSTR("MSG"); break;
	case SLT_WARN: this->m_TagStringSet[SLT_WARN] = _SSTR("WARN"); break;
	case SLT_ERR: this->m_TagStringSet[SLT_ERR] = _SSTR("ERR"); break;
	case SLT_DBG: this->m_TagStringSet[SLT_DBG] = _SSTR("DBG"); break;
	case SLT_SYS: this->m_TagStringSet[SLT_SYS] = _SSTR("SYS"); break;
	default: break;
	}
}
SString cSanLogAgent::iGetLogFormat() const
{
	return this->m_strLogFormat;
}
SString cSanLogAgent::iGetTimeFormat() const
{
	return this->m_strTimeFormat;
}
SString cSanLogAgent::iGetClockFormat() const
{
	return this->m_strClockFormat;
}
SString cSanLogAgent::iGetTagString(const eSANLOGTYPE Type) const
{
	const auto It = this->m_TagStringSet.find(Type);
	if (It == this->m_TagStringSet.end()){ return _SSTR(""); }

	return  It->second;
}
bool cSanLogAgent::iPushLog(const SANLOG &Log)
{
	if (this->_LogAcceptable(Log)) { return this->_PushLog(Log); }
	return false;
}
SString cSanLogAgent::iGenerateLogString(const SANLOG &Log) const
{
	const int32 Size = this->m_strLogFormat.size();
	const int32 BufferSize = 2048;

	SString strString(64, 0);

	schar Buffer[BufferSize]; //For speed
	cSanMemoryFuncSet::iMemSet(Buffer, 0, BufferSize);

	int32 Index = 0;

	for (int32 seek = 0; seek < Size; seek = seek + 1)
	{
		if (Index >= BufferSize) { break; }

		schar Val = this->m_strLogFormat[seek];

		if (Val != _SSTR('%'))
		{
			Buffer[Index] = Val;
			Index = Index + 1;
			continue;
		}

		seek = seek + 1;

		Val = this->m_strLogFormat[seek];

		switch (Val)
		{
		case _SSTR('T'):
			strString = this->iTimeToString(Log.TimeStamp, this->m_strTimeFormat);
			break;
		case _SSTR('C'):
			strString = this->iClockToString(Log.ClockStamp, this->m_strClockFormat);
			break;
		case _SSTR('t'):
			strString = this->iGetTagString(Log.Type);
			break;
		case _SSTR('l'):
			strString = ::gloIToS(Log.Level);
			break;
		case _SSTR('c'):
			strString = ::gloIToS(Log.Code);
			break;
		case _SSTR('m'):
			strString = Log.strLog;
			break;
		case _SSTR('d'):
			strString = Log.strFile.empty() ? _SSTR("") : Log.strFile + _SSTR(":") + ::gloIToS(Log.Line) + _SSTR("\t0x") + ::gloIToS(uint32(Log.MemAddr), 8);
			break;
		case _SSTR('D'):
			strString = Log.strFile.empty() ? _SSTR("") : _SSTR("\r\n\t") + Log.strFile + _SSTR(":") + ::gloIToS(Log.Line) + _SSTR("\t0x") + ::gloIToS(uint32(Log.MemAddr), 8);
			break;
		default:
			strString = Val;
			break;
		}

		const int32 Length = min(BufferSize - Index, strString.size());

		for (int seek = 0; seek < Length; seek = seek + 1, Index = Index + 1)
		{
			Buffer[Index] = strString[seek];
		}
	}

	return Buffer;
}
vector<cSanLogAgent::LOG_ITEM> cSanLogAgent::iGenerateLogStringPairSet(const SANLOG &Log) const
{
	const int32 Size = this->m_strLogFormat.size();

	SString strString(64, 0);

	vector<LOG_ITEM> ItemSet;
	ItemSet.reserve(16);

	int32 Begin = 0;
	for (int32 seek = 0; seek < Size; seek = seek + 1)
	{
		if (this->m_strLogFormat[seek] != _SSTR('%')) { continue; }

		ItemSet.push_back(LOG_ITEM(IT_DEFAULT, this->m_strLogFormat.substr(Begin, seek - Begin)));

		seek = seek + 1;

		switch (this->m_strLogFormat[seek])
		{
		case _SSTR('T'):
			ItemSet.push_back(LOG_ITEM(IT_TIME, this->iTimeToString(Log.TimeStamp, this->m_strTimeFormat)));
			break;
		case _SSTR('C'):
			ItemSet.push_back(LOG_ITEM(IT_CLOCK, this->iClockToString(Log.ClockStamp, this->m_strClockFormat)));
			break;
		case _SSTR('t'):
			ItemSet.push_back(LOG_ITEM(IT_TAG, this->iGetTagString(Log.Type)));
			break;
		case _SSTR('l'):
			ItemSet.push_back(LOG_ITEM(IT_LEVEL, ::gloIToS(Log.Level)));
			break;
		case _SSTR('c'):
			ItemSet.push_back(LOG_ITEM(IT_CODE, ::gloIToS(Log.Code)));
			break;
		case _SSTR('m'):
			ItemSet.push_back(LOG_ITEM(IT_MSG, Log.strLog));
			break;
		case _SSTR('d'):
			if (!Log.strFile.empty())
			{
				strString = Log.strFile + _SSTR(":") + ::gloIToS(Log.Line) + _SSTR("\t0x") + ::gloIToS(uint32(Log.MemAddr), 8);
				ItemSet.push_back(LOG_ITEM(IT_DEBUG, strString));
			}
			break;
		case _SSTR('D'):
			if (!Log.strFile.empty())
			{
				strString = Log.strFile + _SSTR(":") + ::gloIToS(Log.Line) + _SSTR("\t0x") + ::gloIToS(uint32(Log.MemAddr), 8);
				ItemSet.push_back(LOG_ITEM(IT_DEFAULT, _SSTR("\r\n\t")));
				ItemSet.push_back(LOG_ITEM(IT_DEBUG, strString));
			}
			break;
		default:
			strString = this->m_strLogFormat[seek];
			ItemSet.push_back(LOG_ITEM(IT_DEFAULT, strString));
			break;
		}

		Begin = seek + 1;
	}

	return ItemSet;
}
SString cSanLogAgent::iTimeToString(const system_clock::time_point &Time, const SString &strFormat)
{
	if (strFormat.empty()){ return strFormat; }

	sachar Buffer[128];
	time_t T = system_clock::to_time_t(Time);
	strftime(Buffer, 128, ::gloTToA(strFormat).c_str(), localtime(&T));

	return ::gloAToT(Buffer);
}
SString cSanLogAgent::iClockToString(const clock_t Clock, const SString &strFormat)
{
	const int32 Size = strFormat.size();

	if (Size == 0){ return strFormat; }

	schar ClockBuffer[128];
	cSanMemoryFuncSet::iMemSet(ClockBuffer, 0, 128);

	uint32 Index = 0;
	uint32 Val = 0;
	SString strVal;

	clock_t Rest = Clock;

	auto WriteValToBufferFunc = [&ClockBuffer, &Index, &Rest, &Val, &strVal](int32 Base, int32 Length) //May increase the process time
	{
		Val = Rest / Base;
		Rest = Rest % Base;

		if (Val >= pow(10, Length))
		{
			strVal = ::gloAToT(to_string(Val));
			for (auto c : strVal) { ClockBuffer[Index] = c; Index = Index + 1; }
		}
		else
		{
			for (int32 seek = Length - 1; seek >= 0; seek = seek - 1)
			{
				ClockBuffer[Index + seek] = '0' + (Val % 10);
				Val = Val / 10;
			}

			Index = Index + Length;
		}
	};

	for (int32 seek = 0; seek < Size;seek=seek+1)
	{
		const schar Val = strFormat[seek];

		if (Val != _SSTR('%'))
		{ 
			ClockBuffer[Index] = Val;
			Index = Index + 1;
			continue; 
		}

		seek = seek + 1;

		if (seek >= Size) { break; }

		switch (strFormat[seek])
		{
		case 'H':
			WriteValToBufferFunc(60 * 60 * 1000, 3);
			break;
		case 'M':
			if ((seek != (Size - 1)) && (strFormat[seek + 1] == 'S'))
			{ 
				WriteValToBufferFunc(1, 3);
				seek = seek + 1;

				break;
			}
			WriteValToBufferFunc(60 * 1000, 2);
			break;
		case 'S':
			WriteValToBufferFunc(1000, 2);
			break;
		default:
			ClockBuffer[Index] = strFormat[seek];
			Index = Index + 1;
			break;
		}
	}

	return ClockBuffer;
}