#include"cSanTerminalLogAgent.h"
using namespace std;
using namespace San;
using namespace San::Device;
cSanTerminalLogAgent::cSanTerminalLogAgent(cSanTerminalDevice* pTerminal, const SString &strLogFormat, const SString &strTimeFormat, const SString &strClockFormat)
	:cSanLogAgent(strLogFormat, strTimeFormat, strClockFormat),
	m_pTerminal(pTerminal)
{
	this->iResetTagColor();
	this->iResetItemColor();
}
cSanTerminalLogAgent::cSanTerminalLogAgent(cSanTerminalDevice* pTerminal, const vector<FILITER> &FiliterArray, const SString &strLogFormat, const SString &strTimeFormat, const SString &strClockFormat)
	:cSanLogAgent(FiliterArray, strLogFormat, strTimeFormat, strClockFormat),
	m_pTerminal(pTerminal)
{
	this->iResetTagColor();
	this->iResetItemColor();
}
cSanTerminalLogAgent::~cSanTerminalLogAgent()
{
}
bool cSanTerminalLogAgent::iCreateAgent()
{
	return true;
}
void cSanTerminalLogAgent::iReleaseAgent()
{
	this->m_pTerminal = nullptr;
}
bool cSanTerminalLogAgent::_PushLog(const SANLOG &Log)
{
	if (this->m_pTerminal == nullptr) { return false; }

	auto PairSet = this->iGenerateLogStringPairSet(Log);
	COLOR_PAIR Color;

	vector<TERMINALOUTPUTOBJ> ObjList;
	ObjList.reserve(PairSet.size() + 1);

	for (auto &Pair : PairSet)
	{
		Color = Pair.Type == IT_TAG ? this->iGetTagColor(Log.Type) : this->iGetItemColor(Pair.Type);
		ObjList.push_back(TERMINALOUTPUTOBJ(Pair.strVal, Color));
	}
	ObjList.push_back(TERMINALOUTPUTOBJ(_SSTR("\r\n")));

	(*this->m_pTerminal) << ObjList;

	return true;
}
void cSanTerminalLogAgent::iSetTagColor(const eSANLOGTYPE Type, const eSANTERMINALCOLOR FontColor, const eSANTERMINALCOLOR BackgroundColor)
{
	auto It = this->m_TagColorSet.find(Type);
	if (It == this->m_TagColorSet.end()) { return; }

	It->second = COLOR_PAIR(FontColor, BackgroundColor);
}
void cSanTerminalLogAgent::iSetItemColor(const cSanTerminalLogAgent::eITEMTYPE Type, const eSANTERMINALCOLOR FontColor, const eSANTERMINALCOLOR BackgroundColor)
{
	auto It = this->m_ItemColorSet.find(Type);
	if (It == this->m_ItemColorSet.end()) { return; }

	It->second = COLOR_PAIR(FontColor, BackgroundColor);
}
void cSanTerminalLogAgent::iResetTagColor()
{
	this->m_TagColorSet.clear();

	this->iResetTagColor(SLT_LOG);
	this->iResetTagColor(SLT_MSG);
	this->iResetTagColor(SLT_WARN);
	this->iResetTagColor(SLT_ERR);
	this->iResetTagColor(SLT_DBG);
	this->iResetTagColor(SLT_SYS);
}
void cSanTerminalLogAgent::iResetTagColor(const eSANLOGTYPE Type)
{
	switch (Type)
	{
	case SLT_LOG: this->m_TagColorSet[SLT_LOG] = COLOR_PAIR(STC_DEFAULT, STC_DEFAULT); break;
	case SLT_MSG: this->m_TagColorSet[SLT_MSG] = COLOR_PAIR(STC_CYAN_HL, STC_DEFAULT); break;
	case SLT_WARN: this->m_TagColorSet[SLT_WARN] = COLOR_PAIR(STC_YELLOW_HL, STC_DEFAULT); break;
	case SLT_ERR: this->m_TagColorSet[SLT_ERR] = COLOR_PAIR(STC_RED_HL, STC_DEFAULT); break;
	case SLT_DBG: this->m_TagColorSet[SLT_DBG] = COLOR_PAIR(STC_GREY, STC_DEFAULT); break;
	case SLT_SYS: this->m_TagColorSet[SLT_SYS] = COLOR_PAIR(STC_MAGENTA_HL, STC_DEFAULT); break;
	default: break;
	}
}
void cSanTerminalLogAgent::iResetItemColor()
{
	this->m_ItemColorSet.clear();

	this->iResetItemColor(IT_TIME);
	this->iResetItemColor(IT_CLOCK);
	this->iResetItemColor(IT_TAG);
	this->iResetItemColor(IT_LEVEL);
	this->iResetItemColor(IT_CODE);
	this->iResetItemColor(IT_MSG);
	this->iResetItemColor(IT_DEBUG);
	this->iResetItemColor(IT_DEFAULT);
}
void cSanTerminalLogAgent::iResetItemColor(const eITEMTYPE Type)
{
	switch (Type)
	{
	case IT_TIME: this->m_ItemColorSet[IT_TIME] = COLOR_PAIR(STC_DEFAULT, STC_DEFAULT); break;
	case IT_CLOCK: this->m_ItemColorSet[IT_CLOCK] = COLOR_PAIR(STC_DEFAULT, STC_DEFAULT); break;
	case IT_TAG: this->m_ItemColorSet[IT_TAG] = COLOR_PAIR(STC_DEFAULT, STC_DEFAULT); break;
	case IT_LEVEL: this->m_ItemColorSet[IT_LEVEL] = COLOR_PAIR(STC_DEFAULT, STC_DEFAULT); break;
	case IT_CODE: this->m_ItemColorSet[IT_CODE] = COLOR_PAIR(STC_DEFAULT, STC_DEFAULT); break;
	case IT_MSG: this->m_ItemColorSet[IT_MSG] = COLOR_PAIR(STC_DEFAULT, STC_DEFAULT); break;
	case IT_DEBUG: this->m_ItemColorSet[IT_DEBUG] = COLOR_PAIR(STC_GREY, STC_DEFAULT); break;
	case IT_DEFAULT: this->m_ItemColorSet[IT_DEFAULT] = COLOR_PAIR(STC_DEFAULT, STC_DEFAULT); break;
	default: break;
	}
}
COLOR_PAIR cSanTerminalLogAgent::iGetTagColor(const eSANLOGTYPE Type) const
{
	const auto It = this->m_TagColorSet.find(Type);
	if (It == this->m_TagColorSet.end()) { return COLOR_PAIR(STC_DEFAULT, STC_DEFAULT); }

	return It->second;
}
COLOR_PAIR cSanTerminalLogAgent::iGetItemColor(const cSanTerminalLogAgent::eITEMTYPE Type) const
{
	const auto It = this->m_ItemColorSet.find(Type);
	if (It == this->m_ItemColorSet.end()) { return COLOR_PAIR(STC_DEFAULT, STC_DEFAULT); }

	return It->second;
}