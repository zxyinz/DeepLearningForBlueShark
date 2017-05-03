#include"cSanFileLogAgent.h"
using namespace std;
using namespace San;
using namespace San::FileIO;
cSanFileLogAgent::cSanFileLogAgent(const SString &strFilePath, const SString &strLogFormat, const SString &strTimeFormat, const SString &strClockFormat)
	:cSanLogAgent(strLogFormat, strTimeFormat, strClockFormat),
	m_strFilePath(strFilePath)
{
}
cSanFileLogAgent::cSanFileLogAgent(const SString &strFilePath, const vector<FILITER> &FiliterArray, const SString &strLogFormat, const SString &strTimeFormat, const SString &strClockFormat)
	:cSanLogAgent(FiliterArray, strLogFormat, strTimeFormat, strClockFormat),
	m_strFilePath(strFilePath)
{
}
cSanFileLogAgent::~cSanFileLogAgent()
{
}
bool cSanFileLogAgent::_PushLog(const SANLOG &Log)
{
	if (!this->m_OutputFile.is_open()) { return false; }

	this->m_OutputFile << this->iGenerateLogString(Log) + _SSTR("\r\n");

	return true;
}
bool cSanFileLogAgent::iCreateAgent()
{
	if (this->m_strFilePath.empty()) { return false; }

	this->m_OutputFile.open(this->m_strFilePath.c_str(), ios_base::out);
	return this->m_OutputFile.is_open();
}
void cSanFileLogAgent::iReleaseAgent()
{
	this->m_strFilePath.clear();

	if (this->m_OutputFile.is_open())
	{
		this->m_OutputFile.close();
	}
}