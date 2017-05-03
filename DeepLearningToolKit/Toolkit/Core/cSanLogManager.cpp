#include"cSanLogManager.h"
using namespace std;
using namespace San;
cSanLogManager::cSanLogManager(const SString &strManagerName, const uint32 TimeSlot)
	:m_strManagerName(strManagerName),
	m_AvailableID(1), m_UpdateProcessTimeSlot(TimeSlot), m_UpdateProcessState(STS_TERMINATED), m_UpdateProcessCount(0)
{
	m_bWriteID = false;
}
cSanLogManager::~cSanLogManager()
{
	this->iReleaseLogManager();
}
uint32 cSanLogManager::_GenerateResponseCode(const eSANLOGTYPE Type, const uint32 Code) const
{
	return (static_cast<uint32>(Type) << 12) | (Code & 0x0fff);
}
uint32 cSanLogManager::_RegisterLogAgent(cSanLogAgent* pAgent)
{
	if (pAgent == nullptr){ return 0; }

	unique_lock<mutex> Lock(this->m_Lock);

	this->m_AgentPtrSet[this->m_AvailableID] = pAgent;
	this->m_AvailableID = this->m_AvailableID + 1;

	return this->m_AgentPtrSet.size();
}
void cSanLogManager::_ReplyResponse(const SANLOG &Log) const
{
	//Lock?
	const auto It = this->m_ResponseFuncPtrMap.find(this->_GenerateResponseCode(Log.Type, Log.Code));
	if (It != this->m_ResponseFuncPtrMap.end())
	{ 
		if (It->second != nullptr) { It->second(Log); }
	}
}
bool cSanLogManager::iCreateLogManager(const SString &strManagerName)
{
	/*if (::gloFoundVector(strManagerName)){ return false; }

	if (::gloRegisterVector(strManagerName, VT_SYS | VT_VAR, (SHANDLE)this))
	{
		this->m_strManagerName = strManagerName;
		return true;
	}

	return false;*/
	return true;
}
void cSanLogManager::iReleaseLogManager()
{
	//if (!this->m_strManagerName.empty()){ ::gloDestoryVector(this->m_strManagerName, VT_SYS | VT_VAR); }

	/*list<cSanLogAgent*>::iterator pAgent = this->m_AgentPtrSet.begin();
	while (pAgent != this->m_AgentPtrSet.end())
	{
		delete *pAgent;
		pAgent++;
	}*/

	this->iSetUpdateProcessState(STS_TERMINATED);

	for (auto &Item : this->m_AgentPtrSet)
	{ 
		if (Item.second == nullptr) { continue; }

		Item.second->iReleaseAgent();

		delete Item.second;
		Item.second = nullptr;
	}

	this->m_AgentPtrSet.clear();
	this->m_ResponseFuncPtrMap.clear();

	this->m_WaitingQueue[0].clear();
	this->m_WaitingQueue[1].clear();
}
void cSanLogManager::iReleaseLogAgent(const uint32 AgentID)
{
	unique_lock<mutex> Lock(this->m_Lock);

	auto It = this->m_AgentPtrSet.find(AgentID);

	if (It == this->m_AgentPtrSet.end()) { return; }
	if (It->second == nullptr) { return; }

	It->second->iReleaseAgent();

	delete It->second;
	It->second = nullptr;

	this->m_AgentPtrSet.erase(It);
}
uint32 cSanLogManager::iRegisterLogAgent(cSanLogAgent* pAgent)
{
	if (pAgent == nullptr) { return 0; }

	if (!pAgent->iCreateAgent()) { return 0; }

	const uint32 ID = this->_RegisterLogAgent(pAgent);
	if (ID == 0) { pAgent->iReleaseAgent(); }

	return ID;
}
bool cSanLogManager::iCreateResponseFunc(const eSANLOGTYPE Type, const uint32 Code, RESPONSEFUNCPTR pFunc)
{
	if ((Type == SLT_MASK) || (pFunc == nullptr)){ return false; }

	if (this->m_ResponseFuncPtrMap.find(Code) != this->m_ResponseFuncPtrMap.end())
	{
		this->m_ResponseFuncPtrMap[this->_GenerateResponseCode(Type, Code)] = pFunc;
		return true;
	}

	return false;
}
void cSanLogManager::iReleaseResponseFunc(const eSANLOGTYPE Type, const uint32 Code, RESPONSEFUNCPTR pFunc)
{
	if (Type == SLT_MASK){ return; }

	auto It = this->m_ResponseFuncPtrMap.find(this->_GenerateResponseCode(Type, Code));
	if (It != this->m_ResponseFuncPtrMap.end()){ this->m_ResponseFuncPtrMap.erase(It); }
}
void cSanLogManager::iPushLog(const SANLOG &Log)
{
	if (this->m_UpdateProcessState != STS_RUNNING) { return; }
	if (Log.Type >= SLT_MASK){ return; }

	this->m_WriteLock.lock();
	this->m_WaitingQueue[this->m_bWriteID].push_back(Log);
	this->m_WriteLock.unlock();

	this->_ReplyResponse(Log);
}
void cSanLogManager::iWriteLog(const SANLOG &Log)
{
	this->iPushLog(Log);
	this->iUpdate();
}
void cSanLogManager::iUpdate()
{
	unique_lock<mutex> Lock(this->m_Lock); // For other _Update func

	const bool ReadID = this->m_bWriteID;
	this->m_bWriteID = this->m_bWriteID ^ true;

	for (const auto &Log : this->m_WaitingQueue[ReadID])
	{
		for (const auto &AgentPair : this->m_AgentPtrSet)
		{
			if (AgentPair.second != nullptr) { AgentPair.second->iPushLog(Log); }
		}
		//this->_ReplyResponse(*pLog); //Mayube too late, move to push
	}

	this->m_WaitingQueue[ReadID].clear();
}
cSanLogManager& cSanLogManager::operator<<(const SANLOG &Log)
{
	this->iPushLog(Log);
	return *this;
}
cSanLogManager& cSanLogManager::operator<<(const SANMSG &Msg)
{
	this->iPushLog(Msg);
	return *this;
}
cSanLogManager& cSanLogManager::operator<<(const SANWARN &Warn)
{
	this->iPushLog(Warn);
	return *this;
}
cSanLogManager& cSanLogManager::operator<<(const SANERR &Err)
{
	this->iPushLog(Err);
	return *this;
}
cSanLogManager& cSanLogManager::operator<<(const SANDBG &Dbg)
{
	this->iPushLog(Dbg);
	return *this;
}
cSanLogManager& cSanLogManager::operator<<(const SANSYS &Sys)
{
	this->iPushLog(Sys);
	return *this;
}