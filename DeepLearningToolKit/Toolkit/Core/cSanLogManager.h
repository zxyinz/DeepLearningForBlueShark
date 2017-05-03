//Project: San Lab Game Engine
//Version: 2.1.1
//Debug State: Add log agent, log input/output
#include"cSanLogAgent.h"
using namespace std;
namespace San
{
#ifndef __CSANLOGMANAGER_H__
#define __CSANLOGMANAGER_H__

	// Current output only
	// Need add: Log reader, search
	class cSanLogManager
	{
	public:
		typedef SRESULT(*RESPONSEFUNCPTR)(const SANLOG&);
	private:
		SANSYSTEMID	m_strManagerName;
		//SString m_strManagerName;
		//SANSYSTEMID m_SystemID;

		unordered_map<uint32, cSanLogAgent*> m_AgentPtrSet;
		unordered_map<uint32, RESPONSEFUNCPTR> m_ResponseFuncPtrMap; //No lock

		uint32 m_AvailableID;

		//Or one link list
		vector<SANLOG> m_WaitingQueue[2];
		atomic_bool m_bWriteID;
		mutex m_WriteLock;

		mutex m_Lock;

		SAN_UNIQUE_THREAD_MEMBER_DEF_UNSAFE(UpdateProcess, private, public, iUpdate());
	protected:
		uint32 _GenerateResponseCode(const eSANLOGTYPE Type, const uint32 Code) const;
		uint32 _RegisterLogAgent(cSanLogAgent* pAgent);
		void _ReplyResponse(const SANLOG &Log) const;
	public:
		cSanLogManager(const SString &strManagerName = _SSTR("SanLogManager"), const uint32 TimeSlot = 100);
		cSanLogManager(const cSanLogManager &Manager) = delete;
		~cSanLogManager();

		cSanLogManager& operator=(const cSanLogManager &Manager) = delete;

		bool iCreateLogManager(const SString &strManagerName = _SSTR(""));
		void iReleaseLogManager();

		uint32 iRegisterLogAgent(cSanLogAgent* pAgent);
		template<class _AgentType> uint32 iRegisterLogAgentT(const _AgentType &Agent)
		{
			cSanLogAgent* pAgent = new _AgentType(Agent);

			uint32 ID = this->iRegisterLogAgent(pAgent);

			if (ID == 0)
			{
				delete pAgent;
				pAgent = nullptr;
			}

			return ID;
		};

		void iReleaseLogAgent(const uint32 AgentID);

		bool iCreateResponseFunc(const eSANLOGTYPE Type, const uint32 Code, RESPONSEFUNCPTR pFunc);
		void iReleaseResponseFunc(const eSANLOGTYPE Type, const uint32 Code, RESPONSEFUNCPTR pFunc);

		/*Push log to the wait list, unblock*/
		void iPushLog(const SANLOG &Log);

		/*Write log to all valid agent, block*/
		void iWriteLog(const SANLOG &Log);

		//May cause error, like: ptr deleted by 3rd func
		template<class _AgentType = cSanLogAgent> const _AgentType* iGetAgentPtr(const uint32 AgentID) const
		{
			const auto It = this->m_AgentPtrSet.find(AgentID);

			if (It == this->m_AgentPtrSet.end()) { return nullptr; }

			return static_cast<const _AgentType*>(It->second);
		};
		//May cause error, like: ptr deleted by 3rd func or change when unlock the manager
		template<class _AgentType = cSanLogAgent> _AgentType* iGetAgentPtr(const uint32 AgentID)
		{
			const auto It = this->m_AgentPtrSet.find(AgentID);

			if (It == this->m_AgentPtrSet.end()) { return nullptr; }

			return static_cast<_AgentType*>(It->second);
		};

		void iUpdate();

		cSanLogManager& operator<<(const SANLOG &Log);
		cSanLogManager& operator<<(const SANMSG &Msg);
		cSanLogManager& operator<<(const SANWARN &Warn);
		cSanLogManager& operator<<(const SANERR &Err);
		cSanLogManager& operator<<(const SANDBG &Dbg);
		cSanLogManager& operator<<(const SANSYS &Sys);
	};
#endif
}