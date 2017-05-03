//Project: San Lab Game Engine
//Version: 2.1.1
//Debug State: Need test
#include"SanContainerDef.h"
using namespace std;
#pragma once
namespace San
{
#ifndef __STDSANTHREAD_H__
#define __STDSANTHREAD_H__
	enum eSANTHREADSTATE
	{
		STS_RUNNING,
		STS_STOPPED,
		STS_TERMINATED
	};

	//Unsafe, may change the next variable/function type, init value
#define SAN_UNIQUE_THREAD_MEMBER_DEF_UNSAFE(__NAME__, __VARTYPE__, __FUNCTYPE__, __UPDATEFUNC__)\
	__VARTYPE__:\
		uint32 m_##__NAME__##TimeSlot;\
		eSANTHREADSTATE m_##__NAME__##State;\
		atomic<uint32> m_##__NAME__##Count;\
		mutex m_##__NAME__##Lock;\
	__FUNCTYPE__:\
		void iSet##__NAME__##State(const eSANTHREADSTATE State = STS_RUNNING)\
		{\
			this->m_##__NAME__##State = State;\
			switch(this->m_##__NAME__##State)\
			{\
			case STS_RUNNING:\
				iStart##__NAME__##Thread(); return;\
			case STS_TERMINATED:\
				while(m_##__NAME__##Count != 0) /*Read*/\
				{\
					::Sleep(this->m_##__NAME__##TimeSlot);\
				}\
			case STS_STOPPED:\
			default:\
				break;\
			}\
		};\
		void iSet##__NAME__##TimeSlot(const uint32 TimeSlot) { this->m_##__NAME__##TimeSlot = max(static_cast<uint32>(1), TimeSlot); };\
		\
		eSANTHREADSTATE iGet##__NAME__##State() const { return this->m_##__NAME__##State; };\
		uint32 iGet##__NAME__##TimeSlot() const { return this->m_##__NAME__##TimeSlot; };\
		\
		void iStart##__NAME__##Thread()\
		{\
			unique_lock<mutex> Lock(this->m_##__NAME__##Lock);\
			\
			if (this->m_##__NAME__##Count != 0) { return; }\
			\
			auto UpdateFunc = [this]()\
			{\
				this->m_##__NAME__##Count++;\
				\
				this->m_##__NAME__##State = STS_RUNNING;\
				while (this->m_##__NAME__##State != STS_TERMINATED)\
				{\
					if (this->m_##__NAME__##State == STS_RUNNING) { __UPDATEFUNC__; }\
					::Sleep(this->m_##__NAME__##TimeSlot);\
				}\
				this->m_##__NAME__##Count--; /*No confliction*/\
			};\
			\
			thread UpdateThread(UpdateFunc);\
			UpdateThread.detach();\
			\
			while(this->m_##__NAME__##Count == 0) { ::Sleep(this->m_##__NAME__##TimeSlot); }\
		}
#endif
}
