#include"../Core/cSanLogAgent.h"
#include"cSanTerminalDeviceWin.h"
#pragma once
using namespace std;
namespace San
{
	namespace Device
	{
#ifndef __CSANTERMINALLOGAGENT_H__
#define __CSANTEMRINALLOGAGENT_H__
		class cSanTerminalLogAgent : public cSanLogAgent
		{
		private:
			cSanTerminalDevice* m_pTerminal;
			unordered_map<eSANLOGTYPE, COLOR_PAIR> m_TagColorSet;
			unordered_map<eITEMTYPE, COLOR_PAIR> m_ItemColorSet;
		protected:
			bool _PushLog(const SANLOG &Log) override final;
		public:
			cSanTerminalLogAgent(cSanTerminalDevice* pTerminal, const SString &strLogFormat = DEFAULT_LOG_FORMAT, const SString &strTimeFormat = DEFAULT_TIME_FORMAT, const SString &strClockFormat = DEFAULT_CLOCK_FORMAT);
			cSanTerminalLogAgent(cSanTerminalDevice* pTerminal, const vector<FILITER> &FiliterArray, const SString &strLogFormat = DEFAULT_LOG_FORMAT, const SString &strTimeFormat = DEFAULT_TIME_FORMAT, const SString &strClockFormat = DEFAULT_CLOCK_FORMAT);
			~cSanTerminalLogAgent();

			bool iCreateAgent() override final;
			void iReleaseAgent() override final;

			void iSetTagColor(const eSANLOGTYPE Type, const eSANTERMINALCOLOR FontColor, const eSANTERMINALCOLOR BackgroundColor);
			void iSetItemColor(const eITEMTYPE Type, const eSANTERMINALCOLOR FontColor, const eSANTERMINALCOLOR BackgroundColor);

			void iResetTagColor();
			void iResetTagColor(const eSANLOGTYPE Type);

			void iResetItemColor();
			void iResetItemColor(const eITEMTYPE Type);

			COLOR_PAIR iGetTagColor(const eSANLOGTYPE Type) const;
			COLOR_PAIR iGetItemColor(const eITEMTYPE Type) const;
		};
#endif
	}
}