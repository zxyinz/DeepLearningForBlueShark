#include"../Core/cSanLogAgent.h"
#include"SanFileIO.h"
#pragma once
using namespace std;
namespace San
{
	namespace FileIO
	{
#ifndef __CSANFILELOGAGENT_H__
#define __CSANFILELOGAGENT_H__
		class cSanFileLogAgent : public cSanLogAgent
		{
		private:
			SString m_strFilePath;
			ofstream m_OutputFile;
		protected:
			bool _PushLog(const SANLOG &Log) override final;
		public:
			cSanFileLogAgent(const SString &strFilePath, const SString &strLogFormat = DEFAULT_LOG_FORMAT, const SString &strTimeFormat = DEFAULT_TIME_FORMAT, const SString &strClockFormat = DEFAULT_CLOCK_FORMAT);
			cSanFileLogAgent(const SString &strFilePath, const vector<FILITER> &FiliterArray, const SString &strLogFormat = DEFAULT_LOG_FORMAT, const SString &strTimeFormat = DEFAULT_TIME_FORMAT, const SString &strClockFormat = DEFAULT_CLOCK_FORMAT);
			~cSanFileLogAgent();

			bool iCreateAgent() override final;
			void iReleaseAgent() override final;
		};
#endif
	}
}