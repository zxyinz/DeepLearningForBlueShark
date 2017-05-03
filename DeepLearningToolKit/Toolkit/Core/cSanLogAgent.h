//Project: San Lab Game Engine
//Version: 2.1.1
//Debug State: Add log agent, log input/output
#include"SanTypes.h"
#include"SanContainer.h"
using namespace std;
using namespace chrono;
namespace San
{
#ifndef __CSANLOGAGENT_H__
#define __CSANLOGAGENT_H__
	enum eSANLOGTYPE
	{
		SLT_LOG		= 0x01,
		SLT_MSG		= 0x02,
		SLT_WARN	= 0x04,
		SLT_ERR		= 0x08,
		SLT_DBG		= 0x10,
		SLT_SYS		= 0x20,
		SLT_MASK	= 0x2f,
	};

	struct SANLOG
	{
	public:
		eSANLOGTYPE	Type;
		uint32	Level;
		uint32	Code;
		SString	strLog;
		system_clock::time_point TimeStamp;
		clock_t ClockStamp;

		//Code location
		SString	strFile;
		uint32	Line;
		SHANDLE	MemAddr;

		/*User data, for response func*/
		_sstream<uint8>	UserData;
	public:
		SANLOG();
		SANLOG(const eSANLOGTYPE Type, const SString &strLog, const uint32 Level = 0, const uint32 Code = 0, const SString &strFile = _SSTR(""), const uint32 Line = 0, const SHANDLE MemAddr = nullptr, const _sstream<uint8> &UserData = _sstream<uint8>());
		~SANLOG();
	};
	typedef SANLOG*	lpSANLOG;

	struct SANMSG : public SANLOG
	{
	public:
		SANMSG(const SString &strLog, const uint32 Level = 0, const uint32 Code = 0, const SString &strFile = _SSTR(""), const uint32 Line = 0, const SHANDLE MemAddr = nullptr, const _sstream<uint8> &UserData = _sstream<uint8>());
		~SANMSG();
	};

	struct SANWARN : public SANLOG
	{
	public:
		SANWARN(const SString &strLog, const uint32 Level = 0, const uint32 Code = 0, const SString &strFile = _SSTR(""), const uint32 Line = 0, const SHANDLE MemAddr = nullptr, const _sstream<uint8> &UserData = _sstream<uint8>());
		~SANWARN();
	};

	struct SANERR : public SANLOG
	{
	public:
		SANERR(const SString &strLog, const uint32 Level = 0, const uint32 Code = 0, const SString &strFile = _SSTR(""), const uint32 Line = 0, const SHANDLE MemAddr = nullptr, const _sstream<uint8> &UserData = _sstream<uint8>());
		~SANERR();
	};

	struct SANDBG : public SANLOG
	{
	public:
		SANDBG(const SString &strLog, const uint32 Level = 0, const uint32 Code = 0, const SString &strFile = _SSTR(""), const uint32 Line = 0, const SHANDLE MemAddr = nullptr, const _sstream<uint8> &UserData = _sstream<uint8>());
		~SANDBG();
	};

	struct SANSYS : public SANLOG
	{
	public:
		SANSYS(const SString &strLog, const uint32 Level = 0, const uint32 Code = 0, const SString &strFile = _SSTR(""), const uint32 Line = 0, const SHANDLE MemAddr = nullptr, const _sstream<uint8> &UserData = _sstream<uint8>());
		~SANSYS();
	};

	// %T: Time stamp
	// %C: Clock stamp
	// %t: Tag
	// %l: Level
	// %c: Message
	// %d: Debug info
	// %D: Debug info in new line

	/*const SString DEFAULT_LOG_FORMAT = _SSTR([ "%T ] [ %C ] [ %t ] %l %c %m %D");
	const SString DEFAULT_TIME_FORMAT = _SSTR("%a %b %d %H:%M:%S %Y");
	const SString DEFAULT_CLOCK_FORMAT = _SSTR("%H:%M:%S:%MS");//*/

	#define DEFAULT_LOG_FORMAT _SSTR("[ %T ] [ %C ] [ %t ] Level:%l Code:%c %m %D")
	#define DEFAULT_TIME_FORMAT _SSTR("%a %b %d %H:%M:%S %Y")
	#define DEFAULT_CLOCK_FORMAT _SSTR("%H:%M:%S:%MS")//*/

	//Add or delete filters
	class cSanLogAgent
	{
	public:
		struct FILITER
		{
		public:
			uint32 Type;
			_srange<uint32> LevelRange;
			_srange<uint32> CodeRange;
			_srange<time_t> TimeRange;
			_srange<clock_t> ClockRange;
			bool bAccept;
		public:
			FILITER(const uint32 Type = SLT_MASK, const _srange<uint32> LevelRange = _srange<uint32>(0, 0), const _srange<uint32> CodeRange = _srange<uint32>(0, 0), const _srange<time_t> TimeRange = _srange<time_t>(0, 0), const _srange<clock_t> ClockRange = _srange<clock_t>(0, 0), const bool bAccept = true);
			~FILITER();

			bool operator==(const FILITER &Filiter) const;
			bool operator!=(const FILITER &Filiter) const;
		};

		enum eITEMTYPE { IT_TIME, IT_CLOCK, IT_TAG, IT_LEVEL, IT_CODE, IT_MSG, IT_DEBUG, IT_DEFAULT, IT_COUNT };

		_SAN_PAIR_DEF(LOG_ITEM, eITEMTYPE, Type, IT_DEFAULT, SString, strVal, _SSTR(""), , );
	protected:
		vector<FILITER> m_FiliterArray; //Order important
		unordered_map<eSANLOGTYPE, SString> m_TagStringSet;
		SString m_strLogFormat;
		SString m_strTimeFormat;
		SString m_strClockFormat;
	protected:
		virtual bool _LogAcceptable(const SANLOG &Log) const;
		virtual bool _PushLog(const SANLOG &Log) = 0;
	public:
		cSanLogAgent(const SString &strLogFormat = DEFAULT_LOG_FORMAT, const SString &strTimeFormat = DEFAULT_TIME_FORMAT, const SString &strClockFormat = DEFAULT_CLOCK_FORMAT);
		cSanLogAgent(const vector<FILITER> &FiliterArray, const SString &strLogFormat = DEFAULT_LOG_FORMAT, const SString &strTimeFormat = DEFAULT_TIME_FORMAT, const SString &strClockFormat = DEFAULT_CLOCK_FORMAT);
		~cSanLogAgent();

		virtual bool iCreateAgent();
		virtual void iReleaseAgent();

		bool iAddFilter(const FILITER &Filiter, const bool bToFront = false);
		
		template<template<class> class _container>
		uint32 iAddFiliter(const typename _container<FILITER>::const_iterator &Begin, const typename _container<FILITER>::const_iterator &End, const bool bToFront = false)
		{
			uint32 Count = 0;
			for (auto It = Begin; It != End; It++) { Count = Count + (this->iAddFiliter(*It, bToFront) ? 1 : 0); }
			return Count;
		};

		bool iDeleteFilter(const FILITER &Filiter);
		
		template<template<class> class _container>
		uint32 iDeleteFiliter(const typename _container<FILITER>::const_iterator &Begin, const typename _container<FILITER>::const_iterator &End)
		{
			uint32 Count = 0;
			for (auto It = Begin; It != End; It++){ Count = Count + (this->iDeleteFiliter(*It) ? 1 : 0); }
			return Count;
		};

		void iSetLogFormat(const SString &strLogFormat = DEFAULT_LOG_FORMAT);
		void iSetTimeFormat(const SString &strTimeFormat = DEFAULT_TIME_FORMAT);
		void iSetClockFormat(const SString &strClockFormat = DEFAULT_CLOCK_FORMAT);

		void iUpdateTagString(const eSANLOGTYPE Type, const SString &strString);

		void iResetTagString();
		void iResetTagString(const eSANLOGTYPE Type);

		SString iGetLogFormat() const;
		SString iGetTimeFormat() const;
		SString iGetClockFormat() const;

		SString iGetTagString(const eSANLOGTYPE Type) const;

		bool iPushLog(const SANLOG &Log);

		SString iGenerateLogString(const SANLOG &Log) const; //Faster
		vector<LOG_ITEM> iGenerateLogStringPairSet(const SANLOG &Log) const;

		static SString iTimeToString(const system_clock::time_point &Time, const SString &strFormat = DEFAULT_TIME_FORMAT);
		static SString iClockToString(const clock_t Clock, const SString &strFormat = DEFAULT_CLOCK_FORMAT);
	};

#endif
}