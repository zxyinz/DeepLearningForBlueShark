//Project: San Lab Game Engine
//Version: 2.1.1
//Debug State: Add functions and need test [buffer operation]
#include"../Core/SanTypes.h"
#include"../Core/SanContainer.h"
#include"../Core/SanMathematics.h"
//#include"../Core/cSanResourceManagerDef.h"
using namespace std;
using namespace San::Mathematics;
#pragma once
namespace San
{
	namespace Device
	{
#ifndef __CSANTERMINALDEVICEWIN_H__
#define __CSANTERMINALDEVICEWIN_H__
		enum eSANTERMINALCOLOR
		{
			STC_BLACK		= 0x0000,
			STC_BLUE		= 0x0001,
			STC_GREEN		= 0x0002,
			STC_CYAN		= 0x0003,
			STC_RED			= 0x0004,
			STC_MAGENTA		= 0x0005,
			STC_YELLOW		= 0x0006,
			STC_GREY		= 0x0007,
			STC_BLUE_HL		= 0x0009,
			STC_GREEN_HL	= 0x000A,
			STC_CYAN_HL		= 0x000B,
			STC_RED_HL		= 0x000C,
			STC_MAGENTA_HL	= 0x000D,
			STC_YELLOW_HL	= 0x000E,
			STC_WHITE		= 0x000F,
			STC_DEFAULT		= 0x00FF,
		};

		_SAN_PAIR_DEF(COLOR_PAIR, eSANTERMINALCOLOR, Font, STC_DEFAULT, eSANTERMINALCOLOR, Background, STC_DEFAULT, , );


		template<class _string>
		struct _TERMINALOUTPUTOBJ
		{
		public:
			_string strString;
			COLOR_PAIR Color;
			SPOINT2 Pos;
			bool bFollowPrevObj;
		public:
			_TERMINALOUTPUTOBJ(const _string &strString = _string(), const COLOR_PAIR &Color = COLOR_PAIR(), const SPOINT2 &Pos = SPOINT2(-1, -1), const bool bFollowPrevObj = false)
				:strString(strString), Color(Color), Pos(Pos), bFollowPrevObj(bFollowPrevObj)
			{
			};
			~_TERMINALOUTPUTOBJ() {};
		};

		typedef _TERMINALOUTPUTOBJ<SStringA> TERMINALOUTPUTOBJA;
		typedef _TERMINALOUTPUTOBJ<SStringW> TERMINALOUTPUTOBJW;

		#ifndef _UNICODE
		typedef TERMINALOUTPUTOBJA TERMINALOUTPUTOBJ;
		#else
		typedef TERMINALOUTPUTOBJW TERMINALOUTPUTOBJ;
		#endif

#ifdef _WINDOWS
		class cSanTerminalDeviceWin
		{
		private:
			SString m_strTerminalName;
			uint32 m_TerminalID;
			SHANDLE m_TerminalHandle;
			SString m_strTittle;
			SStringA m_strCodeLocate;
			eSANTERMINALCOLOR m_TextDefColor;
			eSANTERMINALCOLOR m_BackgroundDefColor;
			_sstream<schar> m_TerminalBuffer;//////////////
			SString m_strBuffer;
			static mutex m_TerminalLock;
		protected:
			void _CreateConsoleDevice();
			void _OutputString(const SStringA &strString, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const;
			void _OutputString(const SStringW &strString, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const;
		public:
			cSanTerminalDeviceWin(SString strTerminalName = _SSTR("SanLabTerminal"), SString strTerminalTittle = _SSTR("San Lab Terminal"));
			~cSanTerminalDeviceWin();

			void iSetTerminalTittle(SString strTittle);
			void iSetTerminalCodeLocate(SStringA strLocate);

			void iSetDefaultTextColor(eSANTERMINALCOLOR Color);
			void iSetDefaultBackgroundColor(eSANTERMINALCOLOR Color);

			bool iSetTerminalBufferSize(uint32 BufferSize);

			void iSetCursorPosition(const SPOINT2 &Pos) const;

			SString iGetTerminalTittle() const;
			SStringA iGetTerminalCodeLocate() const;

			eSANTERMINALCOLOR iGetDefaultTextColor() const;
			eSANTERMINALCOLOR iGetDefaultBackgroundColor() const;

			uint32 iGetTerminalBufferSize() const;
			SPOINT2 iGetCursorPosition() const;

			SString iGetInputString(schar DelLim = _SSTR('\n'));

			void iOutputString(const SStringA &strString, eSANTERMINALCOLOR TextColor = STC_DEFAULT, eSANTERMINALCOLOR BackgroundColor = STC_DEFAULT) const;
			void iOutputString(const SStringW &strString, eSANTERMINALCOLOR TextColor = STC_DEFAULT, eSANTERMINALCOLOR BackgroundColor = STC_DEFAULT) const;

			void iOutputString(const SStringA &strString, const SPOINT2 &Pos, eSANTERMINALCOLOR TextColor = STC_DEFAULT, eSANTERMINALCOLOR BackgroundColor = STC_DEFAULT) const;
			void iOutputString(const SStringW &strString, const SPOINT2 &Pos, eSANTERMINALCOLOR TextColor = STC_DEFAULT, eSANTERMINALCOLOR BackgroundColor = STC_DEFAULT) const;

			void iOutputStringLine(const SStringA &strString, eSANTERMINALCOLOR TextColor = STC_DEFAULT, eSANTERMINALCOLOR BackgroundColor = STC_DEFAULT) const;
			void iOutputStringLine(const SStringW &strString, eSANTERMINALCOLOR TextColor = STC_DEFAULT, eSANTERMINALCOLOR BackgroundColor = STC_DEFAULT) const;

			void iOutputStringLine(const SStringA &strString, const SPOINT2 &Pos, eSANTERMINALCOLOR TextColor = STC_DEFAULT, eSANTERMINALCOLOR BackgroundColor = STC_DEFAULT) const;
			void iOutputStringLine(const SStringW &strString, const SPOINT2 &Pos, eSANTERMINALCOLOR TextColor = STC_DEFAULT, eSANTERMINALCOLOR BackgroundColor = STC_DEFAULT) const;

			void iOutput(const TERMINALOUTPUTOBJA &Obj) const;
			void iOutput(const TERMINALOUTPUTOBJW &Obj) const;

			void iOutput(const vector<TERMINALOUTPUTOBJA> &ObjList) const;
			void iOutput(const vector<TERMINALOUTPUTOBJW> &ObjList) const;

			void iOutputLine(const TERMINALOUTPUTOBJA &Obj) const;
			void iOutputLine(const TERMINALOUTPUTOBJW &Obj) const;

			//void iOutputLine(const vector<TERMINALOUTPUTOBJA> &ObjList) const;
			//void iOutputLine(const vector<_TERMINALOUTPUTOBJ<SStringW>> &ObjList) const;

			void iClearLine();/////////////////////
			void iClearScreen() const;

			cSanTerminalDeviceWin& operator<<(const SStringA &strString);
			cSanTerminalDeviceWin& operator<<(const SStringW &strString);

			cSanTerminalDeviceWin& operator<<(const TERMINALOUTPUTOBJA &Obj);
			cSanTerminalDeviceWin& operator<<(const TERMINALOUTPUTOBJW &Obj);

			cSanTerminalDeviceWin& operator<<(const vector<TERMINALOUTPUTOBJA> &ObjList);
			cSanTerminalDeviceWin& operator<<(const vector<TERMINALOUTPUTOBJW> &ObjList);

//#ifndef _UNICODE
//#define iOutputString iOutputStringA
//#define iOutputStringLine iOutputStringLineA
//#else
//#define iOutputString iOutputStringW
//#define iOutputStringLine iOutputStringLineW
//#endif
		};

		//typedef cSanTerminalDeviceWin cSanTerminalDevice;
#define cSanTerminalDevice cSanTerminalDeviceWin

#endif
#endif
	}
}