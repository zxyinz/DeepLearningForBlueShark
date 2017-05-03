//Project: San Lab Game Engine
//Version: 2.1.1
//Debug State: Add functions and need test [buffer operation]
#include"../Core/SanTypes.h"
#include"../Core/SanContainer.h"
#include"../Core/SanMathematics.h"
//#include"../Core/cSanResourceManagerDef.h"
#include"cSanTerminalDeviceWin.h"
using namespace std;
using namespace San::Mathematics;
#pragma once
namespace San
{
	namespace Device
	{
#ifndef __CSANTERMINALDEVICEUNIX_H__
#define __CSANTERMINALDEVICEUNIX_H__

#ifdef _UNIX
		static const int32 TerminalColorTable[] = { 30, 34, 32, 36, 31, 35, 33, 30, 34, 32, 36, 31, 35, 33, 37 };

		class cSanTerminalDeviceUnix
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
			void _OutputStringA(SStringA strString, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const;
			void _OutputStringW(SStringW strString, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const;
		public:
			cSanTerminalDeviceUnix(SString strTerminalName = _SSTR("SanLabTerminal"), SString strTerminalTittle = _SSTR("San Lab Terminal"));
			~cSanTerminalDeviceUnix();

			void iSetTerminalTittle(SString strTittle);
			void iSetTerminalCodeLocate(SStringA strLocate);

			void iSetDefaultTextColor(eSANTERMINALCOLOR Color);
			void iSetDefaultBackgroundColor(eSANTERMINALCOLOR Color);

			bool iSetTerminalBufferSize(uint32 BufferSize);

			SString iGetTerminalTittle() const;
			SStringA iGetTerminalCodeLocate() const;

			eSANTERMINALCOLOR iGetDefaultTextColor() const;
			eSANTERMINALCOLOR iGetDefaultBackgroundColor() const;

			uint32 iGetTerminalBufferSize() const;
			SPOINT2 iGetCurrentCursorPosition() const;

			SString iGetInputString(schar DelLim = _SSTR('\n'));

			void iOutputStringA(SStringA strString, eSANTERMINALCOLOR TextColor = STC_DEFAULT, eSANTERMINALCOLOR BackgroundColor = STC_DEFAULT) const;
			void iOutputStringW(SStringW strString, eSANTERMINALCOLOR TextColor = STC_DEFAULT, eSANTERMINALCOLOR BackgroundColor = STC_DEFAULT) const;

			void iOutputStringA(SStringA strString, const SPOINT2 &Pos, eSANTERMINALCOLOR TextColor = STC_DEFAULT, eSANTERMINALCOLOR BackgroundColor = STC_DEFAULT) const;
			void iOutputStringW(SStringW strString, const SPOINT2 &Pos, eSANTERMINALCOLOR TextColor = STC_DEFAULT, eSANTERMINALCOLOR BackgroundColor = STC_DEFAULT) const;

			void iOutputStringLineA(SStringA strString, eSANTERMINALCOLOR TextColor = STC_DEFAULT, eSANTERMINALCOLOR BackgroundColor = STC_DEFAULT) const;
			void iOutputStringLineW(SStringW strString, eSANTERMINALCOLOR TextColor = STC_DEFAULT, eSANTERMINALCOLOR BackgroundColor = STC_DEFAULT) const;

			void iOutputStringLineA(SStringA strString, const SPOINT2 &Pos, eSANTERMINALCOLOR TextColor = STC_DEFAULT, eSANTERMINALCOLOR BackgroundColor = STC_DEFAULT) const;
			void iOutputStringLineW(SStringW strString, const SPOINT2 &Pos, eSANTERMINALCOLOR TextColor = STC_DEFAULT, eSANTERMINALCOLOR BackgroundColor = STC_DEFAULT) const;

			void iClearLine();/////////////////////
			void iClearScreen() const;

			cSanTerminalDeviceUnix& operator<<(const SStringA &strString);
			cSanTerminalDeviceUnix& operator<<(const SStringW &strString);

#ifndef _UNICODE
#define iOutputString iOutputStringA
#define iOutputStringLine iOutputStringLine
#else
#define iOutputString iOutputStringW
#define iOutputStringLine iOutputStringLineW
#endif
		};

		//typedef cSanTerminalDeviceUnix cSanTerminalDevice;
#define cSanTerminalDevice cSanTerminalDeviceUnix

#endif
#endif
	}
}