#include"cSanTerminalDeviceUnix.h"
using namespace std;
using namespace San;
using namespace San::Device;
#ifdef _UNIX
mutex cSanTerminalDeviceUnix::m_TerminalLock;
cSanTerminalDeviceUnix::cSanTerminalDeviceUnix(SString strTerminalName, SString strTerminalTittle)
	:m_strTerminalName(strTerminalName),
	m_TerminalID(0),
	m_TerminalHandle(nullptr),
	m_strTittle(strTerminalTittle),
	m_strCodeLocate(""),
	m_TextDefColor(STC_WHITE),
	m_BackgroundDefColor(STC_BLACK),
	m_TerminalBuffer(1024, 0),
	m_strBuffer(_SSTR(" "))
{
	/*this->m_TerminalHandle = ::GetStdHandle(STD_OUTPUT_HANDLE);
	if (this->m_TerminalHandle == nullptr)
	{
		this->_CreateConsoleDevice();
	}*/

	//this->m_TerminalHandle = ::GetStdHandle(STD_OUTPUT_HANDLE);
	//::SetConsoleTitle(this->m_strTittle.c_str());

	this->m_strCodeLocate = ::setlocale(LC_ALL, this->m_strCodeLocate.c_str());

	printf("\x1b[%dm\x1b[%dm", TerminalColorTable[this->m_TextDefColor], TerminalColorTable[this->m_BackgroundDefColor]);
};
cSanTerminalDeviceUnix::~cSanTerminalDeviceUnix()
{
};
void cSanTerminalDeviceUnix::iSetTerminalTittle(SString strTittle)
{
	if (!strTittle.empty())
	{
		this->m_strTittle = strTittle;
		//::SetConsoleTitle(this->m_strTittle.c_str());
	}
}
void cSanTerminalDeviceUnix::iSetTerminalCodeLocate(SStringA strLocate)
{
	this->m_strCodeLocate = ::setlocale(LC_ALL, strLocate.c_str());
}
void cSanTerminalDeviceUnix::iSetDefaultTextColor(eSANTERMINALCOLOR Color)
{
	if (Color != STC_DEFAULT)
	{
		this->m_TextDefColor = Color;
	}
}
void cSanTerminalDeviceUnix::iSetDefaultBackgroundColor(eSANTERMINALCOLOR Color)
{
	if (Color != STC_DEFAULT)
	{
		this->m_BackgroundDefColor = Color;
	}
}
void cSanTerminalDeviceUnix::_CreateConsoleDevice()
{
	//::AllocConsole();
	//::AttachConsole(ATTACH_PARENT_PROCESS);
	::freopen("CONIN$", "r+t", stdin);
	::freopen("CONOUT$", "w+t", stdout);
}
void cSanTerminalDeviceUnix::_OutputStringA(SStringA strString, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const
{
	TextColor = TextColor == STC_DEFAULT ? this->m_TextDefColor : TextColor;
	BackgroundColor = BackgroundColor == STC_DEFAULT ? this->m_BackgroundDefColor : BackgroundColor;

	printf("\x1b[%dm\x1b[%dm", TerminalColorTable[TextColor], TerminalColorTable[BackgroundColor]);
	::cout << strString.c_str();
	printf("\x1b[%dm\x1b[%dm", TerminalColorTable[this->m_TextDefColor], TerminalColorTable[this->m_BackgroundDefColor]);
}
void cSanTerminalDeviceUnix::_OutputStringW(SStringW strString, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const
{
	TextColor = TextColor == STC_DEFAULT ? this->m_TextDefColor : TextColor;
	BackgroundColor = BackgroundColor == STC_DEFAULT ? this->m_BackgroundDefColor : BackgroundColor;

	printf("\x1b[%dm\x1b[%dm", TerminalColorTable[TextColor], TerminalColorTable[BackgroundColor]);
	::wcout << strString.c_str();
	printf("\x1b[%dm\x1b[%dm", TerminalColorTable[this->m_TextDefColor], TerminalColorTable[this->m_BackgroundDefColor]);
}
bool cSanTerminalDeviceUnix::iSetTerminalBufferSize(uint32 BufferSize)
{
	this->m_TerminalBuffer.iReSizeStream(BufferSize);
	this->m_TerminalBuffer.iSetStream(0);

	return true;
}
SString cSanTerminalDeviceUnix::iGetTerminalTittle() const
{
	return this->m_strTittle;
}
SStringA cSanTerminalDeviceUnix::iGetTerminalCodeLocate() const
{
	return this->m_strCodeLocate;
}
eSANTERMINALCOLOR cSanTerminalDeviceUnix::iGetDefaultTextColor() const
{
	return this->m_TextDefColor;
}
eSANTERMINALCOLOR cSanTerminalDeviceUnix::iGetDefaultBackgroundColor() const
{
	return this->m_BackgroundDefColor;
}
uint32 cSanTerminalDeviceUnix::iGetTerminalBufferSize() const
{
	return this->m_TerminalBuffer.iGetSize();
}
SPOINT2 cSanTerminalDeviceUnix::iGetCurrentCursorPosition() const
{
	SStringA strGetPos = "\033[6n";
	//CONSOLE_SCREEN_BUFFER_INFO BufferInfo;
	//::GetConsoleScreenBufferInfo(this->m_TerminalHandle, &BufferInfo);
	return SPOINT2(0, 0);// BufferInfo.dwCursorPosition.X, BufferInfo.dwCursorPosition.Y, 0);
}
SString cSanTerminalDeviceUnix::iGetInputString(schar DelLim)
{
#ifndef _UNICODE
	std::getline(cin, this->m_strBuffer, DelLim);
#else
	std::getline(wcin, this->m_strBuffer, DelLim);
#endif
	return this->m_strBuffer;
}
void cSanTerminalDeviceUnix::iOutputStringA(SStringA strString, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const
{
	if (strString.empty())
	{
		return;
	}
	this->m_TerminalLock.lock();
	this->_OutputStringA(strString, TextColor, BackgroundColor);
	this->m_TerminalLock.unlock();
}
void cSanTerminalDeviceUnix::iOutputStringW(SStringW strString, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const
{
	if (strString.empty())
	{
		return;
	}
	this->m_TerminalLock.lock();
	this->_OutputStringW(strString, TextColor, BackgroundColor);
	this->m_TerminalLock.unlock();
}
void cSanTerminalDeviceUnix::iOutputStringA(SStringA strString, const SPOINT2 &Pos, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const
{
	this->m_TerminalLock.lock();

	SPOINT2 OrigionalPos = this->iGetCurrentCursorPosition();

	//Set Pos
	printf("\033[%d;%dH", (int32)Pos.x, (int32)Pos.y);
	this->_OutputStringA(strString, TextColor, BackgroundColor);
	printf("\033[%d;%dH", (int32)OrigionalPos.x, (int32)OrigionalPos.y);

	this->m_TerminalLock.unlock();
}
void cSanTerminalDeviceUnix::iOutputStringW(SStringW strString, const SPOINT2 &Pos, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const
{
	this->m_TerminalLock.lock();

	SPOINT2 OrigionalPos = this->iGetCurrentCursorPosition();

	//Set Pos
	printf("\033[%d;%dH", (int32)Pos.x, (int32)Pos.y);
	this->_OutputStringW(strString, TextColor, BackgroundColor);
	printf("\033[%d;%dH", (int32)OrigionalPos.x, (int32)OrigionalPos.y);

	this->m_TerminalLock.unlock();
}
void cSanTerminalDeviceUnix::iOutputStringLineA(SStringA strString, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const
{
	this->iOutputStringA(strString + "\r\n", TextColor, BackgroundColor);
}
void cSanTerminalDeviceUnix::iOutputStringLineW(SStringW strString, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const
{
	this->iOutputStringW(strString + L"\r\n", TextColor, BackgroundColor);
}
void cSanTerminalDeviceUnix::iOutputStringLineA(SStringA strString, const SPOINT2 &Pos, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const
{
	this->iOutputStringLineA(strString + "\r\n", Pos, TextColor, BackgroundColor);
}
void cSanTerminalDeviceUnix::iOutputStringLineW(SStringW strString, const SPOINT2 &Pos, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const
{
	this->iOutputStringLineW(strString + L"\r\n", Pos, TextColor, BackgroundColor);
}
//void cSanTerminalDeviceUnix::iClearLine()
//{
void cSanTerminalDeviceUnix::iClearScreen() const
{
	this->m_TerminalLock.lock();
	::system("cls");
	this->m_TerminalLock.unlock();
}
cSanTerminalDeviceUnix& cSanTerminalDeviceUnix::operator<<(const SStringA &strString)
{
	this->iOutputStringA(strString);
	return *this;
}
cSanTerminalDeviceUnix& cSanTerminalDeviceUnix::operator<<(const SStringW &strString)
{
	this->iOutputStringW(strString);
	return *this;
}
#endif