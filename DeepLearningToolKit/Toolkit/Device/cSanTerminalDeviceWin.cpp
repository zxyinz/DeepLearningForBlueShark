#include"cSanTerminalDeviceWin.h"
using namespace std;
using namespace San;
using namespace San::Device;
#ifdef _WINDOWS
mutex cSanTerminalDeviceWin::m_TerminalLock;
cSanTerminalDeviceWin::cSanTerminalDeviceWin(SString strTerminalName, SString strTerminalTittle)
	:m_strTerminalName(strTerminalName),
	m_TerminalID(0),
	m_TerminalHandle(nullptr),
	m_strTittle(strTerminalTittle),
	m_strCodeLocate(""),
	m_TextDefColor(STC_WHITE),
	m_BackgroundDefColor(STC_BLACK),
	m_TerminalBuffer(1024,0),
	m_strBuffer(_SSTR(" "))
{
	this->m_TerminalHandle = ::GetStdHandle(STD_OUTPUT_HANDLE);
	if (this->m_TerminalHandle == nullptr)
	{
		this->_CreateConsoleDevice();
	}

	this->m_TerminalHandle = ::GetStdHandle(STD_OUTPUT_HANDLE);

	::SetConsoleTitle(this->m_strTittle.c_str());
	this->m_strCodeLocate = ::setlocale(LC_ALL, this->m_strCodeLocate.c_str());

	::SetConsoleTextAttribute(this->m_TerminalHandle, (this->m_TextDefColor) | (this->m_BackgroundDefColor << 4));
};
cSanTerminalDeviceWin::~cSanTerminalDeviceWin()
{
};
void cSanTerminalDeviceWin::iSetTerminalTittle(SString strTittle)
{
	if(!strTittle.empty())
	{
		this->m_strTittle=strTittle;
		::SetConsoleTitle(this->m_strTittle.c_str());
	}
}
void cSanTerminalDeviceWin::iSetTerminalCodeLocate(SStringA strLocate)
{
	this->m_strCodeLocate=::setlocale(LC_ALL,strLocate.c_str());
}
void cSanTerminalDeviceWin::iSetDefaultTextColor(eSANTERMINALCOLOR Color)
{
	if(Color!=STC_DEFAULT)
	{
		this->m_TextDefColor=Color;
		::SetConsoleTextAttribute(this->m_TerminalHandle,this->m_TextDefColor|(this->m_BackgroundDefColor<<4));
	}
}
void cSanTerminalDeviceWin::iSetDefaultBackgroundColor(eSANTERMINALCOLOR Color)
{
	if(Color!=STC_DEFAULT)
	{
		this->m_BackgroundDefColor=Color;
		::SetConsoleTextAttribute(this->m_TerminalHandle,this->m_TextDefColor|(this->m_BackgroundDefColor<<4));
	}
}
void cSanTerminalDeviceWin::_CreateConsoleDevice()
{
	::AllocConsole();
	::AttachConsole(ATTACH_PARENT_PROCESS);
	::freopen("CONIN$","r+t",stdin);
	::freopen("CONOUT$","w+t",stdout);
}
void cSanTerminalDeviceWin::_OutputString(const SStringA &strString, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const
{
	if ((TextColor == STC_DEFAULT) && (BackgroundColor == STC_DEFAULT))
	{
		//::printf("%s",strString.c_str());
		::cout << strString.c_str();
	}
	else
	{
		TextColor = TextColor == STC_DEFAULT ? this->m_TextDefColor : TextColor;
		BackgroundColor = BackgroundColor == STC_DEFAULT ? this->m_BackgroundDefColor : BackgroundColor;

		::SetConsoleTextAttribute(this->m_TerminalHandle, TextColor | (BackgroundColor << 4));
		//::printf("%s",strString.c_str());
		::cout << strString.c_str();
		::SetConsoleTextAttribute(this->m_TerminalHandle, this->m_TextDefColor | (this->m_BackgroundDefColor << 4));
	}
}
void cSanTerminalDeviceWin::_OutputString(const SStringW &strString, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const
{
	if ((TextColor == STC_DEFAULT) && (BackgroundColor == STC_DEFAULT))
	{
		//::wprintf(L"%s",strString.c_str());
		::wcout << strString.c_str();
	}
	else
	{
		TextColor = TextColor == STC_DEFAULT ? this->m_TextDefColor : TextColor;
		BackgroundColor = BackgroundColor == STC_DEFAULT ? this->m_BackgroundDefColor : BackgroundColor;

		::SetConsoleTextAttribute(this->m_TerminalHandle, TextColor | (BackgroundColor << 4));
		//::wprintf(L"%s",strString.c_str());
		::wcout << strString.c_str();
		::SetConsoleTextAttribute(this->m_TerminalHandle, this->m_TextDefColor | (this->m_BackgroundDefColor << 4));
	}
}
bool cSanTerminalDeviceWin::iSetTerminalBufferSize(uint32 BufferSize)
{
	this->m_TerminalBuffer.iReSizeStream(BufferSize);
	this->m_TerminalBuffer.iSetStream(0);

	return true;
}
void cSanTerminalDeviceWin::iSetCursorPosition(const SPOINT2 &Pos) const
{
	COORD Coord;
	Coord.X = Pos.x;
	Coord.Y = Pos.y;

	::SetConsoleCursorPosition(this->m_TerminalHandle, Coord);
}
SString cSanTerminalDeviceWin::iGetTerminalTittle() const
{
	return this->m_strTittle;
}
SStringA cSanTerminalDeviceWin::iGetTerminalCodeLocate() const
{
	return this->m_strCodeLocate;
}
eSANTERMINALCOLOR cSanTerminalDeviceWin::iGetDefaultTextColor() const
{
	return this->m_TextDefColor;
}
eSANTERMINALCOLOR cSanTerminalDeviceWin::iGetDefaultBackgroundColor() const
{
	return this->m_BackgroundDefColor;
}
uint32 cSanTerminalDeviceWin::iGetTerminalBufferSize() const
{
	return this->m_TerminalBuffer.iGetSize();
}
SPOINT2 cSanTerminalDeviceWin::iGetCursorPosition() const
{
	CONSOLE_SCREEN_BUFFER_INFO Info;
	::GetConsoleScreenBufferInfo(this->m_TerminalHandle, &Info);

	return SPOINT2(Info.dwCursorPosition.X, Info.dwCursorPosition.Y);
}
SString cSanTerminalDeviceWin::iGetInputString(schar DelLim)
{
#ifndef _UNICODE
	::getline(cin,this->m_strBuffer,DelLim);
#else
	::getline(wcin,this->m_strBuffer,DelLim);
#endif
	return this->m_strBuffer;
}
void cSanTerminalDeviceWin::iOutputString(const SStringA &strString,eSANTERMINALCOLOR TextColor,eSANTERMINALCOLOR BackgroundColor) const
{
	if(strString.empty()) { return; }
	unique_lock<mutex> Lock(this->m_TerminalLock);

	this->_OutputString(strString, TextColor, BackgroundColor);
}
void cSanTerminalDeviceWin::iOutputString(const SStringW &strString,eSANTERMINALCOLOR TextColor,eSANTERMINALCOLOR BackgroundColor) const
{
	if(strString.empty()) { return; }
	unique_lock<mutex> Lock(this->m_TerminalLock);

	this->_OutputString(strString, TextColor, BackgroundColor);
}
void cSanTerminalDeviceWin::iOutputString(const SStringA &strString, const SPOINT2 &Pos, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const
{
	unique_lock<mutex> Lock(this->m_TerminalLock);

	const SPOINT2 PrevPos = this->iGetCursorPosition();

	this->iSetCursorPosition(Pos);
	this->_OutputString(strString, TextColor, BackgroundColor);
	this->iSetCursorPosition(PrevPos);
}
void cSanTerminalDeviceWin::iOutputString(const SStringW &strString, const SPOINT2 &Pos, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const
{
	unique_lock<mutex> Lock(this->m_TerminalLock);

	const SPOINT2 PrevPos = this->iGetCursorPosition();

	this->iSetCursorPosition(Pos);
	this->_OutputString(strString, TextColor, BackgroundColor);
	this->iSetCursorPosition(PrevPos);
}
void cSanTerminalDeviceWin::iOutputStringLine(const SStringA &strString, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const
{
	this->iOutputString(strString + "\r\n", TextColor, BackgroundColor);
}
void cSanTerminalDeviceWin::iOutputStringLine(const SStringW &strString, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const
{
	this->iOutputString(strString + L"\r\n", TextColor, BackgroundColor);
}
void cSanTerminalDeviceWin::iOutputStringLine(const SStringA &strString, const SPOINT2 &Pos, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const
{
	this->iOutputStringLine(strString + "\r\n", Pos, TextColor, BackgroundColor);
}
void cSanTerminalDeviceWin::iOutputStringLine(const SStringW &strString, const SPOINT2 &Pos, eSANTERMINALCOLOR TextColor, eSANTERMINALCOLOR BackgroundColor) const
{
	this->iOutputStringLine(strString + L"\r\n", Pos, TextColor, BackgroundColor);
}
void cSanTerminalDeviceWin::iOutput(const TERMINALOUTPUTOBJA &Obj) const
{
	(Obj.Pos.x < 0) && (Obj.Pos.y < 0) ? this->iOutputString(Obj.strString, Obj.Color.Font, Obj.Color.Background) : this->iOutputString(Obj.strString, Obj.Pos, Obj.Color.Font, Obj.Color.Background);
}
void cSanTerminalDeviceWin::iOutput(const TERMINALOUTPUTOBJW &Obj) const
{
	(Obj.Pos.x < 0) && (Obj.Pos.y < 0) ? this->iOutputString(Obj.strString, Obj.Color.Font, Obj.Color.Background) : this->iOutputString(Obj.strString, Obj.Pos, Obj.Color.Font, Obj.Color.Background);
}
void cSanTerminalDeviceWin::iOutput(const vector<TERMINALOUTPUTOBJA> &ObjList) const
{
	bool bHasPos = false;
	for (const auto &Obj : ObjList)
	{
		if ((Obj.Pos.x >= 0) && (Obj.Pos.y >= 0)) { bHasPos = true; }
	}

	unique_lock<mutex> Lock(this->m_TerminalLock);

	if (!bHasPos)
	{
		for (const auto &Obj : ObjList)
		{
			this->_OutputString(Obj.strString, Obj.Color.Font, Obj.Color.Background);
		}
	}
	else
	{
		SPOINT2 PrevPos = this->iGetCursorPosition();

		for (const auto &Obj : ObjList)
		{
			if ((Obj.Pos.x < 0) && (Obj.Pos.y < 0))
			{
				if (!Obj.bFollowPrevObj) { this->iSetCursorPosition(PrevPos); }
				this->_OutputString(Obj.strString, Obj.Color.Font, Obj.Color.Background);
			}
			else
			{
				PrevPos = this->iGetCursorPosition();

				this->iSetCursorPosition(Obj.Pos);
				this->_OutputString(Obj.strString, Obj.Color.Font, Obj.Color.Background);
			}
		}
	}
}
void cSanTerminalDeviceWin::iOutput(const vector<TERMINALOUTPUTOBJW> &ObjList) const
{
	bool bHasPos = false;
	for (const auto &Obj : ObjList)
	{
		if ((Obj.Pos.x >= 0) && (Obj.Pos.y >= 0)) { bHasPos = true; }
	}

	unique_lock<mutex> Lock(this->m_TerminalLock);

	if (!bHasPos)
	{
		for (const auto &Obj : ObjList)
		{
			this->_OutputString(Obj.strString, Obj.Color.Font, Obj.Color.Background);
		}
	}
	else
	{
		SPOINT2 PrevPos = this->iGetCursorPosition();

		for (const auto &Obj : ObjList)
		{
			if ((Obj.Pos.x < 0) && (Obj.Pos.y < 0))
			{
				if (!Obj.bFollowPrevObj) { this->iSetCursorPosition(PrevPos); }
				this->_OutputString(Obj.strString, Obj.Color.Font, Obj.Color.Background);
			}
			else
			{
				PrevPos = this->iGetCursorPosition();

				this->iSetCursorPosition(Obj.Pos);
				this->_OutputString(Obj.strString, Obj.Color.Font, Obj.Color.Background);
			}
		}
	}
}
void cSanTerminalDeviceWin::iOutputLine(const TERMINALOUTPUTOBJA &Obj) const
{
	(Obj.Pos.x < 0) && (Obj.Pos.y < 0) ? this->iOutputStringLine(Obj.strString, Obj.Color.Font, Obj.Color.Background) : this->iOutputStringLine(Obj.strString, Obj.Pos, Obj.Color.Font, Obj.Color.Background);
}
void cSanTerminalDeviceWin::iOutputLine(const TERMINALOUTPUTOBJW &Obj) const
{
	(Obj.Pos.x < 0) && (Obj.Pos.y < 0) ? this->iOutputStringLine(Obj.strString, Obj.Color.Font, Obj.Color.Background) : this->iOutputStringLine(Obj.strString, Obj.Pos, Obj.Color.Font, Obj.Color.Background);
}
//void cSanTerminalDeviceWin::iOutputLine(const vector<TERMINALOUTPUTOBJA> &ObjList) const
//{
//
//}
//void cSanTerminalDeviceWin::iOutputLine(const vector<TERMINALOUTPUTOBJW> &ObjList) const
//{
//
//}
//void cSanTerminalDeviceWin::iClearLine()
//{
void cSanTerminalDeviceWin::iClearScreen() const
{
	unique_lock<mutex> Lock(this->m_TerminalLock);
	::system("cls");
}
cSanTerminalDeviceWin& cSanTerminalDeviceWin::operator<<(const SStringA &strString)
{
	this->iOutputString(strString);
	return *this;
}
cSanTerminalDeviceWin& cSanTerminalDeviceWin::operator<<(const SStringW &strString)
{
	this->iOutputString(strString);
	return *this;
}
cSanTerminalDeviceWin& cSanTerminalDeviceWin::operator<<(const TERMINALOUTPUTOBJA &Obj)
{
	this->iOutput(Obj);
	return *this;
}
cSanTerminalDeviceWin& cSanTerminalDeviceWin::operator<<(const TERMINALOUTPUTOBJW &Obj)
{
	this->iOutput(Obj);
	return *this;
}
cSanTerminalDeviceWin& cSanTerminalDeviceWin::operator<<(const vector<TERMINALOUTPUTOBJA> &ObjList)
{
	this->iOutput(ObjList);
	return *this;
}
cSanTerminalDeviceWin& cSanTerminalDeviceWin::operator<<(const vector<TERMINALOUTPUTOBJW> &ObjList)
{
	this->iOutput(ObjList);
	return *this;
}
#endif