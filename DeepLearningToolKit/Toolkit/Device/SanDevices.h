#include"../Core/SanTypes.h"
using namespace std;
#pragma once
namespace San
{
	namespace Device
	{
#ifndef __STDSANDEVICES_H__
#define __STDSANDEVICES_H__

#define KBD_SIZE	256
#define KBT_BUTTONUP	0x01
#define KBT_BUTTONDOWN	0x02
		enum eSANVIRTUALKEY
		{
			SVK_TAB			= 0x0f,
			SVK_CAPSLOCK	= 0x3a,
			SVK_SPACE		= 0x39,
			SVK_BACKSPACE	= 0x0e,
			SVK_ENTER		= 0x1c,
			SVK_SHIFT_L		= 0x2a,
			SVK_SHIFT_R		= 0x36,
			SVK_CTRL_L		= 0x1d,
			SVK_CTRL_R		= 157,
			SVK_ALT_L		= 0x38,
			SVK_ALT_R		= 184,
			SVK_ESC			= 0x01,
			SVK_F1			= 0x3b,
			SVK_F2			= 0x3c,
			SVK_F3			= 0x3d,
			SVK_F4			= 0x3e,
			SVK_F5			= 0x3f,
			SVK_F6			= 0x40,
			SVK_F7			= 0x41,
			SVK_F8			= 0x42,
			SVK_F9			= 0x43,
			SVK_F10			= 0x44,
			SVK_F11			= 0x57,
			SVK_F12			= 0x58,
			SVK_INSERT		= 210,
			SVK_PRNTSCR		= 183,
			SVK_DELETE		= 211,
			SVK_HOME		= 199,
			SVK_PGUP		= 201,
			SVK_PGDN		= 209,
			SVK_END			= 207,
			SVK_UP			= 200,
			SVK_DOWN		= 208,
			SVK_LEFT		= 203,
			SVK_RIGHT		= 205,
		};

#define MD_SIZE 64
#define MDS_LBUTTON 0x01
#define MDS_RBUTTON 0x02
#define MDS_MBUTTON 0x04
#define MDS_BUTTONDOWN 0x08
#define MDS_BUTTONUP 0x10
#define MDS_MOVE 0x20

		struct stSANDEVICEDESCRIPTION
		{
			unsigned int DeviceScreen_w;
			unsigned int DeviceScreen_h;
			unsigned int PixFormat;
			unsigned int MaxTextureSize;
			unsigned int DXVersion;
			unsigned int MultiSampleLv;
			SString strGraphicsCardName;
		};
		typedef stSANDEVICEDESCRIPTION	SANDEVICEDESC;
		typedef stSANDEVICEDESCRIPTION*	LPSANDEVICEDESC;

#endif
	}
}