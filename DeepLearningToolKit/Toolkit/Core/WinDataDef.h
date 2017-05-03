#include"../RenderSystem.h"
using namespace std;
namespace San
{
#ifndef __STDWINDATADEF_H__
#define __STDWINDATADEF_H__

	#ifndef _WINDEF_
	typedef unsigned char	BYTE;
	typedef unsigned short	WORD;
	typedef unsigned long	DWORD;

	typedef void	VOID;
	typedef char	CHAR;
	typedef short	SHORT;
	typedef long	LONG;
	typedef int		INT;

	typedef unsigned char	UCHAR;
	typedef unsigned short	USHORT;
	typedef unsigned long	ULONG;
	typedef unsigned int	UINT;

	typedef struct stPOINT
	{
		long x;
		long y;
	} POINT, *PPOINT, *LPPOINT;

	typedef struct stSIZE
	{
		long cx;
		long cy;
	} SIZE, *PSIZE, *LPSIZE;

	typedef struct stRECT
	{
		long top;
		long bottom;
		long left;
		long right;
	} RECT, *PRECT, *LPRECT;
#endif

#ifndef _WINGDI_
#pragma pack(1)
	typedef struct stBITMAPFILEHEADER
	{
		WORD	bfType;
		DWORD	bfSize;
		WORD	bfReserved1;
		WORD	dfReserved2;
		DWORD	bfOffbits;
	} BITMAPFILEHEADER, *PBITMAPFILEHEADER, *LPBITMAPFILEHEADER;

	typedef struct stBITMAPINFOHEADER
	{
		DWORD	biSize;
		LONG	biWidth;
		LONG	biHeight;
		WORD	biPlanes;
		WORD	biBitCount;
		DWORD	biCompression;
		DWORD	biSizeImage;
		LONG	biXPelsPerMeter;
		LONG	biYPelsPerMeter;
		DWORD	biClrUsed;
		DWORD	biClrImportant;
	} BITMAPINFOHEADER, *PBITMAPINFOHEADER, *LPBITMAPINFOHEADER;
#pragma pack()
#endif

#endif
}