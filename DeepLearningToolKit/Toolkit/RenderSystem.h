/*			Select Operating System			*/
#ifndef _WINDOWS
#define _WINDOWS
#pragma message("Msg: define _WINDOWS	File: RenderSystem.h")
#endif//*/

/*#ifndef _UNIX
#define _UNIX
#pragma message("Msg: define _UNIX	File: RenderSystem.h")
#endif//*/

/*			Select CPU Architecture			*/
#ifndef _X32
#define _X32
#pragma message("Msg: define _X32	File: RenderSystem.h")
#endif//*/

/*#ifndef _X64
#define _X64
#pragma message("Msg: define _X64	File: RenderSystem.h")
#endif//*/

/*			Select Render System			*/
#ifndef _OPENGL
#define _OPENGL
#pragma message("Msg define _OPENGL	File: RenderSystem.h")
#endif//*/

/*#ifndef _DIRECTX
#define _DIRECTX
#pragma message("Msg: define _DIRECTX	File: RenderSystem.h")
#endif//*/

/*			Select Physics Engine			*/
/*#ifndef _PHYSX
#define _PHYSX
#pragma message("Msg: define _PHYSX	File RenderSystem.h")
#endif//*/

/*			Select Audio Engine				*/
/*#ifndef _OPENAL
#define _OPENAL
#pragma message("Msg: define _OPENAL File RenderSystem.h")
#endif//*/

/*			Select System Build Environment			*/
#ifndef _CPP_VERSION_CPP0x
#define _CPP_VERSION_CPP0x
#pragma message("Msg: define _CPP_VERSION_CPP0x	File: RenderSystem.h")
#endif

#ifndef _FLOAT_TYPE_FLOAT
#define _FLOAT_TYPE_FLOAT
#pragma message("Msg: define _FLOAT_TYPE_FLOAT	File: RenderSystem.h")
#endif//*/

/*#ifndef _FLOAT_TYPT_DOUBLE
#define _FLOAT_TYPE_DOUBLE
#pragma message("Msg: define _FLOAT_TYPE_DOUBLE	File: RenderSystem.h")
#endif//*/

/*#ifndef _UNICODE
#define _UNICODE
#pragma message("Msg: define _UNICODE		File: RenderSystem.h")
#endif//*/

/*			Include Operating System			*/
#ifdef _WINDOWS

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#pragma message("Msg: define WIN32_LEAN_AND_MEAN	File: RenderSystem.h")
#endif

#ifdef _PHYSX
#define NOMINMAX
#pragma message("Msg: define NOMINMAX	File: RenderSystem.h")
#endif

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#pragma message("Msg: define _CRT_SECURE_NO_WARNINGS	File: RenderSystem.h")
#endif

#include "windows.h"
#pragma message("Msg: include file windows.h	File: RenderSystem.h")
//#include "Gdiplus.h"
//#pragma message("Msg: include file Gdiplus.h	File: RenderSystem.h")
#endif

#ifdef _UNIX
#include"stdio.h"
#include"stdlib.h"
#include"sys/types.h"
#endif

/*			Include Graphics API Header			*/

#ifdef _OPENGL
//#include"GL/glew.h"
//#include"glut.h"
//#pragma comment(lib,"glew32.lib")
#endif

#ifdef _DIRECTX
#include"d3d11.h"
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dx11.lib")
#endif

/*			Include Physics API Header			*/

#ifdef _PHYSX
#include"NxPhysics.h"
#endif

/*			Include OpenAL API Header			*/
#ifdef _OPENAL
#include"al.h"
#include"alc.h"
#include"efx.h"
#include"Framework.h"
#endif