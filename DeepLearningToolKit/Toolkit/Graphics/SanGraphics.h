#include"../RenderSystem.h"
#include"../Core/SanTypes.h"
#include"../Core/SanContainer.h"
#include"../Core/SanMathematics.h"
#pragma once
using namespace std;
using namespace San::Mathematics;
namespace San
{
	namespace Graphics
	{
#ifndef __STDSANGRAPHICS_H__
#define __STDSANGRAPHICS_H__

		enum eSANPIXTYPE
		{
			PT_RGB		= 3,
			PT_RGBA		= 4,
			PT_RGB_F	= 3*sizeof(sfloat),
			PT_RGBA_F	= 4*sizeof(sfloat),
		};

		struct stSANCOLORF;

		struct stSANCOLOR
		{
		public:
			union
			{
				struct
				{
					uint8	r;
					uint8	g;
					uint8	b;
					uint8	a;
				};
				uint8	color_a[4];
				uint32	c_32;
			};
		public:
			stSANCOLOR(){};
			stSANCOLOR(const uint8 red,const uint8 green,const uint8 blue,const uint8 alpha)
				:r(red),g(green),b(blue),a(alpha)
			{};
			stSANCOLOR(const uint32 color32)
				:c_32(color32)
			{};
			~stSANCOLOR(){};
			stSANCOLOR& operator+(const stSANCOLOR& Color);
			stSANCOLOR& operator-(const stSANCOLOR& Color);
			stSANCOLOR& operator=(const stSANCOLORF& Color);
		};
		typedef stSANCOLOR	SANCOLOR;
		typedef stSANCOLOR*	lpSANCOLOR;

		struct stSANCOLORF
		{
		public:
			union
			{
				struct
				{
					sfloat r;
					sfloat g;
					sfloat b;
					sfloat a;
				};
				sfloat color_a[4];
			};
		public:
			stSANCOLORF(){};
			stSANCOLORF(const sfloat red,const sfloat green,const sfloat blue,const sfloat alpha)
				:r(red),g(green),b(blue),a(alpha)
			{};
			stSANCOLORF(const SANCOLOR &color)
				:r(color.r/255.0f),g(color.g/255.0f),b(color.b/255.0f),a(color.a/255.0f)
			{};
			~stSANCOLORF(){};
			stSANCOLORF& operator+(const stSANCOLORF& Color);
			stSANCOLORF& operator-(const stSANCOLORF& Color);
			stSANCOLORF& operator=(const stSANCOLOR& Color);
		};
		typedef stSANCOLORF	SANCOLORF;
		typedef SANCOLORF*	lpSANCOLORF;

		typedef SANCOLOR	SANPIX;
		typedef SANPIX*		lpSANPIX;
		typedef SANCOLORF	SANPIXF;
		typedef SANPIXF*	lpSANPIXF;

		enum eSANTEXTYPE
		{
			STT_TEX1D		= 0x0100,//Not Support Yet
			STT_TEX2D		= 0x0200,
			STT_TEX3D		= 0x0300,
		};

		enum eSANTEXVERSION
		{
			STV_STATIC		= 0x0000,
			STV_DYNAMIC		= 0x0001,
			STV_VIDEO		= 0x0002,
		};

		struct stSANPIXMAP2D
		{
			lpSANPIX pPixMap;
			uint32 width;
			uint32 height;
		};
		typedef stSANPIXMAP2D	SANPIXMAP2D;
		typedef stSANPIXMAP2D*	lpSANPIXMAP2D;
		
		struct stSANPIXMAP3D
		{
			lpSANPIX pPixMap;
			uint32 width;
			uint32 height;
			uint32 depth;
		};
		typedef stSANPIXMAP3D	SANPIXMAP3D;
		typedef stSANPIXMAP3D*	lpSANPIXMAP3D;

		struct stSANPIXMAPF2D
		{
			lpSANPIXF pPixMap;
			uint32 width;
			uint32 height;
		};
		typedef stSANPIXMAPF2D	SANPIXMAPF2D;
		typedef stSANPIXMAPF2D*	lpSANPIXMAPF2D;
		
		struct stSANPIXMAPF3D
		{
			lpSANPIXF pPixMap;
			uint32 width;
			uint32 height;
			uint32 depth;
		};
		typedef stSANPIXMAPF3D	SANPIXMAPF3D;
		typedef stSANPIXMAPF3D*	lpSANPIXMAPF3D;

		struct stSANTEXTURE
		{
		public:
			eSANTEXVERSION Version;
			eSANTEXTYPE	Type;
			eSANPIXTYPE	PixFormat;
			SHANDLE DataHandle;
			uint32	DataSize;
			SVALUE	CurrentParam;
			SVALUE* ParamList;
			uint32	ParamListSize;
		public:
			/*stSANTEXTURE();
			stSANTEXTURE(eSANTEXTYPE Type,eSANPIXTYPE PixFormat,SHANDLE pBuffer,uint32 BufferSize,uint32 Width,uint32 Height,uint32 Depth);
			stSANTEXTURE(const stSANTEXTURE& Texture);
			~stSANTEXTURE();
			virtual void iSetTexType(eSANTEXTYPE Type);
			virtual void iSetPixFormat(eSANPIXTYPE PixFormat);
			virtual void iSetBufferData(SHANDLE pBuffer,uint32 BufferSize);
			virtual void iSetTextureSize(SPOINT3 TexSize);
			virtual void iUpdateTexture(eSANTEXTYPE Type,eSANPIXTYPE PixFormat,SHANDLE pBuffer,uint32 BufferSize,uint32 Width,uint32 Height,uint32 Depth);
			virtual eSANTEXTYPE iGetTexType() const;
			virtual eSANPIXTYPE iGetPixFormat() const;
			virtual uint32 iGetBufferData(SHANDLE* pBuffer,uint32 BeginIndex=0,uint32 EndIndex=0) const;
			virtual SPOINT3 iGetTextureSize() const;*/
			virtual void iUpdate(){};
			/*stSANTEXTURE& operator=(const stSANTEXTURE& Texture);*/
		};
		typedef stSANTEXTURE	SANTEXTURE;
		typedef stSANTEXTURE*	lpSANTEXTURE;
		typedef list<SANTEXTURE>	SANTEXTURELIST;
		typedef list<SANTEXTURE>*	lpSANTEXTURELIST;

		struct stSANACTION
		{
			uint32 Identity;
			SString strName;
			uint32 Begin;
			uint32 End;
			SVECTOR3 Translation;
			SVECTOR3 Rotation;
			SANMATRIX4X4 Matrix;
			list<SHANDLE> ParamList;
		};
		
		struct stSANGRAPHICSPOINT
		{
		public:
			SVECTOR3 v;
			SVECTOR3 vn;
			lpSVECTOR3	p_vt;
			uint8 vt_size;
			SANCOLOR vc;
		public:
			stSANGRAPHICSPOINT();
			stSANGRAPHICSPOINT(const stSANGRAPHICSPOINT& P);
			~stSANGRAPHICSPOINT();
			stSANGRAPHICSPOINT& operator=(const stSANGRAPHICSPOINT& P);
		};
		typedef stSANGRAPHICSPOINT	SANGRAPHICSPOINT;
		typedef stSANGRAPHICSPOINT*	lpSANGRAPHICSPOINT;
		typedef SANGRAPHICSPOINT	SANVERTEX;
		typedef SANGRAPHICSPOINT*	lpSANVERTEX;

#endif
		}
}