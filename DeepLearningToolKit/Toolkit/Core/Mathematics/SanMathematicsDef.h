//Project: San Lab Game Engine
//Version: 2.1.1
//Debug State: Add functions [vector and matrix based functions, and optimization]
#include"cmath"
#include"xmmintrin.h"
#include"../../RenderSystem.h"
#include"../SanContainer.h"
#pragma once
using namespace std;
namespace San
{
	namespace Mathematics
	{
#ifndef __STDSANMATHEMATICSDEF_H__
#define __STDSANMATHEMATICSDEF_H__

		static const float64 S_PI = 3.14159265358979323846;//3.14159265357;
		static const float64 S_PI_2 = S_PI / 2.0;

#ifdef _FLOAT_TYPE_DOUBLE
		static const float64 DEFAULT_FLOAT_COMPARE_BOUNDARY = 0.000000000000001;
#else
		static const float64 DEFAULT_FLOAT_COMPARE_BOUNDARY = 0.000001;
#endif

		struct SPOINT2;
		struct SPOINT3;
		struct SPOINT4;

		struct SVECTOR2;
		struct SVECTOR3;
		struct SVECTOR4;

		template<class ST, class _Zero, class _One> struct _SPOINT2;
		template<class ST, class _Zero, class _One> struct _SPOINT3;
		template<class ST, class _Zero, class _One> struct _SPOINT4;

		template<class ST, class _Zero, class _One> struct _SVECTOR2;
		template<class ST, class _Zero, class _One> struct _SVECTOR3;
		template<class ST, class _Zero, class _One> struct _SVECTOR4;

		struct SANMATRIX2X2;
		struct SANMATRIX3X3;
		struct SANMATRIX4X4;

		/*template<class T>
		using _SVETOR = _stSANVECTOR<T>;

		template<class T>
		using _SMATRIX = _stSANMATRIX<T>;*/

		SVECTOR2	gloAbs(const SVECTOR2 &vector2);
		SVECTOR3	gloAbs(const SVECTOR3 &vector3);
		SVECTOR4	gloAbs(const SVECTOR4 &vector4);
		void		gloAbs(sfloat* pDest, const sfloat* pSrc, size_t size);

		sfloat		gloMod(const SVECTOR2 &vector2);
		sfloat		gloMod(const SVECTOR3 &vector3);//mod modulus
		sfloat		gloMod(const SVECTOR4 &vector4);

		SVECTOR2	gloNormalize(const SVECTOR2 &vector2);
		SVECTOR3	gloNormalize(const SVECTOR3 &vector3);
		SVECTOR4	gloNormalize(const SVECTOR4 &vector4);

		sfloat		gloDot(const SVECTOR2 &vec2_1, const SVECTOR2 &vec2_2);
		sfloat		gloDot(const SVECTOR3 &vec3_1, const SVECTOR3 &vec3_2);
		sfloat		gloDot(const SVECTOR4 &vec4_1, const SVECTOR4 &vec4_2);

		SVECTOR2	gloCross(const SVECTOR2 &vec2_1, const SVECTOR3 &vec2_2);
		SVECTOR3	gloCross(const SVECTOR3 &vec3_1, const SVECTOR3 &vec3_2);
		SVECTOR4	gloCross(const SVECTOR4 &vec4_1, const SVECTOR4 &vec4_2);

		sfloat		gloLength(const SVECTOR2 &vector2);
		sfloat		gloLength(const SVECTOR3 &vector3);
		sfloat		gloLength(const SVECTOR4 &vector4);

		SVECTOR2	gloCos(const SVECTOR2 &vector2);
		SVECTOR3	gloCos(const SVECTOR3 &vector3);
		SVECTOR4	gloCos(const SVECTOR4 &vector4);

		SVECTOR2	gloACos(const SVECTOR2 &vector2, const sfloat Length);
		SVECTOR3	gloACos(const SVECTOR3 &vector3, const sfloat Length);
		SVECTOR4	gloACos(const SVECTOR4 &vector4, const sfloat Length);

		void		gloSetTranslateMatrix(SANMATRIX4X4 &matrix, const SVECTOR3 &value);
		void		gloSetRotateMatrix(SANMATRIX4X4 &matrix, const SVECTOR3 &angle, const SVECTOR3 &axle);
		void		gloSetScaleMatrix(SANMATRIX4X4 &matrix, const SVECTOR3 &rate);

		bool		gloIsFloatEqual32(const float32 &f1, const float32 &f2, const float64 &bound = DEFAULT_FLOAT_COMPARE_BOUNDARY);
		bool		gloIsFloatEqual64(const float64 &f1, const float64 &f2, const float64 &bound = DEFAULT_FLOAT_COMPARE_BOUNDARY);

#ifdef _FLOAT_TYPE_DOUBLE
#define gloIsFloatEqual gloIsFloatEqual64
#else
#define gloIsFloatEqual gloIsFloatEqual32
#endif

		template<class ST> inline ST gloMin(const ST &v1, const ST &v2){ return v1 > v2 ? v2 : v1; };
		template<class ST> inline ST gloMax(const ST &v1, const ST &v2){ return v1 > v2 ? v1 : v2; };

#endif
	}
}