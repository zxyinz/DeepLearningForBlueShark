//Project: San Lab Game Engine
//Version: 2.1.1
//Debug State: Add functions and need test [_vector iterator]
#include"SanMathematicsDef.h"
#pragma once
using namespace std;
namespace San
{
	namespace Mathematics
	{
#ifndef __STDSANMATHEMATICSVECTOR_H__
#define __STDSANMATIEMATICSVECTOR_H__

		struct stSPOINT2
		{
		public:
			union
			{
				struct
				{ 
					sfloat x;
					sfloat y;
				};
				sfloat p[2];
			};
		public:
			stSPOINT2(const sfloat x = 0.0, const sfloat y = 0.0);
			~stSPOINT2();
		};

		struct stSPOINT3
		{
		public:
			union
			{
				struct
				{
					sfloat x;
					sfloat y;
					sfloat z;
				};
				sfloat p[3];
			};
		public:
			stSPOINT3(const sfloat x = 0.0, const sfloat y = 0.0, const sfloat z = 0.0);
			~stSPOINT3();
		};

		struct stSPOINT4
		{
		public:
			union
			{
				struct
				{
					sfloat x;
					sfloat y;
					sfloat z;
					sfloat w;
				};
				sfloat p[4];
			};
		public:
			stSPOINT4(const sfloat x = 0.0, const sfloat y = 0.0, const sfloat z = 0.0, const sfloat w = 0.0);
			~stSPOINT4();
		};

		struct SPOINT2 : public stSPOINT2
		{
		public:
			SPOINT2(const sfloat Val = 0.0);
			SPOINT2(const sfloat x, const sfloat y);
			SPOINT2(const sfloat* pArray);
			SPOINT2(const SPOINT2 &Point2);

			explicit SPOINT2(const SPOINT3 &Point3);
			explicit SPOINT2(const SPOINT4 &Point4);

			explicit SPOINT2(const SVECTOR2 &Vec2);
			explicit SPOINT2(const SVECTOR3 &Vec3);
			explicit SPOINT2(const SVECTOR4 &Vec4);

			~SPOINT2();

			const sfloat& operator[](const uint32 Position) const;
			sfloat& operator[](const uint32 Position);

			SPOINT2& operator = (const SPOINT2 &Point2);
			SPOINT2& operator = (const SPOINT3 &Point3);
			SPOINT2& operator = (const SPOINT4 &Point4);
			SPOINT2& operator = (const SVECTOR2 &Vec2);
			SPOINT2& operator = (const SVECTOR3 &Vec3);
			SPOINT2& operator = (const SVECTOR4 &Vec4);

			SPOINT2 operator + (const SPOINT2 &Point2) const;
			SPOINT2 operator - (const SPOINT2 &Point2) const;
			SPOINT2 operator * (const SPOINT2 &Point2) const;
			SPOINT2 operator + (const sfloat Val) const;
			SPOINT2 operator - (const sfloat Val) const;
			SPOINT2 operator * (const sfloat Val) const;

			bool operator == (const SPOINT2 &Point2) const;
			bool operator != (const SPOINT2 &Point2) const;

			bool operator < (const SPOINT2 &Point2) const;
			bool operator > (const SPOINT2 &Point2) const;

			bool operator <= (const SPOINT2 &Point2) const;
			bool operator >= (const SPOINT2 &Point2) const;

			SPOINT2	iGetHomogeneousPoint() const;
			void	iHomogenzation();
		};
		typedef SPOINT2*	lpSPOINT2;

		struct SVECTOR2 :public stSPOINT2
		{
		public:
			SVECTOR2(const sfloat Val = 0.0);
			SVECTOR2(const sfloat x, const sfloat y);
			SVECTOR2(const sfloat* pArray);
			SVECTOR2(const SVECTOR2 &Vec2);

			explicit SVECTOR2(const SPOINT2 &Point2);
			explicit SVECTOR2(const SPOINT3 &Point3);
			explicit SVECTOR2(const SPOINT4 &Point4);

			explicit SVECTOR2(const SVECTOR3 &Vec3);
			explicit SVECTOR2(const SVECTOR4 &Vec4);

			~SVECTOR2();

			const sfloat& operator[](const uint32 Position) const;
			sfloat& operator[](const uint32 Position);

			SVECTOR2& operator = (const SPOINT2 &Point2);
			SVECTOR2& operator = (const SPOINT3 &Point3);
			SVECTOR2& operator = (const SPOINT4 &Point4);
			SVECTOR2& operator = (const SVECTOR2 &Vec2);
			SVECTOR2& operator = (const SVECTOR3 &Vec3);
			SVECTOR2& operator = (const SVECTOR4 &Vec4);

			SVECTOR2 operator + (const SVECTOR2 &Vec2) const;
			SVECTOR2 operator - (const SVECTOR2 &Vec2) const;
			SVECTOR2 operator * (const SVECTOR2 &Vec2) const;
			SVECTOR2 operator + (const sfloat Val) const;
			SVECTOR2 operator - (const sfloat Val) const;
			SVECTOR2 operator * (const sfloat Val) const;

			bool operator == (const SVECTOR3 &Vec2) const;
			bool operator != (const SVECTOR3 &Vec2) const;

			bool operator < (const SVECTOR3 &Vec2) const;
			bool operator > (const SVECTOR3 &Vec2) const;

			bool operator <= (const SVECTOR3 &Vec2) const;
			bool operator >= (const SVECTOR3 &Vec2) const;

			SVECTOR2	iGetHomogeneousVector() const;
			void		iHomogenzation();
		};
		typedef SVECTOR2*	lpSVECTOR2;

		struct SPOINT3 :public stSPOINT3
		{
		public:
			SPOINT3(const sfloat Val = 0.0);
			SPOINT3(const sfloat x, const sfloat y, const sfloat z);
			SPOINT3(const sfloat* pArray);
			SPOINT3(const SPOINT3 &Point3);

			explicit SPOINT3(const SPOINT2 &Point2, const sfloat z = 0.0);
			explicit SPOINT3(const SPOINT4 &Point4);
			explicit SPOINT3(const SVECTOR2 &Vec2, const sfloat z = 0.0);

			explicit SPOINT3(const SVECTOR3 &Vec3);
			explicit SPOINT3(const SVECTOR4 &Vec4);

			~SPOINT3();

			const sfloat& operator[](const uint32 Position) const;
			sfloat& operator[](const uint32 Position);

			//assignment operators
			SPOINT3& operator = (const SPOINT2 &Point2);
			SPOINT3& operator = (const SPOINT3 &Point3);
			SPOINT3& operator = (const SPOINT4 &Point4);
			SPOINT3& operator = (const SVECTOR2 &Vec2);
			SPOINT3& operator = (const SVECTOR3 &Vec3);
			SPOINT3& operator = (const SVECTOR4 &Vec4);

			//binary operators
			SPOINT3 operator + (const SPOINT3 &Point3) const;
			SPOINT3 operator - (const SPOINT3 &Point3) const;
			SPOINT3 operator * (const SPOINT3 &Point3) const;
			SPOINT3 operator + (const sfloat Val) const;
			SPOINT3 operator - (const sfloat Val) const;
			SPOINT3 operator * (const sfloat Val) const;

			bool operator == (const SPOINT3 &Point3) const;
			bool operator != (const SPOINT3 &Point3) const;

			bool operator < (const SPOINT3 &Point3) const;
			bool operator > (const SPOINT3 &Point3) const;

			bool operator <= (const SPOINT3 &Point3) const;
			bool operator >= (const SPOINT3 &Point3) const;

			SPOINT3	iGetHomogeneousPoint() const;
			void	iHomogenzation();
		};
		typedef SPOINT3*	lpSPOINT3;

		struct SVECTOR3 :public stSPOINT3
		{
		public:
			SVECTOR3(const sfloat Val = 0.0);
			SVECTOR3(const sfloat x, const sfloat y, const sfloat z);
			SVECTOR3(const sfloat* pArray);
			SVECTOR3(const SVECTOR3 &Vec3);

			explicit SVECTOR3(const SPOINT2 &Point2, const sfloat z = 0.0);
			explicit SVECTOR3(const SPOINT3 &Point3);
			explicit SVECTOR3(const SPOINT4 &Point4);

			explicit SVECTOR3(const SVECTOR2 &Vec2, const sfloat z = 0.0);
			explicit SVECTOR3(const SVECTOR4 &Vec4);

			~SVECTOR3();

			const sfloat& operator[](const uint32 Position) const;
			sfloat& operator[](const uint32 Position);

			SVECTOR3& operator = (const SPOINT2 &Point2);
			SVECTOR3& operator = (const SPOINT3 &Point3);
			SVECTOR3& operator = (const SPOINT4 &Point4);
			SVECTOR3& operator = (const SVECTOR2 &Vec2);
			SVECTOR3& operator = (const SVECTOR3 &Vec3);
			SVECTOR3& operator = (const SVECTOR4 &Vec4);

			SVECTOR3 operator + (const SVECTOR3 &Vec3) const;
			SVECTOR3 operator - (const SVECTOR3 &Vec3) const;
			SVECTOR3 operator * (const SVECTOR3 &Vec3) const;
			SVECTOR3 operator + (const sfloat Val) const;
			SVECTOR3 operator - (const sfloat Val) const;
			SVECTOR3 operator * (const sfloat Val) const;

			bool operator == (const SVECTOR3 &Vec3) const;
			bool operator != (const SVECTOR3 &Vec3) const;

			bool operator < (const SVECTOR3 &Vec3) const;
			bool operator > (const SVECTOR3 &Vec3) const;

			bool operator <= (const SVECTOR3 &Vec3) const;
			bool operator >= (const SVECTOR3 &Vec3) const;

			SVECTOR3	iGetHomogeneousVector() const;
			void		iHomogenzation();
		};
		typedef SVECTOR3*	lpSVECTOR3;

		struct SPOINT4 :public stSPOINT4
		{
		public:
			SPOINT4(const sfloat x = 0.0);
			SPOINT4(const sfloat x, const sfloat y, const sfloat z, const sfloat w);
			SPOINT4(const sfloat* pArray);
			SPOINT4(const SPOINT4 &Point4);

			explicit SPOINT4(const SPOINT2 &Point2, const sfloat z = 0.0, const sfloat w = 0.0);
			explicit SPOINT4(const SPOINT3 &Point3, const sfloat w = 0.0);

			explicit SPOINT4(const SVECTOR2 &Vec2, const sfloat z = 0.0, const sfloat w = 0.0);
			explicit SPOINT4(const SVECTOR3 &Vec3, const sfloat w = 0.0);
			explicit SPOINT4(const SVECTOR4 &Vec4);

			~SPOINT4();

			const sfloat& operator[](const uint32 Position) const;
			sfloat& operator[](const uint32 Position);

			SPOINT4& operator = (const SPOINT2 &Point2);
			SPOINT4& operator = (const SPOINT3 &Point3);
			SPOINT4& operator = (const SPOINT4 &Point4);
			SPOINT4& operator = (const SVECTOR2 &Vec2);
			SPOINT4& operator = (const SVECTOR3 &Vec3);
			SPOINT4& operator = (const SVECTOR4 &Vec4);

			SPOINT4 operator + (const SPOINT4 &Point4) const;
			SPOINT4 operator - (const SPOINT4 &Point4) const;
			SPOINT4 operator * (const SPOINT4 &Point4) const;
			SPOINT4 operator + (const sfloat Val) const;
			SPOINT4 operator - (const sfloat Val) const;
			SPOINT4 operator * (const sfloat Val) const;

			bool operator == (const SPOINT4 &Point4) const;
			bool operator != (const SPOINT4 &Point4) const;

			bool operator < (const SPOINT4 &Point4) const;
			bool operator > (const SPOINT4 &Point4) const;

			bool operator <= (const SPOINT4 &Point4) const;
			bool operator >= (const SPOINT4 &Point4) const;

			SPOINT4	iGetHomogeneousPoint() const;
			void	iHomogenzation();
		};
		typedef SPOINT4*	lpSPOINT4;

		struct SVECTOR4 :public stSPOINT4
		{
		public:
			SVECTOR4(const sfloat x = 0.0);
			SVECTOR4(const sfloat x, const sfloat y, const sfloat z, const sfloat w);
			SVECTOR4(const sfloat* pArray);
			SVECTOR4(const SVECTOR4 &Vec4);

			explicit SVECTOR4(const SPOINT2 &Point2, const sfloat z = 0.0, const sfloat w = 0.0);
			explicit SVECTOR4(const SPOINT3 &Point3, const sfloat w = 0.0);
			explicit SVECTOR4(const SPOINT4 &Point4);

			explicit SVECTOR4(const SVECTOR2 &Vec2, const sfloat z = 0.0, const sfloat w = 0.0);
			explicit SVECTOR4(const SVECTOR3 &Vec3, const sfloat w = 0.0);

			~SVECTOR4();

			const sfloat& operator[](const uint32 Position) const;
			sfloat& operator[](const uint32 Position);

			SVECTOR4& operator = (const SPOINT2 &Point2);
			SVECTOR4& operator = (const SPOINT3 &Point3);
			SVECTOR4& operator = (const SPOINT4 &Point4);
			SVECTOR4& operator = (const SVECTOR2 &Vec2);
			SVECTOR4& operator = (const SVECTOR3 &Vec3);
			SVECTOR4& operator = (const SVECTOR4 &Vec4);

			SVECTOR4 operator + (const SVECTOR4 &Vec4) const;
			SVECTOR4 operator - (const SVECTOR4 &Vec4) const;
			SVECTOR4 operator * (const SVECTOR4 &Vec4) const;
			SVECTOR4 operator + (const sfloat Val) const;
			SVECTOR4 operator - (const sfloat Val) const;
			SVECTOR4 operator * (const sfloat Val) const;

			bool operator == (const SVECTOR4 &Vec4) const;
			bool operator != (const SVECTOR4 &Vec4) const;

			bool operator < (const SVECTOR4 &Vec4) const;
			bool operator > (const SVECTOR4 &Vec4) const;

			bool operator <= (const SVECTOR4 &Vec4) const;
			bool operator >= (const SVECTOR4 &Vec4) const;

			SVECTOR4	iGetHomogeneousVector() const;
			void		iHomogenzation();
		};
		typedef SVECTOR4*	lpSVECTOR4;
#endif
	}
}