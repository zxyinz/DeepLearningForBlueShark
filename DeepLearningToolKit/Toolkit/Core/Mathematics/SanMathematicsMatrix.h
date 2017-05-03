//Project: San Lab Game Engine
//Version: 2.1.1
//Debug State: Add structures and functions [_smatrix iterator]
#include"SanMathematicsDef.h"
#include"SanMathematicsVector.h"
#pragma once
using namespace std;
namespace San
{
	namespace Mathematics
	{
#ifndef __STDSANMATHEMATICSMATRIX_H__
#define __STDSANMATIEMATICSMATRIX_H__
		struct stSANMATRIX2X2
		{
		public:
			union
			{
				struct
				{
					sfloat e11, e12;
					sfloat e21, e22;
				};
				sfloat m[2][2];
				sfloat ma[4];
			};
		public:
			stSANMATRIX2X2(const sfloat v11 = 0.0, const sfloat v12 = 0.0, const sfloat v21 = 0.0, const sfloat v22 = 0.0);
			~stSANMATRIX2X2();
		};

		struct stSANMATRIX3X3
		{
		public:
			union
			{
				struct
				{
					sfloat e11, e12, e13;
					sfloat e21, e22, e23;
					sfloat e31, e32, e33;
				};
				sfloat	m[3][3];
				sfloat	ma[9];
			};
		public:
			stSANMATRIX3X3(const sfloat v11 = 0.0, const sfloat v12 = 0.0, const sfloat v13 = 0.0,
				const sfloat v21 = 0.0, const sfloat v22 = 0.0, const sfloat v23 = 0.0,
				const sfloat v31 = 0.0, const sfloat v32 = 0.0, const sfloat v33 = 0.0);
			~stSANMATRIX3X3();
		};

		struct stSANMATRIX4X4
		{
		public:
			union
			{
				struct
				{
					sfloat e11, e12, e13, e14;
					sfloat e21, e22, e23, e24;
					sfloat e31, e32, e33, e34;
					sfloat e41, e42, e43, e44;
				};
				sfloat	m[4][4];
				sfloat	ma[16];
			};
		public:
			stSANMATRIX4X4(const sfloat v11 = 0.0, const sfloat v12 = 0.0, const sfloat v13 = 0.0, const sfloat v14 = 0.0,
				const sfloat v21 = 0.0, const sfloat v22 = 0.0, const sfloat v23 = 0.0, const sfloat v24 = 0.0,
				const sfloat v31 = 0.0, const sfloat v32 = 0.0, const sfloat v33 = 0.0, const sfloat v34 = 0.0,
				const sfloat v41 = 0.0, const sfloat v42 = 0.0, const sfloat v43 = 0.0, const sfloat v44 = 0.0);
			~stSANMATRIX4X4();
		};

		struct SANMATRIX2X2 : public stSANMATRIX2X2
		{
		public:
			SANMATRIX2X2();
			SANMATRIX2X2(const SANMATRIX2X2 &Mat2x2);
			SANMATRIX2X2(const SVECTOR2 &Vec2_1, const SVECTOR2 &Vec2_2);
			SANMATRIX2X2(const sfloat e11, const sfloat e12, const sfloat e21, const sfloat e22);
			SANMATRIX2X2(const sfloat* pArray);

			SANMATRIX2X2& operator=(const SANMATRIX2X2 &Mat2x2);
			//SANMATRIX2X2& operator=(const sfloat* pArray);

			/*const sfloat* operator[](const uint32 Position) const;
			sfloat* operator[](const uint32 Position);*/

			SANMATRIX2X2 operator+(const SANMATRIX2X2 &Mat2x2) const;
			SANMATRIX2X2 operator-(const SANMATRIX2X2 &Mat2x2) const;
			SANMATRIX2X2 operator*(const SANMATRIX2X2 &Mat2x2) const;
			SANMATRIX2X2 operator*(const sfloat Value) const;

			bool operator == (const SANMATRIX2X2 &Mat2x2) const;
			bool operator != (const SANMATRIX2X2 &Mat2x2) const;

			void iIdentity();
			void iAdd(const SVECTOR2 &Vec2, const uint8 Index, bool bIsRow = false);
			void iSub(const SVECTOR2 &Vec2, const uint8 Index, bool bIsRow = false);
			sfloat iSum() const;
			void iSetRow(const SVECTOR2 &Vec2, const uint8 Index);
			void iSetCol(const SVECTOR2 &Vec2, const uint8 Index);
			SVECTOR2 iGetRow(const uint8 Index) const;
			SVECTOR2 iGetCol(const uint8 Index) const;
		};
		typedef SANMATRIX2X2*	lpSANMATRIX2X2;

		struct SANMATRIX3X3 : public stSANMATRIX3X3
		{
		public:
			SANMATRIX3X3(){};
			SANMATRIX3X3(const SANMATRIX3X3 &Mat3x3);
			SANMATRIX3X3(const SVECTOR3 &Vec3_1,
				const SVECTOR3 &Vec3_2,
				const SVECTOR3 &Vec3_3);
			SANMATRIX3X3(const sfloat e11, const sfloat e12, const sfloat e13,
				const sfloat e21, const sfloat e22, const sfloat e23,
				const sfloat e31, const sfloat e32, const sfloat e33);
			SANMATRIX3X3(const sfloat* pArray);

			SANMATRIX3X3& operator=(const SANMATRIX3X3 &Mat3x3);
			//SANMATRIX3X3& operator=(const sfloat* pArray);

			/*const sfloat* operator[](const uint32 Position) const;
			sfloat* operator[](const uint32 Position);*/

			SANMATRIX3X3 operator+(const SANMATRIX3X3 &Mat3x3) const;
			SANMATRIX3X3 operator-(const SANMATRIX3X3 &Mat3x3) const;
			SANMATRIX3X3 operator*(const SANMATRIX3X3 &Mat3x3) const;
			SANMATRIX3X3 operator*(const sfloat Value) const;

			bool operator == (const SANMATRIX3X3 &Mat3x3) const;
			bool operator != (const SANMATRIX3X3 &Mat3x3) const;

			void iIdentity();
			void iAdd(const SVECTOR3 &Vec3, const uint8 Index, bool bIsRow = false);
			void iSub(const SVECTOR3 &Vec3, const uint8 Index, bool bIsRow = false);
			sfloat iSum() const;
			void iSetRow(const SVECTOR3 &Vec3, const uint8 Index);
			void iSetCol(const SVECTOR3 &Vec3, const uint8 Index);
			SVECTOR3 iGetRow(const uint8 Index) const;
			SVECTOR3 iGetCol(const uint8 Index) const;
		};
		typedef SANMATRIX3X3*	lpSANMATRIX3X3;

		struct SANMATRIX4X4 : public stSANMATRIX4X4
		{
		public:
			SANMATRIX4X4();
			SANMATRIX4X4(const SANMATRIX4X4 &Mat4x4);
			SANMATRIX4X4(const SPOINT4 &Point4_1,
				const SPOINT4 &Point4_2,
				const SPOINT4 &Point4_3,
				const SPOINT4 &Point4_4);
			SANMATRIX4X4(const SVECTOR4 &Vec4_1,
				const SVECTOR4 &Vec4_2,
				const SVECTOR4 &Vec4_3,
				const SVECTOR4 &Vec4_4);
			SANMATRIX4X4(const sfloat e11, const sfloat e12, const sfloat e13, const sfloat e14,
				const sfloat e21, const sfloat e22, const sfloat e23, const sfloat e24,
				const sfloat e31, const sfloat e32, const sfloat e33, const sfloat e34,
				const sfloat e41, const sfloat e42, const sfloat e43, const sfloat e44);
			SANMATRIX4X4(const sfloat* pArray);

			SANMATRIX4X4& operator=(const SANMATRIX4X4 &Mat4x4);
			//SANMATRIX4X4& operator=(const sfloat* pArray);

			/*const sfloat* operator[](const uint32 Position) const;
			sfloat* operator[](const uint32 Position);*/

			SANMATRIX4X4 operator+(const SANMATRIX4X4 &Mat4x4) const;
			SANMATRIX4X4 operator-(const SANMATRIX4X4 &Mat4x4) const;
			SANMATRIX4X4 operator*(const SANMATRIX4X4 &Mat4x4) const;
			SANMATRIX4X4 operator*(const sfloat Value) const;

			bool operator==(const SANMATRIX4X4 &Mat4x4) const;
			bool operator!=(const SANMATRIX4X4 &Mat4x4) const;

			void iIdentity();
			void iAdd(const SPOINT4 &Point4, const uint8 Index, bool bIsRow = false);
			void iAdd(const SVECTOR4 &Vec4, const uint8 Index, bool bIsRow = false);
			void iSub(const SPOINT4 &Point4, const uint8 Index, bool bIsRow = false);
			void iSub(const SVECTOR4 &Vec4, const uint8 Index, bool bIsRow = false);
			sfloat iSum() const;
			void iSetRow(const SPOINT4 &Point4, const uint8 Index);
			void iSetRow(const SVECTOR4 &Vec4, const uint8 Index);
			void iSetCol(const SPOINT4 &Point4, const uint8 Index);
			void iSetCol(const SVECTOR4 &Vec4, const uint8 Index);
			SVECTOR4 iGetRow(const uint8 Index) const;
			SVECTOR4 iGetCol(const uint8 Index) const;
		};
		typedef SANMATRIX4X4*	lpSANMATRIX4X4;

		SVECTOR3 operator*(const SVECTOR3 &Vec3, const SANMATRIX4X4 &Mat16);
		SVECTOR4 operator*(const SVECTOR4 &Vec4, const SANMATRIX4X4 &Mat16);
#endif
	}
}