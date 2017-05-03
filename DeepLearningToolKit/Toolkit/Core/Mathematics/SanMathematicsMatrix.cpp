#include"SanMathematicsMatrix.h"
using namespace std;
using namespace San;
using namespace San::Mathematics;
San::Mathematics::stSANMATRIX2X2::stSANMATRIX2X2(const sfloat v11, const sfloat v12, const sfloat v21, const sfloat v22)
	:e11(v11), e12(v12), e21(v21), e22(v22)
{
}
San::Mathematics::stSANMATRIX2X2::~stSANMATRIX2X2()
{
}
San::Mathematics::stSANMATRIX3X3::stSANMATRIX3X3(const sfloat v11, const sfloat v12, const sfloat v13,
	const sfloat v21, const sfloat v22, const sfloat v23,
	const sfloat v31, const sfloat v32, const sfloat v33)
	:e11(v11), e12(v12), e13(v13),
	e21(v21), e22(v22), e23(v23),
	e31(v31), e32(v32), e33(v33)
{
}
San::Mathematics::stSANMATRIX3X3::~stSANMATRIX3X3()
{
}
San::Mathematics::stSANMATRIX4X4::stSANMATRIX4X4(const sfloat v11, const sfloat v12, const sfloat v13, const sfloat v14,
	const sfloat v21, const sfloat v22, const sfloat v23, const sfloat v24,
	const sfloat v31, const sfloat v32, const sfloat v33, const sfloat v34,
	const sfloat v41, const sfloat v42, const sfloat v43, const sfloat v44)
	:e11(v11), e12(v12), e13(v13), e14(v14),
	e21(v21), e22(v22), e23(v23), e24(v24),
	e31(v31), e32(v32), e33(v33), e34(v34),
	e41(v41), e42(v42), e43(v43), e44(v44)
{
}
San::Mathematics::stSANMATRIX4X4::~stSANMATRIX4X4()
{
}
San::Mathematics::SANMATRIX2X2::SANMATRIX2X2()
	:stSANMATRIX2X2(1.0, 0.0, 0.0, 1.0)
{
}
San::Mathematics::SANMATRIX2X2::SANMATRIX2X2(const SANMATRIX2X2 &Mat2x2)
	:stSANMATRIX2X2(Mat2x2.e11, Mat2x2.e12, Mat2x2.e21, Mat2x2.e22)
{
}
San::Mathematics::SANMATRIX2X2::SANMATRIX2X2(const SVECTOR2 &Vec2_1, const SVECTOR2 &Vec2_2)
	:stSANMATRIX2X2(Vec2_1.x, Vec2_1.y, Vec2_2.x, Vec2_2.y)
{
}
San::Mathematics::SANMATRIX2X2::SANMATRIX2X2(const sfloat e11, const sfloat e12, const sfloat e21, const sfloat e22)
	:stSANMATRIX2X2(e11, e12, e21, e22)
{
}
San::Mathematics::SANMATRIX2X2::SANMATRIX2X2(const sfloat* pArray)
	:stSANMATRIX2X2(pArray[0], pArray[1], pArray[2], pArray[3])
{
}
San::Mathematics::SANMATRIX2X2& San::Mathematics::SANMATRIX2X2::operator=(const SANMATRIX2X2 &Mat2x2)
{
	this->e11 = Mat2x2.e11;
	this->e12 = Mat2x2.e12;
	this->e21 = Mat2x2.e21;
	this->e22 = Mat2x2.e22;

	return *this;
}
//San::Mathematics::SANMATRIX2X2& San::Mathematics::SANMATRIX2X2::operator=(const sfloat* pArray)
//{
//
//}
//const sfloat* San::Mathematics::SANMATRIX2X2::operator[](const uint32 Position) const
//{
//
//}
//sfloat* San::Mathematics::SANMATRIX2X2::operator[](const uint32 Position)
//{
//
//}
San::Mathematics::SANMATRIX2X2 San::Mathematics::SANMATRIX2X2::operator+(const SANMATRIX2X2 &Mat2x2) const
{
	return SANMATRIX2X2(this->e11 + Mat2x2.e11, this->e12 + Mat2x2.e12, this->e21 + Mat2x2.e21, this->e22 + Mat2x2.e22);
}
San::Mathematics::SANMATRIX2X2 San::Mathematics::SANMATRIX2X2::operator-(const SANMATRIX2X2 &Mat2x2) const
{
	return SANMATRIX2X2(this->e11 - Mat2x2.e11, this->e12 - Mat2x2.e12, this->e21 - Mat2x2.e21, this->e22 - Mat2x2.e22);
}
San::Mathematics::SANMATRIX2X2 San::Mathematics::SANMATRIX2X2::operator*(const SANMATRIX2X2 &Mat2x2) const
{
	return SANMATRIX2X2(this->e11 + Mat2x2.e11, this->e12 + Mat2x2.e12, this->e21 + Mat2x2.e21, this->e22 + Mat2x2.e22);
}
San::Mathematics::SANMATRIX2X2 San::Mathematics::SANMATRIX2X2::operator*(const sfloat Value) const
{
	return SANMATRIX2X2(this->e11 * Value, this->e12 * Value, this->e21 * Value, this->e22 * Value);
}
bool San::Mathematics::SANMATRIX2X2::operator==(const SANMATRIX2X2 &Mat2x2) const
{
	for (int8 seek = 0; seek < 4; seek = seek + 1)
	{
		if (!::gloIsFloatEqual(this->ma[seek], Mat2x2.ma[seek])){ return false; }
	}

	return true;
}
bool San::Mathematics::SANMATRIX2X2::operator!=(const SANMATRIX2X2 &Mat2x2) const
{
	for (int8 seek = 0; seek < 4; seek = seek + 1)
	{
		if (!::gloIsFloatEqual(this->ma[seek], Mat2x2.ma[seek])){ return true; }
	}

	return false;
}
void San::Mathematics::SANMATRIX2X2::iIdentity()
{
	this->e11 = 1.0;
	this->e12 = 0.0;
	this->e21 = 0.0;
	this->e22 = 1.0;
}
void San::Mathematics::SANMATRIX2X2::iAdd(const SVECTOR2 &Vec2, const uint8 Index, bool bIsRow)
{
	if (Index > 1)
	{
		return;
	}
	if (bIsRow)
	{
		this->m[Index][0] = this->m[Index][0] + Vec2.x;
		this->m[Index][1] = this->m[Index][1] + Vec2.y;
	}
	else
	{
		this->m[0][Index] = this->m[0][Index] + Vec2.x;
		this->m[1][Index] = this->m[1][Index] + Vec2.y;
	}
}
void San::Mathematics::SANMATRIX2X2::iSub(const SVECTOR2 &Vec2, const uint8 Index, bool bIsRow)
{
	if (Index > 1)
	{
		return;
	}
	if (bIsRow)
	{
		this->m[Index][0] = this->m[Index][0] - Vec2.x;
		this->m[Index][1] = this->m[Index][1] - Vec2.y;
	}
	else
	{
		this->m[0][Index] = this->m[0][Index] - Vec2.x;
		this->m[1][Index] = this->m[1][Index] - Vec2.y;
	}
}
sfloat San::Mathematics::SANMATRIX2X2::iSum() const
{
	//Error
	return this->e11 + this->e12 + this->e21 + this->e22;
}
void San::Mathematics::SANMATRIX2X2::iSetRow(const SVECTOR2 &Vec2, const uint8 Index)
{

}
void San::Mathematics::SANMATRIX2X2::iSetCol(const SVECTOR2 &Vec2, const uint8 Index)
{

}
SVECTOR2 San::Mathematics::SANMATRIX2X2::iGetRow(const uint8 Index) const
{
	if (Index <= 1)
	{
		return SVECTOR2(this->m[Index]);
	}
	return SVECTOR2(0.0, 0.0);
}
SVECTOR2 San::Mathematics::SANMATRIX2X2::iGetCol(const uint8 Index) const
{
	if (Index <= 1)
	{
		return SVECTOR2(this->m[0][Index], this->m[1][Index]);
	}
	return SVECTOR2(0.0, 0.0);
}
San::Mathematics::SANMATRIX3X3::SANMATRIX3X3(const SANMATRIX3X3 &Mat3x3)
{
	this->e11 = Mat3x3.e11; this->e12 = Mat3x3.e12; this->e13 = Mat3x3.e13;
	this->e21 = Mat3x3.e21; this->e22 = Mat3x3.e22; this->e23 = Mat3x3.e23;
	this->e31 = Mat3x3.e31; this->e32 = Mat3x3.e32; this->e33 = Mat3x3.e33;
}
San::Mathematics::SANMATRIX3X3::SANMATRIX3X3(const SVECTOR3 &Vec3_1, const SVECTOR3 &Vec3_2, const SVECTOR3 &Vec3_3)
{
	this->e11 = Vec3_1.x; this->e12 = Vec3_1.y; this->e13 = Vec3_1.z;
	this->e21 = Vec3_2.x; this->e22 = Vec3_2.y; this->e23 = Vec3_2.z;
	this->e31 = Vec3_3.x; this->e32 = Vec3_3.y; this->e33 = Vec3_3.z;
}
San::Mathematics::SANMATRIX3X3::SANMATRIX3X3(const sfloat e11, const sfloat e12, const sfloat e13,
	const sfloat e21, const sfloat e22, const sfloat e23,
	const sfloat e31, const sfloat e32, const sfloat e33)
{
	this->e11 = e11; this->e12 = e12; this->e13 = e13;
	this->e21 = e21; this->e22 = e22; this->e23 = e23;
	this->e31 = e31; this->e32 = e32; this->e33 = e33;
}
San::Mathematics::SANMATRIX3X3::SANMATRIX3X3(const sfloat* pArray)
{
	cSanMemoryFuncSet::iMemCopy((void*) this->ma, (const void*) pArray, sizeof(sfloat) * 9);
}
San::Mathematics::SANMATRIX3X3& San::Mathematics::SANMATRIX3X3::operator=(const SANMATRIX3X3 &Mat3x3)
{
	this->e11 = Mat3x3.e11; this->e12 = Mat3x3.e12; this->e13 = Mat3x3.e13;
	this->e21 = Mat3x3.e21; this->e22 = Mat3x3.e22; this->e23 = Mat3x3.e23;
	this->e31 = Mat3x3.e31; this->e32 = Mat3x3.e32; this->e33 = Mat3x3.e33;
	return *this;
}
//San::Mathematics::SANMATRIX3X3& San::Mathematics::SANMATRIX3X3::operator=(const sfloat* pArray)
//{
//	::gloMemCopy((void*) this->ma, (const void*) pArray, sizeof(sfloat) * 9);
//	return *this;
//}
San::Mathematics::SANMATRIX3X3 San::Mathematics::SANMATRIX3X3::operator+(const SANMATRIX3X3 &Mat3x3) const
{
	return SANMATRIX3X3(this->e11 + Mat3x3.e11, this->e12 + Mat3x3.e12, this->e13 + Mat3x3.e13,
		this->e21 + Mat3x3.e21, this->e22 + Mat3x3.e22, this->e23 + Mat3x3.e23,
		this->e31 + Mat3x3.e31, this->e32 + Mat3x3.e32, this->e33 + Mat3x3.e33);
}
San::Mathematics::SANMATRIX3X3 San::Mathematics::SANMATRIX3X3::operator-(const SANMATRIX3X3 &Mat3x3) const
{
	return SANMATRIX3X3(this->e11 - Mat3x3.e11, this->e12 - Mat3x3.e12, this->e13 - Mat3x3.e13,
		this->e21 - Mat3x3.e21, this->e22 - Mat3x3.e22, this->e23 - Mat3x3.e23,
		this->e31 - Mat3x3.e31, this->e32 - Mat3x3.e32, this->e33 - Mat3x3.e33);
}
San::Mathematics::SANMATRIX3X3 San::Mathematics::SANMATRIX3X3::operator*(const SANMATRIX3X3 &Mat3x3) const
{
	SANMATRIX3X3 Mat;
	SVECTOR3 Vec;
	for (uint8 s_row = 0; s_row < 3; s_row = s_row + 1)
	{
		for (uint8 s_col = 0; s_col < 3; s_col = s_col + 1)
		{
			Vec.x = this->m[s_row][0] * Mat3x3.m[0][s_col];
			Vec.y = this->m[s_row][1] * Mat3x3.m[1][s_col];
			Vec.z = this->m[s_row][2] * Mat3x3.m[2][s_col];
			Mat.m[s_row][s_col] = Vec.x + Vec.y + Vec.z;
		}
	}
	return Mat;
}
San::Mathematics::SANMATRIX3X3 San::Mathematics::SANMATRIX3X3::operator*(const sfloat Value) const
{
	return SANMATRIX3X3(this->e11*Value, this->e12*Value, this->e13*Value,
		this->e21*Value, this->e22*Value, this->e23*Value,
		this->e31*Value, this->e32*Value, this->e33*Value);
}
bool San::Mathematics::SANMATRIX3X3::operator==(const SANMATRIX3X3 &Mat3x3) const
{
	for (int8 seek = 0; seek < 3; seek = seek + 1)
	{
		if (!::gloIsFloatEqual(this->ma[seek], Mat3x3.ma[seek])){ return false; }
	}

	return true;
}
bool San::Mathematics::SANMATRIX3X3::operator!=(const SANMATRIX3X3 &Mat3x3) const
{
	for (int8 seek = 0; seek < 3; seek = seek + 1)
	{
		if (!::gloIsFloatEqual(this->ma[seek], Mat3x3.ma[seek])){ return true; }
	}

	return false;
}
void San::Mathematics::SANMATRIX3X3::iIdentity()
{
	this->e11 = 1.0; this->e12 = 0.0; this->e13 = 0.0;
	this->e21 = 0.0; this->e22 = 1.0; this->e23 = 0.0;
	this->e31 = 0.0; this->e32 = 0.0; this->e33 = 1.0;
}
void San::Mathematics::SANMATRIX3X3::iAdd(const SVECTOR3 &Vec3, const uint8 Index, bool bIsRow)
{
	if (Index > 2)
	{
		return;
	}
	if (bIsRow)
	{
		this->m[Index][0] = this->m[Index][0] + Vec3.x;
		this->m[Index][1] = this->m[Index][1] + Vec3.y;
		this->m[Index][2] = this->m[Index][2] + Vec3.z;
	}
	else
	{
		this->m[0][Index] = this->m[0][Index] + Vec3.x;
		this->m[1][Index] = this->m[1][Index] + Vec3.y;
		this->m[2][Index] = this->m[2][Index] + Vec3.z;
	}
}
void San::Mathematics::SANMATRIX3X3::iSub(const SVECTOR3 &Vec3, const uint8 Index, bool bIsRow)
{
	if (Index > 2)
	{
		return;
	}
	if (bIsRow)
	{
		this->m[Index][0] = this->m[Index][0] - Vec3.x;
		this->m[Index][1] = this->m[Index][1] - Vec3.y;
		this->m[Index][2] = this->m[Index][2] - Vec3.z;
	}
	else
	{
		this->m[0][Index] = this->m[0][Index] - Vec3.x;
		this->m[1][Index] = this->m[1][Index] - Vec3.y;
		this->m[2][Index] = this->m[2][Index] - Vec3.z;
	}
}
sfloat San::Mathematics::SANMATRIX3X3::iSum() const
{
	return this->e11*this->e22*this->e33 + this->e12*this->e23*this->e31 + this->e13*this->e21*this->e32 -
		this->e13*this->e22*this->e31 - this->e12*this->e21*this->e33 - this->e11*this->e23*this->e32;
}
void San::Mathematics::SANMATRIX3X3::iSetRow(const SVECTOR3 &Vec3, const uint8 Index)
{
	if (Index <= 2)
	{
		this->m[Index][0] = Vec3.x;
		this->m[Index][1] = Vec3.y;
		this->m[Index][2] = Vec3.z;
	}
}
void San::Mathematics::SANMATRIX3X3::iSetCol(const SVECTOR3 &Vec3, const uint8 Index)
{
	if (Index <= 2)
	{
		this->m[0][Index] = Vec3.x;
		this->m[1][Index] = Vec3.y;
		this->m[2][Index] = Vec3.z;
	}
}
San::Mathematics::SVECTOR3 San::Mathematics::SANMATRIX3X3::iGetRow(const uint8 Index) const
{
	if (Index <= 2)
	{
		return SVECTOR3(this->m[Index]);
	}
	return SVECTOR3(0.0, 0.0, 0.0);
}
San::Mathematics::SVECTOR3 San::Mathematics::SANMATRIX3X3::iGetCol(const uint8 Index) const
{
	if (Index <= 2)
	{
		return SVECTOR3(this->m[0][Index], this->m[1][Index], this->m[2][Index]);
	}
	return SVECTOR3(0.0, 0.0, 0.0);
}
San::Mathematics::SANMATRIX4X4::SANMATRIX4X4()
{
	this->e11 = 0.0; this->e12 = 0.0; this->e13 = 0.0; this->e14 = 0.0;
	this->e21 = 0.0; this->e22 = 0.0; this->e23 = 0.0; this->e24 = 0.0;
	this->e31 = 0.0; this->e32 = 0.0; this->e33 = 0.0; this->e34 = 0.0;
	this->e41 = 0.0; this->e42 = 0.0; this->e43 = 0.0; this->e44 = 0.0;
}
San::Mathematics::SANMATRIX4X4::SANMATRIX4X4(const SANMATRIX4X4 &Mat4x4)
{
	this->e11 = Mat4x4.e11; this->e12 = Mat4x4.e12; this->e13 = Mat4x4.e13; this->e14 = Mat4x4.e14;
	this->e21 = Mat4x4.e21; this->e22 = Mat4x4.e22; this->e23 = Mat4x4.e23; this->e24 = Mat4x4.e24;
	this->e31 = Mat4x4.e31; this->e32 = Mat4x4.e32; this->e33 = Mat4x4.e33; this->e34 = Mat4x4.e34;
	this->e41 = Mat4x4.e41; this->e42 = Mat4x4.e42; this->e43 = Mat4x4.e43; this->e44 = Mat4x4.e44;
}
San::Mathematics::SANMATRIX4X4::SANMATRIX4X4(const SPOINT4 &Point4_1, const SPOINT4 &Point4_2, const SPOINT4 &Point4_3, const SPOINT4 &Point4_4)
{
	this->e11 = Point4_1.x; this->e12 = Point4_1.y; this->e13 = Point4_1.z; this->e14 = Point4_1.w;
	this->e21 = Point4_2.x; this->e22 = Point4_2.y; this->e23 = Point4_2.z; this->e24 = Point4_2.w;
	this->e31 = Point4_3.x; this->e32 = Point4_3.y; this->e33 = Point4_3.z; this->e34 = Point4_3.w;
	this->e41 = Point4_4.x; this->e42 = Point4_4.y; this->e43 = Point4_4.z; this->e44 = Point4_4.w;
}
San::Mathematics::SANMATRIX4X4::SANMATRIX4X4(const SVECTOR4 &Vec4_1, const SVECTOR4 &Vec4_2, const SVECTOR4 &Vec4_3, const SVECTOR4 &Vec4_4)
{
	this->e11 = Vec4_1.x; this->e12 = Vec4_1.y; this->e13 = Vec4_1.z; this->e14 = Vec4_1.w;
	this->e21 = Vec4_2.x; this->e22 = Vec4_2.y; this->e23 = Vec4_2.z; this->e24 = Vec4_2.w;
	this->e31 = Vec4_3.x; this->e32 = Vec4_3.y; this->e33 = Vec4_3.z; this->e34 = Vec4_3.w;
	this->e41 = Vec4_4.x; this->e42 = Vec4_4.y; this->e43 = Vec4_4.z; this->e44 = Vec4_4.w;
}
San::Mathematics::SANMATRIX4X4::SANMATRIX4X4(const sfloat e11, const sfloat e12, const sfloat e13, const sfloat e14,
	const sfloat e21, const sfloat e22, const sfloat e23, const sfloat e24,
	const sfloat e31, const sfloat e32, const sfloat e33, const sfloat e34,
	const sfloat e41, const sfloat e42, const sfloat e43, const sfloat e44)
{
	this->e11 = e11; this->e12 = e12; this->e13 = e13; this->e14 = e14;
	this->e21 = e21; this->e22 = e22; this->e23 = e23; this->e24 = e24;
	this->e31 = e31; this->e32 = e32; this->e33 = e33; this->e34 = e34;
	this->e41 = e41; this->e42 = e42; this->e43 = e43; this->e44 = e44;
}
San::Mathematics::SANMATRIX4X4::SANMATRIX4X4(const sfloat* pArray)
{
	cSanMemoryFuncSet::iMemCopy((void*)this->ma, (const void*) pArray, sizeof(sfloat) * 16);
}
San::Mathematics::SANMATRIX4X4& San::Mathematics::SANMATRIX4X4::operator=(const SANMATRIX4X4 &Mat4x4)
{
	this->e11 = Mat4x4.e11; this->e12 = Mat4x4.e12; this->e13 = Mat4x4.e13; this->e14 = Mat4x4.e14;
	this->e21 = Mat4x4.e21; this->e22 = Mat4x4.e22; this->e23 = Mat4x4.e23; this->e24 = Mat4x4.e24;
	this->e31 = Mat4x4.e31; this->e32 = Mat4x4.e32; this->e33 = Mat4x4.e33; this->e34 = Mat4x4.e34;
	this->e41 = Mat4x4.e41; this->e42 = Mat4x4.e42; this->e43 = Mat4x4.e43; this->e44 = Mat4x4.e44;
	return *this;
}
//San::Mathematics::SANMATRIX4X4& San::Mathematics::SANMATRIX4X4::operator=(const sfloat* pArray)
//{
//	::gloMemCopy((void*)this->ma, (const void*) pArray, sizeof(sfloat) * 16);
//	return *this;
//}
San::Mathematics::SANMATRIX4X4 San::Mathematics::SANMATRIX4X4::operator+(const SANMATRIX4X4 &Mat4x4) const
{
	return SANMATRIX4X4(this->e11 + Mat4x4.e11, this->e12 + Mat4x4.e12, this->e13 + Mat4x4.e13, this->e14 + Mat4x4.e14,
		this->e21 + Mat4x4.e21, this->e22 + Mat4x4.e22, this->e23 + Mat4x4.e23, this->e24 + Mat4x4.e24,
		this->e31 + Mat4x4.e31, this->e32 + Mat4x4.e32, this->e33 + Mat4x4.e33, this->e34 + Mat4x4.e34,
		this->e41 + Mat4x4.e41, this->e42 + Mat4x4.e42, this->e43 + Mat4x4.e43, this->e44 + Mat4x4.e44);
}
San::Mathematics::SANMATRIX4X4 San::Mathematics::SANMATRIX4X4::operator-(const SANMATRIX4X4 &Mat4x4) const
{
	return SANMATRIX4X4(this->e11 - Mat4x4.e11, this->e12 - Mat4x4.e12, this->e13 - Mat4x4.e13, this->e14 - Mat4x4.e14,
		this->e21 - Mat4x4.e21, this->e22 - Mat4x4.e22, this->e23 - Mat4x4.e23, this->e24 - Mat4x4.e24,
		this->e31 - Mat4x4.e31, this->e32 - Mat4x4.e32, this->e33 - Mat4x4.e33, this->e34 - Mat4x4.e34,
		this->e41 - Mat4x4.e41, this->e42 - Mat4x4.e42, this->e43 - Mat4x4.e43, this->e44 - Mat4x4.e44);
}
San::Mathematics::SANMATRIX4X4 San::Mathematics::SANMATRIX4X4::operator*(const SANMATRIX4X4 &Mat4x4) const
{
	SANMATRIX4X4 Mat;
	SVECTOR4 Vec;
	for (uint8 s_row = 0; s_row < 4; s_row = s_row + 1)
	{
		for (uint8 s_col = 0; s_col < 4; s_col = s_col + 1)
		{
			Vec.x = this->m[s_row][0] * Mat4x4.m[0][s_col];
			Vec.y = this->m[s_row][1] * Mat4x4.m[1][s_col];
			Vec.z = this->m[s_row][2] * Mat4x4.m[2][s_col];
			Vec.w = this->m[s_row][3] * Mat4x4.m[3][s_col];
			Mat.m[s_row][s_col] = Vec.x + Vec.y + Vec.z + Vec.w;
		}
	}
	return Mat;
}
San::Mathematics::SANMATRIX4X4 San::Mathematics::SANMATRIX4X4::operator*(const sfloat Value) const
{
	return SANMATRIX4X4(this->e11*Value, this->e12*Value, this->e13*Value, this->e14*Value,
		this->e21*Value, this->e22*Value, this->e23*Value, this->e24*Value,
		this->e31*Value, this->e32*Value, this->e33*Value, this->e34*Value,
		this->e41*Value, this->e42*Value, this->e43*Value, this->e44*Value);
}
bool San::Mathematics::SANMATRIX4X4::operator==(const SANMATRIX4X4 &Mat4x4) const
{
	for (int8 seek = 0; seek < 16; seek = seek + 1)
	{
		if (!::gloIsFloatEqual(this->ma[seek], Mat4x4.ma[seek])){ return false; }
	}

	return true;
}
bool San::Mathematics::SANMATRIX4X4::operator!=(const SANMATRIX4X4 &Mat4x4) const
{
	for (uint8 seek = 0; seek < 16; seek = seek + 1)
	{
		if (!::gloIsFloatEqual(this->ma[seek], Mat4x4.ma[seek])){ return true; }
	}

	return false;
}
void San::Mathematics::SANMATRIX4X4::iIdentity()
{
	this->e11 = 1.0f; this->e12 = 0.0; this->e13 = 0.0; this->e14 = 0.0;
	this->e21 = 0.0; this->e22 = 1.0f; this->e23 = 0.0; this->e24 = 0.0;
	this->e31 = 0.0; this->e32 = 0.0; this->e33 = 1.0f; this->e34 = 0.0;
	this->e41 = 0.0; this->e42 = 0.0; this->e43 = 0.0; this->e44 = 1.0f;
}
void San::Mathematics::SANMATRIX4X4::iAdd(const SPOINT4 &Point4, const uint8 Index, bool bIsRow)
{
	if (Index>3)
	{
		return;
	}
	if (bIsRow)
	{
		this->m[Index][0] = this->m[Index][0] + Point4.x;
		this->m[Index][1] = this->m[Index][1] + Point4.y;
		this->m[Index][2] = this->m[Index][2] + Point4.z;
		this->m[Index][3] = this->m[Index][3] + Point4.w;
	}
	else
	{
		this->m[0][Index] = this->m[0][Index] + Point4.x;
		this->m[1][Index] = this->m[1][Index] + Point4.y;
		this->m[2][Index] = this->m[2][Index] + Point4.z;
		this->m[3][Index] = this->m[3][Index] + Point4.w;
	}
}
void San::Mathematics::SANMATRIX4X4::iAdd(const SVECTOR4 &Vec4, const uint8 Index, bool bIsRow)
{
	if (Index>3)
	{
		return;
	}
	if (bIsRow)
	{
		this->m[Index][0] = this->m[Index][0] + Vec4.x;
		this->m[Index][1] = this->m[Index][1] + Vec4.y;
		this->m[Index][2] = this->m[Index][2] + Vec4.z;
		this->m[Index][3] = this->m[Index][3] + Vec4.w;
	}
	else
	{
		this->m[0][Index] = this->m[0][Index] + Vec4.x;
		this->m[1][Index] = this->m[1][Index] + Vec4.y;
		this->m[2][Index] = this->m[2][Index] + Vec4.z;
		this->m[3][Index] = this->m[3][Index] + Vec4.w;
	}
}
void San::Mathematics::SANMATRIX4X4::iSub(const SPOINT4 &Point4, const uint8 Index, bool bIsRow)
{
	if (Index>3)
	{
		return;
	}
	if (bIsRow)
	{
		this->m[Index][0] = this->m[Index][0] - Point4.x;
		this->m[Index][1] = this->m[Index][1] - Point4.y;
		this->m[Index][2] = this->m[Index][2] - Point4.z;
		this->m[Index][3] = this->m[Index][3] - Point4.w;
	}
	else
	{
		this->m[0][Index] = this->m[0][Index] - Point4.x;
		this->m[1][Index] = this->m[1][Index] - Point4.y;
		this->m[2][Index] = this->m[2][Index] - Point4.z;
		this->m[3][Index] = this->m[3][Index] - Point4.w;
	}
}
void San::Mathematics::SANMATRIX4X4::iSub(const SVECTOR4 &Vec4, const uint8 Index, bool bIsRow)
{
	if (Index>3)
	{
		return;
	}
	if (bIsRow)
	{
		this->m[Index][0] = this->m[Index][0] - Vec4.x;
		this->m[Index][1] = this->m[Index][1] - Vec4.y;
		this->m[Index][2] = this->m[Index][2] - Vec4.z;
		this->m[Index][3] = this->m[Index][3] - Vec4.w;
	}
	else
	{
		this->m[0][Index] = this->m[0][Index] - Vec4.x;
		this->m[1][Index] = this->m[1][Index] - Vec4.y;
		this->m[2][Index] = this->m[2][Index] - Vec4.z;
		this->m[3][Index] = this->m[3][Index] - Vec4.w;
	}
}
sfloat San::Mathematics::SANMATRIX4X4::iSum() const
{
	return 0.0;
}
void San::Mathematics::SANMATRIX4X4::iSetRow(const SPOINT4 &Point4, const uint8 Index)
{
	if (Index <= 3)
	{
		this->m[Index][0] = Point4.x;
		this->m[Index][1] = Point4.y;
		this->m[Index][2] = Point4.z;
		this->m[Index][3] = Point4.w;
	}
}
void San::Mathematics::SANMATRIX4X4::iSetRow(const SVECTOR4 &Vec4, const uint8 Index)
{
	if (Index <= 3)
	{
		this->m[Index][0] = Vec4.x;
		this->m[Index][1] = Vec4.y;
		this->m[Index][2] = Vec4.z;
		this->m[Index][3] = Vec4.w;
	}
}
void San::Mathematics::SANMATRIX4X4::iSetCol(const SPOINT4 &Point4, const uint8 Index)
{
	if (Index <= 3)
	{
		this->m[0][Index] = Point4.x;
		this->m[1][Index] = Point4.y;
		this->m[2][Index] = Point4.z;
		this->m[3][Index] = Point4.w;
	}
}
void San::Mathematics::SANMATRIX4X4::iSetCol(const SVECTOR4 &Vec4, const uint8 Index)
{
	if (Index <= 3)
	{
		this->m[0][Index] = Vec4.x;
		this->m[1][Index] = Vec4.y;
		this->m[2][Index] = Vec4.z;
		this->m[3][Index] = Vec4.w;
	}
}
San::Mathematics::SVECTOR4 San::Mathematics::SANMATRIX4X4::iGetRow(const uint8 Index) const
{
	if (Index <= 3)
	{
		return SVECTOR4(this->m[Index]);
	}
	return SVECTOR4(0.0, 0.0, 0.0, 0.0);
}
San::Mathematics::SVECTOR4 San::Mathematics::SANMATRIX4X4::iGetCol(const uint8 Index) const
{
	if (Index <= 3)
	{
		return SVECTOR4(this->m[0][Index], this->m[1][Index], this->m[2][Index], this->m[3][Index]);
	}
	return SVECTOR4(0.0, 0.0, 0.0, 0.0);
}