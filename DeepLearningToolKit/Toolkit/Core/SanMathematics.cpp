#include"SanMathematics.h"
using namespace std;
using namespace San;
using namespace San::Mathematics;
SVECTOR3 San::Mathematics::gloAbs(const SVECTOR3 &vector3)
{
	return SVECTOR3(vector3.x>=0.0?(vector3.x):(-vector3.x),
					vector3.y>=0.0?(vector3.y):(-vector3.y),
					vector3.z>=0.0?(vector3.z):(-vector3.z));
}
SVECTOR4 San::Mathematics::gloAbs(const SVECTOR4 &vector4)
{
	return SVECTOR4(vector4.x>=0.0?(vector4.x):(-vector4.x),
					vector4.y>=0.0?(vector4.y):(-vector4.y),
					vector4.z>=0.0?(vector4.z):(-vector4.z),
					vector4.w>=0.0?(vector4.w):(-vector4.w));
}
void San::Mathematics::gloAbs(sfloat* pDest,const sfloat* pSrc,size_t Size)
{
	if((pDest==nullptr)||(pSrc==nullptr)||(Size==0))
	{
		return;
	}
	for(size_t seek=0;seek<Size;seek=seek+1)
	{
		pDest[seek]=(pSrc[seek]<=0.0)?(pSrc[seek]):(-pSrc[seek]);
	}
}
San::sfloat San::Mathematics::gloMod(const SVECTOR3 &vector3) //mod modulus
{
	return sqrt(vector3.x*vector3.x + vector3.y*vector3.y + vector3.z*vector3.z);
}
San::sfloat San::Mathematics::gloMod(const SVECTOR4 &vector4)
{
	return sqrt(vector4.x*vector4.x+vector4.y*vector4.y+vector4.z*vector4.z+vector4.w*vector4.w);
}
SVECTOR3 San::Mathematics::gloNormalize(const SVECTOR3 &vector3)
{
	sfloat mod=::gloMod(vector3);
	if (!::gloIsFloatEqual(mod, 0.0))
	{
		return SVECTOR3(vector3.x / mod, vector3.y / mod, vector3.z / mod);
	}
	return SVECTOR3(0.0,0.0,0.0);
}
SVECTOR4 San::Mathematics::gloNormalize(const SVECTOR4 &vector4)
{
	sfloat mod=::gloMod(vector4);
	if (!::gloIsFloatEqual(mod, 0.0))
	{
		return SVECTOR4(vector4.x/mod,vector4.y/mod,vector4.z/mod,vector4.w/mod);
	}
	return SVECTOR4(0.0,0.0,0.0,0.0);
}
San::sfloat San::Mathematics::gloDot(const SVECTOR3 &vec3_1,const SVECTOR3 &vec3_2)
{
	return vec3_1.x*vec3_2.x+vec3_1.y*vec3_2.y+vec3_1.z*vec3_2.z;
}
San::sfloat San::Mathematics::gloDot(const SVECTOR4 &vec4_1,const SVECTOR4 &vec4_2)
{
	return vec4_1.x*vec4_2.x+vec4_1.y*vec4_2.y+vec4_1.z*vec4_2.z+vec4_1.w*vec4_2.w;
}
SVECTOR3 San::Mathematics::gloCross(const SVECTOR3 &vec3_1,const SVECTOR3 &vec3_2)
{
	return SVECTOR3(vec3_1.y*vec3_2.z-vec3_1.z*vec3_2.y,vec3_1.z*vec3_2.x-vec3_1.x*vec3_2.z,vec3_1.x*vec3_2.y-vec3_1.y*vec3_2.x);
}
SVECTOR4 San::Mathematics::gloCross(const SVECTOR4 &vec4_1,const SVECTOR4 &vec4_2)
{
	return SVECTOR4(vec4_1.x*vec4_2.w-vec4_1.y*vec4_2.x-vec4_1.z*vec4_2.y-vec4_1.w*vec4_2.z,
		vec4_1.y*vec4_2.w+vec4_1.x*vec4_2.x+vec4_1.z*vec4_2.z-vec4_1.w*vec4_2.y,
		vec4_1.z*vec4_2.w+vec4_1.x*vec4_2.y+vec4_1.w*vec4_2.x-vec4_1.y*vec4_2.z,
		vec4_1.w*vec4_2.w+vec4_1.z*vec4_2.x+vec4_1.y*vec4_2.y-vec4_1.x*vec4_2.z);
}
sfloat San::Mathematics::gloLength(const SVECTOR3 &vector3)
{
	return ::sqrt(vector3.x*vector3.x+vector3.y*vector3.y+vector3.z*vector3.z);
}
sfloat San::Mathematics::gloLength(const SVECTOR4 &vector4)
{
	return ::sqrt(vector4.x*vector4.x+vector4.y*vector4.y+vector4.z*vector4.z+vector4.w*vector4.w);
}
SVECTOR3 San::Mathematics::gloCos(const SVECTOR3 &vector3)
{
	return SVECTOR3(::cos(vector3.x),::cos(vector3.y),::cos(vector3.z));
}
SVECTOR4 San::Mathematics::gloCos(const SVECTOR4 &vector4)
{
	return SVECTOR4(::cos(vector4.x),::cos(vector4.y),::cos(vector4.z),::cos(vector4.w));
}
SVECTOR3 San::Mathematics::gloACos(const SVECTOR3 &vector3,const sfloat Length)
{
	SVECTOR3 Theta(0.0,0.0,0.0);
	if (::gloIsFloatEqual(Length, 0.0))
	{
		return Theta;
	}
	Theta = vector3*(1.0 / Length);
	Theta.x=::acos(Theta.x);
	Theta.y=::acos(Theta.y);
	Theta.z=::acos(Theta.z);
	return Theta;
}
SVECTOR4 San::Mathematics::gloACos(const SVECTOR4 &vector4,const sfloat Length)
{
	SVECTOR4 Theta(0.0,0.0,0.0,0.0);
	if (::gloIsFloatEqual(Length, 0.0))
	{
		return Theta;
	}
	Theta = vector4*(1.0 / Length);
	Theta.x = ::acos(Theta.x);
	Theta.y = ::acos(Theta.y);
	Theta.z = ::acos(Theta.z);
	Theta.w = ::acos(Theta.w);
	return Theta;
}
void San::Mathematics::gloSetTranslateMatrix(SANMATRIX4X4 &matrix, const SVECTOR3 &value)
{
	SANMATRIX4X4 TransMatrix;
	TransMatrix.iIdentity();
	TransMatrix.iSetCol((SVECTOR4)value, 3);
	matrix = matrix*TransMatrix;
}
void San::Mathematics::gloSetRotateMatrix(SANMATRIX4X4 &matrix, const SVECTOR3 &angle, const SVECTOR3 &axle)
{
	if (::gloIsFloatEqual(axle.x, 1.0))
	{
		SANMATRIX4X4 RotateMatrix;
		sfloat cos_theta = ::cos(angle.x);
		sfloat sin_theta = ::sin(angle.x);
		RotateMatrix.iIdentity();
		RotateMatrix.e22 = cos_theta;
		RotateMatrix.e23 = -sin_theta;
		RotateMatrix.e32 = sin_theta;
		RotateMatrix.e33 = cos_theta;
		matrix = matrix*RotateMatrix;
	}
	if (::gloIsFloatEqual(axle.y, 1.0))
	{
		SANMATRIX4X4 RotateMatrix;
		sfloat cos_theta = ::cos(angle.y);
		sfloat sin_theta = ::sin(angle.y);
		RotateMatrix.iIdentity();
		RotateMatrix.e11 = cos_theta;
		RotateMatrix.e13 = sin_theta;
		RotateMatrix.e31 = -sin_theta;
		RotateMatrix.e33 = cos_theta;
		matrix = matrix*RotateMatrix;
	}
	if (::gloIsFloatEqual(axle.z, 1.0))
	{
		SANMATRIX4X4 RotateMatrix;
		sfloat cos_theta = ::cos(angle.z);
		sfloat sin_theta = ::sin(angle.z);
		RotateMatrix.iIdentity();
		RotateMatrix.e11 = cos_theta;
		RotateMatrix.e12 = -sin_theta;
		RotateMatrix.e21 = sin_theta;
		RotateMatrix.e22 = cos_theta;
		matrix = matrix*RotateMatrix;
	}
}
void San::Mathematics::gloSetScaleMatrix(SANMATRIX4X4 &matrix, const SVECTOR3 &rate)
{
	SANMATRIX4X4 ScaleMatrix;
	ScaleMatrix.iIdentity();
	ScaleMatrix.e11 = rate.x;
	ScaleMatrix.e22 = rate.y;
	ScaleMatrix.e33 = rate.z;
	matrix = matrix*ScaleMatrix;
}
bool San::Mathematics::gloIsFloatEqual32(const float32 &f1, const float32 &f2, const float64 &bound)
{
	return (f1 <= (f2 + bound)) && (f1 >= (f2 - bound));
}
bool San::Mathematics::gloIsFloatEqual64(const float64 &f1, const float64 &f2, const float64 &bound)
{
	return (f1 <= (f2 + bound)) && (f1 >= (f2 - bound));
}