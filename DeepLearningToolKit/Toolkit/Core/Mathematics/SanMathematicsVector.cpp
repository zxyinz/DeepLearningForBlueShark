#include"SanMathematicsVector.h"
using namespace std;
using namespace San;
using namespace San::Mathematics;
San::Mathematics::stSPOINT2::stSPOINT2(const sfloat x, const sfloat y)
	:x(x), y(y)
{
}
San::Mathematics::stSPOINT2::~stSPOINT2()
{
}
San::Mathematics::stSPOINT3::stSPOINT3(const sfloat x, const sfloat y, const sfloat z)
	:x(x), y(y), z(z)
{
}
San::Mathematics::stSPOINT3::~stSPOINT3()
{
}
San::Mathematics::stSPOINT4::stSPOINT4(const sfloat x, const sfloat y, const sfloat z, const sfloat w)
	:x(x), y(y), z(z), w(w)
{
}
San::Mathematics::stSPOINT4::~stSPOINT4()
{
}
San::Mathematics::SPOINT2::SPOINT2(const sfloat Val)
	:stSPOINT2(Val, Val)
{
}
San::Mathematics::SPOINT2::SPOINT2(const sfloat x, const sfloat y)
	:stSPOINT2(x, y)
{
}
San::Mathematics::SPOINT2::SPOINT2(const sfloat* pArray)
	:stSPOINT2(0.0, 0.0)
{
	if (pArray != nullptr)
	{
		this->x = pArray[0];
		this->y = pArray[1];
	}
}
San::Mathematics::SPOINT2::SPOINT2(const SPOINT2 &Point2)
	:stSPOINT2(Point2.x, Point2.y)
{
}
San::Mathematics::SPOINT2::SPOINT2(const SPOINT3 &Point3)
	:stSPOINT2(Point3.x, Point3.y)
{
}
San::Mathematics::SPOINT2::SPOINT2(const SPOINT4 &Point4)
	:stSPOINT2(Point4.x, Point4.y)
{
}
San::Mathematics::SPOINT2::SPOINT2(const SVECTOR2 &Vec2)
	:stSPOINT2(Vec2.x, Vec2.y)
{
}
San::Mathematics::SPOINT2::SPOINT2(const SVECTOR3 &Vec3)
	:stSPOINT2(Vec3.x, Vec3.y)
{
}
San::Mathematics::SPOINT2::SPOINT2(const SVECTOR4 &Vec4)
	:stSPOINT2(Vec4.x, Vec4.y)
{
}
San::Mathematics::SPOINT2::~SPOINT2()
{
}
const sfloat& San::Mathematics::SPOINT2::operator[](const uint32 Position) const
{
	return this->p[Position > 2 ? 2 : Position];
}
sfloat& San::Mathematics::SPOINT2::operator[](const uint32 Position)
{
	return this->p[Position > 2 ? 2 : Position];
}
San::Mathematics::SPOINT2& San::Mathematics::SPOINT2::operator=(const SPOINT2 &Point2)
{
	this->x = Point2.x;
	this->y = Point2.y;

	return *this;
}
San::Mathematics::SPOINT2& San::Mathematics::SPOINT2::operator=(const SPOINT3 &Point3)
{
	this->x = Point3.x;
	this->y = Point3.y;

	return *this;
}
San::Mathematics::SPOINT2& San::Mathematics::SPOINT2::operator=(const SPOINT4 &Point4)
{
	this->x = Point4.x;
	this->y = Point4.y;

	return *this;
}
San::Mathematics::SPOINT2& San::Mathematics::SPOINT2::operator=(const SVECTOR2 &Vec2)
{
	this->x = Vec2.x;
	this->y = Vec2.y;

	return *this;
}
San::Mathematics::SPOINT2& San::Mathematics::SPOINT2::operator=(const SVECTOR3 &Vec3)
{
	this->x = Vec3.x;
	this->y = Vec3.y;

	return *this;
}
San::Mathematics::SPOINT2& San::Mathematics::SPOINT2::operator=(const SVECTOR4 &Vec4)
{
	this->x = Vec4.x;
	this->y = Vec4.y;

	return *this;
}
San::Mathematics::SPOINT2 San::Mathematics::SPOINT2::operator+(const SPOINT2 &Point2) const
{
	return SPOINT2(this->x + Point2.x, this->y + Point2.y);
}
San::Mathematics::SPOINT2 San::Mathematics::SPOINT2::operator-(const SPOINT2 &Point2) const
{
	return SPOINT2(this->x - Point2.x, this->y - Point2.y);
}
San::Mathematics::SPOINT2 San::Mathematics::SPOINT2::operator*(const SPOINT2 &Point2) const
{
	return SPOINT2(this->x * Point2.x, this->y * Point2.y);
}
San::Mathematics::SPOINT2 San::Mathematics::SPOINT2::operator+(const sfloat Val) const
{
	return SPOINT2(this->x + Val, this->y + Val);
}
San::Mathematics::SPOINT2 San::Mathematics::SPOINT2::operator-(const sfloat Val) const
{
	return SPOINT2(this->x - Val, this->y - Val);
}
San::Mathematics::SPOINT2 San::Mathematics::SPOINT2::operator*(const sfloat Val) const
{
	return SPOINT2(this->x * Val, this->y * Val);
}
bool San::Mathematics::SPOINT2::operator==(const SPOINT2 &Point2) const
{
	if (!::gloIsFloatEqual(this->x, Point2.x)){ return false; }
	if (!::gloIsFloatEqual(this->y, Point2.y)){ return false; }

	//return (::gloIsFloatEqual(this->x, Point2.x) && ::gloIsFloatEqual(this->y, Point2.y)) ? true : false;
	return true;
}
bool San::Mathematics::SPOINT2::operator!=(const SPOINT2 &Point2) const
{
	return !((*this) == Point2);
}
bool San::Mathematics::SPOINT2::operator<(const SPOINT2 &Point2) const
{
	return (this->x < Point2.x) && (this->y < Point2.y);
}
bool San::Mathematics::SPOINT2::operator>(const SPOINT2 &Point2) const
{
	return (this->x > Point2.x) && (this->y > Point2.y);
}
bool San::Mathematics::SPOINT2::operator<=(const SPOINT2 &Point2) const
{
	return (this->x <= Point2.x) && (this->y <= Point2.y);
}
bool San::Mathematics::SPOINT2::operator>=(const SPOINT2 &Point2) const
{
	return (this->x >= Point2.x) && (this->y >= Point2.y);
}
San::Mathematics::SPOINT2 San::Mathematics::SPOINT2::iGetHomogeneousPoint() const
{
	if (::gloIsFloatEqual(this->y, 0.0))
	{
		return SPOINT2(this->x, 1.0);
	}

	return SPOINT2(this->x/this->y,1.0);
}
void San::Mathematics::SPOINT2::iHomogenzation()
{
	if (::gloIsFloatEqual(this->y, 0.0))
	{ 
		this->y = 1.0; 
		return; 
	}

	this->x = this->x / this->y;
	this->y = 1.0;
}
San::Mathematics::SVECTOR2::SVECTOR2(const sfloat Val)
	:stSPOINT2(Val, Val)
{
}
San::Mathematics::SVECTOR2::SVECTOR2(const sfloat x, const sfloat y)
	:stSPOINT2(x, y)
{
}
San::Mathematics::SVECTOR2::SVECTOR2(const sfloat* pArray)
	:stSPOINT2(0.0, 0.0)
{
	if (pArray != nullptr)
	{
		this->x = pArray[0];
		this->y = pArray[1];
	}
}
San::Mathematics::SVECTOR2::SVECTOR2(const SVECTOR2 &Vec2)
	:stSPOINT2(Vec2.x, Vec2.y)
{
}
San::Mathematics::SVECTOR2::SVECTOR2(const SPOINT2 &Point2)
	:stSPOINT2(Point2.x, Point2.y)
{
}
San::Mathematics::SVECTOR2::SVECTOR2(const SPOINT3 &Point3)
	:stSPOINT2(Point3.x, Point3.y)
{
}
San::Mathematics::SVECTOR2::SVECTOR2(const SPOINT4 &Point4)
	:stSPOINT2(Point4.x, Point4.y)
{
}
San::Mathematics::SVECTOR2::SVECTOR2(const SVECTOR3 &Vec3)
	:stSPOINT2(Vec3.x, Vec3.y)
{
}
San::Mathematics::SVECTOR2::SVECTOR2(const SVECTOR4 &Vec4)
	:stSPOINT2(Vec4.x, Vec4.y)
{
}
San::Mathematics::SVECTOR2::~SVECTOR2()
{
}
const sfloat& San::Mathematics::SVECTOR2::operator[](const uint32 Position) const
{
	return this->p[Position > 2 ? 2 : Position];
}
sfloat& San::Mathematics::SVECTOR2::operator[](const uint32 Position)
{
	return this->p[Position > 2 ? 2 : Position];
}
San::Mathematics::SVECTOR2& San::Mathematics::SVECTOR2::operator=(const SPOINT2 &Point2)
{
	this->x = Point2.x;
	this->y = Point2.y;

	return *this;
}
San::Mathematics::SVECTOR2& San::Mathematics::SVECTOR2::operator=(const SPOINT3 &Point3)
{
	this->x = Point3.x;
	this->y = Point3.y;

	return *this;
}
San::Mathematics::SVECTOR2& San::Mathematics::SVECTOR2::operator=(const SPOINT4 &Point4)
{
	this->x = Point4.x;
	this->y = Point4.y;

	return *this;
}
San::Mathematics::SVECTOR2& San::Mathematics::SVECTOR2::operator=(const SVECTOR2 &Vec2)
{
	this->x = Vec2.x;
	this->y = Vec2.y;

	return *this;
}
San::Mathematics::SVECTOR2& San::Mathematics::SVECTOR2::operator=(const SVECTOR3 &Vec3)
{
	this->x = Vec3.x;
	this->y = Vec3.y;

	return *this;
}
San::Mathematics::SVECTOR2& San::Mathematics::SVECTOR2::operator=(const SVECTOR4 &Vec4)
{
	this->x = Vec4.x;
	this->y = Vec4.y;

	return *this;
}
San::Mathematics::SVECTOR2 San::Mathematics::SVECTOR2::operator+(const SVECTOR2 &Vec2) const
{
	return SVECTOR2(this->x + Vec2.x, this->y + Vec2.y);
}
San::Mathematics::SVECTOR2 San::Mathematics::SVECTOR2::operator-(const SVECTOR2 &Vec2) const
{
	return SVECTOR2(this->x - Vec2.x, this->y - Vec2.y);
}
San::Mathematics::SVECTOR2 San::Mathematics::SVECTOR2::operator*(const SVECTOR2 &Vec2) const
{
	return SVECTOR2(this->x * Vec2.x, this->y * Vec2.y);
}
San::Mathematics::SVECTOR2 San::Mathematics::SVECTOR2::operator+(const sfloat Val) const
{
	return SVECTOR2(this->x + Val, this->y + Val);
}
San::Mathematics::SVECTOR2 San::Mathematics::SVECTOR2::operator-(const sfloat Val) const
{
	return SVECTOR2(this->x - Val, this->y - Val);
}
San::Mathematics::SVECTOR2 San::Mathematics::SVECTOR2::operator*(const sfloat Val) const
{
	return SVECTOR2(this->x * Val, this->y * Val);
}
bool San::Mathematics::SVECTOR2::operator==(const SVECTOR3 &Vec2) const
{
	if (!::gloIsFloatEqual(this->x, Vec2.x)){ return false; }
	if (!::gloIsFloatEqual(this->y, Vec2.y)){ return false; }

	return true;
}
bool San::Mathematics::SVECTOR2::operator!=(const SVECTOR3 &Vec2) const
{
	return !((*this) == Vec2);
}
bool San::Mathematics::SVECTOR2::operator<(const SVECTOR3 &Vec2) const
{
	return (this->x < Vec2.x) && (this->y < Vec2.y);
}
bool San::Mathematics::SVECTOR2::operator>(const SVECTOR3 &Vec2) const
{
	return (this->x > Vec2.x) && (this->y > Vec2.y);
}
bool San::Mathematics::SVECTOR2::operator<=(const SVECTOR3 &Vec2) const
{
	return (this->x <= Vec2.x) && (this->y <= Vec2.y);
}
bool San::Mathematics::SVECTOR2::operator>=(const SVECTOR3 &Vec2) const
{
	return (this->x >= Vec2.x) && (this->y >= Vec2.y);
}
San::Mathematics::SVECTOR2 San::Mathematics::SVECTOR2::iGetHomogeneousVector() const
{
	return SVECTOR2(this->x, 0.0);
}
void San::Mathematics::SVECTOR2::iHomogenzation()
{
	this->y = 0.0;
}
San::Mathematics::SPOINT3::SPOINT3(const sfloat Val)
	:stSPOINT3(Val, Val, Val)
{
}
San::Mathematics::SPOINT3::SPOINT3(const sfloat x, const sfloat y, const sfloat z)
	:stSPOINT3(x, y, z)
{
}
San::Mathematics::SPOINT3::SPOINT3(const sfloat* pArray)
	:stSPOINT3(0.0, 0.0, 0.0)
{
	if (pArray != nullptr)
	{
		this->x = pArray[0];
		this->y = pArray[1];
		this->z = pArray[2];
	}
}
San::Mathematics::SPOINT3::SPOINT3(const SPOINT3& Point3)
	:stSPOINT3(Point3.x, Point3.y, Point3.z)
{
}
San::Mathematics::SPOINT3::SPOINT3(const SPOINT2 &Point2, const sfloat z)
	:stSPOINT3(Point2.x, Point2.y, z)
{
}
San::Mathematics::SPOINT3::SPOINT3(const SPOINT4 &Point4)
	:stSPOINT3(Point4.x, Point4.y, Point4.z)
{
	/*if (!::gloIsFloatEqual(Point4.w, 1.0))
	{
		if (!gloIsFloatEqual(Point4.w, 0.0))
		{
			this->x = this->x / Point4.w;
			this->y = this->y / Point4.w;
			this->z = this->z / Point4.w;
		}
	}*/
}
San::Mathematics::SPOINT3::SPOINT3(const SVECTOR2 &Vec2, const sfloat z)
	:stSPOINT3(Vec2.x, Vec2.y, z)
{
}
San::Mathematics::SPOINT3::SPOINT3(const SVECTOR3 &Vec3)
	:stSPOINT3(Vec3.x, Vec3.y, Vec3.z)
{
}
San::Mathematics::SPOINT3::SPOINT3(const SVECTOR4 &Vec4)
	:stSPOINT3(Vec4.x, Vec4.y, Vec4.z)
{
}
San::Mathematics::SPOINT3::~SPOINT3()
{
}
const sfloat& San::Mathematics::SPOINT3::operator[](const uint32 Position) const
{
	return this->p[Position > 3 ? 3 : Position];
}
sfloat& San::Mathematics::SPOINT3::operator[](const uint32 Position)
{
	return this->p[Position > 3 ? 3 : Position];
}
San::Mathematics::SPOINT3& San::Mathematics::SPOINT3::operator=(const SPOINT2 &Point2)
{
	this->x = Point2.x;
	this->y = Point2.y;
	this->z = 0.0;

	return *this;
}
San::Mathematics::SPOINT3& San::Mathematics::SPOINT3::operator=(const SPOINT3 &Point3)
{
	this->x = Point3.x;
	this->y = Point3.y;
	this->z = Point3.z;

	return *this;
}
San::Mathematics::SPOINT3& San::Mathematics::SPOINT3::operator=(const SPOINT4 &Point4)
{
	this->x = Point4.x;
	this->y = Point4.y;
	this->z = Point4.z;
	/*if (!::gloIsFloatEqual(Point4.w, 1.0))
	{
		if (!gloIsFloatEqual(Point4.w, 0.0))
		{
			this->x = this->x / Point4.w;
			this->y = this->y / Point4.w;
			this->z = this->z / Point4.w;
		}
	}*/

	return *this;
}
San::Mathematics::SPOINT3& San::Mathematics::SPOINT3::operator=(const SVECTOR2 &Vec2)
{
	this->x = Vec2.x;
	this->y = Vec2.y;
	this->z = 0.0;

	return *this;
}
San::Mathematics::SPOINT3& San::Mathematics::SPOINT3::operator=(const SVECTOR3 &Vec3)
{
	this->x = Vec3.x;
	this->y = Vec3.y;
	this->z = Vec3.z;

	return *this;
}
San::Mathematics::SPOINT3& San::Mathematics::SPOINT3::operator=(const SVECTOR4 &Vec4)
{
	this->x = Vec4.x;
	this->y = Vec4.y;
	this->z = Vec4.z;

	return *this;
}
San::Mathematics::SPOINT3 San::Mathematics::SPOINT3::operator+(const SPOINT3 &Point3) const
{
	return SPOINT3(this->x + Point3.x, this->y + Point3.y, this->z + Point3.z);
}
San::Mathematics::SPOINT3 San::Mathematics::SPOINT3::operator-(const SPOINT3 &Point3) const
{
	return SPOINT3(this->x - Point3.x, this->y - Point3.y, this->z - Point3.z);
}
San::Mathematics::SPOINT3 San::Mathematics::SPOINT3::operator*(const SPOINT3 &Point3) const
{
	return SPOINT3(this->x * Point3.x, this->y * Point3.y, this->z * Point3.z);
}
San::Mathematics::SPOINT3 San::Mathematics::SPOINT3::operator+(const sfloat Val) const
{
	return SPOINT3(this->x + Val, this->y + Val, this->z + Val);
}
San::Mathematics::SPOINT3 San::Mathematics::SPOINT3::operator-(const sfloat Val) const
{
	return SPOINT3(this->x - Val, this->y - Val, this->z - Val);
}
San::Mathematics::SPOINT3 San::Mathematics::SPOINT3::operator*(const sfloat Val) const
{
	return SPOINT3(this->x * Val, this->y * Val, this->z * Val);
}
bool San::Mathematics::SPOINT3::operator==(const SPOINT3 &Point3) const
{
	if (!::gloIsFloatEqual(this->x, Point3.x)){ return false; }
	if (!::gloIsFloatEqual(this->y, Point3.y)){ return false; }
	if (!::gloIsFloatEqual(this->z, Point3.z)){ return false; }

	return true;
}
bool San::Mathematics::SPOINT3::operator!=(const SPOINT3 &Point3) const
{
	return !((*this) == Point3);
}
bool San::Mathematics::SPOINT3::operator<(const SPOINT3 &Point3) const
{
	return (this->x < Point3.x) && (this->y < Point3.y) && (this->z < Point3.z);
}
bool San::Mathematics::SPOINT3::operator>(const SPOINT3 &Point3) const
{
	return (this->x > Point3.x) && (this->y > Point3.y) && (this->z > Point3.z);
}
bool San::Mathematics::SPOINT3::operator<=(const SPOINT3 &Point3) const
{
	return (this->x <= Point3.x) && (this->y <= Point3.y) && (this->z <= Point3.z);
}
bool San::Mathematics::SPOINT3::operator>=(const SPOINT3 &Point3) const
{
	return (this->x >= Point3.x) && (this->y >= Point3.y) && (this->z >= Point3.z);
}
San::Mathematics::SPOINT3 San::Mathematics::SPOINT3::iGetHomogeneousPoint() const
{
	if (::gloIsFloatEqual(this->z, 0.0))
	{
		return SPOINT3(this->x, this->y, 1.0);
	}

	return SPOINT3(this->x / this->z, this->y / this->z, 1.0);
}
void San::Mathematics::SPOINT3::iHomogenzation()
{
	if (::gloIsFloatEqual(this->z, 0.0))
	{
		this->z = 1.0;
	}

	this->x = this->x / this->z;
	this->y = this->y / this->z;
	this->z = 1.0;
}
San::Mathematics::SVECTOR3::SVECTOR3(const sfloat Val)
	:stSPOINT3(Val, Val, Val)
{
}
San::Mathematics::SVECTOR3::SVECTOR3(const sfloat x, const sfloat y, const sfloat z)
	:stSPOINT3(x, y, z)
{
}
San::Mathematics::SVECTOR3::SVECTOR3(const sfloat* pArray)
	:stSPOINT3(0.0, 0.0, 0.0)
{
	if (pArray != nullptr)
	{
		this->x = pArray[0];
		this->y = pArray[1];
		this->z = pArray[2];
	}
}
San::Mathematics::SVECTOR3::SVECTOR3(const SVECTOR3 &Vec3)
	:stSPOINT3(Vec3.x, Vec3.y, Vec3.z)
{
}
San::Mathematics::SVECTOR3::SVECTOR3(const SPOINT2 &Point2, const sfloat z)
	:stSPOINT3(Point2.x, Point2.y, z)
{
}
San::Mathematics::SVECTOR3::SVECTOR3(const SPOINT3 &Point3)
	:stSPOINT3(Point3.x, Point3.y, Point3.z)
{
}
San::Mathematics::SVECTOR3::SVECTOR3(const SPOINT4 &Point4)
	:stSPOINT3(Point4.x, Point4.y, Point4.z)
{
}
San::Mathematics::SVECTOR3::SVECTOR3(const SVECTOR2 &Vec2, const sfloat z)
	:stSPOINT3(Vec2.x, Vec2.y, z)
{
}
San::Mathematics::SVECTOR3::SVECTOR3(const SVECTOR4 &Vec4)
	:stSPOINT3(Vec4.x, Vec4.y, Vec4.z)
{
}
San::Mathematics::SVECTOR3::~SVECTOR3()
{
}
const sfloat& San::Mathematics::SVECTOR3::operator[](const uint32 Position) const
{
	return this->p[Position > 3 ? 3 : Position];
}
sfloat& San::Mathematics::SVECTOR3::operator[](const uint32 Position)
{
	return this->p[Position > 3 ? 3 : Position];
}
San::Mathematics::SVECTOR3& San::Mathematics::SVECTOR3::operator=(const SPOINT2 &Point2)
{
	this->x = Point2.x;
	this->y = Point2.y;
	this->z = 0.0;

	return *this;
}
San::Mathematics::SVECTOR3& San::Mathematics::SVECTOR3::operator=(const SPOINT3 &Point3)
{
	this->x = Point3.x;
	this->y = Point3.y;
	this->z = Point3.z;

	return *this;
}
San::Mathematics::SVECTOR3& San::Mathematics::SVECTOR3::operator=(const SPOINT4 &Point4)
{
	this->x = Point4.x;
	this->y = Point4.y;
	this->z = Point4.z;

	return *this;
}
San::Mathematics::SVECTOR3& San::Mathematics::SVECTOR3::operator=(const SVECTOR2 &Vec2)
{
	this->x = Vec2.x;
	this->y = Vec2.y;
	this->z = 0.0;

	return *this;
}
San::Mathematics::SVECTOR3& San::Mathematics::SVECTOR3::operator=(const SVECTOR3 &Vec3)
{
	this->x = Vec3.x;
	this->y = Vec3.y;
	this->z = Vec3.z;

	return *this;
}
San::Mathematics::SVECTOR3& San::Mathematics::SVECTOR3::operator=(const SVECTOR4 &Vec4)
{
	this->x = Vec4.x;
	this->y = Vec4.y;
	this->z = Vec4.z;

	return *this;
}
San::Mathematics::SVECTOR3 San::Mathematics::SVECTOR3::operator+(const SVECTOR3 &Vec3) const
{
	return SVECTOR3(this->x + Vec3.x, this->y + Vec3.y, this->z + Vec3.z);
}
San::Mathematics::SVECTOR3 San::Mathematics::SVECTOR3::operator-(const SVECTOR3 &Vec3) const
{
	return SVECTOR3(this->x - Vec3.x, this->y - Vec3.y, this->z - Vec3.z);
}
San::Mathematics::SVECTOR3 San::Mathematics::SVECTOR3::operator*(const SVECTOR3 &Vec3) const
{
	return SVECTOR3(this->x * Vec3.x, this->y * Vec3.y, this->z * Vec3.z);
}
San::Mathematics::SVECTOR3 San::Mathematics::SVECTOR3::operator+(const sfloat Val) const
{
	return SVECTOR3(this->x + Val, this->y + Val, this->z + Val);
}
San::Mathematics::SVECTOR3 San::Mathematics::SVECTOR3::operator-(const sfloat Val) const
{
	return SVECTOR3(this->x - Val, this->y - Val, this->z - Val);
}
San::Mathematics::SVECTOR3 San::Mathematics::SVECTOR3::operator*(const sfloat Val) const
{
	return SVECTOR3(this->x * Val, this->y * Val, this->z * Val);
}
bool San::Mathematics::SVECTOR3::operator==(const SVECTOR3 &Vec3) const
{
	if (!::gloIsFloatEqual(this->x, Vec3.x)){ return false; }
	if (!::gloIsFloatEqual(this->y, Vec3.y)){ return false; }
	if (!::gloIsFloatEqual(this->z, Vec3.z)){ return false; }

	return true;
}
bool San::Mathematics::SVECTOR3::operator!=(const SVECTOR3 &Vec3) const
{
	return !((*this) == Vec3);
}
bool San::Mathematics::SVECTOR3::operator<(const SVECTOR3 &Vec3) const
{
	return (this->x < Vec3.x) && (this->y < Vec3.y) && (this->z < Vec3.z);
}
bool San::Mathematics::SVECTOR3::operator>(const SVECTOR3 &Vec3) const
{
	return (this->x > Vec3.x) && (this->y > Vec3.y) && (this->z > Vec3.z);
}
bool San::Mathematics::SVECTOR3::operator<=(const SVECTOR3 &Vec3) const
{
	return (this->x <= Vec3.x) && (this->y <= Vec3.y) && (this->z <= Vec3.z);
}
bool San::Mathematics::SVECTOR3::operator>=(const SVECTOR3 &Vec3) const
{
	return (this->x >= Vec3.x) && (this->y >= Vec3.y) && (this->z >= Vec3.z);
}
San::Mathematics::SVECTOR3 San::Mathematics::SVECTOR3::iGetHomogeneousVector() const
{
	return SVECTOR3(this->x, this->y, 0.0);
}
void San::Mathematics::SVECTOR3::iHomogenzation()
{
	this->z = 0.0;
}
San::Mathematics::SPOINT4::SPOINT4(const sfloat Val)
	:stSPOINT4(Val, Val, Val, Val)
{
}
San::Mathematics::SPOINT4::SPOINT4(const sfloat x, const sfloat y, const sfloat z, const sfloat w)
	:stSPOINT4(x, y, z, w)
{
	/*if (!::gloIsFloatEqual(w, 1.0))
	{
	if (!gloIsFloatEqual(w, 0.0))
	{
	this->x = this->x / w;
	this->y = this->y / w;
	this->z = this->z / w;
	}
	}*/
}
San::Mathematics::SPOINT4::SPOINT4(const sfloat* pArray)
	:stSPOINT4(0.0, 0.0, 0.0, 0.0)
{
	if (pArray != nullptr)
	{
		this->x = pArray[0];
		this->y = pArray[1];
		this->z = pArray[2];
		this->w = pArray[3];
	}

	/*if (!::gloIsFloatEqual(pArray[3], 1.0))
	{
	if (!gloIsFloatEqual(pArray[3], 0.0))
	{
	this->x = this->x / pArray[3];
	this->y = this->y / pArray[3];
	this->z = this->z / pArray[3];
	}
	}*/
}
San::Mathematics::SPOINT4::SPOINT4(const SPOINT4 &Point4)
	:stSPOINT4(Point4.x, Point4.y, Point4.z, Point4.w)
{
	/*if (!::gloIsFloatEqual(Point4.w, 1.0))
	{
	if (!gloIsFloatEqual(Point4.w, 0.0))
	{
	this->x = this->x / Point4.w;
	this->y = this->y / Point4.w;
	this->z = this->z / Point4.w;
	}
	}*/
}
San::Mathematics::SPOINT4::SPOINT4(const SPOINT2 &Point2, const sfloat z, const sfloat w)
	:stSPOINT4(Point2.x, Point2.y, z, w)
{
}
San::Mathematics::SPOINT4::SPOINT4(const SPOINT3 &Point3, const sfloat w)
	:stSPOINT4(Point3.x, Point3.y, Point3.z, w)
{
}
San::Mathematics::SPOINT4::SPOINT4(const SVECTOR2 &Vec2, const sfloat z, const sfloat w)
	:stSPOINT4(Vec2.x, Vec2.y, z, w)
{
}
San::Mathematics::SPOINT4::SPOINT4(const SVECTOR3 &Vec3, const sfloat w)
	:stSPOINT4(Vec3.x, Vec3.y, Vec3.z, w)
{

}
San::Mathematics::SPOINT4::SPOINT4(const SVECTOR4 &Vec4)
	:stSPOINT4(Vec4.x, Vec4.y, Vec4.z, Vec4.w)
{
}
San::Mathematics::SPOINT4::~SPOINT4()
{
}
const sfloat& San::Mathematics::SPOINT4::operator[](const uint32 Position) const
{
	return this->p[Position > 4 ? 4 : Position];
}
sfloat& San::Mathematics::SPOINT4::operator[](const uint32 Position)
{
	return this->p[Position > 4 ? 4 : Position];
}
San::Mathematics::SPOINT4& San::Mathematics::SPOINT4::operator=(const SPOINT2 &Point2)
{
	this->x = Point2.x;
	this->y = Point2.y;
	this->z = 0.0;
	this->w = 0.0;

	return *this;
}
San::Mathematics::SPOINT4& San::Mathematics::SPOINT4::operator=(const SPOINT3 &Point3)
{
	this->x = Point3.x;
	this->y = Point3.y;
	this->z = Point3.z;
	this->w = 0.0;

	return *this;
}
San::Mathematics::SPOINT4& San::Mathematics::SPOINT4::operator=(const SPOINT4 &Point4)
{
	this->x = Point4.x;
	this->y = Point4.y;
	this->z = Point4.z;
	this->w = Point4.w;

	/*if (!::gloIsFloatEqual(Point4.w, 1.0))
	{
		if (!gloIsFloatEqual(Point4.w, 0.0))
		{
			this->x = this->x / Point4.w;
			this->y = this->y / Point4.w;
			this->z = this->z / Point4.w;
		}
	}*/
	return *this;
}
San::Mathematics::SPOINT4& San::Mathematics::SPOINT4::operator = (const SVECTOR2 &Vec2)
{
	this->x = Vec2.x;
	this->y = Vec2.y;
	this->z = 0.0;
	this->w = 0.0;

	return *this;
}
San::Mathematics::SPOINT4& San::Mathematics::SPOINT4::operator = (const SVECTOR3 &Vec3)
{
	this->x = Vec3.x;
	this->y = Vec3.y;
	this->z = Vec3.z;
	this->w = 0.0;

	return *this;
}
San::Mathematics::SPOINT4& San::Mathematics::SPOINT4::operator=(const SVECTOR4 &Vec4)
{
	this->x = Vec4.x;
	this->y = Vec4.y;
	this->z = Vec4.z;
	this->w = Vec4.w;

	return *this;
}
San::Mathematics::SPOINT4 San::Mathematics::SPOINT4::operator+(const SPOINT4 &Point4) const
{
	/*SPOINT4 HomoPointSrc(*this);
	SPOINT4 HomoPointDesc(Point4);*/
	return SPOINT4(this->x + Point4.x, this->y + Point4.y, this->z + Point4.z, this->w + Point4.w);
}
San::Mathematics::SPOINT4 San::Mathematics::SPOINT4::operator-(const SPOINT4 &Point4) const
{
	/*SPOINT4 HomoPointSrc(*this);
	SPOINT4 HomoPointDesc(Point4);*/
	return SPOINT4(this->x - Point4.x, this->y - Point4.y, this->z - Point4.z, this->w - Point4.w);
}
San::Mathematics::SPOINT4 San::Mathematics::SPOINT4::operator*(const SPOINT4 &Point4) const
{
	/*SPOINT4 HomoPointSrc(*this);
	SPOINT4 HomoPointDesc(Point4);*/
	return SPOINT4(this->x * Point4.x, this->y * Point4.y, this->z * Point4.z, this->w * Point4.w);
}
San::Mathematics::SPOINT4 San::Mathematics::SPOINT4::operator+(const sfloat Val) const
{
	return SPOINT4(this->x + Val, this->y + Val, this->z + Val, this->w + Val);
}
San::Mathematics::SPOINT4 San::Mathematics::SPOINT4::operator-(const sfloat Val) const
{
	return SPOINT4(this->x - Val, this->y - Val, this->z - Val, this->w - Val);
}
San::Mathematics::SPOINT4 San::Mathematics::SPOINT4::operator*(const sfloat Val) const
{
	return SPOINT4(this->x * Val, this->y * Val, this->z * Val, this->w * Val);
}
bool San::Mathematics::SPOINT4::operator==(const SPOINT4 &Point4) const
{
	if (!::gloIsFloatEqual(this->x, Point4.x)){ return false; }
	if (!::gloIsFloatEqual(this->y, Point4.y)){ return false; }
	if (!::gloIsFloatEqual(this->z, Point4.z)){ return false; }
	if (!::gloIsFloatEqual(this->w, Point4.w)){ return false; }
	return true;
}
bool San::Mathematics::SPOINT4::operator!=(const SPOINT4 &Point4) const
{
	return !((*this) == Point4);
}
bool San::Mathematics::SPOINT4::operator<(const SPOINT4 &Point4) const
{
	return (this->x < Point4.x) && (this->y < Point4.y) && (this->z < Point4.z) && (this->w < Point4.w);
}
bool San::Mathematics::SPOINT4::operator>(const SPOINT4 &Point4) const
{
	return (this->x > Point4.x) && (this->y > Point4.y) && (this->z > Point4.z) && (this->w > Point4.w);
}
bool San::Mathematics::SPOINT4::operator<=(const SPOINT4 &Point4) const
{
	return (this->x <= Point4.x) && (this->y <= Point4.y) && (this->z <= Point4.z) && (this->w <= Point4.w);
}
bool San::Mathematics::SPOINT4::operator>=(const SPOINT4 &Point4) const
{
	return (this->x >= Point4.x) && (this->y >= Point4.y) && (this->z >= Point4.z) && (this->w >= Point4.w);
}
SPOINT4 San::Mathematics::SPOINT4::iGetHomogeneousPoint() const
{
	if (::gloIsFloatEqual(this->w, 0.0))
	{
		return SPOINT4(this->x, this->y, this->z, 1.0);
	}

	return SPOINT4(this->x / this->w, this->y / this->w, this->z / this->w, 1.0);
}
void San::Mathematics::SPOINT4::iHomogenzation()
{
	if (::gloIsFloatEqual(this->w, 0.0))
	{
		this->w = 1.0;
	}

	this->x = this->x / this->w;
	this->y = this->y / this->w;
	this->z = this->z / this->w;
	this->w = 1.0;
}
San::Mathematics::SVECTOR4::SVECTOR4(const sfloat Val)
	:stSPOINT4(Val, Val, Val, Val)
{
}
San::Mathematics::SVECTOR4::SVECTOR4(const sfloat x, const sfloat y, const sfloat z, const sfloat w)
	:stSPOINT4(x, y, z, w)
{
	/*if (!::gloIsFloatEqual(w, 1.0))
	{
	if (!gloIsFloatEqual(w, 0.0))
	{
	this->x = this->x / w;
	this->y = this->y / w;
	this->z = this->z / w;
	}
	}*/
}
San::Mathematics::SVECTOR4::SVECTOR4(const sfloat* pArray)
	:stSPOINT4(0.0, 0.0, 0.0, 0.0)
{
	if (pArray != nullptr)
	{
		this->x = pArray[0];
		this->y = pArray[1];
		this->z = pArray[2];
		this->w = pArray[3];
	}

	/*if (!::gloIsFloatEqual(pArray[3], 1.0))
	{
	if (!gloIsFloatEqual(pArray[3], 0.0))
	{
	this->x = this->x / pArray[3];
	this->y = this->y / pArray[3];
	this->z = this->z / pArray[3];
	}
	}*/
}
San::Mathematics::SVECTOR4::SVECTOR4(const SVECTOR4 &Vec4)
	:stSPOINT4(Vec4.x, Vec4.y, Vec4.z, Vec4.w)
{
}
San::Mathematics::SVECTOR4::SVECTOR4(const SPOINT2 &Point2, const sfloat z, const sfloat w)
	:stSPOINT4(Point2.x, Point2.y, z, w)
{
}
San::Mathematics::SVECTOR4::SVECTOR4(const SPOINT3 &Point3, const sfloat w)
	:stSPOINT4(Point3.x, Point3.y, Point3.z, w)
{
}
San::Mathematics::SVECTOR4::SVECTOR4(const SPOINT4 &Point4)
	:stSPOINT4(Point4.x, Point4.y, Point4.z, Point4.w)
{
	/*if (!::gloIsFloatEqual(Point4.w, 1.0))
	{
		if (!gloIsFloatEqual(Point4.w, 0.0))
		{
			this->x = this->x / Point4.w;
			this->y = this->y / Point4.w;
			this->z = this->z / Point4.w;
		}
	}*/
}
San::Mathematics::SVECTOR4::SVECTOR4(const SVECTOR2 &Vec2, const sfloat z, const sfloat w)
	:stSPOINT4(Vec2.x, Vec2.y, z, w)
{
}
San::Mathematics::SVECTOR4::SVECTOR4(const SVECTOR3 &Vec3, const sfloat w)
	:stSPOINT4(Vec3.x, Vec3.y, Vec3.z, w)
{
}
San::Mathematics::SVECTOR4::~SVECTOR4()
{
}
const sfloat& San::Mathematics::SVECTOR4::operator[](const uint32 Position) const
{
	return this->p[Position > 4 ? 4 : Position];
}
sfloat& San::Mathematics::SVECTOR4::operator[](const uint32 Position)
{
	return this->p[Position > 4 ? 4 : Position];
}
San::Mathematics::SVECTOR4& San::Mathematics::SVECTOR4::operator=(const SPOINT2 &Point2)
{
	this->x = Point2.x;
	this->y = Point2.y;
	this->z = 0.0;
	this->w = 0.0;

	return *this;
}
San::Mathematics::SVECTOR4& San::Mathematics::SVECTOR4::operator=(const SPOINT3 &Point3)
{
	this->x = Point3.x;
	this->y = Point3.y;
	this->z = Point3.z;
	this->w = 0.0;

	return *this;
}
San::Mathematics::SVECTOR4& San::Mathematics::SVECTOR4::operator=(const SPOINT4 &Point4)
{
	this->x = Point4.x;
	this->y = Point4.y;
	this->z = Point4.z;
	this->w = Point4.w;

	/*if (!::gloIsFloatEqual(Point4.w, 1.0))
	{
		if (!gloIsFloatEqual(Point4.w, 0.0))
		{
			this->x = this->x / Point4.w;
			this->y = this->y / Point4.w;
			this->z = this->z / Point4.w;
		}
	}*/

	return *this;
}
San::Mathematics::SVECTOR4& San::Mathematics::SVECTOR4::operator=(const SVECTOR2 &Vec2)
{
	this->x = Vec2.x;
	this->y = Vec2.y;
	this->z = 0.0;
	this->w = 0.0;

	return *this;
}
San::Mathematics::SVECTOR4& San::Mathematics::SVECTOR4::operator=(const SVECTOR3 &Vec3)
{
	this->x = Vec3.x;
	this->y = Vec3.y;
	this->z = Vec3.z;
	this->w = 0.0;

	return *this;
}
San::Mathematics::SVECTOR4& San::Mathematics::SVECTOR4::operator=(const SVECTOR4 &Vec4)
{
	this->x = Vec4.x;
	this->y = Vec4.y;
	this->z = Vec4.z;
	this->w = Vec4.w;

	return *this;
}
San::Mathematics::SVECTOR4 San::Mathematics::SVECTOR4::operator+(const SVECTOR4 &Vec4) const
{
	return SVECTOR4(this->x + Vec4.x, this->y + Vec4.y, this->z + Vec4.z, 0.0);
}
San::Mathematics::SVECTOR4 San::Mathematics::SVECTOR4::operator-(const SVECTOR4 &Vec4) const
{
	return SVECTOR4(this->x - Vec4.x, this->y - Vec4.y, this->z - Vec4.z, 0.0);
}
San::Mathematics::SVECTOR4 San::Mathematics::SVECTOR4::operator*(const SVECTOR4 &Vec4) const
{
	return SVECTOR4(this->x * Vec4.x, this->y * Vec4.y, this->z * Vec4.z, 0.0);
}
San::Mathematics::SVECTOR4 San::Mathematics::SVECTOR4::operator+(const sfloat Val) const
{
	return SVECTOR4(this->x + Val, this->y + Val, this->z + Val, 0.0);
}
San::Mathematics::SVECTOR4 San::Mathematics::SVECTOR4::operator-(const sfloat Val) const
{
	return SVECTOR4(this->x - Val, this->y - Val, this->z - Val, 0.0);
}
San::Mathematics::SVECTOR4 San::Mathematics::SVECTOR4::operator*(const sfloat Val) const
{
	return SVECTOR4(this->x * Val, this->y * Val, this->z * Val, 0.0);
}
bool San::Mathematics::SVECTOR4::operator==(const SVECTOR4 &Vec4) const
{
	if (!::gloIsFloatEqual(this->x, Vec4.x)){ return false; }
	if (!::gloIsFloatEqual(this->y, Vec4.y)){ return false; }
	if (!::gloIsFloatEqual(this->z, Vec4.z)){ return false; }
	if (!::gloIsFloatEqual(this->w, Vec4.w)){ return false; }

	return true;
}
bool San::Mathematics::SVECTOR4::operator!=(const SVECTOR4 &Vec4) const
{
	return !((*this) == Vec4);
}
bool San::Mathematics::SVECTOR4::operator<(const SVECTOR4 &Vec4) const
{
	return (this->x < Vec4.x) && (this->y < Vec4.y) && (this->z < Vec4.z) && (this->w < Vec4.w);
}
bool San::Mathematics::SVECTOR4::operator>(const SVECTOR4 &Vec4) const
{
	return (this->x > Vec4.x) && (this->y > Vec4.y) && (this->z > Vec4.z) && (this->w > Vec4.w);
}
bool San::Mathematics::SVECTOR4::operator<=(const SVECTOR4 &Vec4) const
{
	return (this->x <= Vec4.x) && (this->y <= Vec4.y) && (this->z <= Vec4.z) && (this->w <= Vec4.w);
}
bool San::Mathematics::SVECTOR4::operator>=(const SVECTOR4 &Vec4) const
{
	return (this->x >= Vec4.x) && (this->y >= Vec4.y) && (this->z >= Vec4.z) && (this->w >= Vec4.w);
}
SVECTOR4 San::Mathematics::SVECTOR4::iGetHomogeneousVector() const
{
	return SVECTOR4(this->x, this->y, this->z, 0.0);
}
void San::Mathematics::SVECTOR4::iHomogenzation()
{
	this->w = 0.0;
}