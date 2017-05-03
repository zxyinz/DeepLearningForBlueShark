//Project: San Lab Game Engine
//Version: 2.1.1
//Debug State: Add functions , vector template version and need test [_vector iterator]
#include"SanMathematicsDef.h"
#pragma once
using namespace std;
namespace San
{
	namespace Mathematics
	{
#ifndef __STDSANMATHEMATICSVECTORT_H__
#define __STDSANMATHEMATICSVECTORT_H__
		template<class ST>
		struct _stSPOINT2
		{
		public:
			union
			{
				struct{ ST x, y; };
				ST p[2];
			};
		public:
			_stSPOINT2(const ST &x, const ST &y);
			~_stSPOINT2();
		};

#pragma region _stSPOINT2 CPP file
		template<class ST>
		_stSPOINT2<ST>::_stSPOINT2(const ST &x, const ST &y)
			:x(x), y(y)
		{
		}
		template<class ST>
		_stSPOINT2<ST>::~_stSPOINT2()
		{
		}
#pragma endregion

		template<class ST>
		struct _stSPOINT3
		{
		public:
			union
			{
				struct{ ST x, y, z; };
				ST p[3];
			};
		public:
			_stSPOINT3(const ST &x, const ST &y, const ST &z);
			~_stSPOINT3();
		};

#pragma region _stSPOINT3 CPP file
		template<class ST>
		_stSPOINT3<ST>::_stSPOINT3(const ST &x, const ST &y, const ST &z)
			:x(x), y(y), z(z)
		{
		}
		template<class ST>
		_stSPOINT3<ST>::~_stSPOINT3()
		{
		}
#pragma endregion

		template<class ST>
		struct _stSPOINT4
		{
		public:
			union
			{
				struct{ ST x, y, z, w; };
				ST p[4];
			};
		public:
			_stSPOINT4(const ST &x, const ST &y, const ST &z, const ST &w);
			~_stSPOINT4();
		};

#pragma region _stSPOINT4 CPP file
		template<class ST>
		_stSPOINT4<ST>::_stSPOINT4(const ST &x, const ST &y, const ST &z, const ST &w)
			:x(x), y(y), z(z), w(w)
		{
		}
		template<class ST>
		_stSPOINT4<ST>::~_stSPOINT4()
		{
		}
#pragma endregion

		template<class ST>
		class _cDefaultZeroGenerator
		{
		public:
			ST operator()(){ return 0.0; }
		};

		template<class ST>
		class _cDefaultOneGenerator
		{
		public:
			ST operator()(){ return 1.0; }
		};

		template<class ST = sfloat, class _Zero = _cDefaultZeroGenerator<ST>, class _One = _cDefaultOneGenerator<ST>>
		struct _SPOINT2 : public _stSPOINT2<ST>
		{
		public:
			_SPOINT2(const ST &Val = _Zero());
			_SPOINT2(const ST &x, const ST &y);
			_SPOINT2(const ST* pArray);
			_SPOINT2(const _SPOINT2<ST, _Zero, _One> &Point2);

			explicit _SPOINT2(const _SPOINT3<ST, _Zero, _One> &Point3);
			explicit _SPOINT2(const _SPOINT4<ST, _Zero, _One> &Point4);

			explicit _SPOINT2(const _SVECTOR2<ST, _Zero, _One> &Vec2);
			explicit _SPOINT2(const _SVECTOR3<ST, _Zero, _One> &Vec3);
			explicit _SPOINT2(const _SVECTOR4<ST, _Zero, _One> &Vec4);

			~_SPOINT2();

			const ST& operator[](const uint32 Position) const;
			ST& operator[](const uint32 Position);

			_SPOINT2<ST, _Zero, _One>& operator=(const _SPOINT2<ST, _Zero, _One> &Point2);
			_SPOINT2<ST, _Zero, _One>& operator=(const _SPOINT3<ST, _Zero, _One> &Point3);
			_SPOINT2<ST, _Zero, _One>& operator=(const _SPOINT4<ST, _Zero, _One> &Point4);
			_SPOINT2<ST, _Zero, _One>& operator=(const _SVECTOR2<ST, _Zero, _One> &Vec2);
			_SPOINT2<ST, _Zero, _One>& operator=(const _SVECTOR3<ST, _Zero, _One> &Vec3);
			_SPOINT2<ST, _Zero, _One>& operator=(const _SVECTOR4<ST, _Zero, _One> &Vec4);

			_SPOINT2<ST, _Zero, _One> operator+(const _SPOINT2<ST, _Zero, _One> &Point2) const;
			_SPOINT2<ST, _Zero, _One> operator-(const _SPOINT2<ST, _Zero, _One> &Point2) const;
			_SPOINT2<ST, _Zero, _One> operator*(const _SPOINT2<ST, _Zero, _One> &Point2) const;
			_SPOINT2<ST, _Zero, _One> operator+(const ST &Val) const;
			_SPOINT2<ST, _Zero, _One> operator-(const ST &Val) const;
			_SPOINT2<ST, _Zero, _One> operator*(const ST &Val) const;

			bool operator==(const _SPOINT2<ST, _Zero, _One> &Point2) const;
			bool operator!=(const _SPOINT2<ST, _Zero, _One> &Point2) const;

			bool operator<(const _SPOINT2<ST, _Zero, _One> &Point2) const;
			bool operator>(const _SPOINT2<ST, _Zero, _One> &Point2) const;

			bool operator<=(const _SPOINT2<ST, _Zero, _One> &Point2) const;
			bool operator>=(const _SPOINT2<ST, _Zero, _One> &Point2) const;

			_SPOINT2<ST, _Zero, _One>	iGetHomogeneousPoint() const;
			void	iHomogenzation();
		};

#pragma region _SPOINT2 CPP file
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One>::_SPOINT2(const ST &Val)
			:_stSPOINT2<ST>(Val, Val)
		{
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One>::_SPOINT2(const _SPOINT2<ST, _Zero, _One> &Point2)
			: _stSPOINT2<ST>(Point2.x, Point2.y)
		{
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One>::_SPOINT2(const _SPOINT3<ST, _Zero, _One> &Point3)
			: _stSPOINT2<ST>(Point3.x, Point3.y)
		{
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One>::_SPOINT2(const _SPOINT4<ST, _Zero, _One> &Point4)
			: _stSPOINT2<ST>(Point4.x, Point4.y)
		{
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One>::_SPOINT2(const _SVECTOR2<ST, _Zero, _One> &Vec2)
			: _stSPOINT2<ST>(Vec2.x, Vec2.y)
		{
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One>::_SPOINT2(const _SVECTOR3<ST, _Zero, _One> &Vec3)
			: _stSPOINT2<ST>(Vec3.x, Vec3.y)
		{
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One>::_SPOINT2(const _SVECTOR4<ST, _Zero, _One> &Vec4)
			: _stSPOINT2<ST>(Vec4.x, Vec4.y)
		{
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One>::_SPOINT2(const ST &x, const ST &y)
			: _stSPOINT2<ST>(x, y)
		{
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One>::_SPOINT2(const ST* pArray)
			: _stSPOINT2<ST>(_Zero(), _Zero())
		{
				if (pArray != nullptr)
				{
					this->x = pArray[0];
					this->y = pArray[1];
				}
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One>::~_SPOINT2()
		{
		}
		template<class ST, class _Zero, class _One>
		const ST& _SPOINT2<ST, _Zero, _One>::operator[](const uint32 Position) const
		{
			return this->p[Position > 2 ? 2 : Position];
		}
		template<class ST, class _Zero, class _One>
		ST& _SPOINT2<ST, _Zero, _One>::operator[](const uint32 Position)
		{
			return this->p[Position > 2 ? 2 : Position];
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One>& _SPOINT2<ST, _Zero, _One>::operator=(const _SPOINT2<ST, _Zero, _One> &Point2)
		{
			this->x = Point2.x;
			this->y = Point2.y;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One>& _SPOINT2<ST, _Zero, _One>::operator=(const _SPOINT3<ST, _Zero, _One> &Point3)
		{
			this->x = Point3.x;
			this->y = Point3.y;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One>& _SPOINT2<ST, _Zero, _One>::operator=(const _SPOINT4<ST, _Zero, _One> &Point4)
		{
			this->x = Point4.x;
			this->y = Point4.y;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One>& _SPOINT2<ST, _Zero, _One>::operator=(const _SVECTOR2<ST, _Zero, _One> &Vec2)
		{
			this->x = Vec2.x;
			this->y = Vec2.y;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One>& _SPOINT2<ST, _Zero, _One>::operator=(const _SVECTOR3<ST, _Zero, _One> &Vec3)
		{
			this->x = Vec3.x;
			this->y = Vec3.y;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One>& _SPOINT2<ST, _Zero, _One>::operator=(const _SVECTOR4<ST, _Zero, _One> &Vec4)
		{
			this->x = Vec4.x;
			this->y = Vec4.y;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One> _SPOINT2<ST, _Zero, _One>::operator+(const _SPOINT2<ST, _Zero, _One> &Point2) const
		{
			return _SPOINT2<ST, _Zero, _One>(this->x + Point2.x, this->y + Point2.y);
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One> _SPOINT2<ST, _Zero, _One>::operator-(const _SPOINT2<ST, _Zero, _One> &Point2) const
		{
			return _SPOINT2<ST, _Zero, _One>(this->x - Point2.x, this->y - Point2.y);
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One> _SPOINT2<ST, _Zero, _One>::operator*(const _SPOINT2<ST, _Zero, _One> &Point2) const
		{
			return _SPOINT2<ST, _Zero, _One>(this->x * Point2.x, this->y * Point2.y);
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One> _SPOINT2<ST, _Zero, _One>::operator+(const ST &Val) const
		{
			return _SPOINT2<ST, _Zero, _One>(this->x + Val, this->y + Val);
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One> _SPOINT2<ST, _Zero, _One>::operator-(const ST &Val) const
		{
			return _SPOINT2<ST, _Zero, _One>(this->x - Val, this->y - Val);
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One> _SPOINT2<ST, _Zero, _One>::operator*(const ST &Val) const
		{
			return _SPOINT2<ST, _Zero, _One>(this->x * Val, this->y * Val);
		}
		template<class ST, class _Zero, class _One>
		bool _SPOINT2<ST, _Zero, _One>::operator==(const _SPOINT2<ST, _Zero, _One> &Point2) const
		{
			if (this->x != Point2.x){ return false; }
			if (this->y != Point2.y){ return false; }

			//return (::gloIsFloatEqual(this->x, Point2.x) && ::gloIsFloatEqual(this->y, Point2.y)) ? true : false;
			return true;
		}
		template<class ST, class _Zero, class _One>
		bool _SPOINT2<ST, _Zero, _One>::operator!=(const _SPOINT2<ST, _Zero, _One> &Point2) const
		{
			return !((*this) == Point2);
		}
		template<class ST, class _Zero, class _One>
		bool _SPOINT2<ST, _Zero, _One>::operator<(const _SPOINT2<ST, _Zero, _One> &Point2) const
		{
			return (this->x < Point2.x) && (this->y < Point2.y);
		}
		template<class ST, class _Zero, class _One>
		bool _SPOINT2<ST, _Zero, _One>::operator>(const _SPOINT2<ST, _Zero, _One> &Point2) const
		{
			return (this->x > Point2.x) && (this->y > Point2.y);
		}
		template<class ST, class _Zero, class _One>
		bool _SPOINT2<ST, _Zero, _One>::operator<=(const _SPOINT2<ST, _Zero, _One> &Point2) const
		{
			return (this->x <= Point2.x) && (this->y <= Point2.y);
		}
		template<class ST, class _Zero, class _One>
		bool _SPOINT2<ST, _Zero, _One>::operator>=(const _SPOINT2<ST, _Zero, _One> &Point2) const
		{
			return (this->x >= Point2.x) && (this->y >= Point2.y);
		}
		template<class ST, class _Zero, class _One>
		_SPOINT2<ST, _Zero, _One> _SPOINT2<ST, _Zero, _One>::iGetHomogeneousPoint() const
		{
			if (this->y == _Zero())
			{
				return _SPOINT2<ST, _Zero, _One>(this->x, _One());
			}

			return _SPOINT2<ST, _Zero, _One>(this->x / this->y, _One());
		}
		template<class ST, class _Zero, class _One>
		void _SPOINT2<ST, _Zero, _One>::iHomogenzation()
		{
			if (this->y == _Zero())
			{
				this->y = _One();
				return;
			}

			this->x = this->x / this->y;
			this->y = _One();
		}
#pragma endregion

		template<class ST = sfloat, class _Zero = _cDefaultZeroGenerator<ST>, class _One = _cDefaultOneGenerator<ST>>
		struct _SVECTOR2 :public _stSPOINT2<ST>
		{
		public:
			_SVECTOR2(const ST &Val = _Zero()());
			_SVECTOR2(const ST &x, const ST &y);
			_SVECTOR2(const ST* pArray);
			_SVECTOR2(const _SVECTOR2<ST, _Zero, _One> &Vec2);

			explicit _SVECTOR2(const _SPOINT2<ST, _Zero, _One> &Point2);
			explicit _SVECTOR2(const _SPOINT3<ST, _Zero, _One> &Point3);
			explicit _SVECTOR2(const _SPOINT4<ST, _Zero, _One> &Point4);

			explicit _SVECTOR2(const _SVECTOR3<ST, _Zero, _One> &Vec3);
			explicit _SVECTOR2(const _SVECTOR4<ST, _Zero, _One> &Vec4);

			~_SVECTOR2();

			const ST& operator[](const uint32 Position) const;
			ST& operator[](const uint32 Position);

			_SVECTOR2<ST, _Zero, _One>& operator=(const _SPOINT2<ST, _Zero, _One> &Point2);
			_SVECTOR2<ST, _Zero, _One>& operator=(const _SPOINT3<ST, _Zero, _One> &Point3);
			_SVECTOR2<ST, _Zero, _One>& operator=(const _SPOINT4<ST, _Zero, _One> &Point4);
			_SVECTOR2<ST, _Zero, _One>& operator=(const _SVECTOR2<ST, _Zero, _One> &Vec2);
			_SVECTOR2<ST, _Zero, _One>& operator=(const _SVECTOR3<ST, _Zero, _One> &Vec3);
			_SVECTOR2<ST, _Zero, _One>& operator=(const _SVECTOR4<ST, _Zero, _One> &Vec4);

			_SVECTOR2<ST, _Zero, _One> operator+(const _SVECTOR2<ST, _Zero, _One> &Vec2) const;
			_SVECTOR2<ST, _Zero, _One> operator-(const _SVECTOR2<ST, _Zero, _One> &Vec2) const;
			_SVECTOR2<ST, _Zero, _One> operator*(const _SVECTOR2<ST, _Zero, _One> &Vec2) const;
			_SVECTOR2<ST, _Zero, _One> operator+(const ST &Val) const;
			_SVECTOR2<ST, _Zero, _One> operator-(const ST &Val) const;
			_SVECTOR2<ST, _Zero, _One> operator*(const ST &Val) const;

			bool operator==(const _SVECTOR2<ST, _Zero, _One> &Vec2) const;
			bool operator!=(const _SVECTOR2<ST, _Zero, _One> &Vec2) const;

			bool operator<(const _SVECTOR2<ST, _Zero, _One> &Vec2) const;
			bool operator>(const _SVECTOR2<ST, _Zero, _One> &Vec2) const;

			bool operator<=(const _SVECTOR2<ST, _Zero, _One> &Vec2) const;
			bool operator>=(const _SVECTOR2<ST, _Zero, _One> &Vec2) const;

			_SVECTOR2<ST, _Zero, _One>	iGetHomogeneousVector() const;
			void		iHomogenzation();
		};

#pragma region _SVECTOR2 CPP file
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One>::_SVECTOR2(const ST &Val)
			:_stSPOINT2<ST>(Val, Val)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One>::_SVECTOR2(const _SPOINT2<ST, _Zero, _One> &Point2)
			: _stSPOINT2<ST>(Point2.x, Point2.y)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One>::_SVECTOR2(const _SPOINT3<ST, _Zero, _One> &Point3)
			: _stSPOINT2<ST>(Point3.x, Point3.y)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One>::_SVECTOR2(const _SPOINT4<ST, _Zero, _One> &Point4)
			: _stSPOINT2<ST>(Point4.x, Point4.y)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One>::_SVECTOR2(const _SVECTOR2<ST, _Zero, _One> &Vec2)
			: _stSPOINT2<ST>(Vec2.x, Vec2.y)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One>::_SVECTOR2(const _SVECTOR3<ST, _Zero, _One> &Vec3)
			: _stSPOINT2<ST>(Vec3.x, Vec3.y)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One>::_SVECTOR2(const _SVECTOR4<ST, _Zero, _One> &Vec4)
			: _stSPOINT2<ST>(Vec4.x, Vec4.y)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One>::_SVECTOR2(const ST &x, const ST &y)
			: _stSPOINT2<ST>(x, y)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One>::_SVECTOR2(const ST* pArray)
			: _stSPOINT2<ST>(_Zero(), _Zero())
		{
				if (pArray != nullptr)
				{
					this->x = pArray[0];
					this->y = pArray[1];
				}
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One>::~_SVECTOR2()
		{
		}
		template<class ST, class _Zero, class _One>
		const ST& _SVECTOR2<ST, _Zero, _One>::operator[](const uint32 Position) const
		{
			return this->p[Position > 2 ? 2 : Position];
		}
		template<class ST, class _Zero, class _One>
		ST& _SVECTOR2<ST, _Zero, _One>::operator[](const uint32 Position)
		{
			return this->p[Position > 2 ? 2 : Position];
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One>& _SVECTOR2<ST, _Zero, _One>::operator=(const _SPOINT2<ST, _Zero, _One> &Point2)
		{
			this->x = Point2.x;
			this->y = Point2.y;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One>& _SVECTOR2<ST, _Zero, _One>::operator=(const _SPOINT3<ST, _Zero, _One> &Point3)
		{
			this->x = Point3.x;
			this->y = Point3.y;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One>& _SVECTOR2<ST, _Zero, _One>::operator=(const _SPOINT4<ST, _Zero, _One> &Point4)
		{
			this->x = Point4.x;
			this->y = Point4.y;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One>& _SVECTOR2<ST, _Zero, _One>::operator=(const _SVECTOR2<ST, _Zero, _One> &Vec2)
		{
			this->x = Vec2.x;
			this->y = Vec2.y;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One>& _SVECTOR2<ST, _Zero, _One>::operator=(const _SVECTOR3<ST, _Zero, _One> &Vec3)
		{
			this->x = Vec3.x;
			this->y = Vec3.y;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One>& _SVECTOR2<ST, _Zero, _One>::operator=(const _SVECTOR4<ST, _Zero, _One> &Vec4)
		{
			this->x = Vec4.x;
			this->y = Vec4.y;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One> _SVECTOR2<ST, _Zero, _One>::operator+(const _SVECTOR2<ST, _Zero, _One> &Vec2) const
		{
			return _SVECTOR2<ST, _Zero, _One>(this->x + Vec2.x, this->y + Vec2.y);
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One> _SVECTOR2<ST, _Zero, _One>::operator-(const _SVECTOR2<ST, _Zero, _One> &Vec2) const
		{
			return _SVECTOR2<ST, _Zero, _One>(this->x - Vec2.x, this->y - Vec2.y);
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One> _SVECTOR2<ST, _Zero, _One>::operator*(const _SVECTOR2<ST, _Zero, _One> &Vec2) const
		{
			return _SVECTOR2<ST, _Zero, _One>(this->x * Vec2.x, this->y * Vec2.y);
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One> _SVECTOR2<ST, _Zero, _One>::operator+(const ST &Val) const
		{
			return _SVECTOR2<ST, _Zero, _One>(this->x + Val, this->y + Val);
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One> _SVECTOR2<ST, _Zero, _One>::operator-(const ST &Val) const
		{
			return _SVECTOR2<ST, _Zero, _One>(this->x - Val, this->y - Val);
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One> _SVECTOR2<ST, _Zero, _One>::operator*(const ST &Val) const
		{
			return _SVECTOR2<ST, _Zero, _One>(this->x * Val, this->y * Val);
		}
		template<class ST, class _Zero, class _One>
		bool _SVECTOR2<ST, _Zero, _One>::operator==(const _SVECTOR2<ST, _Zero, _One> &Vec2) const
		{
			if (this->x != Vec2.x){ return false; }
			if (this->y != Vec2.y){ return false; }

			return true;
		}
		template<class ST, class _Zero, class _One>
		bool _SVECTOR2<ST, _Zero, _One>::operator!=(const _SVECTOR2<ST, _Zero, _One> &Vec2) const
		{
			return !((*this) == Vec2);
		}
		template<class ST, class _Zero, class _One>
		bool _SVECTOR2<ST, _Zero, _One>::operator<(const _SVECTOR2<ST, _Zero, _One> &Vec2) const
		{
			return (this->x < Vec2.x) && (this->y < Vec2.y);
		}
		template<class ST, class _Zero, class _One>
		bool _SVECTOR2<ST, _Zero, _One>::operator>(const _SVECTOR2<ST, _Zero, _One> &Vec2) const
		{
			return (this->x > Vec2.x) && (this->y > Vec2.y);
		}
		template<class ST, class _Zero, class _One>
		bool _SVECTOR2<ST, _Zero, _One>::operator<=(const _SVECTOR2<ST, _Zero, _One> &Vec2) const
		{
			return (this->x <= Vec2.x) && (this->y <= Vec2.y);
		}
		template<class ST, class _Zero, class _One>
		bool _SVECTOR2<ST, _Zero, _One>::operator>=(const _SVECTOR2<ST, _Zero, _One> &Vec2) const
		{
			return (this->x >= Vec2.x) && (this->y >= Vec2.y);
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR2<ST, _Zero, _One> _SVECTOR2<ST, _Zero, _One>::iGetHomogeneousVector() const
		{
			return _SVECTOR2<ST, _Zero, _One>(this->x, _Zero());
		}
		template<class ST, class _Zero, class _One>
		void _SVECTOR2<ST, _Zero, _One>::iHomogenzation()
		{
			this->y = _Zero();
		}
#pragma endregion

		template<class ST = sfloat, class _Zero = _cDefaultZeroGenerator<ST>, class _One = _cDefaultOneGenerator<ST>>
		struct _SPOINT3 :public _stSPOINT3<ST>
		{
		public:
			_SPOINT3(const ST &Val = _Zero());
			_SPOINT3(const ST &x, const ST &y, const ST &z);
			_SPOINT3(const ST* pArray);
			_SPOINT3(const _SPOINT3<ST, _Zero, _One> &Point3);

			explicit _SPOINT3(const _SPOINT2<ST, _Zero, _One> &Point2, const ST &z = _Zero());
			explicit _SPOINT3(const _SPOINT4<ST, _Zero, _One> &Point4);

			explicit _SPOINT3(const _SVECTOR2<ST, _Zero, _One> &Vec2, const ST &z = _Zero());
			explicit _SPOINT3(const _SVECTOR3<ST, _Zero, _One> &Vec3);
			explicit _SPOINT3(const _SVECTOR4<ST, _Zero, _One> &Vec4);

			~_SPOINT3();

			const ST& operator[](const uint32 Position) const;
			ST& operator[](const uint32 Position);

			//assignment operators
			_SPOINT3<ST, _Zero, _One>& operator=(const _SPOINT2<ST, _Zero, _One> &Point2);
			_SPOINT3<ST, _Zero, _One>& operator=(const _SPOINT3<ST, _Zero, _One> &Point3);
			_SPOINT3<ST, _Zero, _One>& operator=(const _SPOINT4<ST, _Zero, _One> &Point4);
			_SPOINT3<ST, _Zero, _One>& operator=(const _SVECTOR2<ST, _Zero, _One> &Vec2);
			_SPOINT3<ST, _Zero, _One>& operator=(const _SVECTOR3<ST, _Zero, _One> &Vec3);
			_SPOINT3<ST, _Zero, _One>& operator=(const _SVECTOR4<ST, _Zero, _One> &Vec4);

			//binary operators
			_SPOINT3<ST, _Zero, _One> operator+(const _SPOINT3<ST, _Zero, _One> &Point3) const;
			_SPOINT3<ST, _Zero, _One> operator-(const _SPOINT3<ST, _Zero, _One> &Point3) const;
			_SPOINT3<ST, _Zero, _One> operator*(const _SPOINT3<ST, _Zero, _One> &Point3) const;
			_SPOINT3<ST, _Zero, _One> operator+(const ST &Val) const;
			_SPOINT3<ST, _Zero, _One> operator-(const ST &Val) const;
			_SPOINT3<ST, _Zero, _One> operator*(const ST &Val) const;

			bool operator==(const _SPOINT3<ST, _Zero, _One> &Point3) const;
			bool operator!=(const _SPOINT3<ST, _Zero, _One> &Point3) const;

			bool operator<(const _SPOINT3<ST, _Zero, _One> &Point3) const;
			bool operator>(const _SPOINT3<ST, _Zero, _One> &Point3) const;

			bool operator<=(const _SPOINT3<ST, _Zero, _One> &Point3) const;
			bool operator>=(const _SPOINT3<ST, _Zero, _One> &Point3) const;

			_SPOINT3<ST, _Zero, _One>	iGetHomogeneousPoint() const;
			void	iHomogenzation();
		};

#pragma region _SPOINT3 CPP file
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One>::_SPOINT3(const ST &Val)
			:_stSPOINT3<ST>(Val, Val, Val)
		{
		}
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One>::_SPOINT3(const _SPOINT2<ST, _Zero, _One> &Point2, const ST &z)
			: _stSPOINT3<ST>(Point2.x, Point2.y, z)
		{
		}
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One>::_SPOINT3(const _SPOINT3<ST, _Zero, _One>& Point3)
			: _stSPOINT3<ST>(Point3.x, Point3.y, Point3.z)
		{
		}
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One>::_SPOINT3(const _SPOINT4<ST, _Zero, _One> &Point4)
			: _stSPOINT3<ST>(Point4.x, Point4.y, Point4.z)
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
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One>::_SPOINT3(const _SVECTOR2<ST, _Zero, _One> &Vec2, const ST &z)
			:_stSPOINT3<ST>(Vec2.x, Vec2.y, z)
		{
		}
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One>::_SPOINT3(const _SVECTOR3<ST, _Zero, _One> &Vec3)
			: _stSPOINT3<ST>(Vec3.x, Vec3.y, Vec3.z)
		{
		}
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One>::_SPOINT3(const _SVECTOR4<ST, _Zero, _One> &Vec4)
			: _stSPOINT3<ST>(Vec4.x, Vec4.y, Vec4.z)
		{
		}
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One>::_SPOINT3(const ST &x, const ST &y, const ST &z)
			: _stSPOINT3<ST>(x, y, z)
		{
		}
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One>::_SPOINT3(const ST* pArray)
			: _stSPOINT3<ST>(_Zero(), _Zero(), _Zero())
		{
				if (pArray != nullptr)
				{
					this->x = pArray[0];
					this->y = pArray[1];
					this->z = pArray[2];
				}
		}
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One>::~_SPOINT3<ST, _Zero, _One>()
		{
		}
		template<class ST, class _Zero, class _One>
		const ST& _SPOINT3<ST, _Zero, _One>::operator[](const uint32 Position) const
		{
			return this->p[Position > 3 ? 3 : Position];
		}
		template<class ST, class _Zero, class _One>
		ST& _SPOINT3<ST, _Zero, _One>::operator[](const uint32 Position)
		{
			return this->p[Position > 3 ? 3 : Position];
		}
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One>& _SPOINT3<ST, _Zero, _One>::operator=(const _SPOINT2<ST, _Zero, _One> &Point2)
		{
			this->x = Point2.x;
			this->y = Point2.y;
			this->z = _Zero();

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One>& _SPOINT3<ST, _Zero, _One>::operator=(const _SPOINT3<ST, _Zero, _One> &Point3)
		{
			this->x = Point3.x;
			this->y = Point3.y;
			this->z = Point3.z;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One>& _SPOINT3<ST, _Zero, _One>::operator=(const _SPOINT4<ST, _Zero, _One> &Point4)
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
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One>& _SPOINT3<ST, _Zero, _One>::operator=(const _SVECTOR2<ST, _Zero, _One> &Vec2)
		{
			this->x = Vec2.x;
			this->y = Vec2.y;
			this->z = _Zero();

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One>& _SPOINT3<ST, _Zero, _One>::operator=(const _SVECTOR3<ST, _Zero, _One> &Vec3)
		{
			this->x = Vec3.x;
			this->y = Vec3.y;
			this->z = Vec3.z;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One>& _SPOINT3<ST, _Zero, _One>::operator=(const _SVECTOR4<ST, _Zero, _One> &Vec4)
		{
			this->x = Vec4.x;
			this->y = Vec4.y;
			this->z = Vec4.z;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One> _SPOINT3<ST, _Zero, _One>::operator+(const _SPOINT3<ST, _Zero, _One> &Point3) const
		{
			return _SPOINT3<ST, _Zero, _One>(this->x + Point3.x, this->y + Point3.y, this->z + Point3.z);
		}
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One> _SPOINT3<ST, _Zero, _One>::operator-(const _SPOINT3<ST, _Zero, _One> &Point3) const
		{
			return _SPOINT3<ST, _Zero, _One>(this->x - Point3.x, this->y - Point3.y, this->z - Point3.z);
		}
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One> _SPOINT3<ST, _Zero, _One>::operator*(const _SPOINT3<ST, _Zero, _One> &Point3) const
		{
			return _SPOINT3<ST, _Zero, _One>(this->x * Point3.x, this->y * Point3.y, this->z * Point3.z);
		}
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One> _SPOINT3<ST, _Zero, _One>::operator+(const ST &Val) const
		{
			return _SPOINT3<ST, _Zero, _One>(this->x + Val, this->y + Val, this->z + Val);
		}
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One> _SPOINT3<ST, _Zero, _One>::operator-(const ST &Val) const
		{
			return _SPOINT3<ST, _Zero, _One>(this->x - Val, this->y - Val, this->z - Val);
		}
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One> _SPOINT3<ST, _Zero, _One>::operator*(const ST &Val) const
		{
			return _SPOINT3<ST, _Zero, _One>(this->x * Val, this->y * Val, this->z * Val);
		}
		template<class ST, class _Zero, class _One>
		bool _SPOINT3<ST, _Zero, _One>::operator==(const _SPOINT3<ST, _Zero, _One> &Point3) const
		{
			if (this->x != Point3.x){ return false; }
			if (this->y != Point3.y){ return false; }
			if (this->z != Point3.z){ return false; }

			return true;
		}
		template<class ST, class _Zero, class _One>
		bool _SPOINT3<ST, _Zero, _One>::operator!=(const _SPOINT3<ST, _Zero, _One> &Point3) const
		{
			return !((*this) == Point3);
		}
		template<class ST, class _Zero, class _One>
		bool _SPOINT3<ST, _Zero, _One>::operator<(const _SPOINT3<ST, _Zero, _One> &Point3) const
		{
			return (this->x < Point3.x) && (this->y < Point3.y) && (this->z < Point3.z);
		}
		template<class ST, class _Zero, class _One>
		bool _SPOINT3<ST, _Zero, _One>::operator>(const _SPOINT3<ST, _Zero, _One> &Point3) const
		{
			return (this->x > Point3.x) && (this->y > Point3.y) && (this->z > Point3.z);
		}
		template<class ST, class _Zero, class _One>
		bool _SPOINT3<ST, _Zero, _One>::operator<=(const _SPOINT3<ST, _Zero, _One> &Point3) const
		{
			return (this->x <= Point3.x) && (this->y <= Point3.y) && (this->z <= Point3.z);
		}
		template<class ST, class _Zero, class _One>
		bool _SPOINT3<ST, _Zero, _One>::operator>=(const _SPOINT3<ST, _Zero, _One> &Point3) const
		{
			return (this->x >= Point3.x) && (this->y >= Point3.y) && (this->z >= Point3.z);
		}
		template<class ST, class _Zero, class _One>
		_SPOINT3<ST, _Zero, _One> _SPOINT3<ST, _Zero, _One>::iGetHomogeneousPoint() const
		{
			if (this->z == _Zero())
			{
				return _SPOINT3<ST, _Zero, _One>(this->x, this->y, _One());
			}

			return _SPOINT3<ST, _Zero, _One>(this->x / this->z, this->y / this->z, _One());
		}
		template<class ST, class _Zero, class _One>
		void _SPOINT3<ST, _Zero, _One>::iHomogenzation()
		{
			if (this->z == _Zero())
			{
				this->z = _One();
			}

			this->x = this->x / this->z;
			this->y = this->y / this->z;
			this->z = _One();
		}
#pragma endregion

		template<class ST = sfloat, class _Zero = _cDefaultZeroGenerator<ST>, class _One = _cDefaultOneGenerator<ST>>
		struct _SVECTOR3 :public _stSPOINT3<ST>
		{
		public:
			_SVECTOR3(const ST &Val = _Zero()());
			_SVECTOR3(const ST &x, const ST &y, const ST &z);
			_SVECTOR3(const ST* pArray);
			_SVECTOR3(const _SVECTOR3<ST, _Zero, _One> &Vec3);

			explicit _SVECTOR3(const _SPOINT2<ST, _Zero, _One> &Point2, const ST &z = _Zero());
			explicit _SVECTOR3(const _SPOINT3<ST, _Zero, _One> &Point3);
			explicit _SVECTOR3(const _SPOINT4<ST, _Zero, _One> &Point4);

			explicit _SVECTOR3(const _SVECTOR2<ST, _Zero, _One> &Vec2, const ST &z = _Zero());
			explicit _SVECTOR3(const _SVECTOR4<ST, _Zero, _One> &Vec4);

			~_SVECTOR3();

			const ST& operator[](const uint32 Position) const;
			ST& operator[](const uint32 Position);

			_SVECTOR3<ST, _Zero, _One>& operator=(const _SPOINT2<ST, _Zero, _One> &Point2);
			_SVECTOR3<ST, _Zero, _One>& operator=(const _SPOINT3<ST, _Zero, _One> &Point3);
			_SVECTOR3<ST, _Zero, _One>& operator=(const _SPOINT4<ST, _Zero, _One> &Point4);
			_SVECTOR3<ST, _Zero, _One>& operator=(const _SVECTOR2<ST, _Zero, _One> &Vec2);
			_SVECTOR3<ST, _Zero, _One>& operator=(const _SVECTOR3<ST, _Zero, _One> &Vec3);
			_SVECTOR3<ST, _Zero, _One>& operator=(const _SVECTOR4<ST, _Zero, _One> &Vec4);

			_SVECTOR3<ST, _Zero, _One> operator+(const _SVECTOR3<ST, _Zero, _One> &Vec3) const;
			_SVECTOR3<ST, _Zero, _One> operator-(const _SVECTOR3<ST, _Zero, _One> &Vec3) const;
			_SVECTOR3<ST, _Zero, _One> operator*(const _SVECTOR3<ST, _Zero, _One> &Vec3) const;
			_SVECTOR3<ST, _Zero, _One> operator+(const ST &Val) const;
			_SVECTOR3<ST, _Zero, _One> operator-(const ST &Val) const;
			_SVECTOR3<ST, _Zero, _One> operator*(const ST &Val) const;

			bool operator==(const _SVECTOR3<ST, _Zero, _One> &Vec3) const;
			bool operator!=(const _SVECTOR3<ST, _Zero, _One> &Vec3) const;

			bool operator<(const _SVECTOR3<ST, _Zero, _One> &Vec3) const;
			bool operator>(const _SVECTOR3<ST, _Zero, _One> &Vec3) const;

			bool operator<=(const _SVECTOR3<ST, _Zero, _One> &Vec3) const;
			bool operator>=(const _SVECTOR3<ST, _Zero, _One> &Vec3) const;

			_SVECTOR3<ST, _Zero, _One>	iGetHomogeneousVector() const;
			void		iHomogenzation();
		};

#pragma region _SVECTOR3 CPP file
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One>::_SVECTOR3(const ST &Val)
			:_stSPOINT3<ST>(Val, Val, Val)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One>::_SVECTOR3(const _SPOINT2<ST, _Zero, _One> &Point2, const ST &z)
			: _stSPOINT3<ST>(Point2.x, Point2.y, z)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One>::_SVECTOR3(const _SPOINT3<ST, _Zero, _One> &Point3)
			: _stSPOINT3<ST>(Point3.x, Point3.y, Point3.z)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One>::_SVECTOR3(const _SPOINT4<ST, _Zero, _One> &Point4)
			: _stSPOINT3<ST>(Point4.x, Point4.y, Point4.z)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One>::_SVECTOR3(const _SVECTOR2<ST, _Zero, _One> &Vec2, const ST &z)
			: _stSPOINT3<ST>(Vec2.x, Vec2.y, z)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One>::_SVECTOR3(const _SVECTOR3<ST, _Zero, _One> &Vec3)
			: _stSPOINT3<ST>(Vec3.x, Vec3.y, Vec3.z)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One>::_SVECTOR3(const _SVECTOR4<ST, _Zero, _One> &Vec4)
			: _stSPOINT3<ST>(Vec4.x, Vec4.y, Vec4.z)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One>::_SVECTOR3(const ST &x, const ST &y, const ST &z)
			: _stSPOINT3<ST>(x, y, z)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One>::_SVECTOR3(const ST* pArray)
			: _stSPOINT3<ST>(_Zero(), _Zero(), _Zero())
		{
				if (pArray != nullptr)
				{
					this->x = pArray[0];
					this->y = pArray[1];
					this->z = pArray[2];
				}
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One>::~_SVECTOR3<ST, _Zero, _One>()
		{
		}
		template<class ST, class _Zero, class _One>
		const ST& _SVECTOR3<ST, _Zero, _One>::operator[](const uint32 Position) const
		{
			return this->p[Position > 3 ? 3 : Position];
		}
		template<class ST, class _Zero, class _One>
		ST& _SVECTOR3<ST, _Zero, _One>::operator[](const uint32 Position)
		{
			return this->p[Position > 3 ? 3 : Position];
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One>& _SVECTOR3<ST, _Zero, _One>::operator=(const _SPOINT2<ST, _Zero, _One> &Point2)
		{
			this->x = Point2.x;
			this->y = Point2.y;
			this->z = _Zero();

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One>& _SVECTOR3<ST, _Zero, _One>::operator=(const _SPOINT3<ST, _Zero, _One> &Point3)
		{
			this->x = Point3.x;
			this->y = Point3.y;
			this->z = Point3.z;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One>& _SVECTOR3<ST, _Zero, _One>::operator=(const _SPOINT4<ST, _Zero, _One> &Point4)
		{
			this->x = Point4.x;
			this->y = Point4.y;
			this->z = Point4.z;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One>& _SVECTOR3<ST, _Zero, _One>::operator=(const _SVECTOR2<ST, _Zero, _One> &Vec2)
		{
			this->x = Vec2.x;
			this->y = Vec2.y;
			this->z = _Zero();

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One>& _SVECTOR3<ST, _Zero, _One>::operator=(const _SVECTOR3<ST, _Zero, _One> &Vec3)
		{
			this->x = Vec3.x;
			this->y = Vec3.y;
			this->z = Vec3.z;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One>& _SVECTOR3<ST, _Zero, _One>::operator=(const _SVECTOR4<ST, _Zero, _One> &Vec4)
		{
			this->x = Vec4.x;
			this->y = Vec4.y;
			this->z = Vec4.z;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One> _SVECTOR3<ST, _Zero, _One>::operator+(const _SVECTOR3<ST, _Zero, _One> &Vec3) const
		{
			return _SVECTOR3<ST, _Zero, _One>(this->x + Vec3.x, this->y + Vec3.y, this->z + Vec3.z);
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One> _SVECTOR3<ST, _Zero, _One>::operator-(const _SVECTOR3<ST, _Zero, _One> &Vec3) const
		{
			return _SVECTOR3<ST, _Zero, _One>(this->x - Vec3.x, this->y - Vec3.y, this->z - Vec3.z);
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One> _SVECTOR3<ST, _Zero, _One>::operator*(const _SVECTOR3<ST, _Zero, _One> &Vec3) const
		{
			return _SVECTOR3<ST, _Zero, _One>(this->x * Vec3.x, this->y * Vec3.y, this->z * Vec3.z);
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One> _SVECTOR3<ST, _Zero, _One>::operator+(const ST &Val) const
		{
			return _SVECTOR3<ST, _Zero, _One>(this->x + Val, this->y + Val, this->z + Val);
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One> _SVECTOR3<ST, _Zero, _One>::operator-(const ST &Val) const
		{
			return _SVECTOR3<ST, _Zero, _One>(this->x - Val, this->y - Val, this->z - Val);
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One> _SVECTOR3<ST, _Zero, _One>::operator*(const ST &Val) const
		{
			return _SVECTOR3<ST, _Zero, _One>(this->x * Val, this->y * Val, this->z * Val);
		}
		template<class ST, class _Zero, class _One>
		bool _SVECTOR3<ST, _Zero, _One>::operator==(const _SVECTOR3<ST, _Zero, _One> &Vec3) const
		{
			if (this->x != Vec3.x){ return false; }
			if (this->y != Vec3.y){ return false; }
			if (this->z != Vec3.z){ return false; }

			return true;
		}
		template<class ST, class _Zero, class _One>
		bool _SVECTOR3<ST, _Zero, _One>::operator!=(const _SVECTOR3<ST, _Zero, _One> &Vec3) const
		{
			return !((*this) == Vec3);
		}
		template<class ST, class _Zero, class _One>
		bool _SVECTOR3<ST, _Zero, _One>::operator<(const _SVECTOR3<ST, _Zero, _One> &Vec3) const
		{
			return (this->x < Vec3.x) && (this->y < Vec3.y) && (this->z < Vec3.z);
		}
		template<class ST, class _Zero, class _One>
		bool _SVECTOR3<ST, _Zero, _One>::operator>(const _SVECTOR3<ST, _Zero, _One> &Vec3) const
		{
			return (this->x > Vec3.x) && (this->y > Vec3.y) && (this->z > Vec3.z);
		}
		template<class ST, class _Zero, class _One>
		bool _SVECTOR3<ST, _Zero, _One>::operator<=(const _SVECTOR3<ST, _Zero, _One> &Vec3) const
		{
			return (this->x <= Vec3.x) && (this->y <= Vec3.y) && (this->z <= Vec3.z);
		}
		template<class ST, class _Zero, class _One>
		bool _SVECTOR3<ST, _Zero, _One>::operator>=(const _SVECTOR3<ST, _Zero, _One> &Vec3) const
		{
			return (this->x >= Vec3.x) && (this->y >= Vec3.y) && (this->z >= Vec3.z);
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR3<ST, _Zero, _One> _SVECTOR3<ST, _Zero, _One>::iGetHomogeneousVector() const
		{
			return _SVECTOR3<ST, _Zero, _One>(this->x, this->y, _Zero());
		}
		template<class ST, class _Zero, class _One>
		void _SVECTOR3<ST, _Zero, _One>::iHomogenzation()
		{
			this->z = _Zero();
		}
#pragma endregion

		template<class ST = sfloat, class _Zero = _cDefaultZeroGenerator<ST>, class _One = _cDefaultOneGenerator<ST>>
		struct _SPOINT4 :public _stSPOINT4<ST>
		{
		public:
			_SPOINT4(const ST &Val = _Zero());
			_SPOINT4(const ST &x, const ST &y, const ST &z, const ST &w);
			_SPOINT4(const ST* pArray);
			_SPOINT4(const _SPOINT4<ST, _Zero, _One> &Point4);

			explicit _SPOINT4(const _SPOINT2<ST, _Zero, _One> &Point2, const ST &z = _Zero(), const ST &w = _Zero());
			explicit _SPOINT4(const _SPOINT3<ST, _Zero, _One> &Point3, const ST &w = _Zero());

			explicit _SPOINT4(const _SVECTOR2<ST, _Zero, _One> &Vec2, const ST &z = _Zero(), const ST &w = _Zero());
			explicit _SPOINT4(const _SVECTOR3<ST, _Zero, _One> &Vec3, const ST &w = _Zero());
			explicit _SPOINT4(const _SVECTOR4<ST, _Zero, _One> &Vec4);

			~_SPOINT4();

			const ST& operator[](const uint32 Position) const;
			ST& operator[](const uint32 Position);

			_SPOINT4<ST, _Zero, _One>& operator=(const _SPOINT2<ST, _Zero, _One> &Point2);
			_SPOINT4<ST, _Zero, _One>& operator=(const _SPOINT3<ST, _Zero, _One> &Point3);
			_SPOINT4<ST, _Zero, _One>& operator=(const _SPOINT4<ST, _Zero, _One> &Point4);
			_SPOINT4<ST, _Zero, _One>& operator=(const _SVECTOR2<ST, _Zero, _One> &Vec2);
			_SPOINT4<ST, _Zero, _One>& operator=(const _SVECTOR3<ST, _Zero, _One> &Vec3);
			_SPOINT4<ST, _Zero, _One>& operator=(const _SVECTOR4<ST, _Zero, _One> &Vec4);

			_SPOINT4<ST, _Zero, _One> operator+(const _SPOINT4<ST, _Zero, _One> &Point4) const;
			_SPOINT4<ST, _Zero, _One> operator-(const _SPOINT4<ST, _Zero, _One> &Point4) const;
			_SPOINT4<ST, _Zero, _One> operator*(const _SPOINT4<ST, _Zero, _One> &Point4) const;
			_SPOINT4<ST, _Zero, _One> operator+(const ST &Val) const;
			_SPOINT4<ST, _Zero, _One> operator-(const ST &Val) const;
			_SPOINT4<ST, _Zero, _One> operator*(const ST &Val) const;

			bool operator==(const _SPOINT4<ST, _Zero, _One> &Point4) const;
			bool operator!=(const _SPOINT4<ST, _Zero, _One> &Point4) const;

			bool operator<(const _SPOINT4<ST, _Zero, _One> &Point4) const;
			bool operator>(const _SPOINT4<ST, _Zero, _One> &Point4) const;

			bool operator<=(const _SPOINT4<ST, _Zero, _One> &Point4) const;
			bool operator>=(const _SPOINT4<ST, _Zero, _One> &Point4) const;

			_SPOINT4<ST, _Zero, _One>	iGetHomogeneousPoint() const;
			void	iHomogenzation();
		};

#pragma region _SPOINT4 CPP file
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One>::_SPOINT4(const ST &Val)
			:_stSPOINT4<ST>(Val, Val, Val, Val)
		{
		}
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One>::_SPOINT4(const _SPOINT2<ST, _Zero, _One> &Point2, const ST &z, const ST &w)
			: _stSPOINT4<ST>(Point2.x, Point2.y, z, w)
		{
		}
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One>::_SPOINT4(const _SPOINT3<ST, _Zero, _One> &Point3, const ST &w)
			: _stSPOINT4<ST>(Point3.x, Point3.y, Point3.z, w)
		{
		}
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One>::_SPOINT4(const _SPOINT4<ST, _Zero, _One> &Point4)
			: _stSPOINT4<ST>(Point4.x, Point4.y, Point4.z, Point4.w)
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
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One>::_SPOINT4(const _SVECTOR2<ST, _Zero, _One> &Vec2, const ST &z, const ST &w)
			:_stSPOINT4<ST>(Vec2.x, Vec2.y, z, w)
		{
		}
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One>::_SPOINT4(const _SVECTOR3<ST, _Zero, _One> &Vec3, const ST &w)
			: _stSPOINT4<ST>(Vec3.x, Vec3.y, Vec3.z, w)
		{

		}
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One>::_SPOINT4(const _SVECTOR4<ST, _Zero, _One> &Vec4)
			: _stSPOINT4<ST>(Vec4.x, Vec4.y, Vec4.z, Vec4.w)
		{
		}
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One>::_SPOINT4(const ST &x, const ST &y, const ST &z, const ST &w)
			: _stSPOINT4<ST>(x, y, z, w)
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
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One>::_SPOINT4(const ST* pArray)
			:_stSPOINT4<ST>(_Zero(), _Zero(), _Zero(), _Zero())
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
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One>::~_SPOINT4<ST, _Zero, _One>()
		{
		}
		template<class ST, class _Zero, class _One>
		const ST& _SPOINT4<ST, _Zero, _One>::operator[](const uint32 Position) const
		{
			return this->p[Position > 4 ? 4 : Position];
		}
		template<class ST, class _Zero, class _One>
		ST& _SPOINT4<ST, _Zero, _One>::operator[](const uint32 Position)
		{
			return this->p[Position > 4 ? 4 : Position];
		}
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One>& _SPOINT4<ST, _Zero, _One>::operator=(const _SPOINT2<ST, _Zero, _One> &Point2)
		{
			this->x = Point2.x;
			this->y = Point2.y;
			this->z = _Zero();
			this->w = _Zero();

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One>& _SPOINT4<ST, _Zero, _One>::operator=(const _SPOINT3<ST, _Zero, _One> &Point3)
		{
			this->x = Point3.x;
			this->y = Point3.y;
			this->z = Point3.z;
			this->w = _Zero();

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One>& _SPOINT4<ST, _Zero, _One>::operator=(const _SPOINT4<ST, _Zero, _One> &Point4)
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
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One>& _SPOINT4<ST, _Zero, _One>::operator=(const _SVECTOR2<ST, _Zero, _One> &Vec2)
		{
			this->x = Vec2.x;
			this->y = Vec2.y;
			this->z = _Zero();
			this->w = _Zero();

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One>& _SPOINT4<ST, _Zero, _One>::operator=(const _SVECTOR3<ST, _Zero, _One> &Vec3)
		{
			this->x = Vec3.x;
			this->y = Vec3.y;
			this->z = Vec3.z;
			this->w = _Zero();

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One>& _SPOINT4<ST, _Zero, _One>::operator=(const _SVECTOR4<ST, _Zero, _One> &Vec4)
		{
			this->x = Vec4.x;
			this->y = Vec4.y;
			this->z = Vec4.z;
			this->w = Vec4.w;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One> _SPOINT4<ST, _Zero, _One>::operator+(const _SPOINT4<ST, _Zero, _One> &Point4) const
		{
			/*_SPOINT4<ST,Zero> HomoPointSrc(*this);
			_SPOINT4<ST,Zero> HomoPointDesc(Point4);*/
			return _SPOINT4<ST, _Zero, _One>(this->x + Point4.x, this->y + Point4.y, this->z + Point4.z, this->w + Point4.w);
		}
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One> _SPOINT4<ST, _Zero, _One>::operator-(const _SPOINT4<ST, _Zero, _One> &Point4) const
		{
			/*_SPOINT4<ST,Zero> HomoPointSrc(*this);
			_SPOINT4<ST,Zero> HomoPointDesc(Point4);*/
			return _SPOINT4<ST, _Zero, _One>(this->x - Point4.x, this->y - Point4.y, this->z - Point4.z, this->w - Point4.w);
		}
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One> _SPOINT4<ST, _Zero, _One>::operator*(const _SPOINT4<ST, _Zero, _One> &Point4) const
		{
			/*_SPOINT4<ST,Zero> HomoPointSrc(*this);
			_SPOINT4<ST,Zero> HomoPointDesc(Point4);*/
			return _SPOINT4<ST, _Zero, _One>(this->x * Point4.x, this->y * Point4.y, this->z * Point4.z, this->w * Point4.w);
		}
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One> _SPOINT4<ST, _Zero, _One>::operator+(const ST &Val) const
		{
			return _SPOINT4<ST, _Zero, _One>(this->x + Val, this->y + Val, this->z + Val, this->w + Val);
		}
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One> _SPOINT4<ST, _Zero, _One>::operator-(const ST &Val) const
		{
			return _SPOINT4<ST, _Zero, _One>(this->x - Val, this->y - Val, this->z - Val, this->w - Val);
		}
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One> _SPOINT4<ST, _Zero, _One>::operator*(const ST &Val) const
		{
			return _SPOINT4<ST, _Zero, _One>(this->x * Val, this->y * Val, this->z * Val, this->w * Val);
		}
		template<class ST, class _Zero, class _One>
		bool _SPOINT4<ST, _Zero, _One>::operator==(const _SPOINT4<ST, _Zero, _One> &Point4) const
		{
			if (this->x != Point4.x){ return false; }
			if (this->y != Point4.y){ return false; }
			if (this->z != Point4.z){ return false; }
			if (this->w != Point4.w){ return false; }
			return true;
		}
		template<class ST, class _Zero, class _One>
		bool _SPOINT4<ST, _Zero, _One>::operator!=(const _SPOINT4<ST, _Zero, _One> &Point4) const
		{
			return !((*this) == Point4);
		}
		template<class ST, class _Zero, class _One>
		bool _SPOINT4<ST, _Zero, _One>::operator<(const _SPOINT4<ST, _Zero, _One> &Point4) const
		{
			return (this->x < Point4.x) && (this->y < Point4.y) && (this->z < Point4.z) && (this->w < Point4.w);
		}
		template<class ST, class _Zero, class _One>
		bool _SPOINT4<ST, _Zero, _One>::operator>(const _SPOINT4<ST, _Zero, _One> &Point4) const
		{
			return (this->x > Point4.x) && (this->y > Point4.y) && (this->z > Point4.z) && (this->w > Point4.w);
		}
		template<class ST, class _Zero, class _One>
		bool _SPOINT4<ST, _Zero, _One>::operator<=(const _SPOINT4<ST, _Zero, _One> &Point4) const
		{
			return (this->x <= Point4.x) && (this->y <= Point4.y) && (this->z <= Point4.z) && (this->w <= Point4.w);
		}
		template<class ST, class _Zero, class _One>
		bool _SPOINT4<ST, _Zero, _One>::operator>=(const _SPOINT4<ST, _Zero, _One> &Point4) const
		{
			return (this->x >= Point4.x) && (this->y >= Point4.y) && (this->z >= Point4.z) && (this->w >= Point4.w);
		}
		template<class ST, class _Zero, class _One>
		_SPOINT4<ST, _Zero, _One> _SPOINT4<ST, _Zero, _One>::iGetHomogeneousPoint() const
		{
			if (this->w == _Zero())
			{
				return _SPOINT4<ST, _Zero, _One>(this->x, this->y, this->z, _One());
			}

			return _SPOINT4<ST, _Zero, _One>(this->x / this->w, this->y / this->w, this->z / this->w, _One());
		}
		template<class ST, class _Zero, class _One>
		void _SPOINT4<ST, _Zero, _One>::iHomogenzation()
		{
			if (this->w == _Zero())
			{
				this->w = _One();
			}

			this->x = this->x / this->w;
			this->y = this->y / this->w;
			this->z = this->z / this->w;
			this->w = _One();
		}
#pragma endregion

		template<class ST = sfloat, class _Zero = _cDefaultZeroGenerator<ST>, class _One = _cDefaultOneGenerator<ST>>
		struct _SVECTOR4 :public _stSPOINT4<ST>
		{
		public:
			_SVECTOR4(const ST &Val = _Zero());
			_SVECTOR4(const ST &x, const ST &y, const ST &z, const ST &w);
			_SVECTOR4(const ST* pArray);
			_SVECTOR4(const _SVECTOR4<ST, _Zero, _One> &Vec4);

			explicit _SVECTOR4(const _SPOINT2<ST, _Zero, _One> &Point2, const ST &z = _Zero(), const ST &w = _Zero());
			explicit _SVECTOR4(const _SPOINT3<ST, _Zero, _One> &Point3, const ST &w = _Zero());
			explicit _SVECTOR4(const _SPOINT4<ST, _Zero, _One> &Point4);

			explicit _SVECTOR4(const _SVECTOR2<ST, _Zero, _One> &Vec2, const ST &z = _Zero(), const ST &w = _Zero());
			explicit _SVECTOR4(const _SVECTOR3<ST, _Zero, _One> &Vec3, const ST &w = _Zero());

			~_SVECTOR4();

			const ST& operator[](const uint32 Position) const;
			ST& operator[](const uint32 Position);

			_SVECTOR4<ST, _Zero, _One>& operator=(const _SPOINT2<ST, _Zero, _One> &Point2);
			_SVECTOR4<ST, _Zero, _One>& operator=(const _SPOINT3<ST, _Zero, _One> &Point3);
			_SVECTOR4<ST, _Zero, _One>& operator=(const _SPOINT4<ST, _Zero, _One> &Point4);
			_SVECTOR4<ST, _Zero, _One>& operator=(const _SVECTOR2<ST, _Zero, _One> &Vec2);
			_SVECTOR4<ST, _Zero, _One>& operator=(const _SVECTOR3<ST, _Zero, _One> &Vec3);
			_SVECTOR4<ST, _Zero, _One>& operator=(const _SVECTOR4<ST, _Zero, _One> &Vec4);

			_SVECTOR4<ST, _Zero, _One> operator+(const _SVECTOR4<ST, _Zero, _One> &Vec4) const;
			_SVECTOR4<ST, _Zero, _One> operator-(const _SVECTOR4<ST, _Zero, _One> &Vec4) const;
			_SVECTOR4<ST, _Zero, _One> operator*(const _SVECTOR4<ST, _Zero, _One> &Vec4) const;
			_SVECTOR4<ST, _Zero, _One> operator+(const ST &Val) const;
			_SVECTOR4<ST, _Zero, _One> operator-(const ST &Val) const;
			_SVECTOR4<ST, _Zero, _One> operator*(const ST &Val) const;

			bool operator==(const _SVECTOR4<ST, _Zero, _One> &Vec4) const;
			bool operator!=(const _SVECTOR4<ST, _Zero, _One> &Vec4) const;

			bool operator<(const _SVECTOR4<ST, _Zero, _One> &Vec4) const;
			bool operator>(const _SVECTOR4<ST, _Zero, _One> &Vec4) const;

			bool operator<=(const _SVECTOR4<ST, _Zero, _One> &Vec4) const;
			bool operator>=(const _SVECTOR4<ST, _Zero, _One> &Vec4) const;

			_SVECTOR4<ST, _Zero, _One>	iGetHomogeneousVector() const;
			void		iHomogenzation();
		};

#pragma region _SVECTOR4 CPP file
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One>::_SVECTOR4(const ST &Val)
			:_stSPOINT4<ST>(Val, Val, Val, Val)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One>::_SVECTOR4(const _SPOINT2<ST, _Zero, _One> &Point2, const ST &z, const ST &w)
			: _stSPOINT4<ST>(Point2.x, Point2.y, z, w)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One>::_SVECTOR4(const _SPOINT3<ST, _Zero, _One> &Point3, const ST &w)
			: _stSPOINT4<ST>(Point3.x, Point3.y, Point3.z, w)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One>::_SVECTOR4(const _SPOINT4<ST, _Zero, _One> &Point4)
			: _stSPOINT4<ST>(Point4.x, Point4.y, Point4.z, Point4.w)
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
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One>::_SVECTOR4(const _SVECTOR2<ST, _Zero, _One> &Vec2, const ST &z, const ST &w)
			:_stSPOINT4<ST>(Vec2.x, Vec2.y, z, w)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One>::_SVECTOR4(const _SVECTOR3<ST, _Zero, _One> &Vec3, const ST &w)
			: _stSPOINT4<ST>(Vec3.x, Vec3.y, Vec3.z, w)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One>::_SVECTOR4(const _SVECTOR4<ST, _Zero, _One> &Vec4)
			: _stSPOINT4<ST>(Vec4.x, Vec4.y, Vec4.z, Vec4.w)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One>::_SVECTOR4(const ST &x, const ST &y, const ST &z, const ST &w)
			: _stSPOINT4<ST>(x, y, z, w)
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
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One>::_SVECTOR4(const ST* pArray)
			:_stSPOINT4<ST>(_Zero(), _Zero(), _Zero(), _Zero())
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
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One>::~_SVECTOR4<ST, _Zero, _One>()
		{
		}
		template<class ST, class _Zero, class _One>
		const ST& _SVECTOR4<ST, _Zero, _One>::operator[](const uint32 Position) const
		{
			return this->p[Position > 4 ? 4 : Position];
		}
		template<class ST, class _Zero, class _One>
		ST& _SVECTOR4<ST, _Zero, _One>::operator[](const uint32 Position)
		{
			return this->p[Position > 4 ? 4 : Position];
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One>& _SVECTOR4<ST, _Zero, _One>::operator=(const _SPOINT2<ST, _Zero, _One> &Point2)
		{
			this->x = Point2.x;
			this->y = Point2.y;
			this->z = _Zero();
			this->w = _Zero();

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One>& _SVECTOR4<ST, _Zero, _One>::operator=(const _SPOINT3<ST, _Zero, _One> &Point3)
		{
			this->x = Point3.x;
			this->y = Point3.y;
			this->z = Point3.z;
			this->w = _Zero();

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One>& _SVECTOR4<ST, _Zero, _One>::operator=(const _SPOINT4<ST, _Zero, _One> &Point4)
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
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One>& _SVECTOR4<ST, _Zero, _One>::operator=(const _SVECTOR2<ST, _Zero, _One> &Vec2)
		{
			this->x = Vec2.x;
			this->y = Vec2.y;
			this->z = _Zero();
			this->w = _Zero();

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One>& _SVECTOR4<ST, _Zero, _One>::operator=(const _SVECTOR3<ST, _Zero, _One> &Vec3)
		{
			this->x = Vec3.x;
			this->y = Vec3.y;
			this->z = Vec3.z;
			this->w = _Zero();

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One>& _SVECTOR4<ST, _Zero, _One>::operator=(const _SVECTOR4<ST, _Zero, _One> &Vec4)
		{
			this->x = Vec4.x;
			this->y = Vec4.y;
			this->z = Vec4.z;
			this->w = Vec4.w;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One> _SVECTOR4<ST, _Zero, _One>::operator+(const _SVECTOR4<ST, _Zero, _One> &Vec4) const
		{
			return _SVECTOR4<ST, _Zero, _One>(this->x + Vec4.x, this->y + Vec4.y, this->z + Vec4.z, _Zero());
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One> _SVECTOR4<ST, _Zero, _One>::operator-(const _SVECTOR4<ST, _Zero, _One> &Vec4) const
		{
			return _SVECTOR4<ST, _Zero, _One>(this->x - Vec4.x, this->y - Vec4.y, this->z - Vec4.z, _Zero());
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One> _SVECTOR4<ST, _Zero, _One>::operator*(const _SVECTOR4<ST, _Zero, _One> &Vec4) const
		{
			return _SVECTOR4<ST, _Zero, _One>(this->x * Vec4.x, this->y * Vec4.y, this->z * Vec4.z, _Zero());
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One> _SVECTOR4<ST, _Zero, _One>::operator+(const ST &Val) const
		{
			return _SVECTOR4<ST, _Zero, _One>(this->x + Val, this->y + Val, this->z + Val, 0.0);
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One> _SVECTOR4<ST, _Zero, _One>::operator-(const ST &Val) const
		{
			return _SVECTOR4<ST, _Zero, _One>(this->x - Val, this->y - Val, this->z - Val, 0.0);
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One> _SVECTOR4<ST, _Zero, _One>::operator*(const ST &Val) const
		{
			return _SVECTOR4<ST, _Zero, _One>(this->x * Val, this->y * Val, this->z * Val, 0.0);
		}
		template<class ST, class _Zero, class _One>
		bool _SVECTOR4<ST, _Zero, _One>::operator==(const _SVECTOR4<ST, _Zero, _One> &Vec4) const
		{
			if (this->x != Vec4.x){ return false; }
			if (this->y != Vec4.y){ return false; }
			if (this->z != Vec4.z){ return false; }
			if (this->w != Vec4.w){ return false; }

			return true;
		}
		template<class ST, class _Zero, class _One>
		bool _SVECTOR4<ST, _Zero, _One>::operator!=(const _SVECTOR4<ST, _Zero, _One> &Vec4) const
		{
			return !((*this) == Vec4);
		}
		template<class ST, class _Zero, class _One>
		bool _SVECTOR4<ST, _Zero, _One>::operator<(const _SVECTOR4<ST, _Zero, _One> &Vec4) const
		{
			return (this->x < Vec4.x) && (this->y < Vec4.y) && (this->z < Vec4.z) && (this->w < Vec4.w);
		}
		template<class ST, class _Zero, class _One>
		bool _SVECTOR4<ST, _Zero, _One>::operator>(const _SVECTOR4<ST, _Zero, _One> &Vec4) const
		{
			return (this->x > Vec4.x) && (this->y > Vec4.y) && (this->z > Vec4.z) && (this->w > Vec4.w);
		}
		template<class ST, class _Zero, class _One>
		bool _SVECTOR4<ST, _Zero, _One>::operator<=(const _SVECTOR4<ST, _Zero, _One> &Vec4) const
		{
			return (this->x <= Vec4.x) && (this->y <= Vec4.y) && (this->z <= Vec4.z) && (this->w <= Vec4.w);
		}
		template<class ST, class _Zero, class _One>
		bool _SVECTOR4<ST, _Zero, _One>::operator>=(const _SVECTOR4<ST, _Zero, _One> &Vec4) const
		{
			return (this->x >= Vec4.x) && (this->y >= Vec4.y) && (this->z >= Vec4.z) && (this->w >= Vec4.w);
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR4<ST, _Zero, _One> _SVECTOR4<ST, _Zero, _One>::iGetHomogeneousVector() const
		{
			return _SVECTOR4<ST, _Zero, _One>(this->x, this->y, this->z, _Zero());
		}
		template<class ST, class _Zero, class _One>
		void _SVECTOR4<ST, _Zero, _One>::iHomogenzation()
		{
			this->w = _Zero();
		}
#pragma endregion

		template<class ST = sfloat, class _Zero = _cDefaultZeroGenerator<ST>, class _One = _cDefaultOneGenerator<ST>>
		struct _SVECTOR
		{
		public:
			_sstream<ST> vec;
			typedef typename _sstream<ST>::iterator iterator;
		public:
			_SVECTOR(const uint32 Dimension = 3, const ST &InitVal = _Zero());
			_SVECTOR(const _SVECTOR<ST, _Zero, _One> &Vec);
			_SVECTOR(const uint32 Dimension, const ST* pArray, const uint32 ArraySize);
			~_SVECTOR();

			const ST& operator[](const uint32 Position) const;
			ST& operator[](const uint32 Position);


			_SVECTOR<ST, _Zero, _One>& operator=(const _SPOINT2<ST, _Zero, _One> &Point2);
			_SVECTOR<ST, _Zero, _One>& operator=(const _SPOINT3<ST, _Zero, _One> &Point3);
			_SVECTOR<ST, _Zero, _One>& operator=(const _SPOINT4<ST, _Zero, _One> &Point4);
			_SVECTOR<ST, _Zero, _One>& operator=(const _SVECTOR2<ST, _Zero, _One> &Vec2);
			_SVECTOR<ST, _Zero, _One>& operator=(const _SVECTOR3<ST, _Zero, _One> &Vec3);
			_SVECTOR<ST, _Zero, _One>& operator=(const _SVECTOR4<ST, _Zero, _One> &Vec4);
			_SVECTOR<ST, _Zero, _One>& operator=(const _SVECTOR<ST, _Zero, _One> &Vec);

			_SVECTOR<ST, _Zero, _One> operator+(const _SVECTOR<ST, _Zero, _One> &Vec) const;
			_SVECTOR<ST, _Zero, _One> operator-(const _SVECTOR<ST, _Zero, _One> &Vec) const;
			_SVECTOR<ST, _Zero, _One> operator*(const _SVECTOR<ST, _Zero, _One> &Vec) const;
			_SVECTOR<ST, _Zero, _One> operator+(const ST &Val) const;
			_SVECTOR<ST, _Zero, _One> operator-(const ST &Val) const;
			_SVECTOR<ST, _Zero, _One> operator*(const ST &Val) const;

			bool operator==(const _SVECTOR<ST, _Zero, _One> &Vec) const;
			bool operator!=(const _SVECTOR<ST, _Zero, _One> &Vec) const;

			_SVECTOR<ST, _Zero, _One> iGetHomogenzationVector() const;
			void iHomogenzation();

			uint32 iGetVectorDimension() const;

			//C++ STL style iterator functions
			iterator begin() const;
			iterator end() const;
			uint32 size() const;
		};

#pragma region _SVECTOR CPP file
		template<class ST, class _Zero, class _One>
		_SVECTOR<ST, _Zero, _One>::_SVECTOR(const uint32 Dimension, const ST &InitVal)
			:vec(Dimension, InitVal)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR<ST, _Zero, _One>::_SVECTOR(const _SVECTOR<ST, _Zero, _One> &Vec)
			:vec(Vec.vec)
		{
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR<ST, _Zero, _One>::_SVECTOR(const uint32 Dimension, const ST* pArray, const uint32 ArraySize)
			:vec(Dimension, _Zero())
		{
			for (uint32 seek = 0; seek < ArraySize; seek = seek + 1)
			{
				this->vec[seek] = pArray[seek];
			}
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR<ST, _Zero, _One>::~_SVECTOR()
		{
		}
		template<class ST, class _Zero, class _One>
		const ST& _SVECTOR<ST, _Zero, _One>::operator[](const uint32 Position) const
		{
			return this->vec[Position];
		}
		template<class ST, class _Zero, class _One>
		ST& _SVECTOR<ST, _Zero, _One>::operator[](const uint32 Position)
		{
			return this->vec[Position];
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR<ST, _Zero, _One>& _SVECTOR<ST, _Zero, _One>::operator=(const _SPOINT2<ST, _Zero, _One> &Point2)
		{
			this->vec.iReSizeStream(2);
			this->vec[0] = Point2.x;
			this->vec[1] = Point2.y;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR<ST, _Zero, _One>& _SVECTOR<ST, _Zero, _One>::operator=(const _SPOINT3<ST, _Zero, _One> &Point3)
		{
			this->vec.iReSizeStream(3);
			this->vec[0] = Point3.x;
			this->vec[1] = Point3.y;
			this->vec[2] = Point3.z;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR<ST, _Zero, _One>& _SVECTOR<ST, _Zero, _One>::operator=(const _SPOINT4<ST, _Zero, _One> &Point4)
		{
			this->vec.iReSizeStream(4);
			this->vec[0] = Point4.x;
			this->vec[1] = Point4.y;
			this->vec[2] = Point4.z;
			this->vec[4] = Point4.w;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR<ST, _Zero, _One>& _SVECTOR<ST, _Zero, _One>::operator=(const _SVECTOR2<ST, _Zero, _One> &Vec2)
		{
			this->vec.iReSizeStream(2);
			this->vec[0] = Vec2.x;
			this->vec[1] = Vec2.y;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR<ST, _Zero, _One>& _SVECTOR<ST, _Zero, _One>::operator=(const _SVECTOR3<ST, _Zero, _One> &Vec3)
		{
			this->vec.iReSizeStream(3);
			this->vec[0] = Vec3.x;
			this->vec[1] = Vec3.y;
			this->vec[2] = Vec3.z;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR<ST, _Zero, _One>& _SVECTOR<ST, _Zero, _One>::operator=(const _SVECTOR4<ST, _Zero, _One> &Vec4)
		{
			this->vec.iReSizeStream(4);
			this->vec[0] = Vec4.x;
			this->vec[1] = Vec4.y;
			this->vec[2] = Vec4.z;
			this->vec[4] = Vec4.w;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR<ST, _Zero, _One>& _SVECTOR<ST, _Zero, _One>::operator=(const _SVECTOR<ST, _Zero, _One> &Vec)
		{
			this->vec = Vec.vec;

			return *this;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR<ST, _Zero, _One> _SVECTOR<ST, _Zero, _One>::operator+(const _SVECTOR<ST, _Zero, _One> &Vec) const
		{
			if (this->vec.Size != Vec.vec.Size()){ return _SVECTOR<ST, _Zero, _One>(); }

			_SVECTOR<ST, _Zero, _One> ResVec(this->vec.Size);

			for (uint32 seek = 0; seek < this->vec.Size(); seek = seek + 1)
			{
				ResVec[seek] = this->vec[seek] + Vec[seek];
			}

			return ResVec;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR<ST, _Zero, _One> _SVECTOR<ST, _Zero, _One>::operator-(const _SVECTOR<ST, _Zero, _One> &Vec) const
		{
			if (this->vec.Size != Vec.vec.Size()){ return _SVECTOR<ST, _Zero, _One>(); }

			_SVECTOR<ST, _Zero, _One> ResVec(this->vec.Size);

			for (uint32 seek = 0; seek < this->vec.Size(); seek = seek + 1)
			{
				ResVec[seek] = this->vec[seek] - Vec[seek];
			}

			return ResVec;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR<ST, _Zero, _One> _SVECTOR<ST, _Zero, _One>::operator*(const _SVECTOR<ST, _Zero, _One> &Vec) const
		{
			if (this->vec.Size != Vec.vec.Size()){ return _SVECTOR<ST, _Zero, _One>(); }

			_SVECTOR<ST, _Zero, _One> ResVec(this->vec.Size);

			for (uint32 seek = 0; seek < this->vec.Size(); seek = seek + 1)
			{
				ResVec[seek] = this->vec[seek] * Vec[seek];
			}

			return ResVec;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR<ST, _Zero, _One> _SVECTOR<ST, _Zero, _One>::operator+(const ST &Val) const
		{
			_SVECTOR<ST, _Zero, _One> ResVec(this->vec.Size);

			for (uint32 seek = 0; seek < this->vec.Size(); seek = seek + 1)
			{
				ResVec[seek] = this->vec[seek] + Val;
			}

			return ResVec;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR<ST, _Zero, _One> _SVECTOR<ST, _Zero, _One>::operator-(const ST &Val) const
		{
			_SVECTOR<ST, _Zero, _One> ResVec(this->vec.Size);

			for (uint32 seek = 0; seek < this->vec.Size(); seek = seek + 1)
			{
				ResVec[seek] = this->vec[seek] - Val;
			}

			return ResVec;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR<ST, _Zero, _One> _SVECTOR<ST, _Zero, _One>::operator*(const ST &Val) const
		{
			_SVECTOR<ST, _Zero, _One> ResVec(this->vec.Size);

			for (uint32 seek = 0; seek < this->vec.Size(); seek = seek + 1)
			{
				ResVec[seek] = this->vec[seek] * Val;
			}

			return ResVec;
		}
		template<class ST, class _Zero, class _One>
		bool _SVECTOR<ST, _Zero, _One>::operator==(const _SVECTOR<ST, _Zero, _One> &Vec) const
		{
			return this->vec == Vec.vec;
		}
		template<class ST, class _Zero, class _One>
		bool _SVECTOR<ST, _Zero, _One>::operator!=(const _SVECTOR<ST, _Zero, _One> &Vec) const
		{
			return this->vec != Vec.vec;
		}
		template<class ST, class _Zero, class _One>
		_SVECTOR<ST, _Zero, _One> _SVECTOR<ST, _Zero, _One>::iGetHomogenzationVector() const
		{
			_SVECTOR<ST, _Zero, _One> Vector(*this);
			Vector.vec->pStream[Vector.vec.Size - 1] = _Zero();

			return Vector;
		}
		template<class ST, class _Zero, class _One>
		void _SVECTOR<ST, _Zero, _One>::iHomogenzation()
		{
			this->vec.pStream[this->vec.Size - 1] = _Zero();
		}
		template<class ST, class _Zero, class _One>
		uint32 _SVECTOR<ST, _Zero, _One>::iGetVectorDimension() const
		{
			return this->vec.Size;
		}
		template<class ST, class _Zero, class _One>
		typename _SVECTOR<ST, _Zero, _One>::iterator _SVECTOR<ST, _Zero, _One>::begin() const
		{
			return this->vec.begin();
		}
		template<class ST, class _Zero, class _One>
		typename _SVECTOR<ST, _Zero, _One>::iterator _SVECTOR<ST, _Zero, _One>::end() const
		{
			return this->vec.end();
		}
		template<class ST, class _Zero, class _One>
		uint32 _SVECTOR<ST, _Zero, _One>::size() const
		{
			return this->vec->Size;
		};
#pragma endregion

		template<class ST, class _Zero = _cDefaultZeroGenerator<ST>, class _One = _cDefaultOneGenerator<ST>>
		using _svector = _SVECTOR<ST, _Zero, _One>;
#endif
	}
}