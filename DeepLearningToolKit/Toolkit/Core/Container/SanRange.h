#include"SanContainerDef.h"
#pragma once
using namespace std;
namespace San
{
#ifndef __STDSANRANGE_H__
#define __STDSANRANGE_H__
	template<class _type, class _Allocator = cSanSystemAllocator>
	struct _srange
	{
	private:
		_type Min;
		_type Max;
		_type Step;
	public:
		_srange(const _type &Min, const _type &Max, const _type &Step = 1)
			:Min(Min), Max(Max), Step(Step)
		{
		};
		_srange(const _srange<_type, _Allocator> &Range)
			:Min(Range.Min), Max(Range.Max), Step(Range.Step)
		{
		};
		~_srange()
		{
		};

		bool operator==(const _srange<_type, _Allocator> &Range) const
		{
			return (this->Min == Range.Min) && (this->Max == Range.Max) && (this->Step == Range.Step);
		};
		bool operator!=(const _srange<_type, _Allocator> &Range) const
		{
			return (this->Min != Range.Min) || (this->Max != Range.Max) || (this->Step != Range.Step);
		};

		bool operator==(const _type &Val) const
		{
			return this->iInRange(Val);
		};
		bool operator!=(const _type &Val) const
		{
			return !this->iInRange(Val);
		};

		bool operator<(const _type &Val) const
		{
			return Val < this->Min;
		};
		bool operator>(const _type &Val) const
		{
			return Val > this->Max;
		};

		bool iInRange(const _type &Val) const
		{
			return (Val >= Min) && (Val < Max); // [Min, Max), Min <= Max
		};

		int32 iCompare(const _type &Val) const
		{
			return this->iInRange(Val) ? 0 : (Val < Min ? -1 : 1);
		};

		void iSetMin(const _type &Min) { this->Min = Min; };
		void iSetMax(const _type &Max) { this->Max = Max; };
		void iSetStep(const _type &Step) { this->Step = Step; }; //Step == 0

		void iSetRange(const _type &Min, const _type &Max) //Opt call set ?
		{
			this->Min = Min;
			this->Max = Max;
		};
		void iSetRange(const _type &Min, const _type &Max, const _type &Step) //Opt call set ?
		{
			this->Min = Min;
			this->Max = Max;
			this->Step = Step;
		};

		const _type& iMin() const { return this->Min; }
		const _type& iMax() const { return this->Max; }
		const _type& iStep() const { return this->Step; }

		//STL
		//begin(), end(), cbegin(), cend(), size() ? ++

		uint32 size() const { return (this->Max - this->Min) / this->Step; /* Step == 0 */ }
		bool empty() const { return this->Min == this->Max; }
	};
#endif
}