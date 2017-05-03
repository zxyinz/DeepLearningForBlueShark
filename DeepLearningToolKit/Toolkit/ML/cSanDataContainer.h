#include"../Core/SanMemory.h"
#include"../Core/SanContainer.h"
#pragma once
using namespace std;
namespace San
{
	namespace MachineLearning
	{
#ifndef __CSANDATACONTAINER_H__
#define __CSANDATACONTAINER_H__
		template<class _data, class _label>
		class cSanDataContainer : public _smemobj<>
		{
		private:
			typedef list<shared_ptr<_data>> _data_list;
		private:
			unordered_map<_label, _data_list> m_Container;
			unordered_map<uint32, _data_list::iterator> m_LookupTable;
			unordered_map<uint32, _label> m_LabelTable;
		public:
			cSanDataContainer();
			cSanDataContainer(const cSanDataContainer<_data, _label> &Container);
			~cSanDataContainer();

			cSanDataContainer<_data, _label>& operator=(const cSanDataContainer<_data, _label> &Container);

			const shared_ptr<_data>& operator[](const uint32 ID) const;
			shared_ptr<_data> operator[](const uint32 ID) const;

			uint32 iInsert(const _data &Data, const _label &Label);
			template<class _it> vector<uint32> iInsert(const _it &Begin, const _it &End, const _label &Label);
			
			void iDelete(const uint32 ID);
			template<class _it> void iDelete(const _it &IDBegin, const _it &IDEnd);

			bool iFind(const uint32 ID);

			_label iGetLabel(const uint32 ID) const;
			uint32 iGetLabelID(const _label &Label) const;

			template<class _it>
			cSanDataContainer<_data, _label> iSubConainer(const _it &Begin, const _it &End) const;

			cSanDataContainer<_data, _label> iDeepCopy() const;

			uint32 iSize() const;
			uint32 iLabelSize() const;

			vector<_label> iGetLabelSet() const;
		};
#endif
	}
}