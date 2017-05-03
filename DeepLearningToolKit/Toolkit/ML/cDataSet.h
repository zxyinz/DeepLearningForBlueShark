#include"random"
#include"../Core/SanContainer.h"
#include"../Core/SanTypes.h"
#pragma once
using namespace std;
namespace San
{
	namespace MachineLearning
	{
#ifndef __CDATASET_H__
#define __CDATASET_H__
		template<class _data, class _label>
		struct _SINSTANCE : public _smemobj<>
		{
		public:
			typedef _data	_data_type;
			typedef _label	_label_type;
		public:
			_data	Data;
			_label	Label;
		public:
			_SINSTANCE(const _data &Data, const _label &Label) :Data(Data), Label(Label){};
			_SINSTANCE(const _spair<_data, _label> &Instance) :Data(Instance.first), Label(Instance.second){};
			_SINSTANCE(const _SINSTANCE<_data, _label> &Instance) :Data(Instance.Data), Label(Instance.Label){};
			~_SINSTANCE(){};

			_SINSTANCE<_data, _label>& operator=(const _SINSTANCE<_data, _label> &Instance)
			{
				this->Data = Instance.Data;
				this->Label = Instance.Label;

				return *this;
			};

			bool operator==(const _SINSTANCE<_data, _label> &Instance) const{ return (this->Label == Instance.Label) && (this->Data == Instance.Data); };

			bool operator<(const _SINSTANCE<_data, _label> &Instance) const { return this->Label < Instance.Label; };
			bool operator>(const _SINSTANCE<_data, _label> &Instance) const { return this->Label > Instance.Label; };
		};

		template<class _data, class _label>
		using _sinstance = _SINSTANCE<_data, _label>;

		template<class _instance, class _iterator>
		struct _CLASS_SET
		{
		public:
			typedef _INDIRECT_CONST_ITERATOR<_instance, typename vector<_iterator>::const_iterator>	const_iterator; // iterator of not general const_iterator
			typedef _INDIRECT_ITERATOR<_instance, typename vector<_iterator>::iterator>				iterator;
		private:
			typename _instance::_label_type Label;
			vector<_iterator> IteratorArray;
		public:
			_CLASS_SET();
			_CLASS_SET(const typename _instance::_label_type &Label);
			//_CLASS_SET(const _CLASS_SET<_instance, _iterator> &ItSet) = delete;
			~_CLASS_SET();

			_CLASS_SET<_instance, _iterator>& operator=(const _CLASS_SET<_instance, _iterator> &ItSet) = delete;

			const _instance& operator[](const uint32 Index) const;
			_instance& operator[](const uint32 Index);

			const typename _instance::_label_type& iGetLabel() const;
			typename _instance::_label_type iGetLabel();

			//STL style function
			const_iterator cbegin() const;
			const_iterator cend() const;

			const_iterator begin() const;
			iterator begin();

			const_iterator end() const;
			iterator end();

			iterator push_back(const _iterator &It);
			iterator erase_by_value(const _iterator &It);

			const_iterator erase(const const_iterator &CIt);
			iterator erase(const iterator &It);

			const _iterator& get(const uint32 Index) const;
			_iterator& get(const uint32 Index);

			void set(const uint32 Index, const _iterator &It);

			void resize(const uint32 Size);

			const vector<_iterator> get_container() const;

			uint32 size() const;
			bool empty() const;
		};

#pragma region _CLASS_SET CPP
#define _M_TYPE typename _CLASS_SET<_instance, _iterator>
		template<class _instance, class _iterator>
		_CLASS_SET<_instance, _iterator>::_CLASS_SET()
		{
		}
		template<class _instance, class _iterator>
		_CLASS_SET<_instance, _iterator>::_CLASS_SET(const typename _instance::_label_type &Label)
			:Label(Label)
		{
		}
		template<class _instance, class _iterator>
		_CLASS_SET<_instance, _iterator>::~_CLASS_SET()
		{
		}
		template<class _instance, class _iterator>
		const _instance& _CLASS_SET<_instance, _iterator>::operator[](const uint32 Index) const
		{
			return *(this->IteratorArray[Index]);
		}
		template<class _instance, class _iterator>
		_instance& _CLASS_SET<_instance, _iterator>::operator[](const uint32 Index) 
		{
			return *(this->IteratorArray[Index]);
		}
		template<class _instance, class _iterator>
		const typename _instance::_label_type& _CLASS_SET<_instance, _iterator>::iGetLabel() const
		{
			return this->Label;
		}
		template<class _instance, class _iterator>
		typename _instance::_label_type _CLASS_SET<_instance, _iterator>::iGetLabel()
		{
			return this->Label;
		}
		template<class _instance, class _iterator>
		_M_TYPE::const_iterator _CLASS_SET<_instance, _iterator>::cbegin() const
		{
			return const_iterator(this->IteratorArray.begin());
		}
		template<class _instance, class _iterator>
		_M_TYPE::const_iterator _CLASS_SET<_instance, _iterator>::cend() const
		{
			return const_iterator(this->IteratorArray.cend());
		}
		template<class _instance, class _iterator>
		_M_TYPE::const_iterator _CLASS_SET<_instance, _iterator>::begin() const
		{
			return const_iterator(this->IteratorArray.begin());
		}
		template<class _instance, class _iterator>
		_M_TYPE::iterator _CLASS_SET<_instance, _iterator>::begin()
		{
			return iterator(this->IteratorArray.begin());
		}
		template<class _instance, class _iterator>
		_M_TYPE::const_iterator _CLASS_SET<_instance, _iterator>::end() const
		{
			return const_iterator(this->IteratorArray.end());
		}
		template<class _instance, class _iterator>
		_M_TYPE::iterator _CLASS_SET<_instance, _iterator>::end()
		{
			return iterator(this->IteratorArray.end());
		}
		template<class _instance, class _iterator>
		_M_TYPE::iterator _CLASS_SET<_instance, _iterator>::push_back(const _iterator &It)
		{
			this->IteratorArray.push_back(It);
			return iterator(prev(this->IteratorArray.end()));
		}
		template<class _instance, class _iterator>
		_M_TYPE::iterator _CLASS_SET<_instance, _iterator>::erase_by_value(const _iterator &It)
		{
			_M_TYPE::iterator pIt = this->IteratorArray.begin();

			while (pIt != this->IteratorArray.end())
			{
				if ((*pIt) == It) { return iterator(this->IteratorArray.erase(pIt)); }
				pIt++;
			}

			return iterator(this->IteratorArray.end());
		}
		template<class _instance, class _iterator>
		_M_TYPE::const_iterator _CLASS_SET<_instance, _iterator>::erase(const _M_TYPE::const_iterator &CIt)
		{
			return const_iterator(this->IteratorArray.erase(CIt.raw_iterator()));
		}
		template<class _instance, class _iterator>
		_M_TYPE::iterator _CLASS_SET<_instance, _iterator>::erase(const _M_TYPE::iterator &It)
		{
			return iterator(this->IteratorArray.erase(It.raw_iterator()));
		}
		template<class _instance, class _iterator>
		const _iterator& _CLASS_SET<_instance, _iterator>::get(const uint32 Index) const
		{
			return this->IteratorArray[Index];
		}
		template<class _instance, class _iterator>
		_iterator& _CLASS_SET<_instance, _iterator>::get(const uint32 Index)
		{
			return this->IteratorArray[Index];
		}
		template<class _instance, class _iterator>
		void _CLASS_SET<_instance, _iterator>::set(const uint32 Index, const _iterator &It)
		{
			if (Index >= this->size()) { return; }
			this->IteratorArray[Index] = It;
		}
		template<class _instance, class _iterator>
		void _CLASS_SET<_instance, _iterator>::resize(const uint32 Size)
		{
			this->IteratorArray.resize(Size);
		}
		template<class _instance, class _iterator>
		const vector<_iterator> _CLASS_SET<_instance, _iterator>::get_container() const
		{
			return this->IteratorArray;
		}
		template<class _instance, class _iterator>
		uint32 _CLASS_SET<_instance, _iterator>::size() const
		{
			return this->IteratorArray.size();
		}
		template<class _instance, class _iterator>
		bool _CLASS_SET<_instance, _iterator>::empty() const
		{
			return this->IteratorArray.empty();
		}
#undef _M_TYPE
#pragma endregion

		template<class _data, class _label>
		class _DATASET : public _smemobj<>
		{
		public:
			typedef _data	_data_type;
			typedef _label	_label_type;

			typedef _DATASET<_data, _label>		_type;
			typedef _sinstance<_data, _label>	_instance;

			typedef typename list<_instance>::const_iterator	const_iterator;
			typedef typename list<_instance>::iterator			iterator;

			typedef _CLASS_SET<_instance, iterator>	_class;

			typedef vector<_spair<uint32, iterator>> _random_access_container;

			//typedef _INDIRECT_CONST_ITERATOR<_data, typename _random_access_container::const_iterator>	random_const_iterator;
			//typedef _INDIRECT_ITERATOR<_data, typename _random_access_container::iterator>				random_iterator;
		protected:
			list<_instance>				InstanceContainer;
			_random_access_container	RandomAccessArray;
		public:
			vector<_class> ClassSet;
		protected:
			static void _DeepCopy(_type &DestDataSet, const _type &SrcDataSet);
		public:
			_DATASET();
			_DATASET(const _type &DataSet);

			//_container: vector, list, map, etc
			template<template<class> class _container> _DATASET(const _container<_instance> &ItemSet);

			//_container: vector, list, map, etc
			template<template<class> class _container> _DATASET(const _container<_data> &DataSet, const _container<_label> &LabelSet);

			//_iterator: _container<_instance>::_iterator
			//_container: vector, list, map, etc
			template<class _iterator> _DATASET(const _iterator &ItemItBegin, const _iterator &ItemItEnd);

			//_iterator: _container<_data>::_iterator
			//_container: vector, list, map, etc
			template<class _iterator> _DATASET(const _iterator &DataItBegin, const _iterator &DataItEnd, const _label &Label);

			~_DATASET();

			_type& operator=(const _type &DataSet);

			//Random access
			const _instance& operator[](const uint32 Index) const;
			//Random access
			_instance& operator[](const uint32 Index);

			/*Push, O(n)*/
			iterator iPush(const _instance &Item);
			vector<iterator> iPush(const _type &DataSet) ;

			//_container: vector, list, map, etc
			template<class _container> vector<iterator> iPush(const _container &ItemSet);
			
			//_container: vector, list, map, etc
			template<template<class> class _container> vector<iterator> iPush(const _container<_data> &DataSet, const _container<_label> &LabelSet);

			//_container: vector, list, map, etc
			template<class _container> vector<iterator> iPush(const _container &DataSet, const _label &Label);

			//_iterator: _container<_instance>::_iterator
			//_container: vector, list, map, etc
			template<class _iterator> vector<iterator> iPush(const _iterator &ItemItBegin, const _iterator &ItemItEnd);
			
			//_iterator: _container<_data>::_iterator
			//_container: vector, list, map, etc
			template<class _iterator> vector<iterator> iPush(const _iterator &DataItBegin, const _iterator &DataItEnd, const _label &Label);//////////////////////////

			/*Delete, O(n)*/
			//_iterator: const_iterator or iterator
			template<class _iterator> void iDelete(const _iterator &Iterator);

			//_container: vector, list, map, etc
			//_iterator: const_iterator or iterator
			template<template<class> class _container, class _iterator> void iDelete(const _container<_iterator> &ItSet);
			
			//_iterator: _container<_element>::_iterator
			//_container: vector, list, map, etc
			//_element: const_iterator or iterator
			template<class _iterator> void iDelete(const _iterator &ItBegin, const _iterator &ItEnd);

			/*SubDataset*/
			//_container: vector, list, map, etc
			//_iterator: const_iterator or iterator
			template<class _container> uint32 iSubDataSet(const _container &ItSet, _type &DataSet) const;
			
			//_iterator: _container<_element>::_iterator
			//_container: vector, list, map, etc
			//_element: const_iterator or iterator
			template<class _iterator> uint32 iSubDataSet(const _iterator &ItBegin, const _iterator &ItEnd, _type &DataSet) const;

			/*O(nlogn)*/
			void iSortByLabel();//unfinished

			void iShuffle(const uint32 RandomSeed);
			template<class _random_engine = std::default_random_engine> void iShuffle(_random_engine &RandomEngine);

			/*O(1)*/
			void iClear();
			bool iEmpty() const;

			uint32 iSize() const;
			uint32 iClassSize() const;

			//STL style function
			const_iterator cbegin() const;
			const_iterator cend() const;

			const_iterator begin() const;
			iterator begin();

			const_iterator end() const;
			iterator end();

			//Random access container
			const_iterator find(const uint32 Index) const;

			//Random access container
			iterator find(const uint32 Index);

			uint32 size() const;

			void clear();
			bool empty() const;

			//Stream operator and + override 
		};

		template<class _data, class _label>
		using _dataset = _DATASET<_data, _label>;

#pragma region _DATASET CPP
#define _M_TYPE typename _DATASET<_data, _label>
		template<class _data, class _label>
		_DATASET<_data, _label>::_DATASET()
		{
		}
		template<class _data, class _label>
		_DATASET<_data, _label>::_DATASET(const _M_TYPE::_type &DataSet)
			:InstanceContainer(DataSet.InstanceContainer)
		{
			this->_DeepCopy(*this, DataSet);
		}
		template<class _data, class _label>
		template<template<class> class _container>
		_DATASET<_data, _label>::_DATASET(const _container<_instance> &ItemSet)
		{
			this->iPush(ItemSet);
		}
		template<class _data, class _label>
		template<template<class> class _container>
		_DATASET<_data, _label>::_DATASET(const _container<_data> &DataSet, const _container<_label> &LabelSet)
		{
			this->iPush(DataSet, LabelSet);
		}
		template<class _data, class _label>
		template<class _iterator>
		_DATASET<_data, _label>::_DATASET(const _iterator &ItemItBegin, const _iterator &ItemItEnd)
		{
			this->iPush(ItemItBegin, ItemItEnd);
		}
		template<class _data, class _label>
		template<class _iterator>
		_DATASET<_data, _label>::_DATASET(const _iterator &DataItBegin, const _iterator &DataItEnd, const _label &Label)
		{
			this->iPush(DataItBegin, DataItEnd, Label);
		}
		template<class _data, class _label>
		_DATASET<_data, _label>::~_DATASET()
		{
			this->iClear();
		}
		template<class _data, class _label>
		void _DATASET<_data, _label>::_DeepCopy(_type &DestDataSet, const _type &SrcDataSet)
		{
			DestDataSet.RandomAccessArray.resize(DestDataSet.InstanceContainer.size());

			const int32 ClassSize = SrcDataSet.ClassSet.size();
			vector<int32> CounterArray(ClassSize, 0);

			/*Copy class set*/
			DestDataSet.ClassSet.clear();
			for (int32 seek = 0; seek < ClassSize; seek = seek + 1)
			{
				DestDataSet.ClassSet.push_back(SrcDataSet.ClassSet[seek].iGetLabel());
				DestDataSet.ClassSet[seek].resize(SrcDataSet.ClassSet[seek].size());
			}

			_M_TYPE::iterator It = DestDataSet.InstanceContainer.begin();

			while (It != DestDataSet.InstanceContainer.end())
			{
				for (int32 seek_class = 0; seek_class < ClassSize; seek_class = seek_class + 1)
				{
					if (SrcDataSet.ClassSet[seek_class].size() == 0){ continue; }

					if (It->Label == SrcDataSet.ClassSet[seek_class].iGetLabel())
					{
						DestDataSet.ClassSet[seek_class].set(CounterArray[seek_class], It);
						CounterArray[seek_class] = CounterArray[seek_class] + 1;
						break;
					}
				}

				It++;
			}

			/*Copy random access array*/
			const uint32 InstanceSize = DestDataSet.InstanceContainer.size();

			vector<_M_TYPE::iterator> IteratorArray(InstanceSize);

			It = DestDataSet.InstanceContainer.begin();

			for (int32 seek = 0; seek < InstanceSize; seek = seek + 1, It++)
			{
				IteratorArray[seek] = It;
			}

			DestDataSet.RandomAccessArray = SrcDataSet.RandomAccessArray;
			for (auto &seek : DestDataSet.RandomAccessArray)
			{
				seek.second = IteratorArray[seek.first];
			}
		}
		template<class _data, class _label>
		_M_TYPE::_type& _DATASET<_data, _label>::operator=(const _M_TYPE::_type &DataSet)
		{
			this->InstanceContainer = DataSet.InstanceContainer;
			this->_DeepCopy(*this, DataSet);

			return *this;
		}
		template<class _data, class _label>
		const _M_TYPE::_instance& _DATASET<_data, _label>::operator[](const uint32 Index) const
		{
			return *(this->RandomAccessArray[Index].second);
		}
		template<class _data, class _label>
		_M_TYPE::_instance& _DATASET<_data, _label>::operator[](const uint32 Index)
		{
			return *(this->RandomAccessArray[Index].second);
		}
		template<class _data, class _label>
		_M_TYPE::iterator _DATASET<_data, _label>::iPush(const _M_TYPE::_instance &Item)
		{
			this->InstanceContainer.push_back(Item);

			_M_TYPE::iterator It = std::prev(this->InstanceContainer.end());

			/*Push instance to random access array*/
			this->RandomAccessArray.push_back(_spair<uint32, iterator>(this->InstanceContainer.size() - 1, It));

			/*Push instance to class set*/
			bool bFound = false;
			for (int32 seek_class = 0; seek_class < this->ClassSet.size(); seek_class = seek_class + 1)
			{
				if (Item.Label == this->ClassSet[seek_class].iGetLabel())
				{
					this->ClassSet[seek_class].push_back(It);
					bFound = true;
					break;
				}
			}

			if (!bFound)
			{
				this->ClassSet.push_back(_class(It->Label));
				this->ClassSet.back().push_back(It);
			}

			return It;
		}
		template<class _data, class _label>
		vector<_M_TYPE::iterator> _DATASET<_data, _label>::iPush(const _M_TYPE::_type &DataSet)
		{
			return this->iPush(DataSet.begin(), DataSet.end());
		}
		template<class _data, class _label>
		template<class _container>
		vector<_M_TYPE::iterator> _DATASET<_data, _label>::iPush(const _container &ItemSet)
		{
			return this->iPush(ItemSet.begin(), ItemSet.end());
		}
		template<class _data, class _label>
		template<class _container>
		vector<_M_TYPE::iterator> _DATASET<_data, _label>::iPush(const _container &DataSet, const _label &Label)
		{ 
			return this->iPush(DataSet.begin(), DataSet.end(), Label);
		}
		template<class _data, class _label>
		template<class _iterator>
		vector<_M_TYPE::iterator> _DATASET<_data, _label>::iPush(const _iterator &ItemItBegin, const _iterator &ItemItEnd)
		{
			uint32 Size = 0;
			for (auto pIt = ItemItBegin; pIt != ItemItEnd; pIt++) { Size = Size + 1; }

			vector<_M_TYPE::iterator> IteratorSet(Size);

			uint32 seek = 0;

			for (auto pIt = ItemItBegin; pIt != ItemItEnd; pIt++)
			{
				IteratorSet[seek] = this->iPush(*pIt);
				seek = seek + 1;
			}

			return IteratorSet;
		}
		template<class _data, class _label>
		template<class _iterator>
		vector<_M_TYPE::iterator> _DATASET<_data, _label>::iPush(const _iterator &DataItBegin, const _iterator &DataItEnd, const _label &Label)//////////////////////////
		{
			const uint32 IDBegin = this->InstanceContainer.size();
			_M_TYPE::iterator It;

			uint32 Count = 0;
			if (IDBegin == 0)
			{
				for (auto seek_data_it = DataItBegin; seek_data_it != DataItEnd; seek_data_it++, Count = Count + 1)
				{
					_M_TYPE::_instance Instance(*seek_data_it, Label);
					this->InstanceContainer.push_back(Instance);
				}

				It = this->InstanceContainer.begin();
			}
			else
			{
				It = std::prev(this->InstanceContainer.end());

				for (auto seek_data_it = DataItBegin; seek_data_it != DataItEnd; seek_data_it++, Count = Count + 1)
				{
					_M_TYPE::_instance Instance(*seek_data_it, Label);
					this->InstanceContainer.push_back(Instance);
				}

				It++;
			}

			/*Find or create class set index*/
			uint32 ClassIndex = 0;

			/*Push instance to class set*/
			bool bFound = false;
			for (int32 seek_class = 0; seek_class < this->ClassSet.size(); seek_class = seek_class + 1)
			{
				if (Label == this->ClassSet[seek_class].iGetLabel())
				{
					ClassIndex = seek_class;
					bFound = true;
					break;
				}
			}

			if (!bFound){ this->ClassSet.push_back(_class(Label)); ClassIndex = this->ClassSet.size() - 1; }

			vector<_M_TYPE::iterator> IteratorSet(Count);
			this->RandomAccessArray.resize(this->RandomAccessArray.size() + Count);

			for (int32 seek = 0; seek < Count; seek = seek + 1)
			{
				/*Push instance to random access array*/
				const uint32 Index = IDBegin + seek;

				this->RandomAccessArray[Index] = _spair<uint32, iterator>(Index, It);
				this->ClassSet[ClassIndex].push_back(It);
				IteratorSet[seek] = It;

				It++;
			}

			return IteratorSet;
		}
		template<class _data, class _label>
		template<class _iterator>
		void _DATASET<_data, _label>::iDelete(const _iterator &Iterator)
		{

		}
		template<class _data, class _label>
		template<template<class> class _container, class _iterator>
		void _DATASET<_data, _label>::iDelete(const _container<_iterator> &ItSet)
		{
			this->iDelete(ItSet.begin(), ItSet.end());
		}
		template<class _data, class _label>
		template<class _iterator>
		void _DATASET<_data, _label>::iDelete(const _iterator &ItBegin, const _iterator &ItEnd)
		{
			for (const auto It = ItBegin; It != ItEnd; It++)
			{
				this->iDelete(It);
			}
		}
		template<class _data, class _label>
		template<class _container>
		uint32 _DATASET<_data, _label>::iSubDataSet(const _container &ItSet, _M_TYPE::_type &DataSet) const
		{
			return this->iSubDataSet(ItSet.begin(), ItSet.end(), DataSet);
		}
		template<class _data, class _label>
		template<class _iterator>
		uint32 _DATASET<_data, _label>::iSubDataSet(const _iterator &ItBegin, const _iterator &ItEnd, _M_TYPE::_type &DataSet) const
		{
			uint32 Size = DataSet.size();
			uint32 Count = 0;

			_iterator seek_item = ItBegin;
			for (_iterator seek_item = ItBegin; seek_item != ItEnd; seek_item++)
			{ 
				Count = Count + 1; 
			}

			DataSet.RandomAccessArray.resize(Size + Count);

			uint32 Index = Size;
			for (auto seek_item = ItBegin; seek_item != ItEnd; seek_item++)
			{
				DataSet.InstanceContainer.push_back(**seek_item);

				iterator It = std::prev(DataSet.InstanceContainer.end());

				/*Push instance to random access array*/
				DataSet.RandomAccessArray[Index] = _spair<uint32, iterator>(Index, It);

				/*Push instance to class set*/
				bool bFound = false;
				for (int32 seek_class = 0; seek_class < DataSet.ClassSet.size(); seek_class = seek_class + 1)
				{
					if (It->Label == DataSet.ClassSet[seek_class].iGetLabel())
					{
						DataSet.ClassSet[seek_class].push_back(It);
						bFound = true;
						break;
					}
				}

				if (!bFound)
				{
					DataSet.ClassSet.push_back(_class(It->Label));
					DataSet.ClassSet.back().push_back(It);
				}

				Index = Index + 1;
			}

			return Count;
		}
		template<class _data, class _label>
		void _DATASET<_data, _label>::iSortByLabel()
		{

		}
		template<class _data, class _label>
		void _DATASET<_data, _label>::iShuffle(const uint32 RandomSeed)
		{
			default_random_engine REngine(RandomSeed);
			this->iShuffle(REngine);
		}
		template<class _data, class _label>
		template<class _random_engine>
		void _DATASET<_data, _label>::iShuffle(_random_engine &RandomEngine)
		{
			const int32 Size = this->InstanceContainer.size();
			std::uniform_real_distribution<> UniformDist(0.0, 1.0);

			vector<_spair<sfloat, _spair<uint32, _M_TYPE::iterator>>> RandomMap(Size);
			_M_TYPE::iterator It = this->InstanceContainer.begin();

			for (int32 seek = 0; seek < Size; seek = seek + 1, It++)
			{
				RandomMap[seek].first = ::rand() % 10000;//UniformDist(RandomEngine);
				RandomMap[seek].second = _spair<uint32, iterator>(seek, It);
			}

			::sort(RandomMap.begin(), RandomMap.end(), [](const _spair<sfloat, _spair<uint32, iterator>> &P1, const _spair<sfloat, _spair<uint32, iterator>> &P2){ return P1.first < P2.first; });

			for (int32 seek = 0; seek < Size; seek = seek + 1)
			{
				this->RandomAccessArray[seek] = RandomMap[seek].second;
			}
		}
		template<class _data, class _label>
		void _DATASET<_data, _label>::iClear()
		{
			this->ClassSet.clear();
			this->RandomAccessArray.clear();
			this->InstanceContainer.clear();
		}
		template<class _data, class _label>
		bool _DATASET<_data, _label>::iEmpty() const
		{
			return this->InstanceContainer.empty();
		}
		template<class _data, class _label>
		uint32 _DATASET<_data, _label>::iSize() const
		{ 
			return this->InstanceContainer.size(); 
		}
		template<class _data, class _label>
		uint32 _DATASET<_data, _label>::iClassSize() const
		{ 
			return this->ClassSet.size();
		}
		template<class _data, class _label>
		_M_TYPE::const_iterator _DATASET<_data, _label>::cbegin() const
		{ 
			return this->InstanceContainer.cbegin(); 
		}
		template<class _data, class _label>
		_M_TYPE::const_iterator _DATASET<_data, _label>::cend() const
		{ 
			return this->InstanceContainer.cend(); 
		};
		template<class _data, class _label>
		_M_TYPE::const_iterator _DATASET<_data, _label>::begin() const
		{
			return this->InstanceContainer.begin();
		}
		template<class _data, class _label>
		_M_TYPE::iterator _DATASET<_data, _label>::begin()
		{
			return this->InstanceContainer.begin();
		}
		template<class _data, class _label>
		_M_TYPE::const_iterator _DATASET<_data, _label>::end() const
		{
			return this->InstanceContainer.end();
		}
		template<class _data, class _label>
		_M_TYPE::iterator _DATASET<_data, _label>::end()
		{
			return this->InstanceContainer.end();
		}
		template<class _data, class _label>
		_M_TYPE::const_iterator _DATASET<_data, _label>::find(const uint32 Index) const
		{
			return this->RandomAccessArray[Index].second;
		}
		template<class _data, class _label>
		_M_TYPE::iterator _DATASET<_data, _label>::find(const uint32 Index)
		{
			return this->RandomAccessArray[Index].second;
		}
		template<class _data, class _label>
		uint32 _DATASET<_data, _label>::size() const
		{
			return this->iSize();
		}
		template<class _data, class _label>
		void _DATASET<_data, _label>::clear()
		{ 
			this->iClear(); 
		}
		template<class _data, class _label>
		bool _DATASET<_data, _label>::empty() const
		{
			return this->iEmpty();
		}
#undef _M_TYPE
#pragma endregion

#endif
	}
}