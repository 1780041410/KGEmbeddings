#pragma once

#include <string>
#include <map>
#include <utility>
#include <set>
#include <list>
#include <vector>
#include <functional>

#include "Triple.h"

namespace sysukg {

template<typename Any>
const Any * look_for(const Any * begin, const Any * end, const Any & target, 
        bool (*cmp)(const Any &, const Any &)) {
    const unsigned num = end - begin;
    unsigned left = 0, right = num, mid = num >> 1;
    while (left < right)
        if (cmp(*(begin + mid), target)) {
            left = mid + 1;
            mid = left + ((right - left) >> 1);
        }
        else if (cmp(target, *(begin + mid))) {
            right = mid;
            mid = left + ((right - left) >> 1);
        } else {
            return begin + mid;
        }
    return nullptr;
}
template<typename Any>
const Any * look_for(const Any * begin, const Any * end, const Any & target) {
    const unsigned num = end - begin;
    unsigned left = 0, right = num, mid = num >> 1;
    while (left < right)
        if (*(begin + mid) < target) {
            left = mid + 1;
            mid = left + ((right - left) >> 1);
        }
        else if (target < *(begin + mid)) {
            right = mid;
            mid = left + ((right - left) >> 1);
        } else {
            return begin + mid;
        }
    return nullptr;
}


inline bool operator<(const std::pair<unsigned, unsigned> & left,
                    const std::pair<unsigned, unsigned> & right) {
    if (left.first < right.first) return true;
    if (left.first > right.first) return false;
    return left.second < right.second;
}

class DataSet {
private:
    typedef std::map<std::string, unsigned> dictionary;

    const std::string _NAME;

    dictionary _entity2id, _relation2id;
    std::string * _id2entity, * _id2relation;

    std::set<unsigned> _outentity;


    Triple * _all, * _pos_hrt, * _ptu, *_pt, * _pos_rht, * _pos_trh;
    Triple * _trainset, * _updateset, * _testset, * _validset;
    unsigned _allsize, _trainsize, _updatesize, _testsize, _validsize, _possize, _ptusize, _ptsize;
    unsigned * _count_by_h, * _count_by_r, * _count_by_t;
    unsigned * _head_by_h, * _head_by_r, * _head_by_t;
    //
    unsigned _entityNum, _relationNum,_outentityNum;




protected:
    void readTriples(const std::string & filename, std::vector<Triple> & target,
            const std::function<void(const Triple &)> & func);

public:
    explicit DataSet(const std::string & name);

    inline const Triple * getIndex_r(unsigned r) const {
        return _pos_rht + _head_by_r[r];
    }

    inline unsigned entityNum() const {
        return _entityNum;
    }
    inline unsigned relationNum() const {
        return _relationNum;
    }
    inline unsigned trainSize() const {
        return _trainsize;
    }
    inline unsigned updateSize() const {
        return _updatesize;
    }
    inline unsigned testSize() const {
        return _testsize;
    }
    inline unsigned validSize() const {
        return _validsize;
    }
    inline unsigned posSize() const {
        return _possize;
    }
    inline unsigned ptuSize() const {
        return _ptusize;
    }
    inline unsigned ptSize() const {
        return _ptsize;
    }

    inline unsigned rcount(unsigned id) const {
        return _count_by_r[id];
    }
    inline unsigned hcount(unsigned id) const {
        return _count_by_h[id];
    }
    inline unsigned tcount(unsigned id) const {
        return _count_by_t[id];
    }

    inline const std::string & getEntityName(unsigned id) const {
        return _id2entity[id];
    }
    inline const std::string & getRelationName(unsigned id) const {
        return _id2relation[id];
    }

    inline const Triple * trainset() const {
        return _trainset;
    }
    inline const Triple * updateset() const {
        return _updateset;
    }
    inline const Triple * testset() const {
        return _testset;
    }
    inline const Triple * validset() const {
        return _validset;
    }
    inline const Triple * ptu() const {
        return _ptu;
    }
    inline const Triple * pt() const {
        return _pt;
    }
    inline const Triple * pos_hrt() const {
        return _pos_hrt;
    }

    inline const std::string & name() const {
        return _NAME;
    }

    inline const std::set<unsigned> & outentity() const{
        return _outentity;
    }

    ~DataSet();
};

}
