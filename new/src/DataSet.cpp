#include "DataSet.h"

#include <fstream>
#include <thread>
#include <mutex>
#include <cstring>
#include <algorithm>
#include <functional>

using namespace sysukg;

void DataSet::readTriples(const std::string & filename, std::vector<Triple> & target,
        const std::function<void(const Triple &)> & func) {
    std::string sth, str, stt;
    short stf;
    Triple temp;
    std::ifstream fin(("data/" + _NAME + "/" + filename).c_str());
    while (fin >> sth >> str >> stt >> stf) {
        temp.h = _entity2id[sth];
        temp.r = _relation2id[str];
        temp.t = _entity2id[stt];
        temp.f = (stf == 1);
        target.push_back(temp);
    
        if (temp.f)
            func(temp);
    }
    fin.close();
}

DataSet::DataSet(const std::string & name) : _NAME(name) {
    std::string str;
    unsigned id;
    std::ifstream fin(("data/" + name + "/" + "entity2id.txt").c_str());
    while (fin >> str >> id)
        _entity2id[str] = id;
    fin.close();
    fin.open(("data/" + name + "/" + "relation2id.txt").c_str());
    while (fin >> str >> id)
        _relation2id[str] = id;
    fin.close();

    if (name.length() <= 5) {
        fin.open(("data/" + name + "/" + "relationClass.txt").c_str());
        while(fin >> str >> id)
            _relationClass[_relation2id[str]] = id;
    }


    if (name.length() > 5) {
        fin.open(("data/" + name + "/" + "ookbEntity.txt").c_str());
        while(fin >> str)
            _outentity.insert(_entity2id[str]);
        fin.close();
    }

    _entityNum = _entity2id.size();
    _relationNum = _relation2id.size();
    _outentityNum = _outentity.size();

    _count_by_h = new unsigned[_entityNum];
    _count_by_r = new unsigned[_relationNum];
    _count_by_t = new unsigned[_entityNum];
    memset(_count_by_h, 0, _entityNum * sizeof(unsigned));
    memset(_count_by_r, 0, _relationNum * sizeof(unsigned));
    memset(_count_by_h, 0, _entityNum * sizeof(unsigned));

    _id2relation = new std::string[_relationNum];
    _id2entity = new std::string[_entityNum];
    for (auto & item : _relation2id)
        _id2relation[item.second] = item.first;
    for (auto & item : _entity2id)
        _id2entity[item.second] = item.first;

    std::vector<Triple> train, update, test, valid, pos, ptu, pt;
    std::thread ** threads = new std::thread *[4];
    std::mutex poslock, ptulock, ptlock;
    std::function<void(const Triple &)>
        func1 = [&pos, &ptu, &poslock, &ptulock](const Triple & t) -> void {
            poslock.lock();
            pos.push_back(t);
            poslock.unlock();
            ptulock.lock();
            ptu.push_back(t);
            ptulock.unlock();
        },
        func2 = [&pos, &poslock](const Triple & t) -> void {
            poslock.lock();
            pos.push_back(t);
            poslock.unlock();
        },
        func3 = [&pos, &ptu, &poslock, &ptulock, &pt, &ptlock](const Triple & t) -> void {
            poslock.lock();
            pos.push_back(t);
            poslock.unlock();
            ptulock.lock();
            ptu.push_back(t);
            ptulock.unlock();
            ptlock.lock();
            pt.push_back(t);
            ptlock.unlock();
        };
    threads[0] = new std::thread(&DataSet::readTriples, this, "train.txt", std::ref(train), std::ref(func3));
    threads[1] = new std::thread(&DataSet::readTriples, this, "update.txt", std::ref(update), std::ref(func1));
    threads[2] = new std::thread(&DataSet::readTriples, this, "test.txt", std::ref(test), std::ref(func2));
    threads[3] = new std::thread(&DataSet::readTriples, this, "valid.txt", std::ref(valid), std::ref(func2));
    for (unsigned short i = 0; i < 4; ++i)
        threads[i]->join();

    _trainsize = train.size();
    _updatesize = update.size();
    _testsize = test.size();
    _validsize = valid.size();
    _allsize = _trainsize + _updatesize + _testsize + _validsize;
    _possize = pos.size();
    _ptusize = ptu.size();
    _ptsize = pt.size();

    for (unsigned short i = 0; i < 4; ++i)
        delete threads[i];
    delete threads;

    _all = new Triple[_allsize];
    _trainset = _all;
    memcpy(_trainset, train.data(), _trainsize * sizeof(Triple));
    _updateset = _trainset + _trainsize;
    memcpy(_updateset, update.data(), _updatesize * sizeof(Triple));
    _testset = _updateset + _updatesize;
    memcpy(_testset, test.data(), _testsize * sizeof(Triple));
    _validset = _testset + _testsize;
    memcpy(_validset, valid.data(), _validsize * sizeof(Triple));

    _ptu = new Triple[_ptusize];
    memcpy(_ptu, ptu.data(), _ptusize * sizeof(Triple));
    _pt = new Triple[_ptsize];
    memcpy(_pt, pt.data(), _ptsize * sizeof(Triple));
    
    _pos_hrt = new Triple[_possize];
    memcpy(_pos_hrt, pos.data(), _possize * sizeof(Triple));
    std::sort(_pos_hrt, _pos_hrt + _possize, Triple_hrt_less);
    _pos_rht = new Triple[_possize];
    memcpy(_pos_rht, pos.data(), _possize * sizeof(Triple));
    std::sort(_pos_rht, _pos_rht + _possize, Triple_rht_less);
    _pos_trh = new Triple[_possize];
    memcpy(_pos_trh, pos.data(), _possize * sizeof(Triple));
    std::sort(_pos_trh, _pos_trh + _possize, Triple_trh_less);

    for (unsigned i = 0; i < _possize; ++i) {
        ++_count_by_h[_pos_hrt[i].h];
        ++_count_by_r[_pos_hrt[i].r];
        ++_count_by_t[_pos_hrt[i].t];
    }
    _head_by_h = new unsigned[1 + _entityNum];
    _head_by_r = new unsigned[1 + _relationNum];
    _head_by_t = new unsigned[1 + _entityNum];
    _head_by_h[0] = 0;
    _head_by_r[0] = 0;
    _head_by_t[0] = 0;
    for (unsigned i = 0; i < _entityNum; ++i) {
        _head_by_h[i + 1] = _head_by_h[i] + _count_by_h[i];
        _head_by_t[i + 1] = _head_by_t[i] + _count_by_t[i];
    }
    for (unsigned i = 0; i < _relationNum; ++i)
        _head_by_r[i + 1] = _head_by_r[i] + _count_by_r[i];
}

DataSet::~DataSet() {
    delete []_all;
    delete []_ptu;
    delete []_pt;
    delete []_pos_hrt;
    delete []_pos_rht;
    delete []_pos_trh;
    delete []_count_by_h;
    delete []_count_by_r;
    delete []_count_by_t;
    delete []_id2entity;
    delete []_id2relation;
    delete []_head_by_h;
    delete []_head_by_r;
    delete []_head_by_t;
}

void DataSet::wash() {
    std::set<Triple> negTriples;
    for (unsigned i = 0; i < _updatesize; ++i)
        if (!_updateset[i].f)
            negTriples.insert(_updateset[i].anti());
    wash(negTriples, _ptu, _ptusize);
    wash(negTriples, _pos_hrt, _possize);
}

void DataSet::wash(const std::set<Triple> negTriples, Triple * array, unsigned & size) {
    for (unsigned i = 0, next = 0; i < size; ++next)
        if (negTriples.find(array[next]) == negTriples.end()) {
            array[i++] = array[next];
            --size;
        }
}
