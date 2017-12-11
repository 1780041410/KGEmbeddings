#include "EmbeddingModel.h"

#include <string>
#include <fstream>
#include <cstring>
#include <thread>
#include <algorithm>
#include <functional>
#include <iostream>
#include <atomic>
#include <mutex>
#include <vector>
#include <queue>

using namespace sysukg;

random_device & EmbeddingModel::_rd = random_device::getInstance();

EmbeddingModel::EmbeddingModel(const DataSet & ds, unsigned dim,
        const std::string & ext, bool use_cache) : _use_cache(use_cache),
    _ds(ds), _dim(dim), _relSize(ds.relationNum()), _entSize(ds.entityNum()) {
    _ed = new EmbeddedData;
    _ed->first = new float*[_relSize];
    for (unsigned i = 0; i < _relSize; ++i)
        _ed->first[i] = new float[dim];
    _ed->second = new float*[_entSize];
    for (unsigned i = 0; i < _entSize; ++i)
        _ed->second[i] = new float[dim];
    if (_use_cache) {
        _ed_cache = new EmbeddedData;
        _ed_cache->first = new float*[_relSize];
        for (unsigned i = 0; i < _relSize; ++i)
            _ed_cache->first[i] = new float[dim];
        _ed_cache->second = new float*[_entSize];
        for (unsigned i = 0; i < _entSize; ++i)
            _ed_cache->second[i] = new float[dim];
    }

    if (ext.empty()) {
        EDreset();
    } else {
        readFromFile(_ed->first, "relation2vec." + ext, _relSize, _dim);
        readFromFile(_ed->second, "entity2vec." + ext, _entSize, _dim);
    }
}

void EmbeddingModel::readFromFile(float ** target, const std::string & filename,
        unsigned n, unsigned m) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << filename << " not found!" << std::endl;
        exit(1);
    }
    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = 0; j < m; ++j)
            fin >> target[i][j];
    fin.close();
}


void EmbeddingModel::single_output(const std::string & filename,
        unsigned num, unsigned dim, float ** mat) {
    std::ofstream fout(filename);
    for (unsigned i = 0; i < num; ++i) {
        fout << mat[i][0];
        for (unsigned j = 1; j < dim; ++j)
            fout << ' ' << mat[i][j];
        fout << std::endl;
    }
    fout.close();
}

void EmbeddingModel::output(const std::string & ext) const {
    static const unsigned short filenum = 2;
    std::thread * threads[filenum] = {
        new std::thread{single_output, std::move("relation2vec." + ext),
            std::ref(_relSize), std::ref(_dim),std::ref(_ed->first)},
        new std::thread{single_output, std::move("entity2vec." + ext),
            std::ref(_entSize), std::ref(_dim), std::ref(_ed->second)}
    };
    for (char i = 0; i < filenum; ++i)
        threads[i]->join();
    for (char i = 0; i < filenum; ++i)
        delete threads[i];
}

void EmbeddingModel::runLinkPredictionTest(std::ostream & os, unsigned threadnum, bool np) const {
    static const unsigned NT_LIMIT = 10;
    typedef std::pair<Triple, float> T_with_Score;
    struct comp {
        bool operator()(const T_with_Score & left, const T_with_Score & right) const {
            return left.second > right.second;
        }
    };

    std::atomic<unsigned> hrsum, trsum, hrhit, trhit,
                          hfsum, tfsum, hfhit, tfhit;
    std::atomic<unsigned> testnum;
    testnum.store(0);
    hrsum.store(0); trsum.store(0); hrhit.store(0); trhit.store(0);
    hfsum.store(0); tfsum.store(0); hfhit.store(0); tfhit.store(0);
    const Triple * ds_pos_hrt_end = _ds.pos_hrt() + _ds.posSize();
    std::mutex os_lock;
    std::function<void(unsigned, unsigned)> func =
        [&](unsigned s, unsigned t) -> void {
            float baseScore, tempScore;
            unsigned hraw, hfiltered, traw, tfiltered;
            std::priority_queue<T_with_Score, std::vector<T_with_Score>, comp> neg_triples;
            Triple hr, tr;
            for (unsigned i = s; i < t; ++i)
                if (_ds.testset()[i].f) {
                    ++testnum;
                    baseScore = calc_sum(_ds.testset()[i]);
                    hraw = traw = 1;
                    hfiltered = tfiltered = 0;
                    hr = tr = _ds.testset()[i];
                    for (unsigned j = 0; j < _entSize; ++j) {
                        hr.h = tr.t = j;
                        tempScore = calc_sum(hr);
                        if (tempScore < baseScore)
                            if (hr.h != _ds.testset()[i].h) {
                                ++hraw;
                                if (look_for(_ds.pos_hrt(), ds_pos_hrt_end, hr, Triple_hrt_less) != nullptr)
                                    ++hfiltered;
                                else
                                    if (np) {
                                        neg_triples.push(T_with_Score(hr, tempScore));
                                        if (neg_triples.size() > NT_LIMIT)
                                            neg_triples.pop();
                                    }
                            }
                        tempScore = calc_sum(tr);
                        if (tempScore < baseScore)
                            if (tr.t != _ds.testset()[i].t) {
                                ++traw;
                                if (look_for(_ds.pos_hrt(), ds_pos_hrt_end, tr, Triple_hrt_less) != nullptr)
                                    ++tfiltered;
                                else
                                    if (np) {
                                        neg_triples.push(T_with_Score(tr, tempScore));
                                        if (neg_triples.size() > NT_LIMIT)
                                            neg_triples.pop();
                                    }
                            }
                    }
                    while (!neg_triples.empty()) {
                        os_lock.lock();
                        os << _ds.getEntityName(neg_triples.top().first.h) << '\t'
                           << _ds.getRelationName(neg_triples.top().first.r) << '\t'
                           << _ds.getEntityName(neg_triples.top().first.t) << '\t'
                           << "-1" << std::endl;
                        os_lock.unlock();
                        neg_triples.pop();
                    }
                    hrsum += hraw;
                    hfiltered = hraw - hfiltered;
                    hfsum += hfiltered;
                    if (hraw <= 10) ++hrhit;
                    if (hfiltered <= 10) ++hfhit;
                    trsum += traw;
                    tfiltered = traw - tfiltered;
                    tfsum += tfiltered;
                    if (traw <= 10) ++trhit;
                    if (tfiltered <= 10) ++tfhit;
                }
        };
    float psize = static_cast<float>(_ds.testSize()) / threadnum;
    std::thread * threads[threadnum];
    float sum = 0;
    unsigned last = 0, temp;
    for (unsigned i = 0; i < threadnum - 1; ++i) {
        sum += psize;
        temp = static_cast<unsigned>(sum + 0.5);
        threads[i] = new std::thread(func, last, temp);
        last = temp;
    }
    threads[threadnum - 1] = new std::thread(func, last, _ds.testSize());
    for (unsigned i = 0; i < threadnum; ++i)
        threads[i]->join();
    float ftestnum = static_cast<float>(testnum);
    if (!np)
        os << hrsum/ftestnum << ' ' << hrhit/ftestnum << ' ' << hfsum/ftestnum << ' ' << hfhit/ftestnum << ' '
           << trsum/ftestnum << ' ' << trhit/ftestnum << ' ' << tfsum/ftestnum << ' ' << tfhit/ftestnum << std::endl;
    for (unsigned i = 0; i < threadnum; ++i)
        delete threads[i];
}

void EmbeddingModel::runClassificationTest(std::ostream & os) const {
    TripleWithScore * valid = new TripleWithScore[_ds.validSize()],
                    * test = new TripleWithScore[_ds.testSize()];
    for (unsigned i = 0; i < _ds.validSize(); ++i) {
        valid[i].t = _ds.validset() + i;
        valid[i].score = calc_sum(*valid[i].t);
    }
    for (unsigned i = 0; i < _ds.testSize(); ++i) {
        test[i].t = _ds.testset() + i;
        test[i].score = calc_sum(*test[i].t);
    }
    std::sort(valid, valid + _ds.validSize());
    float bestThres = 0;
    unsigned bestCorrect = 0, leftpos = 0;
    for (unsigned i = 1; i < _ds.validSize(); ++i) {
        if (valid[i - 1].t->f)
            ++leftpos;
        if (leftpos + (_ds.validSize() >> 1) - (i - leftpos) > bestCorrect) {
            bestCorrect = leftpos + (_ds.validSize() >> 1) - (i - leftpos);
            bestThres = (valid[i].score + valid[i - 1].score) / 2.0;
        }
    }

    unsigned correctnum = 0;
    for (unsigned i = 0; i < _ds.testSize(); ++i)
        if (test[i].t->f ^ test[i].score > bestThres)
            ++correctnum;
    os << static_cast<float>(correctnum)/_ds.testSize() << std::endl;

    delete[] test;
    delete[] valid;
}

EmbeddingModel::~EmbeddingModel() {
    for (unsigned i = 0; i < _relSize; ++i)
        delete[] _ed->first[i];
    delete[] _ed->first;
    for (unsigned i = 0; i < _entSize; ++i)
        delete[] _ed->second[i];
    delete[] _ed->second;
    delete _ed;
    if (_use_cache) {
        for (unsigned i = 0; i < _entSize; ++i)
            delete[] _ed_cache->second[i];
        for (unsigned i = 0; i < _relSize; ++i)
            delete[] _ed_cache->first[i];
        delete[] _ed_cache->first;
        delete[] _ed_cache->second;
        delete _ed_cache;
    }
}

float EmbeddingModel::update(const std::pair<Triple, Triple> * samples, unsigned size,
                             float rate, float margin) {
    float posval, negval, res = 0;
    for (unsigned i = 0; i < size; ++i) {
        posval = calc_sum(samples[i].first);
        negval = calc_sum(samples[i].second);
        if (posval + margin > negval) {
            res += margin + posval - negval;
            update_core(samples[i].first, 1, rate);
            update_core(samples[i].second, -1, rate);
            if (_use_cache)
                norm_cache(samples[i]);
            else
                norm(samples[i]);
        }
    }
    return res;
}

void EmbeddingModel::resetNegTriples() {
    for (unsigned i = 0; i < _ds.updateSize(); ++i)
        if (!(_ds.updateset() + i)->f) {
            vecReset(vh(*(_ds.updateset() + i)), _dim);
            //vecReset(vr(*(_ds.updateset() + i)), _dim);
            vecReset(vt(*(_ds.updateset() + i)), _dim);
        }
}

void EmbeddingModel::resetOutEntity() {
    const std::set<unsigned> & outEntity = _ds.outentity();
    std::set<unsigned>::iterator it;

    for (unsigned i = 0; i < _ds.updateSize(); ++i){
        std::vector<unsigned> neighborH,neighborT;
        const Triple & tmpi = _ds.updateset()[i];
        it = outEntity.find(tmpi.h);
        if (it != outEntity.end()) {  //head pooling
            for (unsigned j = 0; j < _ds.updateSize(); ++j)
                if (i != j) {
                    const Triple & tmpj = _ds.updateset()[j];
                    if(tmpi.r == tmpj.r && tmpi.t == tmpj.t)
                        neighborH.push_back(tmpj.h);
                }

            vecPooling(tmpi.h, neighborH);
        }


        it = outEntity.find(tmpi.t);
        if(it != outEntity.end()) {
            for (unsigned j = 0; j < _ds.updateSize(); ++j)
                if(i != j) {
                    const Triple & tmpj = _ds.updateset()[j];
                    if(tmpi.r == tmpj.r && tmpi.h == tmpj.h)
                        neighborT.push_back(tmpj.t);
                }
            vecPooling(tmpi.t,neighborT); // repalce with maxPooling()    
        }
    }
}

void EmbeddingModel::vecPooling(unsigned eid, const std::vector<unsigned> & neighbors) {
    if (!neighbors.empty()) {
        float * vec = _ed->second[eid];
        for (unsigned i = 0; i < _dim; ++i)
            vec[i] = 0;
        for (auto & neighbor : neighbors)
            for (unsigned i = 0; i < _dim; ++i)
                vec[i] += _ed->second[neighbor][i];
        norm(vec, _dim);
    }
}

void EmbeddingModel::maxPooling(unsigned eid, const std::vector<unsigned> & neighbors) {
    if (!neighbors.empty()) {
        float * vec = _ed->second[eid];
        for (unsigned i = 0; i < _dim; ++i)
            vec[i] = -1;
        for (auto & neighbor : neighbors)
            for (unsigned i = 0; i < _dim; ++i)
                vec[i] = std::max(vec[i], _ed->second[neighbor][i]);
        norm(vec, _dim);
    }
}
