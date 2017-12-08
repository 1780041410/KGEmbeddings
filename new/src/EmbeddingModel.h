#pragma once

#include <cmath>
#include <cstdlib>
#include <string>
#include <ostream>
#include <cstring>

#include "DataSet.h"
#include "Triple.h"
#include "random.h"

namespace sysukg {

class EmbeddingModel {
protected:
    struct TripleWithScore {
        const Triple * t;
        float score;
        inline bool operator<(const TripleWithScore & other) const {
            return score < other.score;
        }
    };
    struct rTripleWithScore {
        Triple t;
        float score;
        inline bool operator<(const TripleWithScore & other) const {
            return score < other.score;
        }
    };

    DataSet _ds;
    static random_device & _rd;
    unsigned _dim, _relSize, _entSize;
    typedef std::pair<float **, float **> EmbeddedData; // first = relations, second = entities
    EmbeddedData * _ed, *_ed_cache;
    const bool _use_cache;

    inline static float sqr(float x) {
        return x * x;
    }
    static inline float dot(const float * x, const float * y, unsigned size) {
        float result = 0;
        for (unsigned i = 0; i < size; ++i)
            result += x[i] * y[i];
        return result;
    }
    static inline float vec_len(float * v, unsigned size) {
        float x = 0;
        for (unsigned i = 0; i < size; ++i)
            x += sqr(v[i]);
        x = sqrt(x);
        return x;
    }
    static inline void norm(float * v, unsigned size) {
        float x = vec_len(v, size);
        if (x > 1)
            for (unsigned i = 0; i < size; ++i)
                v[i] /= x;
    }
    static inline void vecReset(float * vec, unsigned size) {
        for (unsigned i = 0; i < size; ++i)
            vec[i] = _rd.randn(0, 1.0 / size, -6 / sqrt(size), 6 / sqrt(size));
        norm(vec, size);
    }

    static inline void vecPooling(float * vec, unsigned size,float * neighbor,int num) {
        for (unsigned i = 0; i < size; ++i){
            vec[i] += neighbor[i] / num ;
        }
        norm(vec,size);
    }

    static inline void matrixReset(float ** matrix, unsigned n, unsigned m) {
        for (unsigned i = 0; i < n; ++i)
            vecReset(matrix[i], m);
    }
    static inline void matrixCopy(float ** target, float ** source, unsigned n, unsigned m) {
        for (unsigned i = 0; i < n; ++i)
            memcpy(target[i], source[i], m * sizeof(float));
    }
    static inline void matrixCopy(float ** target, const float ** source, unsigned n, unsigned m) {
        for (unsigned i = 0; i < n; ++i)
            memcpy(target[i], source[i], m * sizeof(float));
    }

    static void readFromFile(float ** target, const std::string & filename, unsigned n, unsigned m);

    inline const float * vh(const Triple & t) const {
        return _ed->second[t.h];
    }
    inline const float * vr(const Triple & t) const {
        return _ed->first[t.r];
    }
    inline const float * vt(const Triple & t) const {
        return _ed->second[t.t];
    }
    inline float * vh(const Triple & t) {
        return _ed->second[t.h];
    }
    inline float * vr(const Triple & t) {
        return _ed->first[t.r];
    }
    inline float * vt(const Triple & t) {
        return _ed->second[t.t];
    }
    inline const float * cvh(const Triple & t) const {
        return _ed_cache->second[t.h];
    }
    inline const float * cvr(const Triple & t) const {
        return _ed_cache->first[t.r];
    }
    inline const float * cvt(const Triple & t) const {
        return _ed_cache->second[t.t];
    }
    inline float * cvh(const Triple & t) {
        return _ed_cache->second[t.h];
    }
    inline float * cvr(const Triple & t) {
        return _ed_cache->first[t.r];
    }
    inline float * cvt(const Triple & t) {
        return _ed_cache->second[t.t];
    }

    virtual inline void norm_cache(const std::pair<Triple, Triple> & sample) {
        norm(cvh(sample.first), _dim);
        norm(cvr(sample.first), _dim);
        norm(cvt(sample.first), _dim);
        if (sample.first.h == sample.second.h)
            norm(cvt(sample.second), _dim);
        else
            norm(cvh(sample.second), _dim);
    }
    virtual inline void norm(const std::pair<Triple, Triple> & sample) {
        norm(vh(sample.first), _dim);
        norm(vr(sample.first), _dim);
        norm(vt(sample.first), _dim);
        if (sample.first.h == sample.second.h)
            norm(vt(sample.second), _dim);
        else
            norm(vh(sample.second), _dim);
    }
    virtual void update_core(const Triple & triple, short label, float rate) = 0;

    static void single_output(const std::string & filename,
                    unsigned num, unsigned dim, float ** mat);

public:
    EmbeddingModel(const DataSet & ds, unsigned dim,
                const std::string & ext = "", bool _use_cache = true);

    inline void EDreset() {
        matrixReset(_ed->first, _relSize, _dim);
        matrixReset(_ed->second, _entSize, _dim);
    }

    virtual void resetNegTriples();

    virtual void resetOutEntity();

    virtual inline void cache_store() {
        if (_use_cache) {
            matrixCopy(_ed_cache->first, _ed->first, _relSize, _dim);
            matrixCopy(_ed_cache->second, _ed->second, _entSize, _dim);
        }
    }
    virtual inline void cache_load() {
        if (_use_cache) {
            matrixCopy(_ed->first, _ed_cache->first, _relSize, _dim);
            matrixCopy(_ed->second, _ed_cache->second, _entSize, _dim);
        }
    }

    void runClassificationTest(std::ostream & os) const;
    void runLinkPredictionTest(std::ostream & os, unsigned threadnum = 4, bool np = false) const;

    inline unsigned dim() const {
        return _dim;
    }
    virtual std::string methodName() const = 0;
    inline const std::string & dsname() const {
        return _ds.name();
    }

    virtual float calc_sum(const Triple & t) const = 0;
    virtual float update(const std::pair<Triple, Triple> * samples, unsigned size,
                            float rate, float margin);
    virtual void output(const std::string & ext) const;

    ~EmbeddingModel();
};

}
