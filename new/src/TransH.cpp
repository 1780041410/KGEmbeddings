#include "TransH.h"
#include <cstring>
#include <cmath>
#include <thread>

using namespace sysukg;

TransH::TransH(const DataSet & ds, unsigned dim, const std::string & ext) :
    EmbeddingModel(ds, dim, ext, true), _last_rate(0) {
    _rp = new float*[_relSize];
    _rp_cache = new float*[_relSize];
    for (unsigned i = 0; i < _relSize; ++i) {
        _rp[i] = new float[dim];
        _rp_cache[i] = new float[dim];
    }
    if (ext.empty())
        matrixReset(_rp, _relSize, dim);
    else
        readFromFile(_rp, "relproj2vec." + ext, _relSize, _dim);
}

float TransH::calc_sum(const Triple & t) const {
    float hdot = dot(vrp(t), vh(t), _dim),
          tdot = dot(vrp(t), vt(t), _dim);
    float result = 0;
    for (unsigned i = 0; i < _dim; ++i)
        result += fabs(frhti(t, hdot, tdot, i));
    return result;
}

void TransH::norm(float * rv, float * rp, float rate) {
    strict_norm(rp);
    if (dot(rv, rp, _dim) > 0.1)
        for (unsigned i = 0; i < _dim; ++i) {
            rv[i] -= rate * rp[i];
            rp[i] -= rate * rv[i];
        }
    strict_norm(rp);
}

void TransH::update_core(const Triple & triple, short label, float rate) {
    const float hdot = dot(vrp(triple), vh(triple), _dim),
                tdot = dot(vrp(triple), vt(triple), _dim);
    float x, sumx = 0;
    for (int i = 0; i < _dim; i++) {
		if (frhti(triple, hdot, tdot, i) < 0) {
			x = label * rate;
			sumx += vrp(triple)[i];
		} else {
			x = -label * rate;
			sumx -= vrp(triple)[i];
		}
		cvr(triple)[i] -= x;
		cvh(triple)[i] -= x;
        cvt(triple)[i] += x;
		cvrp(triple)[i] += x * (hdot - tdot);
	}
	for (int i = 0; i < _dim; i++)
		cvrp(triple)[i] += label * rate * sumx * (vh(triple)[i] - vt(triple)[i]);
}

void TransH::output(const std::string & ext) const {
    static const unsigned short filenum = 3;
    std::thread * threads[filenum] = {
        new std::thread{single_output, std::move("relation2vec." + ext),
            std::ref(_relSize), std::ref(_dim),std::ref(_ed->first)},
        new std::thread{single_output, std::move("entity2vec." + ext),
            std::ref(_entSize), std::ref(_dim), std::ref(_ed->second)},
        new std::thread{single_output, std::move("relproj2vec." + ext),
            std::ref(_relSize), std::ref(_dim), std::ref(_rp)}
    };
    for (char i = 0; i < filenum; ++i)
        threads[i]->join();
    for (char i = 0; i < filenum; ++i)
        delete threads[i];
}

TransH::~TransH() {
    for (unsigned i = 0; i < _relSize; ++i) {
        delete[] _rp[i];
        delete[] _rp_cache[i];
    }
    delete[] _rp;
    delete[] _rp_cache;
}

void TransH::resetNegTriples() {
    for (unsigned i = 0; i < _ds.updateSize(); ++i)
        if (!(_ds.updateset() + i)->f) {
            vecReset(vh(*(_ds.updateset() + i)), _dim);
            //vecReset(vr(*(_ds.updateset() + i)), _dim);
            vecReset(vt(*(_ds.updateset() + i)), _dim);
            //vecReset(vrp(*(_ds.updateset() + i)), _dim);
        }
}
