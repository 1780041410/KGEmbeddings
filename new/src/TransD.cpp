#include "TransD.h"

#include <thread>

using namespace sysukg;

void TransD::norm(float * ev, float * epv, float * rpv, float rate) {
    const float edot = dot(ev, epv, _dim);
    float x = 0, temp, sum = 0;
    for (unsigned i = 0; i < _dim; ++i) {
        temp = edot * rpv[i] + ev[i];
        x += sqr(temp);
        sum += temp * rpv[i];
    }
    if (x > 1) {
        sum *= 2;
        for (unsigned i = 0; i < _dim; ++i) {
            temp = 2 * (edot * rpv[i] + ev[i]);
            rpv[i] -= rate * temp * edot;
            ev[i] -= rate * temp;
        }
        for (unsigned i = 0; i < _dim; ++i) {
            ev[i] -= rate * sum * epv[i];
            epv[i] -= rate * sum * ev[i];
        }
    }
}

TransD::TransD(const DataSet & ds, unsigned dim, const std::string & ext) :
    EmbeddingModel(ds, dim, ext, true), _last_rate(0) {
    _rp = new float*[_relSize];
    _rp_cache = new float*[_relSize];
    for (unsigned i = 0; i < _relSize; ++i) {
        _rp[i] = new float[dim];
        _rp_cache[i] = new float[dim];
    }

    _ep = new float*[_entSize];
    _ep_cache = new float*[_entSize];
    for (unsigned i = 0; i < _entSize; ++i) {
        _ep[i] = new float[dim];
        _ep_cache[i] = new float[dim];
    }
    if (ext.empty()) {
        matrixReset(_rp, _relSize, dim);
        matrixReset(_ep, _entSize, dim);
    } else {
        readFromFile(_rp, "relproj2vec." + ext, _relSize, dim);
        readFromFile(_ep, "entproj2vec." + ext, _entSize, dim);
    }
}

TransD::~TransD() {
    for (unsigned i = 0; i < _relSize; ++i) {
        delete[] _rp[i];
        delete[] _rp_cache[i];
    }
    delete[] _rp;
    delete[] _rp_cache;
    for (unsigned i = 0; i < _entSize; ++i) {
        delete[] _ep[i];
        delete[] _ep_cache[i];
    }
    delete[] _ep;
    delete[] _ep_cache;
}

float TransD::calc_sum(const Triple & t) const {
    const float hdot = dot(vh(t), vhp(t), _dim), tdot = dot(vt(t), vtp(t), _dim);
    float result = 0, tmph, tmpt;
    for (unsigned i = 0; i < _dim; ++i) {
        tmph = hdot * vrp(t)[i] + vh(t)[i];
        tmpt = tdot * vrp(t)[i] + vt(t)[i];
        result += fabs(tmph + vr(t)[i] - tmpt);
    }
    return result;
}

void TransD::update_core(const Triple & triple, short label, float rate) {
    const float hdot = dot(vh(triple), vhp(triple), _dim),
                tdot = dot(vt(triple), vtp(triple), _dim);
    float x, s = 0, tmph, tmpt;
	for (int i = 0; i < _dim; i++) {
        tmph = hdot * vrp(triple)[i] + vh(triple)[i];
        tmpt = tdot * vrp(triple)[i] + vt(triple)[i];
		if (tmpt - tmph - vr(triple)[i] < 0)
			x = label * rate;
		else
			x = -label * rate;
		s += x * vrp(triple)[i];
		cvrp(triple)[i] += x * (tdot - hdot);
		cvr(triple)[i] -= x;
        cvh(triple)[i] -= x;
        cvt(triple)[i] += x;
	}
	for (int i = 0; i < _dim; i++) {
		cvh(triple)[i] -= s * vhp(triple)[i];
		cvhp(triple)[i] -= s * vh(triple)[i];
		cvt(triple)[i] += s * vtp(triple)[i];
		cvtp(triple)[i] += s * vt(triple)[i];
	}
}

void TransD::output(const std::string & ext) const {
    static const unsigned short filenum = 4;
    std::thread * threads[filenum] = {
        new std::thread{single_output, std::move("relation2vec." + ext),
            std::ref(_relSize), std::ref(_dim),std::ref(_ed->first)},
        new std::thread{single_output, std::move("entity2vec." + ext),
            std::ref(_entSize), std::ref(_dim), std::ref(_ed->second)},
        new std::thread{single_output, std::move("relproj2vec." + ext),
            std::ref(_relSize), std::ref(_dim), std::ref(_rp)},
        new std::thread{single_output, std::move("entproj2vec." + ext),
            std::ref(_relSize), std::ref(_dim), std::ref(_rp)}
    };
    for (char i = 0; i < filenum; ++i)
        threads[i]->join();
    for (char i = 0; i < filenum; ++i)
        delete threads[i];
}

void TransD::resetNegTriples() {
    for (unsigned i = 0; i < _ds.updateSize(); ++i)
        if (!(_ds.updateset() + i)->f) {
            vecReset(vh(*(_ds.updateset() + i)), _dim);
            vecReset(vr(*(_ds.updateset() + i)), _dim);
            vecReset(vt(*(_ds.updateset() + i)), _dim);
            vecReset(vhp(*(_ds.updateset() + i)), _dim);
            vecReset(vrp(*(_ds.updateset() + i)), _dim);
            vecReset(vtp(*(_ds.updateset() + i)), _dim);
        }
}
