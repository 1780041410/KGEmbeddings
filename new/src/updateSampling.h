#pragma once

#include <utility>
#include <cmath>
#include <ostream>

#include "unifSampling.h"

namespace sysukg {

class updateSampling : public unifSampling {
private:
    typedef double floatType;

    struct node {
        unsigned terminal;
        floatType weight;
        node * next;
        node(unsigned t, floatType w, node * n) :
            terminal(t), weight(w), next(n) {}
    };
    struct alias_node {
        unsigned id;
        floatType prob;
        alias_node(unsigned a, floatType b) : id(a), prob(b) {}
    };

    const DataSet & _ds;
    const unsigned _relNum;
    const floatType _alpha;
    floatType * _pool_weight;
    floatType * _ew, * _rw;
    floatType * _alias_prob;
    unsigned * _alias; 

    constexpr static const floatType INF_DISTANCE = -1;

    inline void undirected_insert(node ** graph, unsigned v1, unsigned v2, floatType weight) {
        node * temp = graph[v1];
        graph[v1] = new node(v2, weight, temp);
        temp = graph[v2];
        graph[v2] = new node(v1, weight, temp);
    }
    inline floatType elementWeight(const floatType distance) {
        if (distance == INF_DISTANCE)
            return 0;
        else
            return exp(- _alpha * distance);
    }
    inline floatType edge_weight(unsigned count) const {
        return 1.0 / count;
    }
    inline floatType triple_weight(const Triple & t) const {
        return _ew[t.h] + _rw[t.r] + _ew[t.t];
    }

    static void norm(floatType * a, unsigned size);
    static void spfa(floatType * distance, unsigned size, node ** graph, std::set<unsigned> sources);
    static floatType avg(const floatType * a, unsigned size);

protected:
    inline Triple getPosSamp() const {
        unsigned id = _rd(_size);
        if (_rd(0, 1) < _alias_prob[id])
            return _pool[id];
        return _pool[_alias[id]];
    }

public:
    explicit updateSampling(const DataSet & ds, floatType alpha = 1, bool with_update_set = true);
    void output(std::ostream & os) const;
    inline std::string methodName() const {
        return "update";
    }
    ~updateSampling();
};

}
