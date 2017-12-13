#pragma once

#include "EmbeddingModel.h"
#include "SamplingModel.h"

#include <string>
#include <sstream>
#include <iomanip>

namespace sysukg {

class Train {
private:
    constexpr static const float _DEFAULT_RATE = 0.001, _DEFAULT_MARGIN = 1;
    constexpr static const unsigned _DEFAULT_THREADS = 4;
    float _rate, _margin;
    unsigned _threads;
    EmbeddingModel * const _em;
    SamplingModel * const _sm;
protected:
    std::string getExt() const {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(3)
           << _em->methodName() << '-' << _em->dsname() << '-' << _sm->methodName()
           << "-dim-" << _em->dim() << "-rate-" << _rate << "-margin-" << _margin;
        return ss.str();
    }
public:
    Train(EmbeddingModel * em, SamplingModel * sm,
          float rate = _DEFAULT_RATE, float margin = _DEFAULT_MARGIN,
          unsigned threads = _DEFAULT_THREADS);
    
    inline void setRate(float rate) {
        _rate = rate;
    }
    inline void setMargin(float margin) {
        _margin = margin;
    }
    inline void setThreads(unsigned threads) {
        _threads = threads;
    }
    inline float rate() const {
        return _rate;
    }
    inline float margin() const {
        return _margin;
    }

    void launch(const std::string & mode, unsigned nepoch, unsigned output, bool silence,const std::string & pooling);
};

}
