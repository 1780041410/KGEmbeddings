#include "Train.h"

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <thread>
#include <iomanip>

using namespace sysukg;

Train::Train(EmbeddingModel * em, SamplingModel * sm, float rate, float margin, unsigned threads) :
    _em(em), _sm(sm), _rate(rate), _margin(margin), _threads(threads) {}

void Train::launch(const std::string & mode, unsigned nepoch, unsigned output, bool silence) {
    if (mode != "testonly") {
        if (mode == "update") {
            //_em->resetNegTriples();
            _em->resetOutEntity();
            if (_em->dsname() == "FB15k" || _em->dsname() == "WN18")
                _em->runLinkPredictionTest(std::cout, _threads);
            else if (_em->dsname() == "FB13" || _em->dsname() == "WN11")
                _em->runClassificationTest(std::cout);
        }
        if (output == 0)
            output = nepoch + 1;
        std::string ext = mode + "-" + getExt();
        float result;
        std::thread ** threads = new std::thread*[_threads];
        for (unsigned i = 0; i < nepoch; ++i) {
            result = 0;
            _em->cache_store();
            for (unsigned j = 0; j < _threads; ++j)
                threads[j] = new std::thread(
                    [this, &result]() -> void {
                        float temp;
                        const unsigned len = _sm->size() / _threads;
                        std::pair<Triple, Triple> * t = new std::pair<Triple, Triple>[len];
                        _sm->sample(t, len);
                        temp = _em->update(t, len, _rate, _margin);
                        result += temp;
                        delete []t;
                    }
                );
            for (unsigned j = 0; j < _threads; ++j)
                threads[j]->join();
            _em->cache_load();
            for (unsigned j = 0; j < _threads; ++j)
                delete threads[j];
            if (!silence)
                std::cout << "epoch " << std::setw(5) << i
                        << " : " << std::fixed << std::setw(10) << std::setprecision(3) << result << std::endl;
            if ((i + 1) % output == 0)
                if (_em->dsname() == "FB15k" || _em->dsname() == "WN18")
                    _em->runLinkPredictionTest(std::cout, _threads);
                else if (_em->dsname() == "FB13" || _em->dsname() == "WN11")
                    _em->runClassificationTest(std::cout);
        }
        std::stringstream ss;
        ss << ext << "-epoch-" << nepoch;
        _em->output(ss.str());
        delete[] threads;
    } else {
        if (_em->dsname() == "FB15k" || _em->dsname() == "WN18")
            _em->runLinkPredictionTest(std::cout, _threads);
        else if (_em->dsname() == "FB13" || _em->dsname() == "WN11")
            _em->runClassificationTest(std::cout);
    }
}
