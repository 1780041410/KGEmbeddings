#pragma once

namespace sysukg {

struct Triple {
    unsigned h, r, t;
    bool f;
    Triple(unsigned h_ = 0, unsigned r_ = 0, unsigned t_ = 0, bool f_ = true) :
        h(h_), r(r_), t(t_), f(f_) {}
    inline Triple symm() const {
        return Triple(t, r, h, f);
    }
    inline Triple anti() const {
        return Triple(h, r, t, !f);
    }
    inline bool operator<(const Triple & other) const {
        if (h < other.h) return true;
        if (h > other.h) return false;
        if (r < other.r) return true;
        if (r > other.r) return false;
        if (t < other.t) return true;
        if (t > other.t) return false;
        return !f && other.f;
    }
};

inline bool Triple_hrt_less(const Triple & left, const Triple & right) {
    return left < right;
}
inline bool Triple_rht_less(const Triple & left, const Triple & right) {
    if (left.r < right.r) return true;
    if (left.r > right.r) return false;
    if (left.h < right.h) return true;
    if (left.h > right.h) return false;
    if (left.t < right.t) return true;
    if (left.t > right.t) return false;
    return !left.f && right.f;
}
inline bool Triple_trh_less(const Triple & left, const Triple & right) {
    if (left.t < right.t) return true;
    if (left.t > right.t) return false;
    if (left.r < right.r) return true;
    if (left.r > right.r) return false;
    if (left.h < right.h) return true;
    if (left.h > right.h) return false;
    return !left.f && right.f;
}

}
