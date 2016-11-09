#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


template<typename float_t>
py::array_t<float_t> inv3(py::array_t<float_t, py::array::c_style> & Ts)
{
    auto Ts_buf = Ts.request();
    float_t *pT = (float_t*)Ts_buf.ptr;

    auto result = py::array_t<float_t, py::array::c_style>(Ts_buf.size);
    auto result_buf = result.request();
    float_t *pR = (float_t*)result_buf.ptr;

    for (size_t idx = 0; idx < Ts_buf.shape[0]; idx++) {
        const float_t T00 = pT[0], T01 = pT[1], T02 = pT[2];
        const float_t T10 = pT[3], T11 = pT[4], T12 = pT[5];
        const float_t T20 = pT[6], T21 = pT[7], T22 = pT[8];
        const double det = T00 * (T22 * T11 - T21 * T12) \
                         - T10 * (T22 * T01 - T21 * T02) \
                         + T20 * (T12 * T01 - T11 * T02);
        double invDet = 1. / det;
        pR[0] =  (T11 * T22 - T21 * T12) * invDet;
        pR[1] = -(T01 * T22 - T02 * T21) * invDet;
        pR[2] =  (T01 * T12 - T02 * T11) * invDet;
        pR[3] = -(T10 * T22 - T12 * T20) * invDet;
        pR[4] =  (T00 * T22 - T02 * T20) * invDet;
        pR[5] = -(T00 * T12 - T10 * T02) * invDet;
        pR[6] =  (T10 * T21 - T20 * T11) * invDet;
        pR[7] = -(T00 * T21 - T20 * T01) * invDet;
        pR[8] =  (T00 * T11 - T10 * T01) * invDet;

        pT += 3*3;
        pR += 3*3;
    }

    return result;
}


PYBIND11_PLUGIN(_fastmath_ext) {
    py::module m("_fastmath_ext");
    m.def("inv3", &inv3<float>);
    m.def("inv3", &inv3<double>);

    return m.ptr();
}
