#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>

namespace py = pybind11;
using namespace Eigen;


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
        const double invDet = 1. / det;
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

template<typename float_t>
py::array_t<float_t> inv2(py::array_t<float_t, py::array::c_style> & Ts)
{
    auto Ts_buf = Ts.request();
    float_t *pT = (float_t*)Ts_buf.ptr;

    auto result = py::array_t<float_t, py::array::c_style>(Ts_buf.size);
    auto result_buf = result.request();
    float_t *pR = (float_t*)result_buf.ptr;

    for (size_t idx = 0; idx < Ts_buf.shape[0]; idx++) {
        const float_t T00 = pT[0], T01 = pT[1];
        const float_t T10 = pT[2], T11 = pT[3];
        const double det = T00 * T11 - T01 * T10;
        const double invDet = 1. / det;
        pR[0] =  T11 * invDet;
        pR[1] = -1 * T01 * invDet;
        pR[2] = -1 * T10 * invDet;
        pR[3] = T00 * invDet;

        pT += 2*2;
        pR += 2*2;
    }

    return result;
}

template<typename float_t>
py::array_t<float_t> matmat(
        py::array_t<float_t, py::array::c_style> & a,
        py::array_t<float_t, py::array::c_style> & b
    )
{
    auto a_buf = a.request();
    float_t *p_a = (float_t*)a_buf.ptr;
    auto b_buf = b.request();
    float_t *p_b = (float_t*)b_buf.ptr;

    auto result = py::array_t<float_t, py::array::c_style>(
            std::vector<size_t>({{a_buf.shape[0], a_buf.shape[1], b_buf.shape[2]}}) );
    auto result_buf = result.request();

    typedef Matrix<float_t, Dynamic, Dynamic, RowMajor> MatrixT;
    for (size_t idx = 0; idx < a_buf.shape[0]; idx++) {
        Map<const MatrixT> a_i(a.data(idx, 0, 0), a_buf.shape[1], a_buf.shape[2]);
        Map<const MatrixT> b_i(b.data(idx, 0, 0), b_buf.shape[1], b_buf.shape[2]);
        Map<MatrixT> r_i(result.mutable_data(idx, 0, 0), result_buf.shape[1], result_buf.shape[2]);
        r_i = a_i * b_i;
    }

    return result;
}

template<typename float_t>
py::array_t<float_t> matvec(
        py::array_t<float_t, py::array::c_style> & mats,
        py::array_t<float_t, py::array::c_style> & vecs
    )
{
    auto mats_buf = mats.request();
    float_t *p_mats = (float_t*)mats_buf.ptr;
    auto vecs_buf = vecs.request();
    float_t *p_vecs = (float_t*)vecs_buf.ptr;

    auto result = py::array_t<float_t, py::array::c_style>(
            std::vector<size_t>({{mats_buf.shape[0], mats_buf.shape[1]}}) );
    auto result_buf = result.request();
    float_t *p_res = (float_t*)result_buf.ptr;

    const size_t mat_stride1 = mats_buf.strides[1] / sizeof(float_t);
    for (size_t idx = 0; idx < mats_buf.shape[0]; idx++) {
        for (size_t row = 0; row < mats_buf.shape[1]; row++) {
            float_t sum = 0.0;
            for (size_t k = 0; k < mats_buf.shape[2]; k++) {
                sum += *(p_mats++) * p_vecs[k];
            }
            *p_res = sum;
            p_res++;
        }
        p_vecs += vecs_buf.shape[1];
    }

    return result;
}


PYBIND11_PLUGIN(_fastmath_ext) {
    py::module m("_fastmath_ext");
    m.def("inv3", &inv3<float>);
    m.def("inv3", &inv3<double>);
    m.def("inv2", &inv2<float>);
    m.def("inv2", &inv2<double>);
    m.def("matmat", &matmat<double>);
    m.def("matvec", &matvec<double>);

    return m.ptr();
}
