#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <Eigen/Core>
#include <igl/polar_dec.h>
#include <igl/polar_svd.h>

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
        const float_t det = T00 * (T22 * T11 - T21 * T12) \
                         - T10 * (T22 * T01 - T21 * T02) \
                         + T20 * (T12 * T01 - T11 * T02);
        const float_t invDet = 1. / det;
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
        const float_t det = T00 * T11 - T01 * T10;
        const float_t invDet = 1. / det;
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
            {a_buf.shape[0], a_buf.shape[1], b_buf.shape[2]});
    auto result_buf = result.request();
    float_t *p_res = (float_t*)result_buf.ptr;

    const size_t n_rows_a = a_buf.shape[1];
    const size_t n_cols_a = a_buf.shape[2];
    const size_t n_rows_b = b_buf.shape[1];
    const size_t n_cols_b = b_buf.shape[2];
    assert(n_cols_a == n_rows_b);
    for (size_t idx = 0; idx < a_buf.shape[0]; idx++) {
        for (size_t row_a = 0; row_a < n_rows_a; row_a++) {
            for (size_t col_b = 0; col_b < n_cols_b; col_b++) {
                float_t sum = 0.0;
                for (size_t k = 0; k < n_cols_a; k++) {
                    const float_t ai = p_a[row_a * n_cols_a + k];
                    const float_t bi = p_b[k * n_cols_b + col_b];
                    sum += ai * bi;
                }
                *p_res = sum;
                p_res++;
            }
        }
        p_a += n_cols_a * n_rows_a;
        p_b += n_cols_b * n_rows_b;
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

    auto result = py::array_t<float_t, py::array::c_style>({mats_buf.shape[0], mats_buf.shape[1]});
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

template<typename float_t>
py::array_t<float_t> cross3(
        py::array_t<float_t, py::array::c_style> & a,
        py::array_t<float_t, py::array::c_style> & b
    )
{
    auto a_buf = a.request();
    float_t *p_a = (float_t*)a_buf.ptr;
    auto b_buf = b.request();
    float_t *p_b = (float_t*)b_buf.ptr;

    auto result = py::array_t<float_t, py::array::c_style>(
            {a_buf.shape[0], a_buf.shape[1]});
    auto result_buf = result.request();
    float_t *p_res = (float_t*)result_buf.ptr;

    for (size_t idx = 0; idx < a_buf.shape[0]; idx++) {
        const double ax = p_a[0];
        const double ay = p_a[1];
        const double az = p_a[2];
        const double bx = p_b[0];
        const double by = p_b[1];
        const double bz = p_b[2];
        p_res[0] = ay * bz - az * by;
        p_res[1] = az * bx - ax * bz;
        p_res[2] = ax * by - ay * bx;
        p_res += 3;
        p_a += 3;
        p_b += 3;
    }

    return result;
}


template<typename float_t>
py::array_t<float_t> multikron(
        py::array_t<float_t, py::array::c_style> & a,
        py::array_t<float_t, py::array::c_style> & b
    )
{
    auto a_buf = a.request();
    float_t *p_a = (float_t*)a_buf.ptr;
    auto b_buf = b.request();
    float_t *p_b = (float_t*)b_buf.ptr;

    const auto n_rows_a = a_buf.shape[1];
    const auto n_cols_a = a_buf.shape[2];
    const auto n_rows_b = b_buf.shape[1];
    const auto n_cols_b = b_buf.shape[2];

    auto result = py::array_t<float_t, py::array::c_style>(
            {a_buf.shape[0], n_rows_a * n_rows_b, n_cols_a * n_cols_b});
    auto result_buf = result.request();
    float_t *p_res = (float_t*)result_buf.ptr;

    for (size_t idx = 0; idx < a_buf.shape[0]; idx++) {
        // iterate over rows of a
        for (size_t row_a = 0; row_a < n_rows_a; row_a++) {
            // iterate over rows of b
            for (size_t row_b = 0; row_b < n_rows_b; row_b++) {
                // iterate over columns of a
                for (size_t col_a = 0; col_a < n_cols_a; col_a++) {
                    // iterate over columns of b
                    for (size_t col_b = 0; col_b < n_cols_b; col_b++) {
                        const float_t ai = p_a[row_a * n_cols_a + col_a];
                        const float_t bi = p_b[row_b * n_cols_b + col_b];
                        *p_res = ai * bi;
                        p_res++;
                    }
                }
            }
        }
        // next matrix
        p_a += n_cols_a * n_rows_a;
        p_b += n_cols_b * n_rows_b;
    }

    return result;
}


std::tuple<py::array_t<double>, py::array_t<double>>
polarDecompose(py::array_t<double, py::array::c_style> Ms)
{
    using RowMat3d = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

    auto Ms_raw = Ms.unchecked<3>();
    // allocate output matrices
    auto Rs = py::array_t<double>({Ms_raw.shape(0), 3l, 3l});
    auto Rs_raw = Rs.mutable_unchecked<3>();
    auto Ss = py::array_t<double>({Ms_raw.shape(0), 3l, 3l});
    auto Ss_raw = Ss.mutable_unchecked<3>();

    RowMat3d U;
    RowMat3d V;
    Eigen::Matrix<double,3,1> S;

    for (int i = 0; i < Ms_raw.shape(0); ++i) {
        const RowMat3d Mi = Eigen::Map<const RowMat3d>(Ms_raw.data(i, 0, 0));
        Eigen::Map<RowMat3d> Ri_map(Rs_raw.mutable_data(i, 0, 0));
        Eigen::Map<RowMat3d> Si_map(Ss_raw.mutable_data(i, 0, 0));
        RowMat3d Ri = Ri_map; // performs copy, since igl::polar_dec does not take Eigen::Map
        RowMat3d Si = Si_map;
        //igl::polar_dec(Mi, Ri, Si);
        igl::polar_svd(Mi, Ri, Si, U, S, V);
        // TODO: use igl::polar_svd3x3?
        // write back results
        Ri_map = Ri;
        Si_map = Si;
    }

    return std::make_tuple(Rs, Ss);
}


PYBIND11_MODULE(_fastmath_ext, m) {
    m.def("inv3", &inv3<float>);
    m.def("inv3", &inv3<double>);
    m.def("inv2", &inv2<float>);
    m.def("inv2", &inv2<double>);
    m.def("matmat", &matmat<double>);
    m.def("matvec", &matvec<double>);
    m.def("cross3", &cross3<double>);
    m.def("multikron", &multikron<double>);
    m.def("polar_dec", &polarDecompose);
}
