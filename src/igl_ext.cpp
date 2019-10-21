#include <limits>
#include <Eigen/Geometry>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <igl/exact_geodesic.h>
#include <igl/is_border_vertex.h>


namespace py = pybind11;
using namespace Eigen;

PYBIND11_MODULE(_igl_ext, m) {
    using namespace pybind11::literals;

    m.def("exact_geodesic", []
        (const MatrixXd& V, 
         const MatrixXi& F, 
         const VectorXi& src_ix)
        -> VectorXd
        {
            VectorXi src_face_ix;
            VectorXi target_ix = VectorXi::LinSpaced(V.rows(), 0, V.rows()-1);
            VectorXi target_face_ix;
            MatrixXd dists;

            igl::exact_geodesic(V, F, 
                    src_ix, src_face_ix, 
                    target_ix, target_face_ix,
                    dists);

            Map<RowVectorXd> dists_flat(dists.data(), dists.size());
            return dists_flat;
        }, 
        py::call_guard<py::gil_scoped_release>(),
        py::arg("verts"), py::arg("tris"), py::arg("src_vert_indices")
    );

    m.def("is_border_vertex", []
        (const MatrixXi& F) 
        -> Matrix<bool, Dynamic, 1>
        {
            std::vector<bool> b = igl::is_border_vertex(F);
            Matrix<bool, Dynamic, 1> bm(b.size());
            for (size_t i=0; i < b.size(); i++) {
                bm(i) = b[i];
            }
            return bm;
        },
        py::call_guard<py::gil_scoped_release>(),
        py::arg("tris")
    );
}

