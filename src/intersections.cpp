#include <limits>
#include <Eigen/Geometry>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include <igl/point_mesh_squared_distance.h>


namespace py = pybind11;
using namespace Eigen;

typedef Eigen::Vector3d Vec3d;
typedef Eigen::Vector2d Vec2d;


std::tuple<py::array_t<int>, py::array_t<double>, py::array_t<int>, py::array_t<double>> 
rayMeshIntersect(
        py::array_t<double> verts_in, 
        py::array_t<int> tris_in,
        py::array_t<double> ray_pts_in, 
        py::array_t<double> ray_dirs_in, 
        double max_distance, 
        double max_angle,
        bool allow_backface_hit) 
{
    auto verts = verts_in.unchecked<2>();
    auto tris = tris_in.unchecked<2>();
    auto rayPts = ray_pts_in.unchecked<2>();
    auto rayDirs = ray_dirs_in.unchecked<2>();

    std::vector<int> hitTris;
    std::vector<Vec2d> hitUVs;
    std::vector<Vec3d> hitPoints;
    std::vector<int> hitRayIndices;

    double maxAngleCos = std::cos(max_angle);
    for(int iray = 0; iray < rayPts.shape(0); iray++) {
        Vec3d rayPos(rayPts(iray, 0), rayPts(iray, 1), rayPts(iray, 2));
        Vec3d rayDir(rayDirs(iray, 0), rayDirs(iray, 1), rayDirs(iray, 2));
        Vec3d rayDirNorm = rayDir.normalized();
        Vec3d bestHit;
        float bestU, bestV, bestAngleCos;
        int bestTri = -1;
        float best_t = std::numeric_limits<float>::infinity();
        for(int itri = 0; itri < tris.shape(0); itri++) {
            // put data into Vec3d for easy dot/cross operations
            auto i1 = tris(itri, 0); auto i2 = tris(itri, 1); auto i3 = tris(itri, 2);
            Vec3d v1(verts(i1, 0), verts(i1, 1), verts(i1, 2));
            Vec3d v2(verts(i2, 0), verts(i2, 1), verts(i2, 2));
            Vec3d v3(verts(i3, 0), verts(i3, 1), verts(i3, 2));
            // perform ray-triangle hit test
            Vec3d edge1 = v2 - v1;
            Vec3d edge2 = v3 - v1;
            Vec3d pvec = rayDir.cross(edge2);
            float det = edge1.dot(pvec);
            if(std::abs(det) < std::numeric_limits<double>::epsilon()) {
                continue;
            }
            float invDet = 1.0 / det;
            Vec3d tvec = rayPos - v1;
            float u = tvec.dot(pvec) * invDet;
            if(u < 0.0 || u > 1.0) {
                continue;
            }
            Vec3d qvec = tvec.cross(edge1);
            float v = rayDir.dot(qvec) * invDet;
            if(v < 0.0 || u + v > 1.0) {
                continue;
            }
            float t = edge2.dot(qvec) * invDet;
            Vec3d hitPoint = v1 + edge1*u + edge2*v;
            // check if nearest hit and if it is valid
            if(fabs(t) < fabs(best_t) && (hitPoint - rayPos).norm() < max_distance) {
                bestHit = hitPoint;
                bestU = u; bestV = v;
                bestTri = itri;
                best_t = t;
                Vec3d normal = edge1.normalized().cross(edge2.normalized()).normalized();
                if (allow_backface_hit) {
                    bestAngleCos = fmax(rayDirNorm.dot(normal), rayDirNorm.dot(normal * -1.f));
                }
                else {
                    bestAngleCos = rayDirNorm.dot(normal);
                }
            }
        }
        if(bestTri > -1) {
            // check ray-normal angle
            if(bestAngleCos < maxAngleCos) {
                continue;
            }
            hitUVs.emplace_back(bestU, bestV);
            hitPoints.push_back(bestHit);
            hitRayIndices.push_back(iray);
            hitTris.push_back(bestTri);
        }
    }

    return std::make_tuple(
            py::array_t<int>(hitTris.size(), hitTris.data()),
            py::array_t<double>({(unsigned long)hitUVs.size(), 2ul}, (double*)hitUVs.data()),
            py::array_t<int>(hitRayIndices.size(), hitRayIndices.data()),
            py::array_t<double>({(unsigned long)hitPoints.size(), 3ul}, (double*)hitPoints.data())
    );
}


std::tuple<VectorXd, VectorXi, MatrixX3d, MatrixX2d>
closestPointOnMesh(const MatrixXd& P, const MatrixXd& V, const MatrixXi& tris) 
{
    VectorXd sq_dists;
    MatrixX3d hit_pts;
    VectorXi tri_ixs;
    igl::point_mesh_squared_distance(P, V, tris, sq_dists, tri_ixs, hit_pts);

    // determine uv coordinates of those closest hits
    MatrixX2d hit_uv(hit_pts.rows(), 2);
    for (int i = 0; i < P.rows(); ++i) {
        Vector3d p = hit_pts.row(i);
        Vector3i tri = tris.row(tri_ixs(i));
        // setup local coordinate frame
        Vector3d e10 = V.row(tri(1)) - V.row(tri(0));
        Vector3d e20 = V.row(tri(2)) - V.row(tri(0));
        Vector3d n = e10.cross(e20); // normal
        Matrix3d F;
        F << e10, e20, n;
        // invert local coordinate frame to get
        // frame-local coordinates
        Vector3d v0 = hit_pts.row(i) - V.row(tri(0));
        Vector3d uvw = F.inverse() * v0;
        hit_uv(i, 0) = uvw(0);
        hit_uv(i, 1) = uvw(1);
    }

    return std::make_tuple(sq_dists, tri_ixs, hit_pts, hit_uv);
};

PYBIND11_PLUGIN(_intersections_ext) {
    using namespace pybind11::literals;

    py::module m("_intersections_ext");
    m.def("ray_mesh_intersect", &rayMeshIntersect,
            "verts"_a, "tris"_a, "ray_pts"_a, "ray_dirs"_a,
            "max_distance"_a = std::numeric_limits<double>::infinity(),
            "max_angle"_a = std::numeric_limits<double>::infinity(),
            "allow_backface_hit"_a = true);

    m.def("closest_points_on_mesh", &closestPointOnMesh,
          "points"_a, "vertices"_a, "triangles"_a);

    return m.ptr();
}

