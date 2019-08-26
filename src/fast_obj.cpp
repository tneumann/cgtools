#include <vector>
#include <iostream>
#include <fstream>
#include <Eigen/Core>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>


inline bool is_white_space(char c)
{
    return ((c) == ' ' || (c) == '\t');
};

inline bool is_valid_digit(char c)
{
    return ((c) >= '0' && (c) <= '9');
};

inline void skip_whitespace(char * &c)
{
    while (is_white_space(*c) ) c++;
}

inline void skip_until_whitespace(char * &c)
{
    while (*c && !is_white_space(*c)) c++;
}

inline int fast_atoi(char * &c)
{
    int val = 0;
    while(is_valid_digit(*c)) {
        val = val*10 + (*c - '0');
        c++;
    }
    return val;
}

// this function was adapted from Tom Van Baaks code here: http://leapsecond.com/tools/fast_atof.c
inline double fast_atod(char * &p)
{
    int frac;
    double sign, value, scale;

    // Get sign, if any.

    sign = 1.0;
    if (*p == '-') {
        sign = -1.0;
        p += 1;

    } else if (*p == '+') {
        p += 1;
    }

    // Get digits before decimal point or exponent, if any.

    for (value = 0.0; is_valid_digit(*p); p += 1) {
        value = value * 10.0 + (*p - '0');
    }

    // Get digits after decimal point, if any.

    if (*p == '.') {
        double pow10 = 10.0;
        p += 1;
        while (is_valid_digit(*p)) {
            value += (*p - '0') / pow10;
            pow10 *= 10.0;
            p += 1;
        }
    }

    // Handle exponent, if any.

    frac = 0;
    scale = 1.0;
    if ((*p == 'e') || (*p == 'E')) {
        unsigned int expon;

        // Get sign of exponent, if any.

        p += 1;
        if (*p == '-') {
            frac = 1;
            p += 1;

        } else if (*p == '+') {
            p += 1;
        }

        // Get digits of exponent, if any.

        for (expon = 0; is_valid_digit(*p); p += 1) {
            expon = expon * 10 + (*p - '0');
        }
        if (expon > 308) expon = 308;

        // Calculate scaling factor.

        while (expon >= 50) { scale *= 1E50; expon -= 50; }
        while (expon >=  8) { scale *= 1E8;  expon -=  8; }
        while (expon >   0) { scale *= 10.0; expon -=  1; }
    }

    // Return signed and scaled floating point result.

    return sign * (frac ? (value / scale) : (value * scale));
}


using RowMatX3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using RowMatXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

std::pair<RowMatX3d, RowMatXi> 
loadOBJFast(std::string filename)
{
    std::ifstream inF(filename, std::fstream::in | std::fstream::binary);
    std::string line;

    std::vector<std::array<double, 3>> verts_vec;
    std::vector<std::array<int, 4>> quads;
    std::vector<std::array<int, 3>> tris;
    int faceDim = 0;

    while (std::getline(inF, line)) {
        char* c = &line[0];
        if (c[0] == 'v' && is_white_space(c[1])) {
            c += 2;
            skip_whitespace(c);
            // parse vertex coordinates
            double x = fast_atod(c);
            skip_whitespace(c);
            double y = fast_atod(c);
            skip_whitespace(c);
            double z = fast_atod(c);
            verts_vec.push_back({x, y, z});
            // TODO: check if there is a vertex color, read that
        }
        else if (c[0] == 'f' && is_white_space(c[1])) {
            c += 2;
            skip_whitespace(c);
            if (faceDim == 0) {
                // determine face dimension - quads or triangles?
                auto c_lookahead = c;
                for (; is_valid_digit(*c_lookahead); faceDim++) {
                    skip_until_whitespace(c_lookahead);
                    skip_whitespace(c_lookahead);
                }
            }
            if (faceDim == 3) {
                // parse triangle
                std::array<int, 3> tri;
                tri[0] = fast_atoi(c) - 1;
                skip_until_whitespace(c); skip_whitespace(c);
                tri[1] = fast_atoi(c) - 1;
                skip_until_whitespace(c); skip_whitespace(c);
                tri[2] = fast_atoi(c) - 1;
                if (tri[0] < 0 || tri[1] < 0 || tri[2] < 0) {
                    std::cerr << "negative face index found, ignoring" << std::endl;
                }
                else {
                    tris.push_back(tri);
                }
            } else {
                // parse quad
                std::array<int, 4> quad;
                quad[0] = fast_atoi(c) - 1;
                skip_until_whitespace(c); skip_whitespace(c);
                quad[1] = fast_atoi(c) - 1;
                skip_until_whitespace(c); skip_whitespace(c);
                quad[2] = fast_atoi(c) - 1;
                skip_until_whitespace(c); skip_whitespace(c);
                quad[3] = fast_atoi(c) - 1;
                if (quad[0] < 0 || quad[1] < 0 || quad[2] < 0 || quad[3] < 0) {
                    std::cerr << "negative face index found, ignoring";
                }
                else {
                    quads.push_back(quad);
                }
            }
        }
    }

    RowMatX3d verts = Eigen::Map<RowMatX3d>((double*)verts_vec.data(), verts_vec.size(), 3);
    RowMatXi faces;
    if (faceDim == 4) {
        faces = Eigen::Map<RowMatXi>((int*)quads.data(), quads.size(), 4);
    }
    else if (faceDim == 3) {
        faces = Eigen::Map<RowMatXi>((int*)tris.data(), tris.size(), 3);
    }
    return std::make_pair(verts, faces);
}


PYBIND11_MODULE(_fastobj_ext, m) {
    m.def("load_obj_fast", &loadOBJFast);
}
