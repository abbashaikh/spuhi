// For Logger
#include "Logger.h"

// For python binding
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// General includes
#include <iostream>
#include <fstream>
#include <boost/format.hpp>

#include <vector>
#include <tuple>
#include <cmath>

// for CGAL::Nef_polyhedron_2
#include <CGAL/Exact_integer.h>
#include <CGAL/Filtered_extended_homogeneous.h>
#include <CGAL/Nef_polyhedron_2.h>

// For CGAL::Polyline_simplification_2
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_with_holes_2.h>
#include <CGAL/Polyline_simplification_2/simplify.h>
#include <CGAL/IO/WKT.h>

// Namespace & Type Definitions
namespace py = pybind11;

// For CGAL::Nef_polyhedron_2
// Define the kernel and Nef polyhedron types
typedef CGAL::Exact_integer RT;
typedef CGAL::Filtered_extended_homogeneous<RT> Extended_kernel;
typedef CGAL::Nef_polyhedron_2<Extended_kernel> Nef_polyhedron;
typedef Extended_kernel::Standard_ray_2 Ray;
 
typedef Nef_polyhedron::Point Point;
typedef Nef_polyhedron::Line  Line;
typedef Nef_polyhedron::Explorer Explorer;

typedef Explorer::Face_const_iterator Face_const_iterator;
typedef Explorer::Hole_const_iterator Hole_const_iterator;
typedef Explorer::Halfedge_around_face_const_circulator Halfedge_around_face_const_circulator;
typedef Explorer::Vertex_const_handle Vertex_const_handle;

// For CGAL::Polyline_simplification_2
namespace PS = CGAL::Polyline_simplification_2;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Polygon_2<K>                   Polygon_2;
typedef CGAL::Polygon_with_holes_2<K>        Polygon_with_holes_2;
typedef PS::Stop_above_cost_threshold        Stop;
typedef PS::Squared_distance_cost            Cost;


// Define the Class for Finding Support Size
// This class is responsible for processing halfplane intersections and calculating the support size
class HalfplaneIntersectionProcessor {
public:
    HalfplaneIntersectionProcessor(int scene_size, double cost_threshold)
        : sceneSize(scene_size), costThreshold(cost_threshold), logger(nullptr) {}

        void set_logger(Logger* log_ptr) {
            logger = log_ptr;
        }

        int get_support_size(py::array_t<double> lines_array) {
            std::vector<Line> lines;
    
            auto array_ref = lines_array.unchecked<2>();
            if (array_ref.ndim() != 2 || array_ref.shape(1) != 3) {
                throw std::runtime_error("Input array must be of shape (n, 3)");
            }
            if (array_ref.shape(0) == 0) {
                throw std::runtime_error("No constraint lines provided.");
            }
            for (ssize_t i = 0; i < array_ref.shape(0); ++i) {
                std::vector<double> coeffs;
                for (ssize_t j = 0; j < array_ref.shape(1); ++j) {
                    coeffs.push_back(array_ref(i, j));
                }
                double a = coeffs[0];
                double b = coeffs[1];
                double c = coeffs[2];
                // ax + by + c <= 0 (2 decimal place precision)
                lines.emplace_back(Line(a * 100, b * 100, c * 100));
            }
        
            Nef_polyhedron intersection_poly = intersect_halfspaces(lines);
            auto poly_vertices = explore(intersection_poly);
            auto simplified_vertices = simplify_polygon(poly_vertices);
            return count_edges(simplified_vertices);
        }

private:
        int sceneSize;
        double costThreshold;
        Logger* logger;
    
        void log(const std::string& msg) const {
            if (logger) logger->log(msg);
        }
    
        Nef_polyhedron intersect_halfspaces(const std::vector<Line>& lines) {
            Nef_polyhedron result(Nef_polyhedron::COMPLETE);
            
            // std::ostringstream ss;
            for (size_t i = 0; i < lines.size(); ++i) {
                const Line& l = lines[i];
        
                // Log the line equation: ax + by + c >= 0
                // ss << "Half-space " << i << ": " << l.a() << "x + " << l.b() << "y + " << l.c() << " >= 0 | ";

                Nef_polyhedron hs(l, Nef_polyhedron::INCLUDED);
                result *= hs;
        
                if (result.is_empty()) {
                    break;
                }
            }

            // log(ss.str());
            return result;
        }

        std::vector<std::pair<double, double>> explore(const Nef_polyhedron& poly) {
            log("Explore Intersection Polygon: ");
    
            std::vector<std::pair<double, double>> poly_vertices;
            std::ostringstream oss;
    
            Explorer explorer = poly.explorer();
            int i = 0;
            for(Face_const_iterator fit = explorer.faces_begin(); fit != explorer.faces_end(); ++fit, i++){
                if (!explorer.mark(fit)) {
                    oss << "Face " << i << " is not marked" << std::endl;
                    continue;
                }

                oss << "Face " << i << " is marked" << std::endl;
                // explore the outer face cycle if it exists
                Halfedge_around_face_const_circulator hafc = explorer.face_cycle(fit);
                if(hafc == Halfedge_around_face_const_circulator()){
                oss << "* has no outer face cycle" << std::endl;
                } else {
                oss << "* outer face cycle" << std::endl;
                oss << "  - halfedges around the face: " << std::endl;
                Halfedge_around_face_const_circulator done(hafc);
                do {
                    char c = (explorer.is_frame_edge(hafc))?'f':'e';
                    oss << c;
                    ++hafc;
                }while (hafc != done);
                oss << " ( f = frame edge, e = ordinary edge)" << std::endl;

                oss << "  - vertices around the face: " << std::endl;
                do {
                    Vertex_const_handle vh = explorer.target(hafc);
                    if (explorer.is_standard(vh)){
                    Point vertex = explorer.point(vh);
                    double vx = CGAL::to_double(vertex.x());
                    double vy = CGAL::to_double(vertex.y());

                    oss << "      Point: " << "(" << vx << "," << vy << ")" << std::endl;
                    
                    poly_vertices.emplace_back(vx,vy);
                    }else{
                    Ray ray = explorer.ray(vh);
                    Point source_point = ray.source();
                    double sx = CGAL::to_double(source_point.x());
                    double sy = CGAL::to_double(source_point.y());

                    Point direction = ray.point(1);
                    double dx = CGAL::to_double(direction.x()); // x-component of direction
                    double dy = CGAL::to_double(direction.y()); // y-component of direction
                    oss << "      Ray: " << "(" << sx << "," << sy << ")" << "->" << "(" << dx << "," << dy << ")" << std::endl;
                    
                    double delX = dx - sx;
                    double delY = dy - sy;
                    double vx, vy;
                    double tol = 1e-3;
                    if (std::fabs(delX) > tol) {
                        double m = delY / delX;
                        double c = sy - m*sx;
                        if (delX > 0){
                            vx = this->sceneSize/2;
                            vy = m*vx + c;
                        }else {
                            vx = -this->sceneSize/2;
                            vy = m*vx + c;
                        }               
                    }else{
                        vx = sx;
                        if (delY > 0) { 
                            vy = this->sceneSize/2;
                        }else {
                            vy = -this->sceneSize/2;
                        } 
                    }
                    poly_vertices.emplace_back(vx,vy);
                    }
                    ++hafc;
                }while (hafc != done);
                }

                // explore the holes if the face has holes
                Hole_const_iterator hit = explorer.holes_begin(fit), end = explorer.holes_end(fit);
                if(hit == end){
                oss << "* has no holes" << std::endl;
                }else{
                oss << "* has holes" << std::endl;
                for(; hit != end; hit++){
                    Halfedge_around_face_const_circulator hafc(hit), done(hit);
                    oss << "  - halfedges around the hole: " << std::endl;
                    do {
                    char c = (explorer.is_frame_edge(hafc))?'f':'e';
                    oss << c;
                    ++hafc;
                    }while (hafc != done);
                    oss << " ( f = frame edge, e = ordinary edge)" << std::endl;
                }
                }
            }
            oss << "done";
            log(oss.str());
            return poly_vertices;
        }

        std::vector<std::pair<double, double>> simplify_polygon(const std::vector<std::pair<double, double>>& poly_vertices) {
            if (poly_vertices.empty()) {
                log("[WARN] simplify_polygon: Received empty polygon vertices.");
            }
            
            // Convert to CGAL::Polygon_2
            Polygon_2 polygon;

            for (const auto& point : poly_vertices) {
                polygon.push_back(K::Point_2(point.first, point.second));
            }
    
            if (polygon.size() < 3) {
                log("[WARN] simplify_polygon: Not enough vertices to form a polygon.");
                return poly_vertices;
            }
    
            // Simplify the polygon
            Cost cost;
            polygon = PS::simplify(polygon, cost, Stop(this->costThreshold));

            std::vector<std::pair<double, double>> simplified_vertices;

            log("Simplified Polygon Vertices: ");
            std::ostringstream oss;
            for (auto it = polygon.vertices_begin(); it != polygon.vertices_end(); ++it) {
                double x = it->x();
                double y = it->y();
                simplified_vertices.push_back({x, y});
                oss << "x: " << x << ", y: " << y << std::endl;
            }
            log(oss.str());
            if (!simplified_vertices.empty()) {
                simplified_vertices.push_back({simplified_vertices[0].first, simplified_vertices[0].second}); // Close the polygon
            }
            return simplified_vertices;
        }

        int count_edges(const std::vector<std::pair<double,double>>& simplified_vertices) {
            int support_size = 0;
        
            for (size_t i = 0; i < simplified_vertices.size() - 1; ++i) {
                size_t j = i + 1;
    
                double x1 = simplified_vertices[i].first;
                double y1 = simplified_vertices[i].second;
                double x2 = simplified_vertices[j].first;
                double y2 = simplified_vertices[j].second;
    
                if (std::abs(std::abs(x1) - std::abs(x2)) < 1e-5 &&
                    std::abs(std::abs(x1) - this->sceneSize/2) < 1e-5) {
                    continue; // Skip vertical lines at the edges
                }
                if (std::abs(std::abs(y1) - std::abs(y2)) < 1e-5 &&
                    std::abs(std::abs(y1) - this->sceneSize/2) < 1e-5) {
                    continue; // Skip horizontal lines at the edges
                }
                if (std::abs(x1) >= this->sceneSize/2 &&
                    std::abs(y1) >= this->sceneSize/2 &&
                    std::abs(x2) >= this->sceneSize/2 &&
                    std::abs(y2) >= this->sceneSize/2) {
                    continue; // Skip edges outside the scene
                }
                support_size += 1;
            }
            log("Calculated Support Size: " + std::to_string(support_size));
            return support_size;
        }
};

// create python binding
PYBIND11_MODULE(halfplane_module, m) {
    py::class_<Logger>(m, "Logger")
        .def(py::init<const std::string&, bool>())
        .def("log", &Logger::log)
        .def("enable", &Logger::enable);

    py::class_<HalfplaneIntersectionProcessor>(m, "HalfplaneIntersectionProcessor")
        .def(py::init<int, double>())
        .def("get_support_size", &HalfplaneIntersectionProcessor::get_support_size)
        .def("set_logger", &HalfplaneIntersectionProcessor::set_logger, py::return_value_policy::reference);
}