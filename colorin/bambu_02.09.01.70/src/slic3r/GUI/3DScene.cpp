#include <GL/glew.h>

#if ENABLE_SMOOTH_NORMALS
#include <igl/per_face_normals.h>
#include <igl/per_corner_normals.h>
#include <igl/per_vertex_normals.h>
#endif // ENABLE_SMOOTH_NORMALS

#include "3DScene.hpp"
#include "GLShader.hpp"
#include "GUI_App.hpp"
#include "GUI_Colors.hpp"
#include "Camera.hpp"

#include "Plater.hpp"
#include "BitmapCache.hpp"

#include "libslic3r/BuildVolume.hpp"
#include "libslic3r/ExtrusionEntity.hpp"
#include "libslic3r/ExtrusionEntityCollection.hpp"
#include "libslic3r/Geometry.hpp"
#include "libslic3r/Print.hpp"
#include "libslic3r/SLAPrint.hpp"
#include "libslic3r/Slicing.hpp"
#include "libslic3r/Format/STL.hpp"
#include "libslic3r/Utils.hpp"
#include "libslic3r/AppConfig.hpp"
#include "libslic3r/PresetBundle.hpp"
#include "libslic3r/ClipperUtils.hpp"
#include "libslic3r/Tesselate.hpp"
#include "libslic3r/PrintConfig.hpp"
#include "libslic3r/QuadricEdgeCollapse.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "libslic3r/Log.hpp"
#include "libslic3r/MixedFilament.hpp"
#include "libslic3r/ImageMap/FacetRasterizer.hpp"
#include "libslic3r/ImageMap/ContinuousColorSolver.hpp"
#include "libslic3r/ImageMap/SimplePmCalibration.hpp"
#include "libslic3r/ImageMap/Sampling.hpp"
#include "MixedColorMatchHelpers.hpp"
#include "GLModel.hpp"
#include "GLTexture.hpp"
#include "GLCanvas3D.hpp"
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <cmath>
#include <functional>
#include <tuple>
#include <unordered_map>
#include <optional>
#include <boost/log/trivial.hpp>
#include <map>
#include <limits>

#include <boost/filesystem/operations.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include <Eigen/Dense>

#ifdef HAS_GLSAFE
void glAssertRecentCallImpl(const char* file_name, unsigned int line, const char* function_name)
{
#if defined(NDEBUG)
    // In release mode, only show OpenGL errors if sufficiently high loglevel.
    if (Slic3r::get_logging_level() < 5)
        return;
#endif // NDEBUG

    GLenum err = glGetError();
    if (err == GL_NO_ERROR)
        return;
    const char* sErr = 0;
    switch (err) {
    case GL_INVALID_ENUM:       sErr = "Invalid Enum";      break;
    case GL_INVALID_VALUE:      sErr = "Invalid Value";     break;
    // be aware that GL_INVALID_OPERATION is generated if glGetError is executed between the execution of glBegin and the corresponding execution of glEnd
    case GL_INVALID_OPERATION:  sErr = "Invalid Operation"; break;
    case GL_STACK_OVERFLOW:     sErr = "Stack Overflow";    break;
    case GL_STACK_UNDERFLOW:    sErr = "Stack Underflow";   break;
    case GL_OUT_OF_MEMORY:      sErr = "Out Of Memory";     break;
    default:                    sErr = "Unknown";           break;
    }
    BOOST_LOG_TRIVIAL(error) << "OpenGL error in " << file_name << ":" << line << ", function " << function_name << "() : " << (int)err << " - " << sErr;
    assert(false);
}
#endif // HAS_GLSAFE

float FullyTransparentMaterialThreshold  = 0.1f;
float FullTransparentModdifiedToFixAlpha = 0.3f;
const float BlackThreshold               = 0.2f;
std::array<float, 4> adjust_color_for_rendering(const std::array<float, 4> &colors)
{
   if (colors[3] < FullyTransparentMaterialThreshold) { // completely transparent
                std::array<float, 4> new_color;
                new_color[0] = 1;
                new_color[1] = 1;
                new_color[2] = 1;
                new_color[3] = FullTransparentModdifiedToFixAlpha;
                return new_color;
    }
    else if ((colors[0] < BlackThreshold) && (colors[1] < BlackThreshold) && (colors[2] < BlackThreshold)) { // black filament deal
        std::array<float, 4> new_color;
        new_color[0] = BlackThreshold;
        new_color[1] = BlackThreshold;
        new_color[2] = BlackThreshold;
        new_color[3] = colors[3];
        return new_color;
    }
    return colors;
}

namespace Slic3r {

static std::map<const TriangleMesh*, std::set<GLVolume*>> g_mesh_volumes_map;

#if ENABLE_SMOOTH_NORMALS
static void smooth_normals_corner(TriangleMesh& mesh, std::vector<stl_normal>& normals)
{
    using MapMatrixXfUnaligned = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::DontAlign>>;
    using MapMatrixXiUnaligned = Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::DontAlign>>;

    std::vector<Vec3f> face_normals = its_face_normals(mesh.its);

    Eigen::MatrixXd vertices = MapMatrixXfUnaligned(mesh.its.vertices.front().data(),
        Eigen::Index(mesh.its.vertices.size()), 3).cast<double>();
    Eigen::MatrixXi indices = MapMatrixXiUnaligned(mesh.its.indices.front().data(),
        Eigen::Index(mesh.its.indices.size()), 3);
    Eigen::MatrixXd in_normals = MapMatrixXfUnaligned(face_normals.front().data(),
        Eigen::Index(face_normals.size()), 3).cast<double>();
    Eigen::MatrixXd out_normals;

    igl::per_corner_normals(vertices, indices, in_normals, 1.0, out_normals);

    normals = std::vector<stl_normal>(mesh.its.vertices.size());
    for (size_t i = 0; i < mesh.its.indices.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) {
            normals[mesh.its.indices[i][j]] = out_normals.row(i * 3 + j).cast<float>();
        }
    }
}

static void smooth_normals_vertex(TriangleMesh& mesh, std::vector<stl_normal>& normals)
{
    using MapMatrixXfUnaligned = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::DontAlign>>;
    using MapMatrixXiUnaligned = Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::DontAlign>>;

    Eigen::MatrixXd vertices = MapMatrixXfUnaligned(mesh.its.vertices.front().data(),
        Eigen::Index(mesh.its.vertices.size()), 3).cast<double>();
    Eigen::MatrixXi indices = MapMatrixXiUnaligned(mesh.its.indices.front().data(),
        Eigen::Index(mesh.its.indices.size()), 3);
    Eigen::MatrixXd out_normals;

//    igl::per_vertex_normals(vertices, indices, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_UNIFORM, out_normals);
//    igl::per_vertex_normals(vertices, indices, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA, out_normals);
    igl::per_vertex_normals(vertices, indices, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, out_normals);
//    igl::per_vertex_normals(vertices, indices, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_DEFAULT, out_normals);

    normals = std::vector<stl_normal>(mesh.its.vertices.size());
    for (size_t i = 0; i < static_cast<size_t>(out_normals.rows()); ++i) {
        normals[i] = out_normals.row(i).cast<float>();
    }
}
#endif // ENABLE_SMOOTH_NORMALS

#if ENABLE_SMOOTH_NORMALS
void GLIndexedVertexArray::load_mesh_full_shading(const TriangleMesh& mesh, bool smooth_normals)
#else
void GLIndexedVertexArray::load_mesh_full_shading(const TriangleMesh& mesh)
#endif // ENABLE_SMOOTH_NORMALS
{
    assert(triangle_indices.empty() && vertices_and_normals_interleaved_size == 0);
    assert(quad_indices.empty() && triangle_indices_size == 0);
    assert(vertices_and_normals_interleaved.size() % 6 == 0 && quad_indices_size == vertices_and_normals_interleaved.size());

#if ENABLE_SMOOTH_NORMALS
    if (smooth_normals) {
        TriangleMesh new_mesh(mesh);
        std::vector<stl_normal> normals;
        smooth_normals_corner(new_mesh, normals);
//        smooth_normals_vertex(new_mesh, normals);

        this->vertices_and_normals_interleaved.reserve(this->vertices_and_normals_interleaved.size() + 3 * 2 * new_mesh.its.vertices.size());
        for (size_t i = 0; i < new_mesh.its.vertices.size(); ++i) {
            const stl_vertex& v = new_mesh.its.vertices[i];
            const stl_normal& n = normals[i];
            this->push_geometry(v(0), v(1), v(2), n(0), n(1), n(2));
        }

        for (size_t i = 0; i < new_mesh.its.indices.size(); ++i) {
            const stl_triangle_vertex_indices& idx = new_mesh.its.indices[i];
            this->push_triangle(idx(0), idx(1), idx(2));
        }
    }
    else {
#endif // ENABLE_SMOOTH_NORMALS
        this->load_its_flat_shading(mesh.its);
#if ENABLE_SMOOTH_NORMALS
    }
#endif // ENABLE_SMOOTH_NORMALS
}

void GLIndexedVertexArray::load_its_flat_shading(const indexed_triangle_set &its)
{
    this->vertices_and_normals_interleaved.reserve(this->vertices_and_normals_interleaved.size() + 3 * 3 * 2 * its.indices.size());
    unsigned int vertices_count = 0;
    for (int i = 0; i < int(its.indices.size()); ++ i) {
        stl_triangle_vertex_indices face        = its.indices[i];
        stl_vertex                  vertex[3]   = { its.vertices[face[0]], its.vertices[face[1]], its.vertices[face[2]] };
        stl_vertex                  n           = face_normal_normalized(vertex);
        for (int j = 0; j < 3; ++j)
            this->push_geometry(vertex[j](0), vertex[j](1), vertex[j](2), n(0), n(1), n(2));
        this->push_triangle(vertices_count, vertices_count + 1, vertices_count + 2);
        vertices_count += 3;
    }
    BOOST_LOG_TRIVIAL(debug) << boost::format("this %1%, indices size %2%, vertices %3%, triangles %4% ")
            %this %its.indices.size() %this->vertices_and_normals_interleaved.size() %this->triangle_indices.size() ;
}

void GLIndexedVertexArray::finalize_geometry(bool opengl_initialized)
{
    assert(this->vertices_and_normals_interleaved_VBO_id == 0);
    assert(this->triangle_indices_VBO_id == 0);
    assert(this->quad_indices_VBO_id == 0);

	if (! opengl_initialized) {
		// Shrink the data vectors to conserve memory in case the data cannot be transfered to the OpenGL driver yet.
		this->shrink_to_fit();
		return;
	}

    BOOST_LOG_TRIVIAL(debug) << boost::format("this %1% ") %this;
    if (! this->vertices_and_normals_interleaved.empty()) {
        glsafe(::glGenBuffers(1, &this->vertices_and_normals_interleaved_VBO_id));
        glsafe(::glBindBuffer(GL_ARRAY_BUFFER, this->vertices_and_normals_interleaved_VBO_id));
        glsafe(::glBufferData(GL_ARRAY_BUFFER, this->vertices_and_normals_interleaved.size() * 4, this->vertices_and_normals_interleaved.data(), GL_STATIC_DRAW));
        glsafe(::glBindBuffer(GL_ARRAY_BUFFER, 0));
        this->vertices_and_normals_interleaved.clear();
    }
    if (! this->triangle_indices.empty()) {
        glsafe(::glGenBuffers(1, &this->triangle_indices_VBO_id));
        glsafe(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->triangle_indices_VBO_id));
        glsafe(::glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->triangle_indices.size() * 4, this->triangle_indices.data(), GL_STATIC_DRAW));
        glsafe(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
        this->triangle_indices.clear();
    }
    if (! this->quad_indices.empty()) {
        glsafe(::glGenBuffers(1, &this->quad_indices_VBO_id));
        glsafe(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->quad_indices_VBO_id));
        glsafe(::glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->quad_indices.size() * 4, this->quad_indices.data(), GL_STATIC_DRAW));
        glsafe(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
        this->quad_indices.clear();
    }
}

void GLIndexedVertexArray::release_geometry()
{
    if (this->vertices_and_normals_interleaved_VBO_id) {
        glsafe(::glDeleteBuffers(1, &this->vertices_and_normals_interleaved_VBO_id));
        this->vertices_and_normals_interleaved_VBO_id = 0;
    }
    if (this->triangle_indices_VBO_id) {
        glsafe(::glDeleteBuffers(1, &this->triangle_indices_VBO_id));
        this->triangle_indices_VBO_id = 0;
    }
    if (this->quad_indices_VBO_id) {
        glsafe(::glDeleteBuffers(1, &this->quad_indices_VBO_id));
        this->quad_indices_VBO_id = 0;
    }
    this->clear();
}

void GLIndexedVertexArray::render(const std::shared_ptr<GLShaderProgram>& shader) const
{
    assert(this->vertices_and_normals_interleaved_VBO_id != 0);
    assert(this->triangle_indices_VBO_id != 0 || this->quad_indices_VBO_id != 0);

    glsafe(::glBindBuffer(GL_ARRAY_BUFFER, this->vertices_and_normals_interleaved_VBO_id));

    int position_id = -1;
    int normal_id = -1;

    position_id = shader->get_attrib_location("v_position");
    if (position_id != -1) {
        glsafe(::glVertexAttribPointer(position_id, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (const void*)(3 * sizeof(float))));
        glsafe(::glEnableVertexAttribArray(position_id));
    }
    normal_id = shader->get_attrib_location("v_normal");
    if (normal_id != -1) {
        glsafe(::glVertexAttribPointer(normal_id, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr));
        glsafe(::glEnableVertexAttribArray(normal_id));
    }

    // Render using the Vertex Buffer Objects.
    if (this->triangle_indices_size > 0) {
        glsafe(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->triangle_indices_VBO_id));
        glsafe(::glDrawElements(GL_TRIANGLES, GLsizei(this->triangle_indices_size), GL_UNSIGNED_INT, nullptr));
        glsafe(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
    }
    if (this->quad_indices_size > 0) {
        glsafe(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->quad_indices_VBO_id));
        glsafe(::glDrawElements(GL_QUADS, GLsizei(this->quad_indices_size), GL_UNSIGNED_INT, nullptr));
        glsafe(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
    }

    if (position_id != -1) {
        glsafe(::glDisableVertexAttribArray(position_id));
    }
    if (normal_id != -1) {
        glsafe(::glDisableVertexAttribArray(normal_id));
    }

    glsafe(::glBindBuffer(GL_ARRAY_BUFFER, 0));
}

void GLIndexedVertexArray::render(
    const std::shared_ptr<GLShaderProgram>& shader,
    const std::pair<size_t, size_t>& tverts_range,
    const std::pair<size_t, size_t>& qverts_range) const
{
    if (0 == vertices_and_normals_interleaved_VBO_id) {
        if (!vertices_and_normals_interleaved.empty()) {
            BOOST_LOG_TRIVIAL(info) << boost::format("finalize_geometry");
            const_cast<GLIndexedVertexArray*>(this)->finalize_geometry(true);
        }
    }
    if (0 == vertices_and_normals_interleaved_VBO_id) {
        return;
    }

    if (0 == triangle_indices_VBO_id && 0 == quad_indices_VBO_id) {
        return;
    }

    // Render using the Vertex Buffer Objects.
    glsafe(::glBindBuffer(GL_ARRAY_BUFFER, this->vertices_and_normals_interleaved_VBO_id));
    int position_id = -1;
    int normal_id = -1;

    position_id = shader->get_attrib_location("v_position");
    if (position_id != -1) {
        glsafe(::glVertexAttribPointer(position_id, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (const void*)(3 * sizeof(float))));
        glsafe(::glEnableVertexAttribArray(position_id));
    }
    normal_id = shader->get_attrib_location("v_normal");
    if (normal_id != -1) {
        glsafe(::glVertexAttribPointer(normal_id, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr));
        glsafe(::glEnableVertexAttribArray(normal_id));
    }

    if (this->triangle_indices_size > 0) {
        glsafe(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->triangle_indices_VBO_id));
        glsafe(::glDrawElements(GL_TRIANGLES, GLsizei(std::min(this->triangle_indices_size, tverts_range.second - tverts_range.first)), GL_UNSIGNED_INT, (const void*)(tverts_range.first * 4)));
        glsafe(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
    }
    if (this->quad_indices_size > 0) {
        glsafe(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->quad_indices_VBO_id));
        glsafe(::glDrawElements(GL_QUADS, GLsizei(std::min(this->quad_indices_size, qverts_range.second - qverts_range.first)), GL_UNSIGNED_INT, (const void*)(qverts_range.first * 4)));
        glsafe(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
    }

    if (position_id != -1) {
        glsafe(::glDisableVertexAttribArray(position_id));
    }
    if (normal_id != -1) {
        glsafe(::glDisableVertexAttribArray(normal_id));
    }

    glsafe(::glBindBuffer(GL_ARRAY_BUFFER, 0));
}

const float GLVolume::SinkingContours::HalfWidth = 0.25f;

void GLVolume::SinkingContours::render(const GUI::Camera &camera, Model &model)
{
    const auto& shader = GUI::wxGetApp().get_shader("flat");
    if (!shader) {
        return;
    }

    GUI::wxGetApp().bind_shader(shader);

    update(model);
    shader->set_uniform("view_model_matrix", camera.get_view_matrix() * Geometry::assemble_transform(m_shift));
    shader->set_uniform("projection_matrix", camera.get_projection_matrix());
    m_model.render_geometry();

    GUI::wxGetApp().unbind_shader();
}

void GLVolume::SinkingContours::update(Model &model)
{
    int object_idx = m_parent.object_idx();
    if (0 <= object_idx && object_idx < (int)model.objects.size() && m_parent.is_sinking() && !m_parent.is_below_printbed()) {
        const BoundingBoxf3& box = m_parent.transformed_convex_hull_bounding_box();
        if (!m_old_box.size().isApprox(box.size()) || m_old_box.min.z() != box.min.z()) {
            m_old_box = box;
            m_shift = Vec3d::Zero();

            const TriangleMesh& mesh = model.objects[object_idx]->volumes[m_parent.volume_idx()]->mesh();

            m_model.reset();
            GUI::GLModel::InitializationData init_data;
            MeshSlicingParams slicing_params;
            slicing_params.trafo = m_parent.world_matrix();
            Polygons polygons = union_(slice_mesh(mesh.its, 0.0f, slicing_params));
            for (ExPolygon &expoly : diff_ex(expand(polygons, float(scale_(HalfWidth))), shrink(polygons, float(scale_(HalfWidth))))) {
                GUI::GLModel::InitializationData::Entity entity;
                entity.type = GUI::GLModel::PrimitiveType::Triangles;
                const std::vector<Vec3d> triangulation = triangulate_expolygon_3d(expoly);
                for (const Vec3d& v : triangulation) {
                    entity.positions.emplace_back(v.cast<float>() + Vec3f(0.0f, 0.0f, 0.015f)); // add a small positive z to avoid z-fighting
                    entity.normals.emplace_back(Vec3f::UnitZ());
                    const size_t positions_count = entity.positions.size();
                    if (positions_count % 3 == 0) {
                        entity.indices.emplace_back(positions_count - 3);
                        entity.indices.emplace_back(positions_count - 2);
                        entity.indices.emplace_back(positions_count - 1);
                    }
                }
                init_data.entities.emplace_back(entity);
            }

            m_model.init_from(init_data);
        }
        else
            m_shift = box.center() - m_old_box.center();
    }
    else
        m_model.reset();
}

std::array<float, 4> GLVolume::DISABLED_COLOR    = { 0.25f, 0.25f, 0.25f, 1.0f };
std::array<float, 4> GLVolume::SLA_SUPPORT_COLOR = { 0.75f, 0.75f, 0.75f, 1.0f };
std::array<float, 4> GLVolume::SLA_PAD_COLOR     = { 0.0f, 0.2f, 0.0f, 1.0f };
// BBS
std::array<float, 4> GLVolume::NEUTRAL_COLOR     = { 0.8f, 0.8f, 0.8f, 1.0f };
std::array<float, 4> GLVolume::UNPRINTABLE_COLOR = { 0.0f, 0.0f, 0.0f, 0.5f };

std::array<float, 4> GLVolume::MODEL_MIDIFIER_COL   = {1.0f, 1.0f, 0.0f, 0.6f};
std::array<float, 4> GLVolume::MODEL_NEGTIVE_COL    = {0.3f, 0.3f, 0.3f, 0.4f};
std::array<float, 4> GLVolume::SUPPORT_ENFORCER_COL = {0.3f, 0.3f, 1.0f, 0.4f};
std::array<float, 4> GLVolume::SUPPORT_BLOCKER_COL  = {1.0f, 0.3f, 0.3f, 0.4f};

std::array<float, 4> GLVolume::MODEL_HIDDEN_COL  = {0.f, 0.f, 0.f, 0.3f};

std::array<std::array<float, 4>, 5> GLVolume::MODEL_COLOR = { {
    { 1.0f, 1.0f, 0.0f, 1.f },
    { 1.0f, 0.5f, 0.5f, 1.f },
    { 0.5f, 1.0f, 0.5f, 1.f },
    { 0.5f, 0.5f, 1.0f, 1.f },
    { 1.0f, 1.0f, 0.0f, 1.f }
} };

float GLVolume::LOD_HIGH_ZOOM          = 3.5f;
float GLVolume::LOD_MIDDLE_ZOOM        = 2.8f;
float GLVolume::LOD_SMALL_ZOOM         = 1.4f;
float GLVolume::LAST_CAMERA_ZOOM_VALUE = 0.0f;
const float ZOOM_THRESHOLD                   = 0.3f;
const unsigned char LOD_UPDATE_FREQUENCY     = 20;
const Vec2i LOD_SCREEN_MIN                   = Vec2i(45, 35);
const Vec2i LOD_SCREEN_MAX                   = Vec2i(65, 55);

Vec2f calc_pt_in_screen(const Vec3d &pt,  const Matrix4d &view_proj_mat, int window_width, int window_height)
{
    auto tran = view_proj_mat;
    Vec4d temp_center(pt.x(), pt.y(), pt.z(), 1.0);
    Vec4d temp_ndc          = tran * temp_center;
    Vec3d screen_box_center = Vec3d(temp_ndc.x(), temp_ndc.y(), temp_ndc.z()) / temp_ndc.w();

    float x = 0.5f * (1 + screen_box_center(0)) * window_width;
    float y = 0.5f * (1 - screen_box_center(1)) * window_height;
    return Vec2f(x, y);
}

LOD_LEVEL calc_volume_box_in_screen_bigger_than_threshold(const BoundingBoxf3 &v_box_in_world,  const Matrix4d &view_proj_mat,
          int window_width, int window_height)
{
    auto & _min = v_box_in_world.min;
    auto & _max = v_box_in_world.max;
    std::array<Vec3d,8>  src_vertices;
    src_vertices[0] = _min;
    src_vertices[1] = Vec3d(_max.x(), _min.y(), _min.z());
    src_vertices[2] = Vec3d(_max.x(), _max.y(), _min.z());
    src_vertices[3] = Vec3d(_min.x(), _max.y(), _min.z());
    src_vertices[4] = Vec3d(_min.x(), _min.y(), _max.z());
    src_vertices[5] = Vec3d(_max.x(), _min.y(), _max.z());
    src_vertices[6] = _max;
    src_vertices[7] = Vec3d(_min.x(), _max.y(), _max.z());
    BoundingBoxf box2d;
    for (int i = 0; i < src_vertices.size(); i++) {
        box2d.merge(calc_pt_in_screen(src_vertices[i], view_proj_mat, window_width, window_height).cast<double>());
    }
    auto size_x = box2d.size().x();
    auto size_y = box2d.size().y();
   if (size_x >= LOD_SCREEN_MAX.x() || size_y >= LOD_SCREEN_MAX.y()) {
       return LOD_LEVEL::HIGH;
   }
   if (size_x <= LOD_SCREEN_MIN.x() && size_y <= LOD_SCREEN_MIN.y()) {
       return LOD_LEVEL::SMALL;
   } else {
       return LOD_LEVEL::MIDDLE;
   }
}

void GLVolume::update_render_colors()
{
    GLVolume::DISABLED_COLOR    = GLColor(RenderColor::colors[RenderCol_Model_Disable]);
    GLVolume::NEUTRAL_COLOR     = GLColor(RenderColor::colors[RenderCol_Model_Neutral]);
    GLVolume::MODEL_COLOR[0]    = GLColor(RenderColor::colors[RenderCol_Modifier]);
    GLVolume::MODEL_COLOR[1]    = GLColor(RenderColor::colors[RenderCol_Negtive_Volume]);
    GLVolume::MODEL_COLOR[2]    = GLColor(RenderColor::colors[RenderCol_Support_Enforcer]);
    GLVolume::MODEL_COLOR[3]    = GLColor(RenderColor::colors[RenderCol_Support_Blocker]);
    GLVolume::UNPRINTABLE_COLOR = GLColor(RenderColor::colors[RenderCol_Model_Unprintable]);

}

void GLVolume::load_render_colors()
{
    RenderColor::colors[RenderCol_Model_Disable]    = IMColor(GLVolume::DISABLED_COLOR);
    RenderColor::colors[RenderCol_Model_Neutral]    = IMColor(GLVolume::NEUTRAL_COLOR);
    RenderColor::colors[RenderCol_Modifier]         = IMColor(GLVolume::MODEL_COLOR[0]);
    RenderColor::colors[RenderCol_Negtive_Volume]   = IMColor(GLVolume::MODEL_COLOR[1]);
    RenderColor::colors[RenderCol_Support_Enforcer] = IMColor(GLVolume::MODEL_COLOR[2]);
    RenderColor::colors[RenderCol_Support_Blocker]  = IMColor(GLVolume::MODEL_COLOR[3]);
    RenderColor::colors[RenderCol_Model_Unprintable]= IMColor(GLVolume::UNPRINTABLE_COLOR);
}

GLVolume::GLVolume(float r, float g, float b, float a, bool create_index_data)
    : m_sla_shift_z(0.0)
    , m_sinking_contours(*this)
    // geometry_id == 0 -> invalid
    , geometry_id(std::pair<size_t, size_t>(0, 0))
    , extruder_id(0)
    , selected(false)
    , disabled(false)
    , printable(true)
    , is_active(true)
    , zoom_to_volumes(true)
    , shader_outside_printer_detection_enabled(false)
    , is_outside(false)
    , partly_inside(false)
    , hover(HS_None)
    , is_modifier(false)
    , slice_error(false)
    , is_wipe_tower(false)
    , is_extrusion_path(false)
    , force_transparent(false)
    , force_native_color(false)
    , force_neutral_color(false)
    , force_sinking_contours(false)
    , picking(false)
    , tverts_range(0, size_t(-1))
    , qverts_range(0, size_t(-1))
    , tverts_range_lod(0, size_t(-1))
    , qverts_range_lod(0, size_t(-1))
{
    color = { r, g, b, a };
    set_render_color(color);
    mmuseg_ts = 0;
    if (create_index_data)
        indexed_vertex_array = std::make_shared<GLIndexedVertexArray>();
}

void GLVolume::set_color(const std::array<float, 4>& rgba)
{
    color = rgba;
}

// BBS
float GLVolume::explosion_ratio = 1.0;
float GLVolume::last_explosion_ratio = 1.0;

void GLVolume::set_render_color(float r, float g, float b, float a)
{
    render_color = { r, g, b, a };
}

void GLVolume::set_render_color(const std::array<float, 4>& rgba)
{
    render_color = rgba;
}

void GLVolume::set_render_color()
{
    bool outside = is_outside || is_below_printbed();

    if (force_native_color || force_neutral_color) {
#ifdef ENABBLE_OUTSIDE_COLOR
        if (outside && shader_outside_printer_detection_enabled)
            set_render_color(OUTSIDE_COLOR);
        else {
#endif
            if (force_native_color)
                set_render_color(color);
            else
                set_render_color(NEUTRAL_COLOR);
#ifdef ENABLE_OUTSIDE_COLOR
        }
#endif
    }
    else {
        /* BBS
        if (hover == HS_Select)
            set_render_color(HOVER_SELECT_COLOR);
        else if (hover == HS_Deselect)
            set_render_color(HOVER_DESELECT_COLOR);
        else if (selected)
            set_render_color(outside ? SELECTED_OUTSIDE_COLOR : SELECTED_COLOR);
        else if (disabled)
        */
        if (disabled)
            set_render_color(DISABLED_COLOR);
#ifdef ENABLE_OUTSIDE_COLOR
        else if (is_outside && shader_outside_printer_detection_enabled)
            set_render_color(OUTSIDE_COLOR);
#endif
        else {
            //to make black not too hard too see
            std::array<float, 4> new_color = adjust_color_for_rendering(color);
            set_render_color(new_color);
        }
    }

    if (force_transparent) {
        if (color[3] < FullyTransparentMaterialThreshold) {
            render_color[3] = FullTransparentModdifiedToFixAlpha;
        } else {
            render_color[3] = color[3];
        }
    }

    //BBS set unprintable color
    if (!printable) {
        render_color[0] = UNPRINTABLE_COLOR[0];
        render_color[1] = UNPRINTABLE_COLOR[1];
        render_color[2] = UNPRINTABLE_COLOR[2];
        render_color[3] = UNPRINTABLE_COLOR[3];
    }

}

std::array<float, 4> color_from_model_volume(const ModelVolume& model_volume)
{
    std::array<float, 4> color = {0.0f, 0.0f, 0.0f, 1.0f};
    if (model_volume.is_negative_volume()) {
        return GLVolume::MODEL_NEGTIVE_COL;
    }
    else if (model_volume.is_modifier()) {
#if ENABLE_MODIFIERS_ALWAYS_TRANSPARENT
        return GLVolume::MODEL_MIDIFIER_COL;
#else
        color[0] = 0.2f;
        color[1] = 1.0f;
        color[2] = 0.2f;
#endif // ENABLE_MODIFIERS_ALWAYS_TRANSPARENT
    }
    else if (model_volume.is_support_blocker()) {
        return GLVolume::SUPPORT_BLOCKER_COL;
    }
    else if (model_volume.is_support_enforcer()) {
        return GLVolume::SUPPORT_ENFORCER_COL;
    }
    return color;
}

bool GLVolume::simplify_mesh(const TriangleMesh &mesh, std::shared_ptr<GLIndexedVertexArray> va, LOD_LEVEL lod) const {
   return simplify_mesh(mesh.its,va,lod);
}
#define SUPER_LARGE_FACES 500000
#define LARGE_FACES 100000
bool GLVolume::simplify_mesh(const indexed_triangle_set &_its, std::shared_ptr<GLIndexedVertexArray> va, LOD_LEVEL lod) const
{
    if (_its.indices.size() == 0 || _its.vertices.size() == 0) { return false; }
    auto its = std::make_unique<indexed_triangle_set>(_its);
    auto m_state = std::make_unique<State>();
    if (lod == LOD_LEVEL::MIDDLE) {
        m_state->config.max_error = 0.5f;
        if (_its.indices.size() > SUPER_LARGE_FACES) {
            m_state->config.max_error = 0.4f;
        } else if (_its.indices.size() > LARGE_FACES) {
            m_state->config.max_error = 0.3f;
        }
    }
    if (lod == LOD_LEVEL::SMALL) {
        m_state->config.max_error = 0.1f;
        if (_its.indices.size() > SUPER_LARGE_FACES) {
            m_state->config.max_error = 0.08f;
        } else if (_its.indices.size() > LARGE_FACES) {
            m_state->config.max_error = 0.05f;
        }
    }

    //std::mutex  m_state_mutex;
    std::thread m_worker = std::thread(
        [va,this](std::unique_ptr<indexed_triangle_set> its, std::unique_ptr<State> state) {
            // Checks that the UI thread did not request cancellation, throws if so.
            std::function<void(void)> throw_on_cancel = []() {
            };
            std::function<void(int)> statusfn = [&state](int percent) {
                state->progress = percent;
            };
            // Initialize.
            uint32_t triangle_count = 0;
            float    max_error      = std::numeric_limits<float>::max();
            {
                if (state->config.use_count)
                    triangle_count = state->config.wanted_count;
                if (!state->config.use_count)
                    max_error = state->config.max_error;
                state->progress = 0;
                state->result.reset();
                state->status = State::Status::running;
            }
            int          init_face_count = its->indices.size();
            TriangleMesh origin_mesh(*its);
            try { // Start the actual calculation.
                its_quadric_edge_collapse(*its, triangle_count, &max_error, throw_on_cancel, statusfn);
            } catch (std::exception&) {
                BOOST_LOG_TRIVIAL(warning) << current_exception_message();
                state->status = State::idle;
            }
            if (state->status == State::Status::running) {
                // We were not cancelled, the result is valid.
                state->status = State::Status::idle;
                state->result = std::move(its);
            }
            if (state->result) {
                int          end_face_count = (*state->result).indices.size();
                if (init_face_count < 200 || (init_face_count < 1000 && end_face_count < init_face_count * 0.5)) {
                    return;
                }
                TriangleMesh mesh(*state->result);
                float        eps        = 1.0f;
                Vec3f        origin_min = origin_mesh.stats().min - Vec3f(eps, eps, eps);
                Vec3f        origin_max = origin_mesh.stats().max + Vec3f(eps, eps, eps);
                if (origin_min.x() < mesh.stats().min.x() && origin_min.y() < mesh.stats().min.y() && origin_min.z() < mesh.stats().min.z()&&
                    origin_max.x() > mesh.stats().max.x() && origin_max.y() > mesh.stats().max.y() && origin_max.z() > mesh.stats().max.z()) {
                    if (va && va.use_count() >= 2) {
                        va->load_mesh(mesh);
                    }
                }
                else {
                    state->status = State::cancelling;
                }
            }
        },
        std::move(its),std::move(m_state));
    if (m_worker.joinable()) {
        m_worker.detach();
    }
    return true;
}

Transform3d GLVolume::world_matrix() const
{
    Transform3d m = m_instance_transformation.get_matrix() * m_volume_transformation.get_matrix();
    Vec3d ofs2ass = m_offset_to_assembly * (GLVolume::explosion_ratio - 1.0);
    Vec3d volofs2obj = m_volume_transformation.get_offset() * (GLVolume::explosion_ratio - 1.0);

    m.translation()(2) += m_sla_shift_z;
    m.translate(ofs2ass + volofs2obj);
    return m;
}

bool GLVolume::is_left_handed() const
{
    // reference <Real-Time Rendering Fourth Edition> page 84
    // link: https://www.realtimerendering.com/
    const auto model_matrix = world_matrix().matrix();
    Eigen::Matrix3d subMatrix = model_matrix.block<3, 3>(0, 0);
    const auto det = subMatrix.determinant();
    if (det < 1e-6f) {
        return true;
    }
    return false;
}

const BoundingBoxf3& GLVolume::transformed_bounding_box() const
{
    if (!m_transformed_bounding_box.has_value() || last_explosion_ratio != explosion_ratio) {
        const BoundingBoxf3& box = bounding_box();
        assert(box.defined || box.min.x() >= box.max.x() || box.min.y() >= box.max.y() || box.min.z() >= box.max.z());
        std::optional<BoundingBoxf3>* trans_box = const_cast<std::optional<BoundingBoxf3>*>(&m_transformed_bounding_box);
        *trans_box = box.transformed(world_matrix());
        last_explosion_ratio = explosion_ratio;
    }
    return *m_transformed_bounding_box;
}

const BoundingBoxf3& GLVolume::transformed_convex_hull_bounding_box() const
{
    if (!m_transformed_convex_hull_bounding_box.has_value()) {
        std::optional<BoundingBoxf3>* trans_box = const_cast<std::optional<BoundingBoxf3>*>(&m_transformed_convex_hull_bounding_box);
        *trans_box = transformed_convex_hull_bounding_box(world_matrix());
    }
    return *m_transformed_convex_hull_bounding_box;
}

BoundingBoxf3 GLVolume::transformed_convex_hull_bounding_box(const Transform3d &trafo) const
{
	return (m_convex_hull && ! m_convex_hull->empty()) ?
		m_convex_hull->transformed_bounding_box(trafo) :
        bounding_box().transformed(trafo);
}

BoundingBoxf3 GLVolume::transformed_non_sinking_bounding_box(const Transform3d& trafo,Model& model) const
{
    if (object_idx() < model.objects.size())
        return model.objects[object_idx()]->volumes[volume_idx()]->mesh().transformed_bounding_box(trafo, 0.0);
    else
        return {};
}

const BoundingBoxf3 &GLVolume::transformed_non_sinking_bounding_box(Model &model) const
{
    if (!m_transformed_non_sinking_bounding_box.has_value()) {
        std::optional<BoundingBoxf3>* trans_box = const_cast<std::optional<BoundingBoxf3>*>(&m_transformed_non_sinking_bounding_box);
        const Transform3d& trafo = world_matrix();
        *trans_box                              = transformed_non_sinking_bounding_box(trafo, model);
    }
    return *m_transformed_non_sinking_bounding_box;
}

void GLVolume::set_range(double min_z, double max_z)
{
    this->qverts_range.first = 0;
    this->qverts_range.second = this->indexed_vertex_array->quad_indices_size;
    this->tverts_range.first = 0;
    this->tverts_range.second = this->indexed_vertex_array->triangle_indices_size;
    if (! this->print_zs.empty()) {
        // The Z layer range is specified.
        // First test whether the Z span of this object is not out of (min_z, max_z) completely.
        if (this->print_zs.front() > max_z || this->print_zs.back() < min_z) {
            this->qverts_range.second = 0;
            this->tverts_range.second = 0;
        } else {
            // Then find the lowest layer to be displayed.
            size_t i = 0;
            for (; i < this->print_zs.size() && this->print_zs[i] < min_z; ++ i);
            if (i == this->print_zs.size()) {
                // This shall not happen.
                this->qverts_range.second = 0;
                this->tverts_range.second = 0;
            } else {
                // Remember start of the layer.
                this->qverts_range.first = this->offsets[i * 2];
                this->tverts_range.first = this->offsets[i * 2 + 1];
                // Some layers are above $min_z. Which?
                for (; i < this->print_zs.size() &&  this->print_zs[i] <= max_z; ++ i);
                if (i < this->print_zs.size()) {
                    this->qverts_range.second = this->offsets[i * 2];
                    this->tverts_range.second = this->offsets[i * 2 + 1];
                }
            }
        }
    }
}

//BBS: add outline related logic
//static unsigned char stencil_data[1284][2944];


// ============================================================================
// Bambu Colorin (Orca FullSpectrum 0.9.18): source-colour preview helpers, copied from Orca
// 3DScene.cpp (SourceColorPreviewJob, start_source_color_preview, ImageMapPreviewPalette).
// Only API differences of Bambu's GLModel were adapted.
// ============================================================================
namespace GUI {
namespace {
constexpr const char* IMAGE_MAP_PREVIEW_PREDICTED_COLORS_KEY = "image_map_preview_predicted_colors";
}
bool image_map_preview_predicted_colors()
{
    return wxGetApp().app_config != nullptr && wxGetApp().app_config->get_bool(IMAGE_MAP_PREVIEW_PREDICTED_COLORS_KEY);
}
} // namespace GUI

namespace {
// Bambu's GLModel::Geometry has no add_vertex(position, normal, colour) for the P3N3C4 layout.
inline void colorin_add_vertex_c4(GUI::GLModel::Geometry& geometry, const Vec3f& position, const Vec3f& normal, const RGBA& color)
{
    geometry.vertices.insert(geometry.vertices.end(), {position.x(), position.y(), position.z(), normal.x(), normal.y(), normal.z(),
                                                       color[0], color[1], color[2], color[3]});
}
} // namespace

enum class SourceColorPreviewStatus : unsigned char { Running, Ready, Uploading, Uploaded, Cancelled, Failed };

struct SourceTexturePreviewPart
{
    std::unique_ptr<GUI::GLModel::Geometry> geometry;
    std::vector<unsigned char>              rgba;
    unsigned int                            width{0};
    unsigned int                            height{0};
    ImageMap::WrapMode                      wrap_u{ImageMap::WrapMode::Repeat};
    ImageMap::WrapMode                      wrap_v{ImageMap::WrapMode::Repeat};
};

struct SourceColorPreviewJob
{
    explicit SourceColorPreviewJob(std::shared_ptr<const ImageMap::VolumeData> source_data) : data(std::move(source_data)) {}

    std::shared_ptr<const ImageMap::VolumeData> data;
    std::atomic<SourceColorPreviewStatus>       status{SourceColorPreviewStatus::Running};
    std::atomic<int>                            progress{1};
    std::atomic<bool>                           cancel{false};
    std::atomic<bool>                           ready_presented{false};
    std::atomic<bool>                           uploaded_presented{false};
    std::mutex                                  result_mutex;
    std::unique_ptr<GUI::GLModel::Geometry>     geometry;
    std::vector<SourceTexturePreviewPart>       texture_parts;
};

namespace {

// Preserve enough facial/texture detail for large scanned OBJs. Rasterization
// is bounded, cancellable and runs off the UI thread, so this can be materially
// higher than the old emergency cap without bringing back transform stalls.
constexpr size_t       k_source_color_preview_triangle_cap = 200'000;
constexpr size_t       k_source_color_preview_cache_cap    = 65'536;
constexpr unsigned int k_source_texture_preview_max_edge   = 2'048;
constexpr size_t       k_source_texture_preview_max_pixels = size_t(2'048) * size_t(2'048);
constexpr unsigned int k_interactive_texture_preview_max_edge   = 2'048;
constexpr size_t       k_interactive_texture_preview_max_pixels = size_t(2) * size_t(1'024) * size_t(1'024);

struct SourceColorPreviewAssignment
{
    uint32_t                                        zone_index{0};
    RGBA                                            target_color{1.f, 1.f, 1.f, 1.f};
    unsigned int                                    filament_id{0};
    RGBA                                            display_color{1.f, 1.f, 1.f, 1.f};
    std::vector<ImageMap::ContinuousColorComponent> components;
};

const SourceColorPreviewAssignment* nearest_source_color_assignment(const std::vector<SourceColorPreviewAssignment>& assignments,
                                                                    const RGBA&                                      color,
                                                                    std::optional<uint32_t> zone_index = std::nullopt)
{
    const SourceColorPreviewAssignment* best_assignment = nullptr;
    double                              best_distance   = std::numeric_limits<double>::infinity();
    for (const SourceColorPreviewAssignment& assignment : assignments) {
        if (zone_index && assignment.zone_index != *zone_index)
            continue;
        double distance = 0.0;
        for (size_t component = 0; component < 3; ++component) {
            const double delta = double(color[component]) - double(assignment.target_color[component]);
            distance += delta * delta;
        }
        if (distance < best_distance) {
            best_distance   = distance;
            best_assignment = &assignment;
        }
    }
    return best_assignment;
}

unsigned int nearest_source_color_filament(const std::vector<SourceColorPreviewAssignment>& assignments,
                                           const RGBA&                                      color,
                                           std::optional<uint32_t>                          zone_index = std::nullopt)
{
    const SourceColorPreviewAssignment* assignment = nearest_source_color_assignment(assignments, color, zone_index);
    return assignment != nullptr ? assignment->filament_id : 0;
}

size_t source_color_preview_signature(const std::vector<ImageMap::ContinuousColorComponent>& components)
{
    size_t seed         = components.size();
    auto   hash_combine = [&seed](size_t value) { seed ^= value + size_t(0x9e3779b9) + (seed << 6) + (seed >> 2); };
    hash_combine(size_t(ImageMap::simple_pm_calibration_revision()));
    for (const ImageMap::ContinuousColorComponent& component : components) {
        hash_combine(std::hash<std::string>{}(component.color_hex));
        hash_combine(component.transmission_distance_mm ? std::hash<double>{}(*component.transmission_distance_mm) : 0u);
        hash_combine(component.material_id ? std::hash<std::string>{}(*component.material_id) : 0u);
    }
    return seed;
}

size_t source_color_preview_zone_signature(const ImageMap::VolumeData& data)
{
    size_t seed = data.zones.size();
    auto hash_combine = [&seed](size_t value) { seed ^= value + size_t(0x9e3779b9) + (seed << 6) + (seed >> 2); };
    auto hash_scalar = [&hash_combine](double value) { hash_combine(std::hash<double>{}(value)); };
    for (const ImageMap::Zone& zone : data.zones) {
        hash_combine(size_t(zone.enabled));
        hash_combine(static_cast<size_t>(zone.render_mode));
        hash_combine(static_cast<size_t>(zone.color_mix_model));
        hash_scalar(zone.modulation_sample_spacing_mm);
        hash_combine(size_t(zone.disable_broad_path_smoothing));
        hash_scalar(zone.gaussian_smoothing_strength);
        hash_scalar(zone.first_path_smoothing_strength);
        hash_scalar(zone.second_path_smoothing_strength);
        hash_scalar(zone.corner_smoothing_radius_mm);
        hash_scalar(zone.tone_gamma);
        hash_scalar(zone.overhang_contrast_percent);
        hash_scalar(zone.image_exposure_ev);
        hash_scalar(zone.image_contrast_percent);
        hash_scalar(zone.image_saturation_percent);
        hash_scalar(zone.image_edge_boost_percent);
    }
    return seed;
}

bool source_color_preview_components_equal(const std::vector<ImageMap::ContinuousColorComponent>& lhs,
                                           const std::vector<ImageMap::ContinuousColorComponent>& rhs)
{
    if (lhs.size() != rhs.size())
        return false;
    for (size_t index = 0; index < lhs.size(); ++index) {
        if (lhs[index].color_hex != rhs[index].color_hex ||
            lhs[index].transmission_distance_mm != rhs[index].transmission_distance_mm ||
            lhs[index].material_id != rhs[index].material_id)
            return false;
    }
    return true;
}

std::shared_ptr<const ImageMap::ContinuousColorSolver> source_color_preview_solver(
    const std::vector<ImageMap::ContinuousColorComponent>& components,
    ImageMap::ColorMixModel                                color_mix_model)
{
    struct CacheEntry
    {
        ImageMap::ColorMixModel                         color_mix_model{ImageMap::ColorMixModel::FullSpectrumKmKs};
        uint64_t                                        calibration_revision{0};
        std::vector<ImageMap::ContinuousColorComponent> components;
        std::shared_ptr<const ImageMap::ContinuousColorSolver> solver;
    };
    struct CacheState
    {
        std::mutex              mutex;
        std::vector<CacheEntry> entries;
    };

    // Solver construction includes generating and predicting the complete
    // printable candidate set. Live controls used to launch that same work in
    // every replacement job; cancellation cannot interrupt the constructor,
    // so those obsolete jobs accumulated at 2%. Serialize cache misses and
    // retain a small set covering the active filament/model configurations.
    // Preview workers are detached and may still be winding down during GUI
    // shutdown, so keep this process-lifetime state out of static destruction.
    static CacheState*          cache = new CacheState;
    std::lock_guard<std::mutex> lock(cache->mutex);
    const uint64_t calibration_revision = ImageMap::simple_pm_calibration_revision();
    const auto cached = std::find_if(cache->entries.begin(), cache->entries.end(), [&components, color_mix_model, calibration_revision](const CacheEntry& entry) {
        return entry.color_mix_model == color_mix_model && entry.calibration_revision == calibration_revision &&
               source_color_preview_components_equal(entry.components, components);
    });
    if (cached != cache->entries.end())
        return cached->solver;

    auto solver = std::make_shared<ImageMap::ContinuousColorSolver>(components, color_mix_model);
    if (!solver->valid())
        return {};
    constexpr size_t maximum_cached_solvers = 12;
    if (cache->entries.size() >= maximum_cached_solvers)
        cache->entries.erase(cache->entries.begin());
    cache->entries.push_back(CacheEntry{color_mix_model, calibration_revision, components, solver});
    return solver;
}

std::vector<ImageMap::ContinuousColorComponent> source_color_preview_components()
{
    std::vector<ImageMap::ContinuousColorComponent> components;
    PresetBundle*                                   preset_bundle = GUI::wxGetApp().preset_bundle;
    if (preset_bundle == nullptr)
        return components;

    const ConfigOptionStrings* colors = preset_bundle->project_config.option<ConfigOptionStrings>("filament_colour");
    if (colors == nullptr)
        return components;
    const ConfigOptionFloats* transmission_distances = preset_bundle->project_config.option<ConfigOptionFloats>(
        "filament_transmission_distance");
    const ConfigOptionStrings* material_ids = preset_bundle->project_config.option<ConfigOptionStrings>(
        "filament_full_spectrum_material_id");

    components.reserve(colors->values.size());
    for (size_t index = 0; index < colors->values.size(); ++index) {
        ImageMap::ContinuousColorComponent component;
        component.color_hex = colors->values[index];
        if (MixedFilamentManager::use_td_for_color_prediction() && transmission_distances != nullptr &&
            index < transmission_distances->values.size()) {
            const double td_mm = transmission_distances->values[index];
            if (std::isfinite(td_mm) && td_mm > EPSILON)
                component.transmission_distance_mm = td_mm;
        }
        if (material_ids != nullptr && index < material_ids->values.size() && !material_ids->values[index].empty())
            component.material_id = material_ids->values[index];
        components.emplace_back(std::move(component));
    }
    return components;
}

uint32_t preview_color_key(const RGBA& color)
{
    auto channel = [](float value) { return uint32_t(std::lround(std::clamp(value, 0.f, 1.f) * 255.f)); };
    return (channel(color[0]) << 16) | (channel(color[1]) << 8) | channel(color[2]);
}

RGBA source_color_preview_rgba(const std::string& color_hex)
{
    const wxColour color = GUI::parse_mixed_color(color_hex);
    return {float(color.Red()) / 255.f, float(color.Green()) / 255.f, float(color.Blue()) / 255.f, 1.f};
}

size_t source_color_preview_assignment_signature(const std::vector<SourceColorPreviewAssignment>& assignments)
{
    size_t seed         = assignments.size();
    auto   hash_combine = [&seed](size_t value) { seed ^= value + size_t(0x9e3779b9) + (seed << 6) + (seed >> 2); };
    for (const SourceColorPreviewAssignment& assignment : assignments) {
        hash_combine(assignment.zone_index);
        hash_combine(preview_color_key(assignment.target_color));
        hash_combine(assignment.filament_id);
        hash_combine(preview_color_key(assignment.display_color));
        for (const ImageMap::ContinuousColorComponent& component : assignment.components) {
            hash_combine(std::hash<std::string>{}(component.color_hex));
            hash_combine(component.transmission_distance_mm ? std::hash<double>{}(*component.transmission_distance_mm) : 0u);
            hash_combine(component.material_id ? std::hash<std::string>{}(*component.material_id) : 0u);
        }
    }
    return seed;
}

std::array<Vec2f, 3> unwrap_source_texture_uvs(const ImageMap::SurfaceSource& source)
{
    std::array<Vec2f, 3> out         = source.uvs;
    auto                 unwrap_axis = [&out](bool use_u_axis, ImageMap::WrapMode wrap_mode) {
        if (wrap_mode != ImageMap::WrapMode::Repeat)
            return;
        std::array<float, 3> values{use_u_axis ? out[0].x() : out[0].y(), use_u_axis ? out[1].x() : out[1].y(),
                                    use_u_axis ? out[2].x() : out[2].y()};
        if (!std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); }))
            return;
        const bool has_repeat_evidence = std::any_of(values.begin(), values.end(), [](float value) {
            constexpr float epsilon = 1e-6f;
            return value < -epsilon || value > 1.f + epsilon;
        });
        auto       span                = [](const std::array<float, 3>& coordinates) {
            return std::max({coordinates[0], coordinates[1], coordinates[2]}) - std::min({coordinates[0], coordinates[1], coordinates[2]});
        };
        const float original_span = span(values);
        if (!has_repeat_evidence || original_span <= 0.5f)
            return;

        std::array<float, 3> best      = values;
        float                best_span = original_span;
        for (size_t anchor = 0; anchor < values.size(); ++anchor) {
            std::array<float, 3> candidate = values;
            for (size_t index = 0; index < candidate.size(); ++index) {
                const float delta = values[index] - values[anchor];
                candidate[index]  = values[anchor] + delta - std::round(delta);
            }
            const float candidate_span = span(candidate);
            if (candidate_span + 1e-6f < best_span) {
                best      = candidate;
                best_span = candidate_span;
            }
        }
        if (best_span >= original_span - 1e-6f)
            return;
        for (size_t index = 0; index < out.size(); ++index) {
            if (use_u_axis)
                out[index].x() = best[index];
            else
                out[index].y() = best[index];
        }
    };
    unwrap_axis(true, source.wrap_u);
    unwrap_axis(false, source.wrap_v);
    return out;
}

std::pair<unsigned int, unsigned int> source_texture_preview_size(const ImageMap::TextureAsset& asset, bool interactive_preview)
{
    if (!asset.valid())
        return {0, 0};
    const unsigned int maximum_edge = interactive_preview ? k_interactive_texture_preview_max_edge : k_source_texture_preview_max_edge;
    const size_t maximum_pixels = interactive_preview ? k_interactive_texture_preview_max_pixels : k_source_texture_preview_max_pixels;
    double scale = std::min({1.0, double(maximum_edge) / double(asset.width), double(maximum_edge) / double(asset.height),
                             std::sqrt(double(maximum_pixels) / (double(asset.width) * double(asset.height)))});
    return {std::max(1u, unsigned(std::floor(double(asset.width) * scale))),
            std::max(1u, unsigned(std::floor(double(asset.height) * scale)))};
}

double source_texture_preview_mm_per_pixel(const indexed_triangle_set&                         its,
                                           const std::vector<const ImageMap::TriangleBinding*>& bindings,
                                           unsigned int                                        preview_width,
                                           unsigned int                                        preview_height)
{
    double model_length_sum = 0.;
    double pixel_length_sum = 0.;
    for (const ImageMap::TriangleBinding* binding : bindings) {
        if (binding == nullptr || binding->triangle_index >= its.indices.size())
            continue;
        const stl_triangle_vertex_indices& indices = its.indices[binding->triangle_index];
        if (indices[0] < 0 || indices[1] < 0 || indices[2] < 0 || size_t(indices[0]) >= its.vertices.size() ||
            size_t(indices[1]) >= its.vertices.size() || size_t(indices[2]) >= its.vertices.size())
            continue;
        const std::array<Vec2f, 3> uvs = unwrap_source_texture_uvs(binding->source);
        for (size_t edge = 0; edge < 3; ++edge) {
            const size_t next = (edge + 1) % 3;
            const Vec2f  pixel_delta((uvs[next].x() - uvs[edge].x()) * float(preview_width),
                                    (uvs[next].y() - uvs[edge].y()) * float(preview_height));
            const double pixel_length = double(pixel_delta.norm());
            const double model_length = double((its.vertices[size_t(indices[next])] - its.vertices[size_t(indices[edge])]).norm());
            if (!std::isfinite(pixel_length) || !std::isfinite(model_length) || pixel_length <= EPSILON || model_length <= EPSILON)
                continue;
            pixel_length_sum += pixel_length;
            model_length_sum += model_length;
        }
    }
    return pixel_length_sum > EPSILON ? std::clamp(model_length_sum / pixel_length_sum, 0.002, 2.0) : 0.05;
}

struct PreviewFilterLayout
{
    // An almost-horizontal mapped surface lies in one slice plane, so its
    // reconstruction neighbourhood is genuinely two-dimensional. Side walls
    // instead use the equal-Z scan lines below; filtering across those lines
    // would mix different print layers in the prepare preview.
    bool                             gaussian_is_two_dimensional{false};
    double                           mm_per_step{0.05};
    std::vector<std::vector<size_t>> equal_z_lines;
};

PreviewFilterLayout source_texture_preview_filter_layout(
    const indexed_triangle_set&                         its,
    const std::vector<const ImageMap::TriangleBinding*>& bindings,
    unsigned int                                        preview_width,
    unsigned int                                        preview_height,
    const Transform3d&                                  mesh_to_world,
    const std::atomic<bool>&                            cancel)
{
    PreviewFilterLayout result;
    if (preview_width == 0 || preview_height == 0)
        return result;

    // Fit the affine texture-pixel -> world-position map represented by the
    // bound triangles. Projection UVs are affine even when triangle edges
    // split the image, and the Z rows remain affine on common curved targets
    // such as cylinders. Using all corners makes the estimate insensitive to
    // the mesh tessellation and gives us the direction tangent to a slice.
    Eigen::Matrix3d normal_matrix = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d right_hand    = Eigen::Matrix3d::Zero();
    size_t          sample_count  = 0;
    for (size_t binding_index = 0; binding_index < bindings.size(); ++binding_index) {
        if ((binding_index & 0x3ffu) == 0u && cancel.load(std::memory_order_relaxed))
            return result;
        const ImageMap::TriangleBinding* binding = bindings[binding_index];
        if (binding == nullptr || binding->triangle_index >= its.indices.size())
            continue;
        const stl_triangle_vertex_indices& indices = its.indices[binding->triangle_index];
        if (indices[0] < 0 || indices[1] < 0 || indices[2] < 0 || size_t(indices[0]) >= its.vertices.size() ||
            size_t(indices[1]) >= its.vertices.size() || size_t(indices[2]) >= its.vertices.size())
            continue;
        const std::array<Vec2f, 3> uvs = unwrap_source_texture_uvs(binding->source);
        for (size_t corner = 0; corner < 3; ++corner) {
            const Eigen::Vector3d predictor(double(uvs[corner].x()) * double(preview_width),
                                            double(uvs[corner].y()) * double(preview_height), 1.0);
            const Vec3d world = mesh_to_world * its.vertices[size_t(indices[corner])].cast<double>();
            if (!predictor.allFinite() || !world.allFinite())
                continue;
            normal_matrix.noalias() += predictor * predictor.transpose();
            right_hand.noalias() += predictor * world.transpose();
            ++sample_count;
        }
    }

    Vec3d world_per_x_pixel = Vec3d::Zero();
    Vec3d world_per_y_pixel = Vec3d::Zero();
    bool  fitted            = false;
    if (sample_count >= 3) {
        Eigen::FullPivLU<Eigen::Matrix3d> decomposition(normal_matrix);
        if (decomposition.rank() == 3) {
            const Eigen::Matrix3d coefficients = decomposition.solve(right_hand);
            if (coefficients.allFinite()) {
                world_per_x_pixel = coefficients.row(0).transpose();
                world_per_y_pixel = coefficients.row(1).transpose();
                fitted            = true;
            }
        }
    }

    const double fallback_mm_per_pixel = source_texture_preview_mm_per_pixel(its, bindings, preview_width, preview_height);
    Vec2d        equal_z_tangent(1.0, 0.0);
    if (fitted) {
        const Vec2d  z_gradient(world_per_x_pixel.z(), world_per_y_pixel.z());
        const double xy_scale = std::max(world_per_x_pixel.head<2>().norm(), world_per_y_pixel.head<2>().norm());
        // When Z changes by less than five percent of an in-plane pixel, the
        // mapped patch is effectively coplanar with a layer and the slicer's
        // world-XY Gaussian really can gather samples in both texture axes.
        result.gaussian_is_two_dimensional = z_gradient.norm() <= std::max(1e-7, 0.05 * xy_scale);
        if (!result.gaussian_is_two_dimensional && z_gradient.squaredNorm() > 1e-14)
            equal_z_tangent = Vec2d(z_gradient.y(), -z_gradient.x()).normalized();
    }

    const bool advance_x = std::abs(equal_z_tangent.x()) >= std::abs(equal_z_tangent.y());
    if (advance_x) {
        const double slope       = std::abs(equal_z_tangent.x()) > 1e-12 ? equal_z_tangent.y() / equal_z_tangent.x() : 0.0;
        const double end_offset  = -slope * double(preview_width - 1);
        const int    minimum_bin = int(std::floor(std::min(0.0, end_offset))) - 1;
        const int    maximum_bin = int(std::ceil(double(preview_height - 1) + std::max(0.0, end_offset))) + 1;
        result.equal_z_lines.resize(size_t(maximum_bin - minimum_bin + 1));
        // X-major insertion leaves every digital scan line in path order.
        for (unsigned int x = 0; x < preview_width; ++x) {
            if ((x & 0x1fu) == 0u && cancel.load(std::memory_order_relaxed))
                return result;
            for (unsigned int y = 0; y < preview_height; ++y) {
                const int bin = int(std::lround(double(y) - slope * double(x)));
                result.equal_z_lines[size_t(bin - minimum_bin)].emplace_back(size_t(y) * size_t(preview_width) + x);
            }
        }
        if (fitted)
            result.mm_per_step = (world_per_x_pixel + slope * world_per_y_pixel).head<2>().norm();
    } else {
        const double slope       = equal_z_tangent.x() / equal_z_tangent.y();
        const double end_offset  = -slope * double(preview_height - 1);
        const int    minimum_bin = int(std::floor(std::min(0.0, end_offset))) - 1;
        const int    maximum_bin = int(std::ceil(double(preview_width - 1) + std::max(0.0, end_offset))) + 1;
        result.equal_z_lines.resize(size_t(maximum_bin - minimum_bin + 1));
        // Y-major insertion leaves every digital scan line in path order.
        for (unsigned int y = 0; y < preview_height; ++y) {
            if ((y & 0x1fu) == 0u && cancel.load(std::memory_order_relaxed))
                return result;
            for (unsigned int x = 0; x < preview_width; ++x) {
                const int bin = int(std::lround(double(x) - slope * double(y)));
                result.equal_z_lines[size_t(bin - minimum_bin)].emplace_back(size_t(y) * size_t(preview_width) + x);
            }
        }
        if (fitted)
            result.mm_per_step = (slope * world_per_x_pixel + world_per_y_pixel).head<2>().norm();
    }
    if (!std::isfinite(result.mm_per_step) || result.mm_per_step <= 0.002)
        result.mm_per_step = fallback_mm_per_pixel;
    result.mm_per_step = std::clamp(result.mm_per_step, 0.002, 2.0);
    return result;
}

bool box_blur_weight_field(const std::vector<float>& source,
                           std::vector<float>&       destination,
                           unsigned int              width,
                           unsigned int              height,
                           size_t                    components,
                           int                       radius,
                           bool                      horizontal,
                           const std::atomic<bool>&  cancel)
{
    if (radius <= 0 || source.empty()) {
        destination = source;
        return true;
    }
    destination.resize(source.size());
    const int extent = horizontal ? int(width) : int(height);
    const int lines  = horizontal ? int(height) : int(width);
    auto index = [width, components, horizontal](int line, int position, size_t component) {
        const size_t x = size_t(horizontal ? position : line);
        const size_t y = size_t(horizontal ? line : position);
        return (y * size_t(width) + x) * components + component;
    };
    const double scale = 1. / double(2 * radius + 1);
    for (int line = 0; line < lines; ++line) {
        if ((line & 0x1f) == 0 && cancel.load(std::memory_order_relaxed))
            return false;
        for (size_t component = 0; component < components; ++component) {
            double sum = double(radius + 1) * source[index(line, 0, component)];
            for (int tap = 1; tap <= radius; ++tap)
                sum += source[index(line, std::min(tap, extent - 1), component)];
            for (int position = 0; position < extent; ++position) {
                destination[index(line, position, component)] = float(sum * scale);
                const int remove_position = std::clamp(position - radius, 0, extent - 1);
                const int add_position    = std::clamp(position + radius + 1, 0, extent - 1);
                sum += source[index(line, add_position, component)] - source[index(line, remove_position, component)];
            }
        }
    }
    return true;
}

bool box_blur_weight_lines(const std::vector<float>&              source,
                           std::vector<float>&                    destination,
                           const std::vector<std::vector<size_t>>& lines,
                           size_t                                 components,
                           int                                    radius,
                           const std::atomic<bool>&                cancel)
{
    if (radius <= 0 || source.empty()) {
        destination = source;
        return true;
    }
    destination.resize(source.size());
    const double scale = 1. / double(2 * radius + 1);
    for (size_t line_index = 0; line_index < lines.size(); ++line_index) {
        if ((line_index & 0x1fu) == 0u && cancel.load(std::memory_order_relaxed))
            return false;
        const std::vector<size_t>& line = lines[line_index];
        if (line.empty())
            continue;
        const int extent = int(line.size());
        for (size_t component = 0; component < components; ++component) {
            auto value = [&source, &line, components, component](int position) {
                return source[line[size_t(position)] * components + component];
            };
            double sum = double(radius + 1) * value(0);
            for (int tap = 1; tap <= radius; ++tap)
                sum += value(std::min(tap, extent - 1));
            for (int position = 0; position < extent; ++position) {
                destination[line[size_t(position)] * components + component] = float(sum * scale);
                const int remove_position = std::clamp(position - radius, 0, extent - 1);
                const int add_position    = std::clamp(position + radius + 1, 0, extent - 1);
                sum += value(add_position) - value(remove_position);
            }
        }
    }
    return true;
}

bool gaussian_blur_weight_field(std::vector<float>&       field,
                                unsigned int              width,
                                unsigned int              height,
                                size_t                    components,
                                double                    sigma_pixels,
                                const PreviewFilterLayout& layout,
                                const std::atomic<bool>&  cancel)
{
    if (sigma_pixels <= 0.15 || field.empty())
        return true;
    constexpr int box_count = 3;
    const double ideal_width = std::sqrt(12. * sigma_pixels * sigma_pixels / double(box_count) + 1.);
    int          lower_width = std::max(1, int(std::floor(ideal_width)));
    if ((lower_width & 1) == 0)
        --lower_width;
    const int upper_width = lower_width + 2;
    const double numerator = 12. * sigma_pixels * sigma_pixels - double(box_count * lower_width * lower_width) -
                             double(4 * box_count * lower_width + 3 * box_count);
    const int lower_count = std::clamp(int(std::lround(numerator / double(-4 * lower_width - 4))), 0, box_count);

    std::vector<float> temporary;
    for (int pass = 0; pass < box_count; ++pass) {
        const int radius = ((pass < lower_count ? lower_width : upper_width) - 1) / 2;
        if (layout.gaussian_is_two_dimensional) {
            if (!box_blur_weight_field(field, temporary, width, height, components, radius, true, cancel) ||
                !box_blur_weight_field(temporary, field, width, height, components, radius, false, cancel))
                return false;
        } else if (!box_blur_weight_lines(field, temporary, layout.equal_z_lines, components, radius, cancel)) {
            return false;
        } else {
            field.swap(temporary);
        }
    }
    return true;
}

void compact_preview_weights(std::vector<float>& field, size_t components)
{
    for (size_t offset = 0; offset < field.size(); offset += components) {
        float maximum = 0.f;
        for (size_t component = 0; component < components; ++component)
            maximum = std::max(maximum, field[offset + component]);
        if (maximum <= EPSILON)
            continue;
        for (size_t component = 0; component < components; ++component)
            field[offset + component] = std::clamp(field[offset + component] / maximum, 0.f, 1.f);
    }
}

bool triangular_preview_smoothing(std::vector<float>&       field,
                                  const PreviewFilterLayout& layout,
                                  size_t                    components,
                                  double                    radius_steps,
                                  const std::atomic<bool>&  cancel)
{
    if (radius_steps <= 0.5 || field.empty())
        return true;
    std::vector<float>  smoothed(field);
    auto range_sum = [](const std::vector<double>& values, int begin, int end) {
        return end >= begin ? values[size_t(end + 1)] - values[size_t(begin)] : 0.;
    };
    for (size_t line_index = 0; line_index < layout.equal_z_lines.size(); ++line_index) {
        if ((line_index & 0x1fu) == 0u && cancel.load(std::memory_order_relaxed))
            return false;
        const std::vector<size_t>& line = layout.equal_z_lines[line_index];
        if (line.size() < 2)
            continue;
        std::vector<double> prefix(line.size() + 1);
        std::vector<double> indexed_prefix(line.size() + 1);
        for (size_t component = 0; component < components; ++component) {
            prefix[0] = indexed_prefix[0] = 0.;
            for (size_t position = 0; position < line.size(); ++position) {
                const double inset = 1. - double(field[line[position] * components + component]);
                prefix[position + 1]         = prefix[position] + inset;
                indexed_prefix[position + 1] = indexed_prefix[position] + inset * double(position);
            }
            for (size_t position = 0; position < line.size(); ++position) {
                const int center = int(position);
                const int left   = std::max(0, int(std::ceil(double(center) - radius_steps)));
                const int right  = std::min(int(line.size()) - 1, int(std::floor(double(center) + radius_steps)));
                const double left_sum     = range_sum(prefix, left, center);
                const double left_indexed = range_sum(indexed_prefix, left, center);
                const double right_sum     = range_sum(prefix, center + 1, right);
                const double right_indexed = range_sum(indexed_prefix, center + 1, right);
                const double weighted_sum = left_sum * (1. - double(center) / radius_steps) + left_indexed / radius_steps +
                                            right_sum * (1. + double(center) / radius_steps) - right_indexed / radius_steps;
                const double left_count   = double(center - left + 1);
                const double right_count  = double(std::max(0, right - center));
                const double left_indices = 0.5 * double(left + center) * left_count;
                const double right_indices = right_count > 0. ? 0.5 * double(center + 1 + right) * right_count : 0.;
                const double weight_sum = left_count * (1. - double(center) / radius_steps) + left_indices / radius_steps +
                                          right_count * (1. + double(center) / radius_steps) - right_indices / radius_steps;
                const size_t offset = line[position] * components + component;
                const double inset  = 1. - double(field[offset]);
                const double filtered_inset = weight_sum > EPSILON ? std::min(inset, weighted_sum / weight_sum) : inset;
                smoothed[offset] = float(std::clamp(1. - filtered_inset, 0., 1.));
            }
        }
    }
    field.swap(smoothed);
    return true;
}

bool slope_limit_preview_weights(std::vector<float>&       field,
                                 const PreviewFilterLayout& layout,
                                 size_t                    components,
                                 double                    sample_spacing_mm,
                                 const std::atomic<bool>&  cancel)
{
    if (field.empty())
        return true;
    constexpr double max_displacement_mm = 0.63;
    // Slice-time limiting runs on the resampled perimeter, where every
    // sample-to-sample edge receives the 0.015 mm transition allowance. A
    // preview texel may cover several of those edges (especially in 0.02 mm
    // Ultra mode), so charging the allowance only once per texel stretches a
    // transition and makes the prepare result visibly softer than the sliced
    // path. Integrate the same allowance over the represented distance.
    const double bounded_sample_spacing = std::clamp(sample_spacing_mm, 0.02, 2.0);
    const double represented_steps      = layout.mm_per_step / bounded_sample_spacing;
    const float  limit                  = float((layout.mm_per_step * 0.35 + 0.015 * represented_steps) /
                               max_displacement_mm);
    for (int pass = 0; pass < 4; ++pass) {
        for (size_t line_index = 0; line_index < layout.equal_z_lines.size(); ++line_index) {
            if ((line_index & 0x1fu) == 0u && cancel.load(std::memory_order_relaxed))
                return false;
            const std::vector<size_t>& line = layout.equal_z_lines[line_index];
            if (line.size() < 2)
                continue;
            for (size_t component = 0; component < components; ++component) {
                for (size_t position = 1; position < line.size(); ++position) {
                    const size_t offset          = line[position] * components + component;
                    const size_t previous_offset = line[position - 1] * components + component;
                    field[offset] = std::max(field[offset], field[previous_offset] - limit);
                }
                for (size_t position = line.size() - 1; position-- > 0;) {
                    const size_t offset      = line[position] * components + component;
                    const size_t next_offset = line[position + 1] * components + component;
                    field[offset] = std::max(field[offset], field[next_offset] - limit);
                }
            }
        }
    }
    return true;
}

bool build_perimeter_modulation_texture_preview(const ImageMap::TextureAsset&                      asset,
                                                const ImageMap::Zone&                              zone,
                                                const ImageMap::ContinuousColorSolver&             solver,
                                                const indexed_triangle_set&                        its,
                                                const std::vector<const ImageMap::TriangleBinding*>& bindings,
                                                unsigned int                                       width,
                                                unsigned int                                       height,
                                                ImageMap::WrapMode                                 wrap_u,
                                                ImageMap::WrapMode                                 wrap_v,
                                                const Transform3d&                                 mesh_to_world,
                                                std::atomic<bool>&                                 cancel,
                                                const std::function<void(float)>&                   progress,
                                                std::vector<unsigned char>&                        output)
{
    const size_t components = solver.component_count();
    if (!solver.valid() || components == 0 || width == 0 || height == 0)
        return false;
    const size_t pixel_count = size_t(width) * size_t(height);
    std::vector<float> weights(pixel_count * components, 0.f);
    std::vector<float> alpha(pixel_count, 1.f);
    constexpr size_t quantized_color_count = 32u * 32u * 32u;
    std::vector<uint16_t> pixel_color_keys(pixel_count, 0);
    std::vector<uint8_t>  used_color_keys(quantized_color_count, 0);

    // Sampling is deliberately separate from physical-mixture lookup. A
    // photographic row may contain thousands of new colours; doing both in
    // one loop left progress at exactly 10% until all those KD searches had
    // completed, which looked like a hung worker.
    for (unsigned int y = 0; y < height; ++y) {
        if ((y & 0x1fu) == 0u && cancel.load(std::memory_order_relaxed))
            return false;
        if ((y & 0x07u) == 0u && progress)
            progress(0.18f * float(y) / float(height));
        for (unsigned int x = 0; x < width; ++x) {
            const Vec2f uv((float(x) + 0.5f) / float(width), (float(y) + 0.5f) / float(height));
            RGBA sampled = ImageMap::sample_processed_texture(asset, uv, wrap_u, wrap_v, zone);
            const size_t pixel = size_t(y) * size_t(width) + x;
            alpha[pixel] = sampled[3];
            for (size_t channel = 0; channel < 3; ++channel)
                sampled[channel] = sampled[channel] * sampled[3] + (1.f - sampled[3]);
            sampled[3] = 1.f;
            const RGBA target = ImageMap::adjusted_modulation_target_color(sampled, zone);
            auto quantized_channel = [&target](size_t channel) {
                return uint32_t(std::lround(std::clamp(target[channel], 0.f, 1.f) * 31.f));
            };
            const uint32_t key = (quantized_channel(0) << 10) | (quantized_channel(1) << 5) | quantized_channel(2);
            pixel_color_keys[pixel] = uint16_t(key);
            used_color_keys[key]    = 1;
        }
    }
    if (progress)
        progress(0.18f);

    std::vector<uint16_t> unique_color_keys;
    unique_color_keys.reserve(quantized_color_count);
    for (size_t key = 0; key < used_color_keys.size(); ++key)
        if (used_color_keys[key] != 0)
            unique_color_keys.emplace_back(uint16_t(key));

    std::vector<float> solved_weights(quantized_color_count * components, 0.f);
    std::atomic<size_t> next_color{0};
    std::atomic<size_t> completed_colors{0};
    const size_t worker_count = std::min<size_t>(4, unique_color_keys.size());
    std::vector<std::thread> solve_workers;
    solve_workers.reserve(worker_count);
    for (size_t worker = 0; worker < worker_count; ++worker) {
        solve_workers.emplace_back([&]() {
            for (;;) {
                const size_t color_index = next_color.fetch_add(1, std::memory_order_relaxed);
                if (color_index >= unique_color_keys.size() || cancel.load(std::memory_order_relaxed))
                    return;
                const uint32_t key   = unique_color_keys[color_index];
                const RGBA target{float((key >> 10) & 31u) / 31.f, float((key >> 5) & 31u) / 31.f,
                                  float(key & 31u) / 31.f, 1.f};
                // The shared solver retains this candidate index, making
                // later slider edits a constant-time lookup.
                std::vector<double> solved = solver.solve_quantized_5bit(target);
                ImageMap::apply_modulation_component_contrast(solved, zone);
                if (solved.size() == components) {
                    const size_t offset = size_t(key) * components;
                    for (size_t component = 0; component < components; ++component)
                        solved_weights[offset + component] = float(solved[component]);
                }
                const size_t completed = completed_colors.fetch_add(1, std::memory_order_relaxed) + 1;
                if ((completed & 0x1fu) == 0u && progress)
                    progress(0.18f + 0.30f * float(completed) / float(std::max<size_t>(unique_color_keys.size(), 1)));
            }
        });
    }
    for (std::thread& worker : solve_workers)
        worker.join();
    if (cancel.load(std::memory_order_relaxed))
        return false;
    if (progress)
        progress(0.48f);

    for (unsigned int y = 0; y < height; ++y) {
        if ((y & 0x1fu) == 0u && cancel.load(std::memory_order_relaxed))
            return false;
        if ((y & 0x0fu) == 0u && progress)
            progress(0.48f + 0.07f * float(y) / float(height));
        for (unsigned int x = 0; x < width; ++x) {
            const size_t pixel         = size_t(y) * size_t(width) + x;
            const size_t source_offset = size_t(pixel_color_keys[pixel]) * components;
            const size_t output_offset = pixel * components;
            std::copy_n(solved_weights.begin() + source_offset, components, weights.begin() + output_offset);
        }
    }
    if (progress)
        progress(0.55f);

    const PreviewFilterLayout filter_layout =
        source_texture_preview_filter_layout(its, bindings, width, height, mesh_to_world, cancel);
    if (cancel.load(std::memory_order_relaxed))
        return false;
    if (progress)
        progress(0.60f);
    const double gaussian_strength = std::clamp(double(zone.gaussian_smoothing_strength), 0., 4.);
    const double base_sigma_mm      = std::max(0.04, 0.45 * double(zone.modulation_sample_spacing_mm));
    if (!gaussian_blur_weight_field(weights, width, height, components,
                                    base_sigma_mm * gaussian_strength / filter_layout.mm_per_step, filter_layout, cancel))
        return false;
    compact_preview_weights(weights, components);
    if (progress)
        progress(0.69f);

    if (!zone.disable_broad_path_smoothing) {
        const double smoothing_base_mm = std::max(0.42, double(zone.corner_smoothing_radius_mm));
        const double first_radius_mm   = std::max(0.30, 1.15 * smoothing_base_mm) *
                                       std::clamp(double(zone.first_path_smoothing_strength), 0., 4.);
        const double second_radius_mm = std::max(0.20, 0.45 * smoothing_base_mm) *
                                        std::clamp(double(zone.second_path_smoothing_strength), 0., 4.);
        if (!triangular_preview_smoothing(weights, filter_layout, components, first_radius_mm / filter_layout.mm_per_step, cancel))
            return false;
        if (progress)
            progress(0.75f);
        if (!slope_limit_preview_weights(weights, filter_layout, components, zone.modulation_sample_spacing_mm, cancel))
            return false;
        if (progress)
            progress(0.80f);
        if (!triangular_preview_smoothing(weights, filter_layout, components, second_radius_mm / filter_layout.mm_per_step, cancel))
            return false;
    } else {
        if (!slope_limit_preview_weights(weights, filter_layout, components, zone.modulation_sample_spacing_mm, cancel))
            return false;
    }
    if (progress)
        progress(0.84f);

    output.resize(pixel_count * 4);
    std::unordered_map<uint64_t, RGBA> prediction_cache;
    prediction_cache.reserve(8192);
    std::vector<double> quantized_weights(components);
    for (size_t pixel = 0; pixel < pixel_count; ++pixel) {
        if ((pixel & 0xffffu) == 0u && cancel.load(std::memory_order_relaxed))
            return false;
        if ((pixel & 0xffffu) == 0u && progress)
            progress(0.84f + 0.16f * float(double(pixel) / double(pixel_count)));
        uint64_t key = 0;
        for (size_t component = 0; component < components; ++component) {
            const uint64_t quantized = uint64_t(std::lround(std::clamp(weights[pixel * components + component], 0.f, 1.f) * 31.f));
            quantized_weights[component] = double(quantized) / 31.;
            key = key * 33u + quantized;
        }
        auto predicted = prediction_cache.find(key);
        if (predicted == prediction_cache.end()) {
            const RGBA color = solver.predict_weights(quantized_weights).value_or(RGBA{1.f, 1.f, 1.f, 1.f});
            predicted = prediction_cache.emplace(key, color).first;
        }
        for (size_t channel = 0; channel < 3; ++channel)
            output[pixel * 4 + channel] = uint8_t(std::lround(std::clamp(predicted->second[channel], 0.f, 1.f) * 255.f));
        output[pixel * 4 + 3] = uint8_t(std::lround(std::clamp(alpha[pixel], 0.f, 1.f) * 255.f));
    }
    if (progress)
        progress(1.f);
    return true;
}

std::shared_ptr<SourceColorPreviewJob> start_source_color_preview(std::shared_ptr<const TriangleMesh>             mesh,
                                                                  std::shared_ptr<const ImageMap::VolumeData>     data,
                                                                  std::vector<ImageMap::ContinuousColorComponent> components,
                                                                  std::vector<SourceColorPreviewAssignment>       assignments,
                                                                  std::string                                     volume_name,
                                                                  ImageMap::RenderMode                            render_mode,
                                                                  bool                                            quantize_to_palette,
                                                                  bool                                            predicted_colors,
                                                                  Transform3d                                     mesh_to_world,
                                                                  bool                                            interactive_preview)
{
    auto job = std::make_shared<SourceColorPreviewJob>(data);
    std::thread([job, mesh = std::move(mesh), data = std::move(data), components = std::move(components),
                 assignments = std::move(assignments), volume_name = std::move(volume_name), render_mode, quantize_to_palette,
                 predicted_colors, mesh_to_world = std::move(mesh_to_world), interactive_preview]() mutable {
        const auto started_at = std::chrono::steady_clock::now();
        try {
            job->progress.store(2, std::memory_order_relaxed);
            const bool adaptive_cycle_preview = predicted_colors && render_mode == ImageMap::RenderMode::AdaptiveLocalizedCycles;
            quantize_to_palette                = predicted_colors && quantize_to_palette;
            auto color_model_index = [](ImageMap::ColorMixModel model) {
                return std::min<size_t>(size_t(model), size_t(ImageMap::ColorMixModel::FilamentMixer));
            };
            std::array<std::shared_ptr<const ImageMap::ContinuousColorSolver>, 2> shared_solvers;
            constexpr size_t                                                     no_solver = std::numeric_limits<size_t>::max();
            std::vector<std::shared_ptr<const ImageMap::ContinuousColorSolver>>   cycle_solvers;
            std::vector<size_t> assignment_solver_indices(assignments.size(), no_solver);
            if (adaptive_cycle_preview) {
                std::unordered_map<uint64_t, size_t> solver_by_filament;
                solver_by_filament.reserve(assignments.size());
                for (size_t assignment_index = 0; assignment_index < assignments.size(); ++assignment_index) {
                    if (job->cancel.load(std::memory_order_relaxed)) {
                        job->status.store(SourceColorPreviewStatus::Cancelled, std::memory_order_release);
                        return;
                    }
                    const unsigned int filament_id = assignments[assignment_index].filament_id;
                    const ImageMap::ColorMixModel color_mix_model = assignments[assignment_index].zone_index < data->zones.size() ?
                                                                        data->zones[assignments[assignment_index].zone_index].color_mix_model :
                                                                        ImageMap::ColorMixModel::FullSpectrumKmKs;
                    const uint64_t solver_key = uint64_t(filament_id) | (uint64_t(color_model_index(color_mix_model)) << 32);
                    const auto existing = solver_by_filament.find(solver_key);
                    if (existing != solver_by_filament.end()) {
                        assignment_solver_indices[assignment_index] = existing->second;
                    } else if (assignments[assignment_index].components.size() >= 2) {
                        const size_t solver_index = cycle_solvers.size();
                        std::shared_ptr<const ImageMap::ContinuousColorSolver> solver =
                            source_color_preview_solver(assignments[assignment_index].components, color_mix_model);
                        if (solver && solver->valid()) {
                            cycle_solvers.emplace_back(std::move(solver));
                            solver_by_filament.emplace(solver_key, solver_index);
                            assignment_solver_indices[assignment_index] = solver_index;
                        }
                    }
                    job->progress.store(2 + int(8 * (assignment_index + 1) / std::max<size_t>(assignments.size(), 1)),
                                        std::memory_order_relaxed);
                }
                // Every palette target sharing a cycle reuses the same solver.
                // Store a guaranteed-attainable fallback as well, so a failed
                // lookup can never leak the original texture RGB into the
                // printable-result preview.
                for (size_t assignment_index = 0; assignment_index < assignments.size(); ++assignment_index) {
                    const size_t solver_index = assignment_solver_indices[assignment_index];
                    if (solver_index == no_solver || solver_index >= cycle_solvers.size())
                        continue;
                    if (const std::optional<RGBA> predicted = cycle_solvers[solver_index]->predict_modulation_color(
                            assignments[assignment_index].target_color))
                        assignments[assignment_index].display_color = *predicted;
                }
            } else if (!quantize_to_palette) {
                for (const ImageMap::Zone& zone : data->zones) {
                    if (job->cancel.load(std::memory_order_relaxed)) {
                        job->status.store(SourceColorPreviewStatus::Cancelled, std::memory_order_release);
                        return;
                    }
                    if (!zone.enabled || zone.render_mode != render_mode)
                        continue;
                    const size_t index = color_model_index(zone.color_mix_model);
                    if (!shared_solvers[index])
                        shared_solvers[index] = source_color_preview_solver(components, zone.color_mix_model);
                }
            }
            if (job->cancel.load(std::memory_order_relaxed)) {
                job->status.store(SourceColorPreviewStatus::Cancelled, std::memory_order_release);
                return;
            }
            job->progress.store(10, std::memory_order_relaxed);

            // Keep texture mappings on their original triangles and let the
            // GPU interpolate the image. Projected decals are intentionally
            // partial and use transparent UV wrapping, so they need a simple
            // background surface plus a clipped texture overlay rather than
            // the subdivided vertex-colour fallback below.
            const indexed_triangle_set&                   its = mesh->its;
            std::vector<const ImageMap::TriangleBinding*> selected(its.indices.size(), nullptr);
            for (const ImageMap::TriangleBinding& binding : data->triangle_bindings) {
                if (binding.triangle_index >= selected.size() || binding.zone_index >= data->zones.size())
                    continue;
                const ImageMap::Zone& zone = data->zones[binding.zone_index];
                if (!zone.enabled || zone.render_mode != render_mode)
                    continue;
                const ImageMap::TriangleBinding* current = selected[binding.triangle_index];
                if (current == nullptr || data->zones[current->zone_index].priority < zone.priority)
                    selected[binding.triangle_index] = &binding;
            }

            bool texture_preview_eligible       = false;
            bool texture_preview_needs_background = false;
            for (const ImageMap::TriangleBinding* binding : selected) {
                if (binding == nullptr) {
                    texture_preview_needs_background = true;
                    continue;
                }
                if (binding->source.kind != ImageMap::SourceKind::Texture || binding->source.texture_asset_index < 0 ||
                    size_t(binding->source.texture_asset_index) >= data->texture_assets.size() ||
                    !data->texture_assets[size_t(binding->source.texture_asset_index)].valid() ||
                    !std::all_of(binding->source.uvs.begin(), binding->source.uvs.end(),
                                 [](const Vec2f& uv) { return std::isfinite(uv.x()) && std::isfinite(uv.y()); })) {
                    texture_preview_eligible = false;
                    break;
                }
                texture_preview_eligible = true;
                texture_preview_needs_background |= binding->source.wrap_u == ImageMap::WrapMode::Transparent ||
                                                    binding->source.wrap_v == ImageMap::WrapMode::Transparent;
            }

            if (texture_preview_eligible) {
                using TextureGroupKey = std::tuple<int32_t, uint32_t, uint8_t, uint8_t>;
                std::map<TextureGroupKey, std::vector<const ImageMap::TriangleBinding*>> groups;
                for (const ImageMap::TriangleBinding* binding : selected) {
                    if (binding == nullptr)
                        continue;
                    groups[{binding->source.texture_asset_index, binding->zone_index, uint8_t(binding->source.wrap_u),
                            uint8_t(binding->source.wrap_v)}]
                        .push_back(binding);
                }

                std::unique_ptr<GUI::GLModel::Geometry> background_geometry;
                if (texture_preview_needs_background) {
                    background_geometry = std::make_unique<GUI::GLModel::Geometry>();
                    background_geometry->format = {GUI::GLModel::PrimitiveType::Triangles,
                                                   GUI::GLModel::Geometry::EVertexLayout::P3N3C4};
                    background_geometry->reserve_vertices(its.indices.size() * 3);
                    background_geometry->reserve_indices(its.indices.size() * 3);
                    background_geometry->color = ColorRGBA::WHITE();
                    unsigned int next_vertex = 0;
                    for (size_t triangle_index = 0; triangle_index < its.indices.size(); ++triangle_index) {
                        const stl_triangle_vertex_indices& indices = its.indices[triangle_index];
                        if (indices[0] < 0 || indices[1] < 0 || indices[2] < 0 || size_t(indices[0]) >= its.vertices.size() ||
                            size_t(indices[1]) >= its.vertices.size() || size_t(indices[2]) >= its.vertices.size())
                            continue;
                        const Vec3f& p0            = its.vertices[size_t(indices[0])];
                        const Vec3f& p1            = its.vertices[size_t(indices[1])];
                        const Vec3f& p2            = its.vertices[size_t(indices[2])];
                        Vec3f        normal        = (p1 - p0).cross(p2 - p0);
                        const float  normal_length = normal.norm();
                        if (normal_length <= EPSILON)
                            continue;
                        normal /= normal_length;
                        const ImageMap::TriangleBinding* binding = selected[triangle_index];
                        const std::array<RGBA, 3> colors = binding != nullptr ? binding->source.corner_colors :
                                                                                 std::array<RGBA, 3>{RGBA{1.f, 1.f, 1.f, 1.f},
                                                                                                     RGBA{1.f, 1.f, 1.f, 1.f},
                                                                                                     RGBA{1.f, 1.f, 1.f, 1.f}};
                        colorin_add_vertex_c4(*background_geometry, p0, normal, colors[0]);
                        colorin_add_vertex_c4(*background_geometry, p1, normal, colors[1]);
                        colorin_add_vertex_c4(*background_geometry, p2, normal, colors[2]);
                        background_geometry->add_triangle(next_vertex, next_vertex + 1, next_vertex + 2);
                        next_vertex += 3;
                    }
                    if (background_geometry->is_empty())
                        background_geometry.reset();
                }

                size_t total_pixels = 0;
                for (const auto& group : groups) {
                    const auto [width, height] =
                        source_texture_preview_size(data->texture_assets[size_t(std::get<0>(group.first))], interactive_preview);
                    total_pixels += size_t(width) * size_t(height);
                }
                size_t                                processed_pixels = 0;
                std::vector<SourceTexturePreviewPart> texture_parts;
                texture_parts.reserve(groups.size());

                for (const auto& [group_key, bindings] : groups) {
                    if (job->cancel.load(std::memory_order_relaxed)) {
                        job->status.store(SourceColorPreviewStatus::Cancelled, std::memory_order_release);
                        return;
                    }
                    const int32_t                 asset_index  = std::get<0>(group_key);
                    const uint32_t                zone_index   = std::get<1>(group_key);
                    const ImageMap::TextureAsset& asset        = data->texture_assets[size_t(asset_index)];
                    const ImageMap::ContinuousColorSolver* shared_solver = nullptr;
                    if (zone_index < data->zones.size()) {
                        const size_t solver_index = color_model_index(data->zones[zone_index].color_mix_model);
                        if (shared_solvers[solver_index] && shared_solvers[solver_index]->valid())
                            shared_solver = shared_solvers[solver_index].get();
                    }
                    const auto [preview_width, preview_height] = source_texture_preview_size(asset, interactive_preview);
                    if (preview_width == 0 || preview_height == 0)
                        continue;

                    SourceTexturePreviewPart part;
                    part.width            = preview_width;
                    part.height           = preview_height;
                    part.wrap_u           = ImageMap::WrapMode(std::get<2>(group_key));
                    part.wrap_v           = ImageMap::WrapMode(std::get<3>(group_key));
                    part.geometry         = std::make_unique<GUI::GLModel::Geometry>();
                    part.geometry->format = {GUI::GLModel::PrimitiveType::Triangles,
                                             GUI::GLModel::Geometry::EVertexLayout::P3N3T2};
                    part.geometry->reserve_vertices(bindings.size() * 3);
                    part.geometry->reserve_indices(bindings.size() * 3);
                    part.geometry->color = ColorRGBA::WHITE();

                    unsigned int next_vertex = 0;
                    for (const ImageMap::TriangleBinding* binding : bindings) {
                        if (binding->triangle_index >= its.indices.size())
                            continue;
                        const stl_triangle_vertex_indices& indices = its.indices[binding->triangle_index];
                        if (indices[0] < 0 || indices[1] < 0 || indices[2] < 0 || size_t(indices[0]) >= its.vertices.size() ||
                            size_t(indices[1]) >= its.vertices.size() || size_t(indices[2]) >= its.vertices.size())
                            continue;
                        const Vec3f& p0            = its.vertices[size_t(indices[0])];
                        const Vec3f& p1            = its.vertices[size_t(indices[1])];
                        const Vec3f& p2            = its.vertices[size_t(indices[2])];
                        Vec3f        normal        = (p1 - p0).cross(p2 - p0);
                        const float  normal_length = normal.norm();
                        if (normal_length <= EPSILON)
                            continue;
                        normal /= normal_length;
                        const std::array<Vec2f, 3> uvs = unwrap_source_texture_uvs(binding->source);
                        part.geometry->add_vertex(p0, normal, uvs[0]);
                        part.geometry->add_vertex(p1, normal, uvs[1]);
                        part.geometry->add_vertex(p2, normal, uvs[2]);
                        part.geometry->add_triangle(next_vertex, next_vertex + 1, next_vertex + 2);
                        next_vertex += 3;
                    }
                    if (part.geometry->is_empty())
                        continue;

                    struct CachedTextureColor
                    {
                        RGBA         color{1.f, 1.f, 1.f, 1.f};
                        unsigned int filament_id{0};
                    };
                    std::unordered_map<uint32_t, CachedTextureColor> color_cache;
                    color_cache.reserve(32u * 32u * 32u);
                    part.rgba.resize(size_t(preview_width) * size_t(preview_height) * 4);

                    auto source_channel = [&asset, preview_width, preview_height](unsigned int x, unsigned int y, size_t channel) {
                        if (preview_width == asset.width && preview_height == asset.height)
                            return float(asset.rgba[(size_t(y) * size_t(asset.width) + x) * 4 + channel]);
                        const float  source_x = std::clamp((float(x) + 0.5f) * float(asset.width) / float(preview_width) - 0.5f, 0.f,
                                                           float(asset.width - 1));
                        const float  source_y = std::clamp((float(y) + 0.5f) * float(asset.height) / float(preview_height) - 0.5f, 0.f,
                                                           float(asset.height - 1));
                        const size_t x0       = size_t(std::floor(source_x));
                        const size_t y0       = size_t(std::floor(source_y));
                        const size_t x1       = std::min(x0 + 1, size_t(asset.width - 1));
                        const size_t y1       = std::min(y0 + 1, size_t(asset.height - 1));
                        const float  tx       = source_x - float(x0);
                        const float  ty       = source_y - float(y0);
                        auto         value    = [&asset, channel](size_t sx, size_t sy) {
                            return float(asset.rgba[(sy * size_t(asset.width) + sx) * 4 + channel]);
                        };
                        const float top    = value(x0, y0) + (value(x1, y0) - value(x0, y0)) * tx;
                        const float bottom = value(x0, y1) + (value(x1, y1) - value(x0, y1)) * tx;
                        return top + (bottom - top) * ty;
                    };
                    const bool raw_source_texture = assignments.empty() && (shared_solver == nullptr || !shared_solver->valid());
                    const ImageMap::Zone* preview_zone = zone_index < data->zones.size() ? &data->zones[zone_index] : nullptr;
                    if (predicted_colors && preview_zone != nullptr &&
                        preview_zone->render_mode == ImageMap::RenderMode::PerimeterModulationV2 && shared_solver != nullptr &&
                        shared_solver->valid()) {
                        const size_t part_pixels = size_t(preview_width) * size_t(preview_height);
                        auto update_part_progress = [job, processed_pixels, part_pixels, total_pixels](float part_progress) {
                            const double completed = double(processed_pixels) +
                                                     double(std::clamp(part_progress, 0.f, 1.f)) * double(part_pixels);
                            const int requested = 10 + int(88. * completed / double(std::max<size_t>(total_pixels, 1)));
                            int       current   = job->progress.load(std::memory_order_relaxed);
                            while (current < requested &&
                                   !job->progress.compare_exchange_weak(current, requested, std::memory_order_relaxed))
                                ;
                        };
                        if (build_perimeter_modulation_texture_preview(asset, *preview_zone, *shared_solver, its, bindings,
                                                                       preview_width, preview_height, part.wrap_u, part.wrap_v,
                                                                       mesh_to_world, job->cancel, update_part_progress, part.rgba)) {
                            processed_pixels += part_pixels;
                            texture_parts.emplace_back(std::move(part));
                            continue;
                        }
                        if (job->cancel.load(std::memory_order_relaxed)) {
                            job->status.store(SourceColorPreviewStatus::Cancelled, std::memory_order_release);
                            return;
                        }
                    }

                    for (unsigned int y = 0; y < preview_height; ++y) {
                        if ((y & 0x1fu) == 0u) {
                            if (job->cancel.load(std::memory_order_relaxed)) {
                                job->status.store(SourceColorPreviewStatus::Cancelled, std::memory_order_release);
                                return;
                            }
                            job->progress.store(10 + int(88 * processed_pixels / std::max<size_t>(total_pixels, 1)),
                                                std::memory_order_relaxed);
                        }
                        for (unsigned int x = 0; x < preview_width; ++x) {
                            RGBA source_color;
                            if (preview_zone != nullptr) {
                                const Vec2f uv((float(x) + 0.5f) / float(preview_width),
                                               (float(y) + 0.5f) / float(preview_height));
                                source_color = ImageMap::sample_processed_texture(asset, uv, part.wrap_u, part.wrap_v, *preview_zone);
                            } else {
                                source_color = {source_channel(x, y, 0) / 255.f, source_channel(x, y, 1) / 255.f,
                                                source_channel(x, y, 2) / 255.f, source_channel(x, y, 3) / 255.f};
                            }
                            const uint8_t source_r     = uint8_t(std::lround(std::clamp(source_color[0], 0.f, 1.f) * 255.f));
                            const uint8_t source_g     = uint8_t(std::lround(std::clamp(source_color[1], 0.f, 1.f) * 255.f));
                            const uint8_t source_b     = uint8_t(std::lround(std::clamp(source_color[2], 0.f, 1.f) * 255.f));
                            const float   source_alpha = source_color[3];
                            RGBA          target{(float(source_r) / 255.f) * source_alpha + (1.f - source_alpha),
                                                 (float(source_g) / 255.f) * source_alpha + (1.f - source_alpha),
                                                 (float(source_b) / 255.f) * source_alpha + (1.f - source_alpha), 1.f};
                            if (preview_zone != nullptr && preview_zone->render_mode != ImageMap::RenderMode::NormalMix)
                                target = ImageMap::adjusted_modulation_target_color(target, *preview_zone);
                            const uint32_t cache_key = (uint32_t(std::lround(target[0] * 31.f)) << 10) |
                                                       (uint32_t(std::lround(target[1] * 31.f)) << 5) |
                                                       uint32_t(std::lround(target[2] * 31.f));
                            auto cached = color_cache.find(cache_key);
                            if (cached == color_cache.end()) {
                                CachedTextureColor                  result{target, 0};
                                const SourceColorPreviewAssignment* assignment = nearest_source_color_assignment(assignments, target,
                                                                                                                 zone_index);
                                if (quantize_to_palette) {
                                    if (assignment != nullptr) {
                                        result.color       = assignment->display_color;
                                        result.filament_id = assignment->filament_id;
                                    }
                                } else if (adaptive_cycle_preview) {
                                    if (assignment != nullptr) {
                                        result.color                  = assignment->display_color;
                                        result.filament_id            = assignment->filament_id;
                                        const size_t assignment_index = size_t(assignment - assignments.data());
                                        const size_t solver_index     = assignment_index < assignment_solver_indices.size() ?
                                                                            assignment_solver_indices[assignment_index] :
                                                                            no_solver;
                                        if (solver_index != no_solver && solver_index < cycle_solvers.size() &&
                                            cycle_solvers[solver_index] != nullptr && cycle_solvers[solver_index]->valid()) {
                                            if (const std::optional<RGBA> predicted = cycle_solvers[solver_index]->predict_modulation_color(
                                                    target))
                                                result.color = *predicted;
                                        }
                                    }
                                } else if (shared_solver != nullptr && shared_solver->valid()) {
                                    if (const std::optional<RGBA> predicted = shared_solver->predict_modulation_color(target))
                                        result.color = *predicted;
                                    result.filament_id = nearest_source_color_filament(assignments, target, zone_index);
                                }
                                cached = color_cache.emplace(cache_key, result).first;
                            }
                            const size_t output_offset = (size_t(y) * size_t(preview_width) + x) * 4;
                            if (raw_source_texture) {
                                part.rgba[output_offset + 0] = source_r;
                                part.rgba[output_offset + 1] = source_g;
                                part.rgba[output_offset + 2] = source_b;
                            } else {
                                for (size_t channel = 0; channel < 3; ++channel)
                                    part.rgba[output_offset + channel] = uint8_t(
                                        std::lround(std::clamp(cached->second.color[channel], 0.f, 1.f) * 255.f));
                            }
                            part.rgba[output_offset + 3] = assignments.empty() ?
                                                                 uint8_t(std::lround(std::clamp(source_alpha, 0.f, 1.f) * 255.f)) :
                                                                 uint8_t(std::min(cached->second.filament_id, 255u));
                            ++processed_pixels;
                        }
                    }
                    texture_parts.emplace_back(std::move(part));
                }

                if (!texture_parts.empty()) {
                    {
                        std::lock_guard<std::mutex> lock(job->result_mutex);
                        job->geometry      = std::move(background_geometry);
                        job->texture_parts = std::move(texture_parts);
                    }
                    job->progress.store(99, std::memory_order_relaxed);
                    job->status.store(SourceColorPreviewStatus::Ready, std::memory_order_release);
                    const auto elapsed_ms =
                        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started_at).count();
                    BOOST_LOG_TRIVIAL(info) << "Image-map UV texture preview prepared"
                                            << " volume=\"" << volume_name << "\""
                                            << " groups=" << groups.size() << " source_triangles=" << its.indices.size()
                                            << " preview_pixels=" << processed_pixels << " elapsed_ms=" << elapsed_ms;
                    return;
                }
            }

            ImageMap::SourceColorRasterizationOptions options;
            options.max_leaf_triangles    = k_source_color_preview_triangle_cap;
            options.source_data_validated = true;
            options.cancelled             = [job]() { return job->cancel.load(std::memory_order_relaxed); };
            options.progress              = [job](int progress) {
                int current = job->progress.load(std::memory_order_relaxed);
                while (current < progress && !job->progress.compare_exchange_weak(current, progress, std::memory_order_relaxed))
                    ;
            };

            ImageMap::SourceColorRasterization rasterized = ImageMap::rasterize_source_colors(*mesh, *data, render_mode,
                                                                                              RGBA{1.f, 1.f, 1.f, 1.f}, options);
            if (job->cancel.load(std::memory_order_relaxed)) {
                job->status.store(SourceColorPreviewStatus::Cancelled, std::memory_order_release);
                return;
            }

            auto geometry                  = std::make_unique<GUI::GLModel::Geometry>();
            geometry->format.type          = GUI::GLModel::PrimitiveType::Triangles;
            geometry->format.vertex_layout = GUI::GLModel::Geometry::EVertexLayout::P3N3C4;
            geometry->reserve_vertices(rasterized.vertices.size());
            geometry->reserve_indices(rasterized.indices.size());
            geometry->indices = std::move(rasterized.indices);
            geometry->color   = ColorRGBA::WHITE();

            const size_t vertex_count  = rasterized.vertices.size();
            int          last_progress = 85;
            struct CachedPreviewColor
            {
                RGBA         color{1.f, 1.f, 1.f, 1.f};
                unsigned int filament_id{0};
            };
            std::unordered_map<uint32_t, CachedPreviewColor> color_cache;
            color_cache.reserve(std::min(vertex_count, k_source_color_preview_cache_cap));
            const ImageMap::ContinuousColorSolver* shared_solver = nullptr;
            for (const ImageMap::Zone& zone : data->zones) {
                if (!zone.enabled || zone.render_mode != render_mode)
                    continue;
                const size_t solver_index = color_model_index(zone.color_mix_model);
                if (shared_solvers[solver_index] && shared_solvers[solver_index]->valid())
                    shared_solver = shared_solvers[solver_index].get();
                break;
            }
            for (size_t vertex_idx = 0; vertex_idx < vertex_count; ++vertex_idx) {
                if ((vertex_idx & 0x3fffu) == 0u) {
                    if (job->cancel.load(std::memory_order_relaxed)) {
                        job->status.store(SourceColorPreviewStatus::Cancelled, std::memory_order_release);
                        return;
                    }
                    const int progress = vertex_count == 0 ? 99 : 85 + int(14 * vertex_idx / vertex_count);
                    if (progress != last_progress) {
                        job->progress.store(progress, std::memory_order_relaxed);
                        last_progress = progress;
                    }
                }
                const ImageMap::SourceColorVertex& vertex = rasterized.vertices[vertex_idx];
                RGBA         result_color = adaptive_cycle_preview ? RGBA{0.55f, 0.55f, 0.55f, vertex.color[3]} : vertex.color;
                unsigned int filament_id  = 0;
                if (quantize_to_palette) {
                    const uint32_t key    = preview_color_key(vertex.color);
                    const auto     cached = color_cache.find(key);
                    if (cached != color_cache.end()) {
                        result_color = cached->second.color;
                        filament_id  = cached->second.filament_id;
                    } else {
                        const RGBA quantized_target{float((key >> 16) & 0xffu) / 255.f, float((key >> 8) & 0xffu) / 255.f,
                                                    float(key & 0xffu) / 255.f, vertex.color[3]};
                        if (const SourceColorPreviewAssignment* assignment = nearest_source_color_assignment(assignments, quantized_target);
                            assignment != nullptr) {
                            result_color = assignment->display_color;
                            filament_id  = assignment->filament_id;
                        }
                        if (color_cache.size() < k_source_color_preview_cache_cap) {
                            RGBA cached_color = result_color;
                            cached_color[3]   = 1.f;
                            color_cache.emplace(key, CachedPreviewColor{cached_color, filament_id});
                        }
                    }
                } else if (adaptive_cycle_preview) {
                    const uint32_t key    = preview_color_key(vertex.color);
                    const auto     cached = color_cache.find(key);
                    if (cached != color_cache.end()) {
                        result_color = cached->second.color;
                        filament_id  = cached->second.filament_id;
                    } else {
                        const RGBA quantized_target{float((key >> 16) & 0xffu) / 255.f, float((key >> 8) & 0xffu) / 255.f,
                                                    float(key & 0xffu) / 255.f, vertex.color[3]};
                        const SourceColorPreviewAssignment* assignment = nearest_source_color_assignment(assignments, quantized_target);
                        if (assignment != nullptr) {
                            filament_id                   = assignment->filament_id;
                            result_color                  = assignment->display_color;
                            const size_t assignment_index = size_t(assignment - assignments.data());
                            const size_t solver_index     = assignment_index < assignment_solver_indices.size() ?
                                                                assignment_solver_indices[assignment_index] :
                                                                no_solver;
                            if (solver_index != no_solver && solver_index < cycle_solvers.size() &&
                                cycle_solvers[solver_index] != nullptr && cycle_solvers[solver_index]->valid()) {
                                if (const std::optional<RGBA> predicted = cycle_solvers[solver_index]->predict_modulation_color(
                                        quantized_target))
                                    result_color = *predicted;
                            }
                        }
                        if (color_cache.size() < k_source_color_preview_cache_cap) {
                            RGBA cached_color = result_color;
                            cached_color[3]   = 1.f;
                            color_cache.emplace(key, CachedPreviewColor{cached_color, filament_id});
                        }
                    }
                } else if (shared_solver != nullptr && shared_solver->valid()) {
                    const uint32_t key    = preview_color_key(vertex.color);
                    const auto     cached = color_cache.find(key);
                    if (cached != color_cache.end()) {
                        result_color = cached->second.color;
                        filament_id  = cached->second.filament_id;
                    } else {
                        RGBA quantized_target{float((key >> 16) & 0xffu) / 255.f, float((key >> 8) & 0xffu) / 255.f,
                                              float(key & 0xffu) / 255.f, vertex.color[3]};
                        if (const std::optional<RGBA> predicted = shared_solver->predict_modulation_color(quantized_target))
                            result_color = *predicted;
                        filament_id = nearest_source_color_filament(assignments, quantized_target);
                        if (color_cache.size() < k_source_color_preview_cache_cap) {
                            RGBA cached_color = result_color;
                            cached_color[3]   = 1.f;
                            color_cache.emplace(key, CachedPreviewColor{cached_color, filament_id});
                        }
                    }
                } else {
                    filament_id = nearest_source_color_filament(assignments, vertex.color);
                }
                // The source-preview shader treats alpha as a compact, transient
                // palette assignment. It restores the volume alpha before
                // lighting, so normal image-map rendering remains opaque.
                result_color[3] = assignments.empty() ? vertex.color[3] : float(std::min(filament_id, 255u)) / 255.f;
                colorin_add_vertex_c4(*geometry, vertex.position, vertex.normal, result_color);
            }

            {
                std::lock_guard<std::mutex> lock(job->result_mutex);
                job->geometry = std::move(geometry);
            }
            // Keep the 99% overlay visible for one frame before the UI thread
            // initializes and uploads the GL model.
            job->progress.store(99, std::memory_order_relaxed);
            job->status.store(SourceColorPreviewStatus::Ready, std::memory_order_release);

            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started_at)
                                        .count();
            BOOST_LOG_TRIVIAL(info) << "Image-map result preview prepared"
                                    << " volume=\"" << volume_name << "\""
                                    << " mode=\"" << (quantize_to_palette ? "quantized" : "optical") << "\""
                                    << " source_triangles=" << rasterized.source_triangle_count
                                    << " preview_triangles=" << rasterized.sampled_leaf_count << " lod_passes=" << rasterized.lod_pass_count
                                    << " elapsed_ms=" << elapsed_ms;
        } catch (const std::exception& error) {
            BOOST_LOG_TRIVIAL(error) << "KM/K-S image-map result preview failed"
                                     << " volume=\"" << volume_name << "\""
                                     << " error=\"" << error.what() << "\"";
            job->status.store(SourceColorPreviewStatus::Failed, std::memory_order_release);
        } catch (...) {
            BOOST_LOG_TRIVIAL(error) << "KM/K-S image-map result preview failed"
                                     << " volume=\"" << volume_name << "\""
                                     << " error=\"unknown exception\"";
            job->status.store(SourceColorPreviewStatus::Failed, std::memory_order_release);
        }
    }).detach();
    return job;
}

} // namespace


struct ImageMapPreviewPalette
{
    const MixedFilamentManager* manager{nullptr};
    size_t                      num_physical{0};
    size_t                      num_total{0};

    unsigned int resolve(const ImageMap::PaletteEntry& entry) const
    {
        if (manager != nullptr && entry.mixed_filament_stable_id != 0) {
            const std::optional<unsigned int> stable = manager->filament_id_from_stable_id(entry.mixed_filament_stable_id, num_physical);
            if (stable && *stable >= 1 && *stable <= num_total)
                return *stable;
        }
        return entry.fallback_filament_id >= 1 && entry.fallback_filament_id <= num_total ? entry.fallback_filament_id : 0u;
    }

    size_t signature(const ImageMap::VolumeData& data, unsigned int base_filament_id) const
    {
        size_t seed = base_filament_id;
        auto hash_combine = [&seed](size_t value) {
            seed ^= value + size_t(0x9e3779b9) + (seed << 6) + (seed >> 2);
        };
        hash_combine(num_physical);
        hash_combine(num_total);
        for (const ImageMap::Zone& zone : data.zones) {
            hash_combine(zone.enabled ? 1u : 0u);
            for (const ImageMap::PaletteEntry& entry : zone.palette)
                hash_combine(resolve(entry));
        }
        return seed;
    }
};

static ImageMapPreviewPalette image_map_preview_palette()
{
    ImageMapPreviewPalette palette;
    if (GUI::wxGetApp().preset_bundle == nullptr)
        return palette;
    palette.manager      = &GUI::wxGetApp().preset_bundle->mixed_filaments;
    palette.num_physical = size_t(std::max(GUI::wxGetApp().filaments_cnt(), 0));
    palette.num_total    = palette.manager->total_filaments(palette.num_physical);
    return palette;
}

// Bambu Colorin (Orca FullSpectrum 0.9.18): Orca's source-colour preview of an OBJ image map.
// Same path as Orca GLVolume::simple_render: the original texture (or the predicted printable colours)
// is prepared on a background thread and drawn with the image_map_texture_preview shader on the mesh UVs,
// so the model looks smooth like in Orca, instead of one flat colour per triangle.
// Returns true when the preview was drawn.
bool GLVolume::colorin_render_source_preview(const GUI::Camera&                                 camera,
                                             ModelVolume&                                       model_volume,
                                             const std::shared_ptr<const ImageMap::VolumeData>& image_map_data,
                                             const std::vector<std::array<float, 4>>&           extruder_colors) const
{
    if (!image_map_data || image_map_data->empty())
        return false;

    std::optional<ImageMap::RenderMode> image_map_render_mode;
    {
        const auto zone_it = std::find_if(image_map_data->zones.begin(), image_map_data->zones.end(),
                                          [](const ImageMap::Zone& zone) { return zone.enabled; });
        if (zone_it != image_map_data->zones.end())
            image_map_render_mode = zone_it->render_mode;
    }
    if (!image_map_render_mode)
        return false;

    const bool quantized_image_map = *image_map_render_mode == ImageMap::RenderMode::NormalMix;
    const bool predicted_image_map = GUI::image_map_preview_predicted_colors();

    std::vector<ImageMap::ContinuousColorComponent> preview_components = !predicted_image_map || quantized_image_map ?
                                                                             std::vector<ImageMap::ContinuousColorComponent>() :
                                                                             source_color_preview_components();
    const ImageMapPreviewPalette preview_palette  = image_map_preview_palette();
    const unsigned int           base_filament_id = unsigned(
        std::clamp(model_volume.extruder_id(), 1, int(std::max<size_t>(1, preview_palette.num_total))));
    std::vector<SourceColorPreviewAssignment> assignments;
    if (predicted_image_map && (quantized_image_map || *image_map_render_mode == ImageMap::RenderMode::AdaptiveLocalizedCycles)) {
        std::optional<MixedFilamentDisplayContext> adaptive_display_context;
        if (*image_map_render_mode == ImageMap::RenderMode::AdaptiveLocalizedCycles) {
            std::vector<std::string> physical_colors;
            physical_colors.reserve(preview_components.size());
            for (const ImageMap::ContinuousColorComponent& component : preview_components)
                physical_colors.emplace_back(component.color_hex);
            adaptive_display_context = GUI::build_mixed_filament_display_context(physical_colors);
        }
        for (const ImageMap::Zone& zone : image_map_data->zones) {
            if (!zone.enabled || zone.render_mode != *image_map_render_mode)
                continue;
            const uint32_t zone_index = uint32_t(&zone - image_map_data->zones.data());
            for (const ImageMap::PaletteEntry& entry : zone.palette) {
                const unsigned int filament_id = preview_palette.resolve(entry);
                if (filament_id == 0)
                    continue;
                RGBA display_color = *image_map_render_mode == ImageMap::RenderMode::AdaptiveLocalizedCycles ?
                                         RGBA{0.55f, 0.55f, 0.55f, 1.f} :
                                         entry.target_color;
                if (quantized_image_map && filament_id <= extruder_colors.size()) {
                    const std::array<float, 4> adjusted = adjust_color_for_rendering(extruder_colors[filament_id - 1]);
                    display_color                       = {adjusted[0], adjusted[1], adjusted[2], adjusted[3]};
                }
                std::vector<ImageMap::ContinuousColorComponent> assignment_components;
                if (*image_map_render_mode == ImageMap::RenderMode::AdaptiveLocalizedCycles) {
                    if (filament_id >= 1 && filament_id <= preview_components.size())
                        display_color = source_color_preview_rgba(preview_components[filament_id - 1].color_hex);
                }
                if (*image_map_render_mode == ImageMap::RenderMode::AdaptiveLocalizedCycles && preview_palette.manager != nullptr &&
                    adaptive_display_context) {
                    const std::optional<MixedFilamentDefinition> definition =
                        preview_palette.manager->mixed_filament_definition_from_id(filament_id, preview_palette.num_physical);
                    if (definition) {
                        display_color = source_color_preview_rgba(compute_mixed_filament_display_color(*definition, *adaptive_display_context));
                        for (const MixedFilamentWeightedComponent& component : definition->recipe.blend.components) {
                            const unsigned int component_id  = component.filament.id;
                            const bool         already_added = std::any_of(assignment_components.begin(), assignment_components.end(),
                                                                           [component_id, &preview_components](
                                                                               const ImageMap::ContinuousColorComponent& existing) {
                                                                               return component_id >= 1 &&
                                                                                      component_id <= preview_components.size() &&
                                                                                      existing.color_hex ==
                                                                                          preview_components[component_id - 1].color_hex;
                                                                           });
                            if (component.percent > 0 && component_id >= 1 && component_id <= preview_components.size() && !already_added)
                                assignment_components.push_back(preview_components[component_id - 1]);
                        }
                    }
                }
                assignments.push_back({zone_index, entry.target_color, filament_id, display_color, std::move(assignment_components)});
            }
        }
    }

    auto hash_mix = [](size_t& seed, size_t value) { seed ^= value + size_t(0x9e3779b9) + (seed << 6) + (seed >> 2); };
    size_t preview_signature = source_color_preview_signature(preview_components);
    hash_mix(preview_signature, size_t(*image_map_render_mode));
    hash_mix(preview_signature, source_color_preview_assignment_signature(assignments));
    hash_mix(preview_signature, preview_palette.signature(*image_map_data, base_filament_id));
    hash_mix(preview_signature, source_color_preview_zone_signature(*image_map_data));
    hash_mix(preview_signature, size_t(predicted_image_map));
    const Transform3d preview_mesh_to_world = world_matrix();
    for (Eigen::Index row = 0; row < 3; ++row)
        for (Eigen::Index column = 0; column < 3; ++column)
            hash_mix(preview_signature, std::hash<double>{}(preview_mesh_to_world.linear()(row, column)));

    if (image_map_data.get() != colorin_src_data || preview_signature != colorin_src_sig) {
        // Keep the last completed preview visible while its replacement is computed.
        if (colorin_src_job)
            colorin_src_job->cancel.store(true, std::memory_order_relaxed);
        colorin_src_data = image_map_data.get();
        colorin_src_sig  = preview_signature;
        colorin_src_job  = start_source_color_preview(model_volume.mesh_shared_ptr(), image_map_data, std::move(preview_components),
                                                      std::move(assignments), model_volume.name, *image_map_render_mode,
                                                      quantized_image_map, predicted_image_map, preview_mesh_to_world, false);
    }

    if (colorin_src_job) {
        const SourceColorPreviewStatus status = colorin_src_job->status.load(std::memory_order_acquire);
        if (status == SourceColorPreviewStatus::Ready) {
            if (!colorin_src_job->ready_presented.exchange(true, std::memory_order_acq_rel)) {
                std::unique_ptr<GUI::GLModel::Geometry> geometry;
                std::vector<SourceTexturePreviewPart>   texture_parts;
                {
                    std::lock_guard<std::mutex> lock(colorin_src_job->result_mutex);
                    geometry      = std::move(colorin_src_job->geometry);
                    texture_parts = std::move(colorin_src_job->texture_parts);
                }
                colorin_src_model.reset();
                colorin_src_tex_models.clear();
                colorin_src_textures.clear();
                colorin_src_wraps.clear();
                if (geometry && geometry->vertices_count() != 0 && geometry->indices_count() != 0) {
                    colorin_src_model = std::make_shared<GUI::GLModel>();
                    colorin_src_model->init_from(std::move(*geometry));
                }
                for (SourceTexturePreviewPart& part : texture_parts) {
                    if (!part.geometry || part.geometry->is_empty() || part.rgba.empty())
                        continue;
                    auto texture = std::make_shared<GUI::GLTexture>();
                    if (!texture->load_from_raw_data(std::move(part.rgba), part.width, part.height, false))
                        continue;
                    glsafe(::glBindTexture(GL_TEXTURE_2D, texture->get_id()));
                    glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
                    glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
                    glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, part.wrap_u == ImageMap::WrapMode::Repeat ? GL_REPEAT : GL_CLAMP_TO_EDGE));
                    glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, part.wrap_v == ImageMap::WrapMode::Repeat ? GL_REPEAT : GL_CLAMP_TO_EDGE));
                    glsafe(::glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0));
                    glsafe(::glBindTexture(GL_TEXTURE_2D, 0));
                    auto model = std::make_shared<GUI::GLModel>();
                    model->init_from(std::move(*part.geometry));
                    colorin_src_tex_models.push_back(std::move(model));
                    colorin_src_textures.push_back(std::move(texture));
                    colorin_src_wraps.push_back({part.wrap_u == ImageMap::WrapMode::Transparent, part.wrap_v == ImageMap::WrapMode::Transparent});
                }
                colorin_src_job->status.store(SourceColorPreviewStatus::Uploaded, std::memory_order_release);
            }
            colorin_src_job.reset();
        } else if (status == SourceColorPreviewStatus::Cancelled || status == SourceColorPreviewStatus::Failed) {
            colorin_src_job.reset();
        } else if (GUI::wxGetApp().plater() != nullptr) {
            // Still computing on the worker thread: keep the canvas redrawing until the result is ready.
            if (GUI::GLCanvas3D* canvas = GUI::wxGetApp().plater()->get_current_canvas3D()) {
                canvas->set_as_dirty();
                canvas->request_extra_frame();
            }
        }
    }

    const bool has_textures = !colorin_src_tex_models.empty();
    const bool has_model    = colorin_src_model && colorin_src_model->is_initialized();
    if (!has_textures && !has_model)
        return false;

    const std::shared_ptr<GLShaderProgram> previous_shader = GUI::wxGetApp().get_current_shader();
    const Transform3d model_matrix       = world_matrix();
    const Transform3d view_model_matrix  = camera.get_view_matrix() * model_matrix;
    const Matrix3d    view_normal_matrix = camera.get_view_matrix().matrix().block(0, 0, 3, 3) *
                                        model_matrix.matrix().block(0, 0, 3, 3).inverse().transpose();
    const float       alpha              = render_color[3];

    // Background: per-vertex colours (vertex-coloured OBJ) or a white base under the texture parts.
    if (has_model) {
        colorin_src_model->set_color(ColorRGBA(1.f, 1.f, 1.f, alpha));
        if (has_textures) {
            // White base drawn with the volume shader already set up by the caller (as Orca does).
            if (previous_shader)
                colorin_src_model->render_geometry();
        } else if (const std::shared_ptr<GLShaderProgram>& vshader = GUI::wxGetApp().get_shader("colorin_vertex_color")) {
            GUI::wxGetApp().bind_shader(vshader);
            vshader->set_uniform("view_model_matrix", view_model_matrix);
            vshader->set_uniform("projection_matrix", camera.get_projection_matrix());
            vshader->set_uniform("view_normal_matrix", view_normal_matrix);
            vshader->set_uniform("uniform_color", std::array<float, 4>{1.f, 1.f, 1.f, alpha});
            colorin_src_model->render_geometry_with_vertice_color();
        }
    }

    if (has_textures) {
        const std::shared_ptr<GLShaderProgram>& tshader = GUI::wxGetApp().get_shader("image_map_texture_preview");
        if (tshader) {
            GUI::wxGetApp().bind_shader(tshader);
            tshader->set_uniform("view_model_matrix", view_model_matrix);
            tshader->set_uniform("projection_matrix", camera.get_projection_matrix());
            tshader->set_uniform("view_normal_matrix", view_normal_matrix);
            tshader->set_uniform("volume_world_matrix", model_matrix);
            tshader->set_uniform("z_range", std::array<float, 2>{std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max()});
            tshader->set_uniform("uniform_texture", 0);
            tshader->set_uniform("image_map_cycle_preview", false);
            tshader->set_uniform("image_map_highlight_filament_id", 0);
            glsafe(::glActiveTexture(GL_TEXTURE0));
            const bool blend_enabled          = glIsEnabled(GL_BLEND) == GL_TRUE;
            const bool polygon_offset_enabled = glIsEnabled(GL_POLYGON_OFFSET_FILL) == GL_TRUE;
            GLint      depth_function         = GL_LESS;
            GLint      blend_src_rgb = GL_ONE, blend_dst_rgb = GL_ZERO, blend_src_alpha = GL_ONE, blend_dst_alpha = GL_ZERO;
            GLfloat    polygon_offset_factor = 0.f, polygon_offset_units = 0.f;
            glsafe(::glGetIntegerv(GL_DEPTH_FUNC, &depth_function));
            glsafe(::glGetIntegerv(GL_BLEND_SRC_RGB, &blend_src_rgb));
            glsafe(::glGetIntegerv(GL_BLEND_DST_RGB, &blend_dst_rgb));
            glsafe(::glGetIntegerv(GL_BLEND_SRC_ALPHA, &blend_src_alpha));
            glsafe(::glGetIntegerv(GL_BLEND_DST_ALPHA, &blend_dst_alpha));
            glsafe(::glGetFloatv(GL_POLYGON_OFFSET_FACTOR, &polygon_offset_factor));
            glsafe(::glGetFloatv(GL_POLYGON_OFFSET_UNITS, &polygon_offset_units));
            glsafe(::glEnable(GL_BLEND));
            glsafe(::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
            if (has_model) {
                glsafe(::glEnable(GL_POLYGON_OFFSET_FILL));
                glsafe(::glPolygonOffset(-1.f, -1.f));
                glsafe(::glDepthFunc(GL_LEQUAL));
            }
            for (size_t index = 0; index < colorin_src_tex_models.size(); ++index) {
                if (!colorin_src_tex_models[index] || !colorin_src_textures[index] || colorin_src_textures[index]->get_id() == 0)
                    continue;
                tshader->set_uniform("transparent_wrap_u", colorin_src_wraps[index][0]);
                tshader->set_uniform("transparent_wrap_v", colorin_src_wraps[index][1]);
                glsafe(::glBindTexture(GL_TEXTURE_2D, colorin_src_textures[index]->get_id()));
                colorin_src_tex_models[index]->set_color(ColorRGBA(1.f, 1.f, 1.f, alpha));
                colorin_src_tex_models[index]->render_geometry();
            }
            glsafe(::glBindTexture(GL_TEXTURE_2D, 0));
            glsafe(::glDepthFunc(depth_function));
            glsafe(::glPolygonOffset(polygon_offset_factor, polygon_offset_units));
            if (!polygon_offset_enabled)
                glsafe(::glDisable(GL_POLYGON_OFFSET_FILL));
            glsafe(::glBlendFuncSeparate(blend_src_rgb, blend_dst_rgb, blend_src_alpha, blend_dst_alpha));
            if (!blend_enabled)
                glsafe(::glDisable(GL_BLEND));
        }
    }

    if (previous_shader)
        GUI::wxGetApp().bind_shader(previous_shader);
    return true;
}


// Bambu Colorin (Orca FullSpectrum 0.9.18): viewport colours of an OBJ image map, exactly as Orca's
// palette view: the image map is rasterized to facets with ImageMap::rasterize_facets (same call and
// palette resolution as Orca GLVolume::simple_render), then each filament is drawn with its colour
// (physical filament colour, or the mixed filament display colour).
static void colorin_build_image_map_preview(const ModelVolume&                       mv,
                                            const ImageMap::VolumeData&              data,
                                            std::vector<GLIndexedVertexArray>&       ivas)
{
    ivas.clear();
    const MixedFilamentManager* manager = GUI::wxGetApp().preset_bundle ? &GUI::wxGetApp().preset_bundle->mixed_filaments : nullptr;
    const size_t num_physical = size_t(std::max(GUI::wxGetApp().filaments_cnt(), 0));
    const size_t num_total    = manager ? manager->total_filaments(num_physical) : num_physical;
    auto resolve = [manager, num_physical, num_total](const ImageMap::PaletteEntry& entry) -> unsigned int {
        if (manager != nullptr && entry.mixed_filament_stable_id != 0) {
            const std::optional<unsigned int> stable = manager->filament_id_from_stable_id(entry.mixed_filament_stable_id, num_physical);
            if (stable && *stable >= 1 && *stable <= num_total)
                return *stable;
        }
        return entry.fallback_filament_id >= 1 && entry.fallback_filament_id <= num_total ? entry.fallback_filament_id : 0u;
    };
    const unsigned int base_filament_id = unsigned(std::clamp(mv.extruder_id(), 1, int(std::max<size_t>(1, num_total))));
    const ImageMap::FacetRasterization rasterized = ImageMap::rasterize_facets(mv.mesh(), data, base_filament_id, resolve);
    std::unique_ptr<FacetsAnnotation> facets = mv.mmu_segmentation_facets.copy_for_slicing();
    for (const ImageMap::RasterizedFacet& facet : rasterized.facets)
        if (!facet.encoded_states.empty())
            facets->set_triangle_from_string(int(facet.triangle_index), facet.encoded_states);
    std::vector<indexed_triangle_set> its_per_color;
    facets->get_facets(mv, its_per_color);

    ivas.resize(its_per_color.size());
    for (size_t idx = 0; idx < its_per_color.size(); ++idx) {
        if (its_per_color[idx].indices.empty())
            continue;
        ivas[idx].load_its_flat_shading(its_per_color[idx]);
        ivas[idx].finalize_geometry(true);
    }
}

// Bambu Colorin: colours of each preview part (cheap, recomputed when filament colours change).
static void colorin_image_map_preview_colors(const ModelVolume&                       mv,
                                             size_t                                   count,
                                             const std::vector<std::array<float, 4>>& extruder_colors,
                                             std::vector<std::array<float, 4>>&       colors)
{
    const MixedFilamentManager* manager = GUI::wxGetApp().preset_bundle ? &GUI::wxGetApp().preset_bundle->mixed_filaments : nullptr;
    const size_t num_physical = size_t(std::max(GUI::wxGetApp().filaments_cnt(), 0));
    const size_t num_total    = manager ? manager->total_filaments(num_physical) : num_physical;
    const unsigned int base_filament_id = unsigned(std::clamp(mv.extruder_id(), 1, int(std::max<size_t>(1, num_total))));
    colors.assign(count, std::array<float, 4>{0.6f, 0.6f, 0.6f, 1.f});
    for (size_t idx = 0; idx < count; ++idx) {
        const unsigned int filament_id = idx == 0 ? base_filament_id : unsigned(idx);
        if (filament_id >= 1 && filament_id <= num_physical && filament_id <= extruder_colors.size()) {
            colors[idx] = extruder_colors[filament_id - 1];
        } else if (manager != nullptr && filament_id > num_physical) {
            const std::optional<MixedFilamentDefinition> definition = manager->mixed_filament_definition_from_id(filament_id, num_physical);
            ColorRGBA rgba;
            if (definition && decode_color(definition->presentation.display_color, rgba))
                colors[idx] = {rgba.r(), rgba.g(), rgba.b(), 1.f};
        }
    }
}

void GLVolume::render(const GUI::Camera &camera, const std::vector<std::array<float, 4>> &colors, Model &model, bool with_outline, const std::array<float, 4> &body_color, bool disable_mmu_colors) const
{
    if (!is_active)
        return;

    if (this->is_left_handed())
        glFrontFace(GL_CW);
    glsafe(::glCullFace(GL_BACK));
    auto                      zoom          = camera.get_zoom();
    auto                      view_matrix   = camera.get_view_matrix();
    Matrix4d                  vier_proj_mat = camera.get_projection_matrix().matrix() * view_matrix.matrix();
    const std::array<int, 4> &viewport      = camera.get_viewport();
    // BBS: add logic for mmu segmentation rendering
    auto render_body = [&]() {
        // Bambu Colorin (Orca FullSpectrum 0.9.18): draw the OBJ image map in colour.
        if (!picking && !disable_mmu_colors && printable && object_idx() < int(model.objects.size())) {
            ModelObject* colorin_mo = model.objects[object_idx()];
            if (volume_idx() < int(colorin_mo->volumes.size())) {
                ModelVolume* colorin_mv = colorin_mo->volumes[volume_idx()];
                const std::shared_ptr<const ImageMap::VolumeData> colorin_data = colorin_mv->image_map_data();
                if (colorin_data && !colorin_data->empty() && colorin_render_source_preview(camera, *colorin_mv, colorin_data, colors))
                    return;
                // While Orca's preview is still being prepared on its thread, draw the plain body instead of
                // rasterizing facets on the UI thread (that blocked the program on big models).
                if (colorin_data && !colorin_data->empty() && !colorin_src_job && colorin_src_data != colorin_data.get()) {
                    size_t colorin_sig = colors.size();
                    for (const auto& cc : colors)
                        for (float f : cc)
                            colorin_sig = colorin_sig * 1000003u + size_t(f * 1000.f);
                    if (GUI::wxGetApp().preset_bundle)
                        colorin_sig = colorin_sig * 1000003u + GUI::wxGetApp().preset_bundle->mixed_filaments.total_filaments(colors.size());
                    // Geometry is rebuilt only when the image map or the number of filaments changes;
                    // colours alone are cheap and are refreshed when the filament colours change.
                    const size_t colorin_geo_sig = GUI::wxGetApp().preset_bundle ?
                        GUI::wxGetApp().preset_bundle->mixed_filaments.total_filaments(size_t(std::max(GUI::wxGetApp().filaments_cnt(), 0))) : 0;
                    bool colorin_geo_changed = false;
                    if (colorin_preview_src != colorin_data.get() || colorin_preview_geo_sig != colorin_geo_sig) {
                        colorin_build_image_map_preview(*colorin_mv, *colorin_data, colorin_preview_ivas);
                        colorin_preview_src     = colorin_data.get();
                        colorin_preview_geo_sig = colorin_geo_sig;
                        colorin_geo_changed     = true;
                    }
                    if (colorin_geo_changed || colorin_preview_sig != colorin_sig || colorin_preview_colors.size() != colorin_preview_ivas.size()) {
                        colorin_image_map_preview_colors(*colorin_mv, colorin_preview_ivas.size(), colors, colorin_preview_colors);
                        colorin_preview_sig = colorin_sig;
                    }
                    const auto colorin_shader = GUI::wxGetApp().get_current_shader();
                    if (colorin_shader && !colorin_preview_ivas.empty()) {
                        for (size_t ci = 0; ci < colorin_preview_ivas.size(); ++ci) {
                            GLIndexedVertexArray& civa = colorin_preview_ivas[ci];
                            if (civa.triangle_indices_size == 0)
                                continue;
                            std::array<float, 4> c = adjust_color_for_rendering(colorin_preview_colors[ci]);
                            if (render_color[3] < 1.f)
                                c[3] = render_color[3];
                            colorin_shader->set_uniform("uniform_color", c);
                            civa.render(colorin_shader);
                        }
                        return;
                    }
                }
            }
        }
        bool color_volume = false;
        ModelObjectPtrs &model_objects = model.objects;
        do {
            // Skip MMU colors if disabled (e.g., when volume color override is active)
            if (disable_mmu_colors)
                break;

            if (mmuseg_detached_from_model) {
                // Keep the colors already uploaded instead of resolving composite_id onto whatever
                // sits at that index now that the Model has moved on.
                color_volume = !mmuseg_ivas.empty();
                break;
            }

            if ((!printable) || object_idx() >= model_objects.size())
                break;

            ModelObject* mo = model_objects[object_idx()];
            if (volume_idx() >= mo->volumes.size())
                break;

            ModelVolume* mv = mo->volumes[volume_idx()];
            if (mv->mmu_segmentation_facets.empty())
                break;

            if (colors.size() == 1) {
                break;
            }

            color_volume = true;
            if (mv->mmu_segmentation_facets.timestamp() != mmuseg_ts) {
                BOOST_LOG_TRIVIAL(debug) << boost::format("this %1%, name %2%, current mmuseg_ts %3%, current color size %4%")
                    %this %this->name %mmuseg_ts %mmuseg_ivas.size() ;
                mmuseg_ivas.clear();
                std::vector<indexed_triangle_set> its_per_color;
                mv->mmu_segmentation_facets.get_facets(*mv, its_per_color);
                mmuseg_ivas.resize(its_per_color.size());

                for (int idx = 0; idx < its_per_color.size(); idx++) {
                    if (its_per_color[idx].indices.size() > 0) {
                        mmuseg_ivas[idx].load_its_flat_shading(its_per_color[idx]);
                        mmuseg_ivas[idx].finalize_geometry(true);
                    }
                }
                mmuseg_ts = mv->mmu_segmentation_facets.timestamp();
                BOOST_LOG_TRIVIAL(debug) << boost::format("this %1%, name %2%, new mmuseg_ts %3%, new color size %4%")
                    %this %this->name %mmuseg_ts  %mmuseg_ivas.size();
            }
        } while (0);

        const auto shader = GUI::wxGetApp().get_current_shader();
        if (color_volume && !picking) {
            //when force_transparent, we need to keep the alpha
            auto cp_colors = colors;
            if (force_native_color && (render_color[3] < 1.0)) {
                for (int index = 0; index < colors.size(); index ++)
                    cp_colors[index][3] = render_color[3];
            }
            for (int idx = 0; idx < mmuseg_ivas.size(); idx++) {
                GLIndexedVertexArray* iva = &mmuseg_ivas[idx];
                if (iva->triangle_indices_size == 0 && iva->quad_indices_size == 0)
                    continue;

                if (shader) {
                    if (idx == 0) {
                        // While detached, composite_id must not be used to index the Model at all;
                        // GLVolume::extruder_id is the copy reload_scene() keeps in sync.
                        int extruder_id = this->extruder_id;
                        if (!mmuseg_detached_from_model) {
                            ModelObject* mo = model_objects[object_idx()];
                            ModelVolume* mv = mo->volumes[volume_idx()];
                            extruder_id = mv->extruder_id();
                        }
                        //to make black not too hard too see
                        if (extruder_id <= 0) { extruder_id = 1; }
                        if (extruder_id > (int)cp_colors.size()) { extruder_id = 1; }
                        std::array<float, 4> new_color = adjust_color_for_rendering(cp_colors[extruder_id - 1]);
                        shader->set_uniform("uniform_color", new_color);
                    }
                    else {
                        if (idx <= colors.size()) {
                            //to make black not too hard too see
                            std::array<float, 4> new_color = adjust_color_for_rendering(cp_colors[idx - 1]);
                            shader->set_uniform("uniform_color", new_color);
                        }
                        else {
                            //to make black not too hard too see
                            std::array<float, 4> new_color = adjust_color_for_rendering(cp_colors[0]);
                            shader->set_uniform("uniform_color", new_color);
                        }
                    }
                }
                iva->render(shader, this->tverts_range, this->qverts_range);
                /*if (force_native_color && (render_color[3] < 1.0)) {
                    BOOST_LOG_TRIVIAL(debug) << boost::format("this %1%, name %2%, tverts_range {%3,%4}, qverts_range{%5%, %6%}")
                     %this %this->name %this->tverts_range.first  %this->tverts_range.second
                     % this->qverts_range.first % this->qverts_range.second;
                }*/
            }
        }
        else {
            auto render_which = [this](std::shared_ptr<GLIndexedVertexArray> cur, const std::shared_ptr<GLShaderProgram>& shader) {
                if (cur->vertices_and_normals_interleaved_VBO_id > 0) {
                    cur->render(shader, tverts_range_lod, qverts_range_lod);
                } else {// if (cur->vertices_and_normals_interleaved_VBO_id == 0)
                    if (cur->triangle_indices.size() > 0) {
                        cur->finalize_geometry(true);
                        cur->render(shader, tverts_range_lod, qverts_range_lod);
                    } else {
                        indexed_vertex_array->render(shader, this->tverts_range, this->qverts_range);
                    }
                }
            };
            m_lod_update_index++;
            if (abs(zoom - LAST_CAMERA_ZOOM_VALUE) > ZOOM_THRESHOLD || m_lod_update_index >= LOD_UPDATE_FREQUENCY){
                m_lod_update_index     = 0;
                LAST_CAMERA_ZOOM_VALUE = zoom;
                if (!picking) {
                    m_cur_lod_level = calc_volume_box_in_screen_bigger_than_threshold(transformed_bounding_box(), vier_proj_mat, viewport[2], viewport[3]);
                }
            }
            if (m_cur_lod_level == LOD_LEVEL::SMALL && indexed_vertex_array_small) {
                render_which(indexed_vertex_array_small, shader);
            } else if (m_cur_lod_level == LOD_LEVEL::MIDDLE && indexed_vertex_array_middle) {
                render_which(indexed_vertex_array_middle, shader);
            } else {
                this->indexed_vertex_array->render(shader, this->tverts_range, this->qverts_range);
            }
        }
    };

    //BBS: add logic of outline rendering
    const auto shader = GUI::wxGetApp().get_current_shader();
    //BOOST_LOG_TRIVIAL(info) << boost::format(": %1%, with_outline %2%, shader %3%.")%__LINE__ %with_outline %shader;
    if (with_outline && shader != nullptr)
    {
        do
        {
            glEnable(GL_STENCIL_TEST);
            glStencilMask(0xFF);
            glStencilOp(GL_KEEP, GL_REPLACE, GL_REPLACE);
            glClear(GL_STENCIL_BUFFER_BIT);
            glStencilFunc(GL_ALWAYS, 0xff, 0xFF);
            //another way use depth buffer
            //glsafe(::glEnable(GL_DEPTH_TEST));
            //glsafe(::glDepthFunc(GL_ALWAYS));
            //glsafe(::glDepthMask(GL_FALSE));
            //glsafe(::glEnable(GL_BLEND));
            //glsafe(::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

            /*GLShaderProgram* outline_shader = GUI::wxGetApp().get_shader("outline");
            if (outline_shader == nullptr)
            {
                glDisable(GL_STENCIL_TEST);
                this->indexed_vertex_array->render(this->tverts_range, this->qverts_range);
                break;
            }
            shader->stop_using();
            outline_shader->start_using();
            //float scale_ratio = 1.02f;
            std::array<float, 4> outline_color = { 0.0f, 1.0f, 0.0f, 1.0f };

            outline_shader->set_uniform("uniform_color", outline_color);*/
#if 0 //dump stencil buffer
            int i = 100, j = 100;
            std::string file_name;
            FILE* file = NULL;
            memset(stencil_data, 0, sizeof(stencil_data));
            glReadPixels(0, 0, 2936, 1083, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, stencil_data);
            for (i = 100; i < 1083; i++)
            {
                for (j = 100; j < 2936; j++)
                {
                    if (stencil_data[i][j] != 0)
                    {
                        file_name = "before_stencil_index_" + std::to_string(i) + "x" + std::to_string(j) + ".a8";
                        break;
                    }
                }

                if (stencil_data[i][j] != 0)
                    break;
            }
            file = fopen(file_name.c_str(), "w");
            if (file)
            {
                fwrite(stencil_data, 2936 * 1083, 1, file);
                fclose(file);
            }
#endif
            render_body();
            //BOOST_LOG_TRIVIAL(info) << boost::format(": %1%, outline render body, shader name %2%")%__LINE__ %shader->get_name();

#if 0 //dump stencil buffer after first rendering
            memset(stencil_data, 0, sizeof(stencil_data));
            glReadPixels(0, 0, 2936, 1083, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, stencil_data);
            for (i = 100; i < 1083; i++)
            {
                for (j = 100; j < 2936; j++)
                    if (stencil_data[i][j] != 0)
                    {
                        file_name = "after_stencil_index_" + std::to_string(i) + "x" + std::to_string(j) + ".a8";
                        break;
                    }

                if (stencil_data[i][j] != 0)
                    break;
            }

            file = fopen(file_name.c_str(), "w");
            if (file)
            {
                fwrite(stencil_data, 2936 * 1083, 1, file);
                fclose(file);
            }
#endif
            // 2nd. render pass: now draw slightly scaled versions of the objects, this time disabling stencil writing.
            // Because the stencil buffer is now filled with several 1s. The parts of the buffer that are 1 are not drawn, thus only drawing
            // the objects' size differences, making it look like borders.
            // -----------------------------------------------------------------------------------------------------------------------------
            /*GLShaderProgram* outline_shader = GUI::wxGetApp().get_shader("outline");
            if (outline_shader == nullptr)
            {
                glDisable(GL_STENCIL_TEST);
                break;
            }
            shader->stop_using();
            outline_shader->start_using();*/
            //outline_shader->stop_using();
            //shader->start_using();

            glStencilFunc(GL_NOTEQUAL, 0xff, 0xFF);
            glStencilMask(0x00);
            float scale = 1.02f;

            shader->set_uniform("uniform_color", body_color);
            shader->set_uniform("is_outline", true);

            Transform3d matrix = world_matrix();
            Transform3d world_tran = matrix;
            matrix.scale(scale);
            shader->set_uniform("view_model_matrix", view_matrix * matrix);
            m_lod_update_index++;
            if (abs(zoom - LAST_CAMERA_ZOOM_VALUE) > ZOOM_THRESHOLD || m_lod_update_index >= LOD_UPDATE_FREQUENCY) {
                m_lod_update_index     = 0;
                LAST_CAMERA_ZOOM_VALUE = zoom;
                if (!picking) {
                    m_cur_lod_level = calc_volume_box_in_screen_bigger_than_threshold(transformed_bounding_box(), vier_proj_mat, viewport[2], viewport[3]);
                }
            }
            if (m_cur_lod_level == LOD_LEVEL::SMALL && indexed_vertex_array_small && indexed_vertex_array_small->vertices_and_normals_interleaved_VBO_id > 0) {
                this->indexed_vertex_array_small->render(shader, this->tverts_range_lod, this->qverts_range_lod);
            } else if (m_cur_lod_level == LOD_LEVEL::MIDDLE && indexed_vertex_array_middle && indexed_vertex_array_middle->vertices_and_normals_interleaved_VBO_id > 0) {
                this->indexed_vertex_array_middle->render(shader, this->tverts_range_lod, this->qverts_range_lod);
            } else {
                this->indexed_vertex_array->render(shader, this->tverts_range, this->qverts_range);
            }
            //BOOST_LOG_TRIVIAL(info) << boost::format(": %1%, outline render for body, shader name %2%")%__LINE__ %shader->get_name();
            shader->set_uniform("is_outline", false);

            //glStencilMask(0xFF);
            //glStencilFunc(GL_ALWAYS, 0, 0xFF);
            glDisable(GL_STENCIL_TEST);
            //glEnable(GL_DEPTH_TEST);
            //outline_shader->stop_using();
            //shader->start_using();
        } while (0);
    }
    else {
        render_body();
        //BOOST_LOG_TRIVIAL(info) << boost::format(": %1%, normal render.")%__LINE__;
    }
    if (this->is_left_handed())
        glFrontFace(GL_CCW);
}

//BBS add render for simple case
void GLVolume::simple_render(const std::shared_ptr<GLShaderProgram>& shader, ModelObjectPtrs &model_objects, std::vector<std::array<float, 4>> &extruder_colors, bool ban_light) const
{
    if (this->is_left_handed())
        glFrontFace(GL_CW);
    glsafe(::glCullFace(GL_BACK));

    bool color_volume = false;
    ModelObject* model_object = nullptr;
    ModelVolume* model_volume = nullptr;
    do {
        if (mmuseg_detached_from_model) {
            // Same as in render(): the Model no longer matches composite_id, so keep what is already
            // uploaded rather than rebuilding from a different ModelVolume.
            color_volume = !mmuseg_ivas.empty();
            break;
        }

        if ((!printable) || object_idx() >= model_objects.size())
            break;
        model_object = model_objects[object_idx()];

        if (volume_idx() >=  model_object->volumes.size())
            break;
        model_volume = model_object->volumes[volume_idx()];
        if (model_volume->mmu_segmentation_facets.empty())
            break;

        color_volume = true;
        if (model_volume->mmu_segmentation_facets.timestamp() != mmuseg_ts) {
            mmuseg_ivas.clear();
            std::vector<indexed_triangle_set> its_per_color;
            model_volume->mmu_segmentation_facets.get_facets(*model_volume, its_per_color);
            mmuseg_ivas.resize(its_per_color.size());
            for (int idx = 0; idx < its_per_color.size(); idx++) {
                mmuseg_ivas[idx].load_its_flat_shading(its_per_color[idx]);
                mmuseg_ivas[idx].finalize_geometry(true);
            }

            mmuseg_ts = model_volume->mmu_segmentation_facets.timestamp();
        }
    } while (0);

    if (color_volume && !picking) {
        for (int idx = 0; idx < mmuseg_ivas.size(); idx++) {
            GLIndexedVertexArray& iva = mmuseg_ivas[idx];
            if (iva.triangle_indices_size == 0 && iva.quad_indices_size == 0)
                continue;

            if (shader) {
                if (idx == 0) {
                    // model_volume stays null while detached from the Model, see above.
                    int extruder_id = model_volume != nullptr ? model_volume->extruder_id() : this->extruder_id;
                    if (extruder_id <= 0) { extruder_id = 1; }
                    //to make black not too hard too see
                    if (extruder_id > (int)extruder_colors.size()) { extruder_id = 1; }
                    std::array<float, 4> new_color = adjust_color_for_rendering(extruder_colors[extruder_id - 1]);
                    if (ban_light) {
                        new_color[3] = (255 - (extruder_id - 1))/255.0f;
                    }
                    shader->set_uniform("uniform_color", new_color);
                }
                else {
                    if (idx <= extruder_colors.size()) {
                        //shader->set_uniform("uniform_color", extruder_colors[idx - 1]);
                        //to make black not too hard too see
                        std::array<float, 4> new_color = adjust_color_for_rendering(extruder_colors[idx - 1]);
                        if (ban_light) {
                            new_color[3] = (255 - (idx - 1))/255.0f;
                        }
                        shader->set_uniform("uniform_color", new_color);
                    }
                    else {
                        //shader->set_uniform("uniform_color", extruder_colors[0]);
                        //to make black not too hard too see
                        std::array<float, 4> new_color = adjust_color_for_rendering(extruder_colors[0]);
                        if (ban_light) {
                            new_color[3] = (255 - 0) / 255.0f;
                        }
                        shader->set_uniform("uniform_color", new_color);
                    }
                }
            }
            iva.render(shader, this->tverts_range, this->qverts_range);
        }
    }
    else {
        this->indexed_vertex_array->render(shader, this->tverts_range, this->qverts_range);
    }

    if (this->is_left_handed())
        glFrontFace(GL_CCW);
}

void GLVolume::set_bounding_boxes_as_dirty()
{
    m_lod_update_index = LOD_UPDATE_FREQUENCY;
    m_transformed_bounding_box.reset();
    m_transformed_convex_hull_bounding_box.reset();
    m_transformed_non_sinking_bounding_box.reset();
}

bool GLVolume::is_sla_support() const { return this->composite_id.volume_id == -int(slaposSupportTree); }
bool GLVolume::is_sla_pad() const { return this->composite_id.volume_id == -int(slaposPad); }

bool GLVolume::is_sinking() const
{
    if (is_modifier || GUI::wxGetApp().preset_bundle->printers.get_edited_preset().printer_technology() == ptSLA)
        return false;
    const BoundingBoxf3& box = transformed_convex_hull_bounding_box();
    return box.min.z() < SINKING_Z_THRESHOLD && box.max.z() >= SINKING_Z_THRESHOLD;
}

bool GLVolume::is_below_printbed() const
{
    return transformed_convex_hull_bounding_box().max.z() < 0.0;
}

void GLVolume::render_sinking_contours(const GUI::Camera &camera, Model &model) {
    m_sinking_contours.render(camera, model);
}

GLWipeTowerVolume::GLWipeTowerVolume(const std::vector<std::array<float, 4>>& colors)
    : GLVolume()
{
    m_colors = colors;
}

void GLWipeTowerVolume::render(const GUI::Camera &camera,
                               const std::vector<std::array<float, 4>> &extruder_colors,
                               Model &model,
                               bool                                     with_outline,
                               const std::array<float, 4> &body_color,
                               bool                                     disable_mmu_colors) const
{
    if (!is_active)
        return;

    if (m_colors.size() == 0 || m_colors.size() != iva_per_colors.size())
        return;

    const auto shader = GUI::wxGetApp().get_current_shader();
    if (!shader) {
        return;
    }

    if (this->is_left_handed())
        glFrontFace(GL_CW);
    glsafe(::glCullFace(GL_BACK));

    for (int i = 0; i < m_colors.size(); i++) {
        if (!picking) {
            ColorRGBA new_color = adjust_color_for_rendering(m_colors[i]);
            std::array<float, 4> final_color;
            final_color[0] = new_color.r();
            final_color[1] = new_color.g();
            final_color[2] = new_color.b();
            final_color[3] = new_color.a();
            shader->set_uniform("uniform_color", final_color);
        }
        this->iva_per_colors[i].render(shader);
    }

    if (this->is_left_handed())
        glFrontFace(GL_CCW);
}

bool GLWipeTowerVolume::IsTransparent() {
    for (size_t i = 0; i < m_colors.size(); i++) {
        if (m_colors[i][3] < 1.0f) {
            return true;
        }
    }
    return false;
}

std::vector<int> GLVolumeCollection::load_object(
    const ModelObject       *model_object,
    int                      obj_idx,
    const std::vector<int>  &instance_idxs,
    const std::string       &color_by,
    bool 					 opengl_initialized,
    bool                    lod_enabled)
{
    std::vector<int> volumes_idx;
    for (int volume_idx = 0; volume_idx < int(model_object->volumes.size()); ++volume_idx)
        for (int instance_idx : instance_idxs)
            volumes_idx.emplace_back(this->GLVolumeCollection::load_object_volume(model_object, obj_idx, volume_idx, instance_idx, color_by, opengl_initialized, false, lod_enabled));
    return volumes_idx;
}

int GLVolumeCollection::load_object_volume(
    const ModelObject   *model_object,
    int                  obj_idx,
    int                  volume_idx,
    int                  instance_idx,
    const std::string   &color_by,
    bool 				 opengl_initialized,
    bool                 in_assemble_view,
    bool                 use_loaded_id,
    bool                 lod_enabled)
{
    // Guard against malformed or partially-loaded objects: the code below
    // dereferences volumes[volume_idx] and, in particular, instances[instance_idx]
    // (e.g. instance->get_transformation()). An object with no instances - or an
    // out-of-range index - would dereference a null/garbage pointer and crash.
    // Seen when opening some MakerWorld 3MFs via the object-color dialog. See #11016.
    if (model_object == nullptr
        || volume_idx < 0 || volume_idx >= (int) model_object->volumes.size()
        || instance_idx < 0 || instance_idx >= (int) model_object->instances.size())
        return -1;
    const ModelVolume   *model_volume = model_object->volumes[volume_idx];
    const int            extruder_id  = model_volume->extruder_id();
    const ModelInstance *instance 	  = model_object->instances[instance_idx];
    const TriangleMesh  &mesh 		  = model_volume->mesh();
    std::array<float, 4> color = GLVolume::MODEL_COLOR[((color_by == "volume") ? volume_idx : obj_idx) % 4];
    color[3] = model_volume->is_model_part() ? 0.7f : 0.4f;
    GLVolume* new_volume = new GLVolume(color[0], color[1], color[2], color[3], false);
    this->volumes.emplace_back(new_volume);
    GLVolume& v = *new_volume;
    v.set_origin_mesh_render_type(model_volume->get_origin_mesh_or_vertice_render());
    v.set_color(color_from_model_volume(*model_volume));
    v.name = model_volume->name;
    v.is_text_shape = model_volume->is_text();

    const TriangleMesh* mesh_ptr = model_volume->mesh_ptr();
    new_volume->ori_mesh = mesh_ptr;
    std::map<const TriangleMesh*, std::set<GLVolume*>>::iterator iter = g_mesh_volumes_map.find(mesh_ptr);
    bool need_create_mesh = true;
    if (iter != g_mesh_volumes_map.end()) {
        std::set<GLVolume*> & volume_set = iter->second;

        if (volume_set.empty()) {
            new_volume->indexed_vertex_array = std::make_shared<GLIndexedVertexArray>();
        }
        else {
            GLVolume* first_volume = *(volume_set.begin());
            new_volume->indexed_vertex_array = first_volume->indexed_vertex_array;
            new_volume->indexed_vertex_array_middle = first_volume->indexed_vertex_array_middle;
            new_volume->indexed_vertex_array_small  = first_volume->indexed_vertex_array_small;
            need_create_mesh = false;
        }
        volume_set.emplace(new_volume);
    }
    else {
        new_volume->indexed_vertex_array = std::make_shared<GLIndexedVertexArray>();

        std::set<GLVolume*> volume_set;
        volume_set.emplace(new_volume);
        g_mesh_volumes_map.emplace(mesh_ptr, std::move(volume_set));
    }
    if (need_create_mesh) {
#if ENABLE_SMOOTH_NORMALS
        v.indexed_vertex_array->load_mesh(mesh, true);
#else
        if (lod_enabled) {
            if (v.indexed_vertex_array_middle == nullptr)
            {
                v.indexed_vertex_array_middle = std::make_shared<GLIndexedVertexArray>();
            }

            v.simplify_mesh(mesh, v.indexed_vertex_array_middle, LOD_LEVEL::MIDDLE); // include finalize_geometry

            if (v.indexed_vertex_array_small == nullptr)
            {
                v.indexed_vertex_array_small = std::make_shared<GLIndexedVertexArray>();
            }
            v.simplify_mesh(mesh, v.indexed_vertex_array_small, LOD_LEVEL::SMALL);
        }

        v.indexed_vertex_array->load_mesh(mesh);
#endif // ENABLE_SMOOTH_NORMALS
        v.indexed_vertex_array->finalize_geometry(opengl_initialized);
    }
    v.composite_id = GLVolume::CompositeID(obj_idx, volume_idx, instance_idx);
    if (model_volume->is_model_part())
    {
        // GLVolume will reference a convex hull from model_volume!
        v.set_convex_hull(model_volume->get_convex_hull_shared_ptr());
        if (extruder_id != -1)
            v.extruder_id = extruder_id;
    }
    v.is_modifier = !model_volume->is_model_part();
    v.shader_outside_printer_detection_enabled = model_volume->is_model_part();
    if (in_assemble_view) {
        v.set_instance_transformation(instance->get_assemble_transformation());
        v.set_offset_to_assembly(instance->get_offset_to_assembly());
        // BBS: in the assembly view the per-volume assemble transformation contributes
        v.set_volume_transformation(model_volume->get_assemble_transformation());
    }
    else {
        v.set_instance_transformation(instance->get_transformation());
        v.set_volume_transformation(model_volume->get_transformation());
    }
    //use object's instance id
    if (use_loaded_id && (instance->loaded_id > 0))
        v.model_object_ID = instance->loaded_id;
    else
        v.model_object_ID = instance->id().id;

    return int(this->volumes.size() - 1);
}

void GLVolumeCollection::clear()
{
    for (auto* v : volumes)
    {
        release_volume(v);
        delete v;
    }
    volumes.clear();
}

void GLVolumeCollection::release_volume (GLVolume* volume)
{
    if (volume->ori_mesh) {
        std::map<const TriangleMesh*, std::set<GLVolume*>>::iterator iter = g_mesh_volumes_map.find(volume->ori_mesh);
        if (iter != g_mesh_volumes_map.end()) {
            std::set<GLVolume*> & volume_set = iter->second;

            volume_set.erase(volume);
            if (volume_set.empty())
                g_mesh_volumes_map.erase(iter);
        }
        else {
            //should not happen
        }
    }
}

const GLVolume* GLVolumeCollection::get_volume_by_composite_id(int obj_id, int vol_id, int instance_id) const
{
    for (const GLVolume* v : volumes)
        if (v->composite_id.object_id == obj_id && v->composite_id.volume_id == vol_id && v->composite_id.instance_id == instance_id)
            return v;
    return nullptr;
}

// Load SLA auxiliary GLVolumes (for support trees or pad).
// This function produces volumes for multiple instances in a single shot,
// as some object specific mesh conversions may be expensive.
void GLVolumeCollection::load_object_auxiliary(
    const SLAPrintObject 		   *print_object,
    int                             obj_idx,
    // pairs of <instance_idx, print_instance_idx>
    const std::vector<std::pair<size_t, size_t>>& instances,
    SLAPrintObjectStep              milestone,
    // Timestamp of the last change of the milestone
    size_t                          timestamp,
    bool 				 			opengl_initialized)
{
    assert(print_object->is_step_done(milestone));
    Transform3d  mesh_trafo_inv = print_object->trafo().inverse();
    // Get the support mesh.
    TriangleMesh mesh = print_object->get_mesh(milestone);
    mesh.transform(mesh_trafo_inv);
    // Convex hull is required for out of print bed detection.
    TriangleMesh convex_hull = mesh.convex_hull_3d();
    for (const std::pair<size_t, size_t>& instance_idx : instances) {
        const ModelInstance& model_instance = *print_object->model_object()->instances[instance_idx.first];
        this->volumes.emplace_back(new GLVolume((milestone == slaposPad) ? GLVolume::SLA_PAD_COLOR : GLVolume::SLA_SUPPORT_COLOR));
        GLVolume& v = *this->volumes.back();
#if ENABLE_SMOOTH_NORMALS
        v.indexed_vertex_array->load_mesh(mesh, true);
#else
        v.indexed_vertex_array->load_mesh(mesh);
#endif // ENABLE_SMOOTH_NORMALS
        v.indexed_vertex_array->finalize_geometry(opengl_initialized);
        v.composite_id = GLVolume::CompositeID(obj_idx, -int(milestone), (int)instance_idx.first);
        v.geometry_id = std::pair<size_t, size_t>(timestamp, model_instance.id().id);
        // Create a copy of the convex hull mesh for each instance. Use a move operator on the last instance.
        if (&instance_idx == &instances.back())
            v.set_convex_hull(std::move(convex_hull));
        else
            v.set_convex_hull(convex_hull);
        v.is_modifier = false;
        v.shader_outside_printer_detection_enabled = (milestone == slaposSupportTree);
        v.set_instance_transformation(model_instance.get_transformation());
        // Leave the volume transformation at identity.
        // v.set_volume_transformation(model_volume->get_transformation());
    }
}

int GLVolumeCollection::load_wipe_tower_preview(
    int obj_idx, float pos_x, float pos_y, float width, float depth, float height,
    float rotation_angle, bool size_unknown, float brim_width, bool opengl_initialized)
{
    int plate_idx = obj_idx - 1000;

    if (depth < 0.01f)
        return int(this->volumes.size() - 1);
    if (height == 0.0f)
        height = 0.1f;

    std::vector<std::array<float, 4>> extruder_colors = GUI::wxGetApp().plater()->get_extruders_colors();
    std::vector<std::array<float, 4>> colors;
    GUI::PartPlateList& ppl = GUI::wxGetApp().plater()->get_partplate_list();
    std::vector<int> plate_extruders = ppl.get_plate(plate_idx)->get_extruders(true);
    TriangleMesh wipe_tower_shell = make_cube(width, depth, height);
    for (int extruder_id : plate_extruders) {
        if (extruder_id > 0 && extruder_id <= (int)extruder_colors.size())
            colors.push_back(extruder_colors[extruder_id - 1]);
        else if (!extruder_colors.empty())
            colors.push_back(extruder_colors[0]);
    }

#if 0
    // We'll make another mesh to show the brim (fixed layer height):
    TriangleMesh brim_mesh = make_cube(width + 2.f * brim_width, depth + 2.f * brim_width, 0.2f);
    brim_mesh.translate(-brim_width, -brim_width, 0.f);
    mesh.merge(brim_mesh);
#endif

    volumes.emplace_back(new GLWipeTowerVolume(colors));
    GLWipeTowerVolume& v = *dynamic_cast<GLWipeTowerVolume*>(volumes.back());
    v.iva_per_colors.resize(colors.size());
    for (int i = 0; i < colors.size(); i++) {
        TriangleMesh color_part = make_cube(width, depth / colors.size(), height);
        color_part.translate({ 0.f, depth * i / colors.size(), 0. });
        v.iva_per_colors[i].load_mesh(color_part);
        v.iva_per_colors[i].finalize_geometry(opengl_initialized);
    }
    v.indexed_vertex_array->load_mesh(wipe_tower_shell);
    v.indexed_vertex_array->finalize_geometry(opengl_initialized);
    v.set_convex_hull(wipe_tower_shell);
    v.set_volume_offset(Vec3d(pos_x, pos_y, 0.0));
    v.set_volume_rotation(Vec3d(0., 0., (M_PI / 180.) * rotation_angle));
    v.composite_id = GLVolume::CompositeID(obj_idx, 0, 0);
    v.geometry_id.first = 0;
    v.geometry_id.second = wipe_tower_instance_id().id + (obj_idx - 1000);
    v.is_wipe_tower = true;
    v.shader_outside_printer_detection_enabled = !size_unknown;
    return int(volumes.size() - 1);
}

int GLVolumeCollection::load_real_wipe_tower_preview(
    int obj_idx, float pos_x, float pos_y, const TriangleMesh& wt_mesh,const TriangleMesh &brim_mesh,bool render_brim, float rotation_angle, bool size_unknown,  bool opengl_initialized)
{
    int plate_idx = obj_idx - 1000;
    if (wt_mesh.its.vertices.empty()) return int(this->volumes.size() - 1);

    std::vector<std::array<float, 4>> extruder_colors = GUI::wxGetApp().plater()->get_extruders_colors();
    GUI::PartPlateList               &ppl              = GUI::wxGetApp().plater()->get_partplate_list();
    std::vector<int>                  plate_extruders  = ppl.get_plate(plate_idx)->get_extruders(true);
    std::vector<std::array<float, 4>>              colors;
    if (!plate_extruders.empty()) {
        int front_id = plate_extruders.front();
        if (front_id > 0 && front_id <= (int)extruder_colors.size())
            colors.push_back(extruder_colors[front_id - 1]);
        else if (!extruder_colors.empty())
            colors.push_back(extruder_colors[0]);
    }
    if (colors.empty()) return int(this->volumes.size() - 1);
    volumes.emplace_back(new GLWipeTowerVolume({colors}));
    GLWipeTowerVolume &v = *dynamic_cast<GLWipeTowerVolume *>(volumes.back());
    auto mesh = wt_mesh;
    if (render_brim) {
        mesh.merge(brim_mesh);
    }
    if (!colors.empty()) {
        v.iva_per_colors.resize(1);
        v.iva_per_colors[0].load_mesh(mesh);
        v.iva_per_colors[0].finalize_geometry(opengl_initialized);
    }
    TriangleMesh wipe_tower_shell = mesh.convex_hull_3d();
    v.indexed_vertex_array->load_mesh(wipe_tower_shell);
    v.indexed_vertex_array->finalize_geometry(opengl_initialized);
    v.set_convex_hull(wipe_tower_shell);
    v.set_volume_offset(Vec3d(pos_x, pos_y, 0.0));
    v.set_volume_rotation(Vec3d(0., 0., (M_PI / 180.) * rotation_angle));
    v.composite_id                             = GLVolume::CompositeID(obj_idx, 0, 0);
    v.geometry_id.first                        = 0;
    v.geometry_id.second                       = wipe_tower_instance_id().id + (obj_idx - 1000);
    v.is_wipe_tower                            = true;
    v.shader_outside_printer_detection_enabled = !size_unknown;
    return int(volumes.size() - 1);
}


GLVolume* GLVolumeCollection::new_toolpath_volume(const std::array<float, 4>& rgba, size_t reserve_vbo_floats)
{
	GLVolume *out = new_nontoolpath_volume(rgba, reserve_vbo_floats);
	out->is_extrusion_path = true;
	return out;
}

GLVolume* GLVolumeCollection::new_nontoolpath_volume(const std::array<float, 4>& rgba, size_t reserve_vbo_floats)
{
	GLVolume *out = new GLVolume(rgba);
	out->is_extrusion_path = false;
	// Reserving number of vertices (3x position + 3x color)
	out->indexed_vertex_array->reserve(reserve_vbo_floats / 6);
	this->volumes.emplace_back(out);
	return out;
}

GLVolumeWithIdAndZList volumes_to_render(const GLVolumePtrs& volumes, GLVolumeCollection::ERenderType type, const Transform3d& view_matrix, std::function<bool(const GLVolume&)> filter_func)
{
    GLVolumeWithIdAndZList list;
    list.reserve(volumes.size());

    for (unsigned int i = 0; i < (unsigned int)volumes.size(); ++i) {
        GLVolume* volume = volumes[i];
        bool is_transparent = (volume->render_color[3] < 1.0f);
        auto tempGlwipeTowerVolume = dynamic_cast<GLWipeTowerVolume *>(volume);
        if (tempGlwipeTowerVolume) {
            is_transparent = tempGlwipeTowerVolume->IsTransparent();
        }
        if (((type == GLVolumeCollection::ERenderType::Opaque && !is_transparent) ||
            (type == GLVolumeCollection::ERenderType::Transparent && is_transparent) ||
             type == GLVolumeCollection::ERenderType::All) &&
            (! filter_func || filter_func(*volume)))
            list.emplace_back(std::make_pair(volume, std::make_pair(i, 0.0)));
    }

    if (type == GLVolumeCollection::ERenderType::Transparent && list.size() > 1) {
        for (GLVolumeWithIdAndZ& volume : list) {
            volume.second.second = volume.first->bounding_box().transformed(view_matrix * volume.first->world_matrix()).max(2);
        }

        std::sort(list.begin(), list.end(),
            [](const GLVolumeWithIdAndZ& v1, const GLVolumeWithIdAndZ& v2) -> bool { return v1.second.second < v2.second.second; }
        );
    }
    else if (type == GLVolumeCollection::ERenderType::Opaque && list.size() > 1) {
        std::sort(list.begin(), list.end(),
            [](const GLVolumeWithIdAndZ& v1, const GLVolumeWithIdAndZ& v2) -> bool { return v1.first->selected && !v2.first->selected; }
        );
    }

    return list;
}

GLVolumeWithIdAndZList volumes_to_render_for_sinking(const GLVolumePtrs &                  volumes,
                                         GLVolumeCollection::ERenderType       type,
                                         const Transform3d &                   view_matrix)
{
    GLVolumeWithIdAndZList list;
    list.reserve(volumes.size());

    for (unsigned int i = 0; i < (unsigned int) volumes.size(); ++i) {
        GLVolume *volume                = volumes[i];
        bool      is_transparent        = (volume->render_color[3] < 1.0f);
        auto      tempGlwipeTowerVolume = dynamic_cast<GLWipeTowerVolume *>(volume);
        if (tempGlwipeTowerVolume) {
            continue;
        }
        if (!volume->selected)
            continue;
        list.emplace_back(std::make_pair(volume, std::make_pair(i, 0.0)));
    }

    if (type == GLVolumeCollection::ERenderType::Transparent && list.size() > 1) {
        for (GLVolumeWithIdAndZ &volume : list) { volume.second.second = volume.first->bounding_box().transformed(view_matrix * volume.first->world_matrix()).max(2); }

        std::sort(list.begin(), list.end(), [](const GLVolumeWithIdAndZ &v1, const GLVolumeWithIdAndZ &v2) -> bool { return v1.second.second < v2.second.second; });
    } else if (type == GLVolumeCollection::ERenderType::Opaque && list.size() > 1) {
        std::sort(list.begin(), list.end(), [](const GLVolumeWithIdAndZ &v1, const GLVolumeWithIdAndZ &v2) -> bool { return v1.first->selected && !v2.first->selected; });
    }

    return list;
}

int GLVolumeCollection::get_selection_support_threshold_angle(bool &enable_support) const
{
    const DynamicPrintConfig& glb_cfg        = GUI::wxGetApp().preset_bundle->prints.get_edited_preset().config;
    enable_support =  glb_cfg.opt_bool("enable_support");
    int support_threshold_angle =  glb_cfg.opt_int("support_threshold_angle");
    return  support_threshold_angle ;
}

//BBS: add outline drawing logic
void GLVolumeCollection::render(GUI::ERenderPipelineStage             render_pipeline_stage,
                                GLVolumeCollection::ERenderType       type,
                                bool                                  disable_cullface,
                                const GUI::Camera &                      camera,
                                const std::vector<std::array<float, 4>>& colors,
                                Model& model,
                                std::function<bool(const GLVolume &)> filter_func,
                                bool                                  with_outline,
                                const std::array<float, 4> &          body_color,
                                bool                                  partly_inside_enable,
                                std::vector<double> *                 printable_heights)
{
    auto     view_matrix   = camera.get_view_matrix();
    auto projection_matrix = camera.get_projection_matrix();
    GLVolumeWithIdAndZList to_render = volumes_to_render(volumes, type, view_matrix, filter_func);
    if (to_render.empty())
        return;

    const auto shader = GUI::wxGetApp().get_current_shader();
    if (shader == nullptr)
        return;

    if (type == ERenderType::Transparent) {
        glsafe(::glEnable(GL_BLEND));
        glsafe(::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    }

    glsafe(::glCullFace(GL_BACK));
    if (disable_cullface) {
        glsafe(::glDisable(GL_CULL_FACE));
    }
    else {
        glsafe(::glEnable(GL_CULL_FACE));
    }

    for (GLVolumeWithIdAndZ& volume : to_render) {
        auto world_box = volume.first->transformed_bounding_box();
        if (!camera.getFrustum().intersects(world_box)) {
            continue;
        }
#if ENABLE_MODIFIERS_ALWAYS_TRANSPARENT
        if (type == ERenderType::Transparent) {
            volume.first->force_transparent = true;
            //BOOST_LOG_TRIVIAL(info) << boost::format("transparent rendering...");
        }
        //else
        //    BOOST_LOG_TRIVIAL(info) << boost::format("opaque rendering...");
#endif // ENABLE_MODIFIERS_ALWAYS_TRANSPARENT
        volume.first->set_render_color();
#if ENABLE_MODIFIERS_ALWAYS_TRANSPARENT
        if (type == ERenderType::Transparent)
            volume.first->force_transparent = false;
#endif // ENABLE_MODIFIERS_ALWAYS_TRANSPARENT

        // render sinking contours of non-hovered volumes
        if (GUI::ERenderPipelineStage::Silhouette != render_pipeline_stage) {
            if (m_show_sinking_contours)
                if (volume.first->is_sinking() && !volume.first->is_below_printbed() &&
                    volume.first->hover == GLVolume::HS_None && !volume.first->force_sinking_contours) {
                    GUI::wxGetApp().unbind_shader();
                    volume.first->render_sinking_contours(camera, model);
                    GUI::wxGetApp().bind_shader(shader);
                }
        }

        if (GUI::ERenderPipelineStage::Silhouette != render_pipeline_stage) {
            shader->set_uniform("is_text_shape", volume.first->is_text_shape);
            // Use volume color override if enabled
            const std::array<float, 4>& effective_color = get_volume_render_color(volume.second.first, volume.first->render_color);
            shader->set_uniform("uniform_color", effective_color);
            shader->set_uniform("z_range", m_z_range);
            shader->set_uniform("clipping_plane", m_clipping_plane);
            shader->set_uniform("use_color_clip_plane", m_use_color_clip_plane);
            shader->set_uniform("color_clip_plane", m_color_clip_plane);
            shader->set_uniform("uniform_color_clip_plane_1", m_color_clip_plane_colors[0]);
            shader->set_uniform("uniform_color_clip_plane_2", m_color_clip_plane_colors[1]);
            shader->set_uniform("use_dovetail_clip", m_use_dovetail_clip);
            shader->set_uniform("dovetail_clip_matrix", m_dovetail_clip_matrix);
            shader->set_uniform("dovetail_clip_params", m_dovetail_clip_params);
            shader->set_uniform("dovetail_clip_tolerance", m_dovetail_clip_tolerance);
            //BOOST_LOG_TRIVIAL(info) << boost::format("set uniform_color to {%1%, %2%, %3%, %4%}, with_outline=%5%, selected %6%")
            //    %volume.first->render_color[0]%volume.first->render_color[1]%volume.first->render_color[2]%volume.first->render_color[3]
            //    %with_outline%volume.first->selected;

            //BBS set print_volume to render volume
            //shader->set_uniform("print_volume.type", static_cast<int>(m_render_volume.type));
            //shader->set_uniform("print_volume.xy_data", m_render_volume.data);
            //shader->set_uniform("print_volume.z_data", m_render_volume.zs);
            if (printable_heights) {
                std::array<float, 3> extruder_printable_heights;
                if ((*printable_heights).size() > 1) {
                    extruder_printable_heights[0] = 2.0f;
                    extruder_printable_heights[1] = (*printable_heights)[0];
                    extruder_printable_heights[2] = (*printable_heights)[1];
                    shader->set_uniform("extruder_printable_heights", extruder_printable_heights);
                    shader->set_uniform("print_volume.xy_data", m_print_volume.data);
                }
                else {
                    extruder_printable_heights[0] = 0.0f;
                    shader->set_uniform("extruder_printable_heights", extruder_printable_heights);
                }
            }
            if (volume.first->partly_inside && partly_inside_enable) {
                //only partly inside volume need to be painted with boundary check
                shader->set_uniform("print_volume.type", static_cast<int>(m_print_volume.type));
                shader->set_uniform("print_volume.z_data", m_print_volume.zs);
                if (!printable_heights || (printable_heights && (*printable_heights).size() == 0)) {
                    shader->set_uniform("print_volume.xy_data", m_print_volume.data);
                }
            }
            else {
                //use -1 ad a invalid type
                shader->set_uniform("print_volume.type", -1);
            }

            bool  enable_support;
            int   support_threshold_angle = get_selection_support_threshold_angle(enable_support);

            float normal_z = -::cos(Geometry::deg2rad((float)support_threshold_angle));

            shader->set_uniform("volume_world_matrix", volume.first->world_matrix());
            shader->set_uniform("slope.actived", m_slope.isGlobalActive && !volume.first->is_modifier && !volume.first->is_wipe_tower);
            shader->set_uniform("slope.volume_world_normal_matrix", static_cast<Matrix3f>(volume.first->world_matrix().matrix().block(0, 0, 3, 3).inverse().transpose().cast<float>()));
            shader->set_uniform("slope.normal_z", normal_z);

#if ENABLE_ENVIRONMENT_MAP
            unsigned int environment_texture_id = GUI::wxGetApp().plater()->get_environment_texture_id();
            bool use_environment_texture = environment_texture_id > 0 && GUI::wxGetApp().app_config->get("use_environment_map") == "1";
            shader->set_uniform("use_environment_tex", use_environment_texture);
            if (use_environment_texture)
                glsafe(::glBindTexture(GL_TEXTURE_2D, environment_texture_id));
#endif // ENABLE_ENVIRONMENT_MAP
            glcheck();

            const Transform3d matrix = view_matrix * volume.first->world_matrix();
            shader->set_uniform("view_model_matrix", matrix);
            shader->set_uniform("projection_matrix", projection_matrix);
            shader->set_uniform("normal_matrix", (Matrix3d)matrix.matrix().block(0, 0, 3, 3).inverse().transpose());

            //BBS: add outline related logic
            auto red_color = std::array<float, 4>({ 1.0f, 0.0f, 0.0f, 1.0f });//slice_error
            // Only disable MMU colors if this specific volume has a color override
            bool this_volume_has_override = m_use_volume_color_override &&
                                           m_volume_color_overrides.find(volume.second.first) != m_volume_color_overrides.end();
            volume.first->render(camera, colors,model,with_outline && volume.first->selected, volume.first->slice_error ? red_color : body_color, this_volume_has_override);
        }
        else {
            if (volume.first->selected) {
                shader->set_uniform("u_model_matrix", volume.first->world_matrix());
                // Only disable MMU colors if this specific volume has a color override
                bool this_volume_has_override = m_use_volume_color_override &&
                                               m_volume_color_overrides.find(volume.second.first) != m_volume_color_overrides.end();
                volume.first->render(camera, colors, model, false, body_color, this_volume_has_override);
            }
        }

#if ENABLE_ENVIRONMENT_MAP
        if (use_environment_texture)
            glsafe(::glBindTexture(GL_TEXTURE_2D, 0));
#endif // ENABLE_ENVIRONMENT_MAP

        glsafe(::glBindBuffer(GL_ARRAY_BUFFER, 0));
        glsafe(::glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0));
    }

    if (GUI::ERenderPipelineStage::Silhouette != render_pipeline_stage) {
        if (m_show_sinking_contours) {
            for (GLVolumeWithIdAndZ& volume : to_render) {
                // render sinking contours of hovered/displaced volumes
                if (volume.first->is_sinking() && !volume.first->is_below_printbed() &&
                    (volume.first->hover != GLVolume::HS_None || volume.first->force_sinking_contours)) {
                    GUI::wxGetApp().unbind_shader();
                    glsafe(::glDepthFunc(GL_ALWAYS));
                    volume.first->render_sinking_contours(camera, model);
                    glsafe(::glDepthFunc(GL_LESS));
                    GUI::wxGetApp().bind_shader(shader);
                }
            }
        }
    }

    if (disable_cullface)
        glsafe(::glEnable(GL_CULL_FACE));

    if (type == ERenderType::Transparent)
        glsafe(::glDisable(GL_BLEND));
}

void GLVolumeCollection::only_render_sinking(GUI::ERenderPipelineStage                render_pipeline_stage,
                                             ERenderType                              type,
                                             bool                                     disable_cullface,
                                             const GUI::Camera &                      camera,
                                             const std::vector<std::array<float, 4>> &colors,
                                             Model &                                  model,
                                             std::function<bool(const GLVolume &)>    filter_func,
                                             bool                                     with_outline,
                                             const std::array<float, 4> &             body_color,
                                             bool                                     partly_inside_enable,
                                             std::vector<double> *                    printable_heights)
{
    auto                   view_matrix       = camera.get_view_matrix();
    auto                   projection_matrix = camera.get_projection_matrix();
    GLVolumeWithIdAndZList to_render         = volumes_to_render_for_sinking(volumes, type, view_matrix);
    if (to_render.empty())
        return;

    const auto shader = GUI::wxGetApp().get_current_shader();
    if (shader == nullptr)
        return;

    if (type == ERenderType::Transparent) {
        glsafe(::glEnable(GL_BLEND));
        glsafe(::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));
    }

    glsafe(::glDisable(GL_CULL_FACE));
    if (GUI::ERenderPipelineStage::Silhouette != render_pipeline_stage) {
        if (m_show_sinking_contours) {
            for (GLVolumeWithIdAndZ &volume : to_render) {
                if (volume.first->is_sinking() && !volume.first->is_below_printbed()) {
                    GUI::wxGetApp().unbind_shader();
                    glsafe(::glDepthFunc(GL_ALWAYS));
                    volume.first->render_sinking_contours(camera, model);
                    glsafe(::glDepthFunc(GL_LESS));
                    GUI::wxGetApp().bind_shader(shader);
                }
            }
        }
    }

    if (disable_cullface)
        glsafe(::glEnable(GL_CULL_FACE));

    if (type == ERenderType::Transparent)
        glsafe(::glDisable(GL_BLEND));
}

bool GLVolumeCollection::check_wipe_tower_outside_state(const Slic3r::BuildVolume &build_volume, int plate_id) const
{
    for (GLVolume *volume : this->volumes) {
        if (volume->is_wipe_tower) {
            int wipe_tower_plate_id = volume->composite_id.object_id - 1000;
            if (wipe_tower_plate_id != plate_id)
                continue;
            const std::vector<Vec2d>& printable_area = build_volume.printable_area();
            Polygon printable_poly = Polygon::new_scale(printable_area);

            // multi-extruder
            Polygons extruder_polys;
            const std::vector<std::vector<Vec2d>> & extruder_areas = build_volume.extruder_areas();
            if (!extruder_areas.empty()) {
                for (size_t i = 0; i < extruder_areas.size(); ++i) {
                    extruder_polys.emplace_back(Polygon::new_scale(extruder_areas[i]));
                }
                extruder_polys = union_(extruder_polys);
                if (extruder_polys.empty())
                    return false;

                printable_poly = extruder_polys[0];
            }

            const BoundingBoxf3 &bbox = volume->transformed_convex_hull_bounding_box();
            Polygon wipe_tower_polygon = bbox.polygon(true);

            Polygons diff_res = diff(wipe_tower_polygon, printable_poly);
            return diff_res.empty();
        }
    }
    return true;
}

bool GLVolumeCollection::check_outside_state(const BuildVolume &build_volume, ModelInstanceEPrintVolumeState *out_state, ObjectFilamentResults *object_results, Model &model) const
{
    if (GUI::wxGetApp().plater() == NULL)
    {
        if (out_state != nullptr)
            *out_state = ModelInstancePVS_Inside;
        return false;
    }

    auto                volume_below       = [](GLVolume& volume) -> bool
        { return volume.object_idx() != -1 && volume.volume_idx() != -1 && volume.is_below_printbed(); };
    // Volume is partially below the print bed, thus a pre-calculated convex hull cannot be used.
    auto                volume_sinking     = [](GLVolume& volume) -> bool
        { return volume.object_idx() != -1 && volume.volume_idx() != -1 && volume.is_sinking(); };
    // Cached bounding box of a volume above the print bed.
    auto                volume_bbox        = [volume_sinking ,&model](GLVolume& volume) -> BoundingBoxf3 {
        return volume_sinking(volume) ? volume.transformed_non_sinking_bounding_box(model) : volume.transformed_convex_hull_bounding_box();
    };
    // Cached 3D convex hull of a volume above the print bed.
    auto                volume_convex_mesh = [volume_sinking, &model](GLVolume& volume) -> const TriangleMesh&
        { return volume_sinking(volume) ? model.objects[volume.object_idx()]->volumes[volume.volume_idx()]->mesh() : *volume.convex_hull(); };

    ModelInstanceEPrintVolumeState overall_state = ModelInstancePVS_Fully_Outside;
    bool contained_min_one = false;

    // Track the print volume state of each object instance across its volumes.
    std::map<int64_t, ModelInstanceEPrintVolumeState> model_state;

    GUI::PartPlate* curr_plate = GUI::wxGetApp().plater()->get_partplate_list().get_selected_plate();
    const Pointfs& pp_bed_shape = curr_plate->get_shape();
    BuildVolume plate_build_volume(pp_bed_shape, build_volume.printable_height(), build_volume.extruder_areas(), build_volume.extruder_heights());
    const std::vector<BoundingBoxf3>& exclude_areas = curr_plate->get_exclude_areas();

    const GlobalFeatureFilaments global_features = GlobalFeatureFilaments::from_config(GUI::wxGetApp().preset_bundle->prints.get_edited_preset().config, GUI::wxGetApp().filaments_cnt());
    std::map<ModelObject*, std::map<int, std::set<int>>> objects_unprintable_filaments;
    int extruder_count = build_volume.get_extruder_area_count();
    std::vector<std::set<int>> unprintable_filament_ids(extruder_count, std::set<int>());
    std::set<ModelObject*> partly_objects_set;
    const ModelObjectPtrs &model_objects = model.objects;
    for (GLVolume* volume : this->volumes)
    {
        std::vector<bool> inside_extruders;
        if (! volume->is_modifier && (volume->shader_outside_printer_detection_enabled || (! volume->is_wipe_tower && volume->composite_id.volume_id >= 0))) {
            BuildVolume::ObjectState state;
            const BoundingBoxf3& bb = volume_bbox(*volume);
            if (volume_below(*volume))
                state = BuildVolume::ObjectState::Below;
            else {
                switch (plate_build_volume.type()) {
                case BuildVolume::Type::Rectangle:
                //FIXME this test does not evaluate collision of a build volume bounding box with non-convex objects.
                    state = plate_build_volume.volume_state_bbox(bb);
                    if ((state == BuildVolume::ObjectState::Inside) && (extruder_count > 1) && !GUI::wxGetApp().plater()->force_ban_check_volume_bbox_state_with_extruder_area())
                    {
                        state = plate_build_volume.check_volume_bbox_state_with_extruder_areas(bb, inside_extruders);
                    }
                    break;
                case BuildVolume::Type::Circle:
                case BuildVolume::Type::Convex:
                //FIXME doing test on convex hull until we learn to do test on non-convex polygons efficiently.
                case BuildVolume::Type::Custom:
                {
                    const indexed_triangle_set& convex_mesh_it = volume_convex_mesh(*volume).its;
                    const Transform3f trafo = volume->world_matrix().cast<float>();
                    state = plate_build_volume.object_state(convex_mesh_it, trafo, volume_sinking(*volume));
                    if ((state == BuildVolume::ObjectState::Inside) && (extruder_count > 1))
                    {
                        state = plate_build_volume.check_object_state_with_extruder_areas(convex_mesh_it, trafo, inside_extruders);
                    }
                    break;
                }
                default:
                    // Ignore, don't produce any collision.
                    state = BuildVolume::ObjectState::Inside;
                    break;
                }
                assert(state != BuildVolume::ObjectState::Below);

                if (state == BuildVolume::ObjectState::Limited) {
                    //unprintable_filament_ids.resize(inside_extruders.size());
                    ModelObject *model_object = model_objects[volume->object_idx()];
                    // Only check for single-color object
                    std::set<int> object_filaments;
                    for (ModelVolume *m_volume : model_object->volumes) {
                        std::vector<int> filaments = m_volume->get_extruders(global_features);
                        object_filaments.insert(filaments.begin(), filaments.end());
                    }
                    if (object_filaments.size() == 1) {
                        ModelVolume *model_volume = model_object->volumes[volume->volume_idx()];
                        for (size_t i = 0; i < inside_extruders.size(); ++i) {
                            if (!inside_extruders[i]) {
                                std::vector<int> filaments = model_volume->get_extruders(global_features);
                                unprintable_filament_ids[i].insert(filaments.begin(), filaments.end());
                                if (object_results) {
                                    std::map<int, std::set<int>> &obj_extruder_filament_maps = objects_unprintable_filaments[model_object];
                                    std::set<int>                &obj_extruder_filaments     = obj_extruder_filament_maps[i + 1];
                                    obj_extruder_filaments.insert(filaments.begin(), filaments.end());
                                }
                            }
                        }
                    }
                }
            }

            const int64_t comp_id = ((int64_t)volume->composite_id.object_id << 32) | ((int64_t)volume->composite_id.instance_id);
            volume->is_outside = (state != BuildVolume::ObjectState::Inside && state != BuildVolume::ObjectState::Limited);
            volume->partly_inside = (state == BuildVolume::ObjectState::Colliding);
            if (volume->printable) {
                if (state == BuildVolume::ObjectState::Colliding)
                {
                    overall_state = ModelInstancePVS_Partly_Outside;
                    partly_objects_set.emplace(model_objects[volume->object_idx()]);
                }
                else if ((state == BuildVolume::ObjectState::Limited) && (overall_state != ModelInstancePVS_Partly_Outside))
                    overall_state = ModelInstancePVS_Limited;
                else if ((state == BuildVolume::ObjectState::Inside) && (overall_state == ModelInstancePVS_Fully_Outside)) {
                    overall_state = ModelInstancePVS_Fully_Outside;
                }
                contained_min_one |= !volume->is_outside;
                ModelInstanceEPrintVolumeState volume_state;
                if (volume->is_outside && (state == BuildVolume::ObjectState::Colliding))
                    volume_state = ModelInstancePVS_Partly_Outside;
                else if (volume->is_outside)
                    volume_state = ModelInstancePVS_Fully_Outside;
                else
                    volume_state = ModelInstancePVS_Inside;

                const auto it = model_state.find(comp_id);
                if (it != model_state.end()) {
                    if (it->second != ModelInstancePVS_Partly_Outside) {
                        if (volume_state == ModelInstancePVS_Partly_Outside || it->second != volume_state)
                            it->second = ModelInstancePVS_Partly_Outside;
                    }
                } else {
                    model_state.emplace(comp_id, volume_state);
                }

                if (model_state[comp_id] == ModelInstancePVS_Partly_Outside) {
                    overall_state = ModelInstancePVS_Partly_Outside;
                    partly_objects_set.emplace(model_objects[volume->object_idx()]);
                }
            }
        }
    }

    for (GLVolume* volume : this->volumes)
    {
        if (!volume->is_modifier && (volume->shader_outside_printer_detection_enabled || (!volume->is_wipe_tower && volume->composite_id.volume_id >= 0))) {
            const int64_t comp_id = ((int64_t)volume->composite_id.object_id << 32) | ((int64_t)volume->composite_id.instance_id);
            const auto it = model_state.find(comp_id);
            if (it != model_state.end() && it->second == ModelInstancePVS_Partly_Outside)
                volume->partly_inside = true;
        }
    }

    std::vector<std::vector<int>> unprintable_filament_vec;
    for (const std::set<int>& filamnt_ids : unprintable_filament_ids) {
        unprintable_filament_vec.emplace_back(std::vector<int>(filamnt_ids.begin(), filamnt_ids.end()));
    }

    if (object_results && !partly_objects_set.empty()) {
        object_results->partly_outside_objects = std::vector<ModelObject*>(partly_objects_set.begin(), partly_objects_set.end());
    }

    //check per-object error for extruder areas
    if (object_results && (extruder_count > 1))
    {
        const auto& project_config = Slic3r::GUI::wxGetApp().preset_bundle->project_config;
        object_results->mode = curr_plate->get_real_filament_map_mode(project_config);
        if (is_auto_filament_map_mode(object_results->mode))
        {
            std::vector<int> conflict_filament_vector;
            for (int index = 0; index < extruder_count; index++ )
            {
                if (!unprintable_filament_vec[index].empty())
                {
                    std::sort (unprintable_filament_vec[index].begin(), unprintable_filament_vec[index].end());
                    if (index == 0)
                        conflict_filament_vector = unprintable_filament_vec[index];
                    else
                    {
                        std::vector<int> result_filaments;
                        //result_filaments.reserve(conflict_filaments.size());
                        std::set_intersection (conflict_filament_vector.begin(), conflict_filament_vector.end(), unprintable_filament_vec[index].begin(), unprintable_filament_vec[index].end(), insert_iterator<vector<int>>(result_filaments, result_filaments.begin()));
                        conflict_filament_vector = result_filaments;
                    }
                }
                else
                {
                    conflict_filament_vector.clear();
                    break;
                }
            }

            if (!conflict_filament_vector.empty())
            {
                std::set<int> conflict_filaments_set(conflict_filament_vector.begin(), conflict_filament_vector.end());
                object_results->filaments = conflict_filament_vector;

                for (auto& object_map: objects_unprintable_filaments)
                {
                    ModelObject *model_object = object_map.first;
                    std::map<int, std::set<int>>& obj_extruder_filament_maps = object_map.second;
                    std::set<int> obj_filaments_set;
                    ObjectFilamentInfo object_filament_info;
                    object_filament_info.object = model_object;

                    for (std::map<int, std::set<int>>::iterator extruder_map_iter = obj_extruder_filament_maps.begin(); extruder_map_iter != obj_extruder_filament_maps.end(); extruder_map_iter++ )
                    {
                        int extruder_id = extruder_map_iter->first;
                        std::set<int>& filaments_set = extruder_map_iter->second;

                        for (int filament: filaments_set)
                        {
                            if (conflict_filaments_set.find(filament) != conflict_filaments_set.end())
                            {
                                obj_filaments_set.emplace(filament);
                            }
                        }
                    }
                    if (!obj_filaments_set.empty()) {
                        object_filament_info.auto_filaments = std::vector<int>(obj_filaments_set.begin(), obj_filaments_set.end());
                        object_results->object_filaments.push_back(std::move(object_filament_info));
                    }
                }
            }
        }
        else
        {
            std::set<int> conflict_filaments_set;
            const auto& project_config = Slic3r::GUI::wxGetApp().preset_bundle->project_config;
            std::vector<int> filament_maps = curr_plate->get_real_filament_maps(project_config);
            for (auto& object_map: objects_unprintable_filaments)
            {
                ModelObject *model_object = object_map.first;
                std::map<int, std::set<int>>& obj_extruder_filament_maps = object_map.second;
                ObjectFilamentInfo object_filament_info;
                object_filament_info.object = model_object;

                for (std::map<int, std::set<int>>::iterator extruder_map_iter = obj_extruder_filament_maps.begin(); extruder_map_iter != obj_extruder_filament_maps.end(); extruder_map_iter++ )
                {
                    int extruder_id = extruder_map_iter->first;
                    std::set<int>& filaments_set = extruder_map_iter->second;

                    for (int filament: filaments_set)
                    {
                        if (filament_maps[filament - 1] == extruder_id)
                        {
                            object_filament_info.manual_filaments.emplace(filament, extruder_id);
                            object_results->filament_maps[filament] = extruder_id;
                            conflict_filaments_set.emplace(filament);
                        }
                    }
                }
                if (!object_filament_info.manual_filaments.empty())
                {
                    object_results->object_filaments.push_back(std::move(object_filament_info));
                }
            }
            if (!conflict_filaments_set.empty()) {
                object_results->filaments = std::vector<int>(conflict_filaments_set.begin(), conflict_filaments_set.end());
            }
        }
    }

    /*for (GLVolume* volume : this->volumes)
    {
        if (! volume->is_modifier && (volume->shader_outside_printer_detection_enabled || (! volume->is_wipe_tower && volume->composite_id.volume_id >= 0)))
        {
            int64_t comp_id = ((int64_t)volume->composite_id.object_id << 32) | ((int64_t)volume->composite_id.instance_id);
            if (model_state.find(comp_id) != model_state.end())
            {
                if (model_state[comp_id] == ModelInstancePVS_Partly_Outside) {
                    volume->partly_inside = true;
                }
                else
                    volume->partly_inside = false;
            }
        }
    }*/

    if (out_state != nullptr)
        *out_state = overall_state;

    return contained_min_one;
}

void GLVolumeCollection::reset_outside_state()
{
    for (GLVolume* volume : this->volumes)
    {
        if (volume != nullptr) {
            volume->is_outside = false;
            volume->partly_inside = false;
        }
    }
}

void GLVolumeCollection::update_colors_by_extruder(const DynamicPrintConfig *config, bool is_update_alpha)
{
    static const float inv_255 = 1.0f / 255.0f;

    struct Color
    {
        std::string text;
        unsigned char rgba[4];

        Color()
            : text("")
        {
            rgba[0] = 255;
            rgba[1] = 255;
            rgba[2] = 255;
            rgba[3] = 255;
        }

        void set(const std::string& text, unsigned char* rgba)
        {
            this->text = text;
            ::memcpy((void*)this->rgba, (const void*)rgba, 4 * sizeof(unsigned char));
        }
    };

    if (config == nullptr)
        return;

    unsigned char rgba[4];
    std::vector<Color> colors;

    if (config->has("printer_technology") && static_cast<PrinterTechnology>(config->opt_int("printer_technology")) == ptSLA)
    {
        const std::string& txt_color = config->opt_string("material_colour").empty() ?
                                       print_config_def.get("material_colour")->get_default_value<ConfigOptionString>()->value :
                                       config->opt_string("material_colour");
        if (Slic3r::GUI::BitmapCache::parse_color4(txt_color, rgba)) {
            colors.resize(1);
            colors[0].set(txt_color, rgba);
        }
    }
    else
    {
        if (!config->has("filament_colour")) {
            return;
        }
        const ConfigOptionStrings* filamemts_opt = dynamic_cast<const ConfigOptionStrings*>(config->option("filament_colour"));
        if (filamemts_opt == nullptr)
            return;

        unsigned int colors_count = (unsigned int)filamemts_opt->values.size();
        if (colors_count == 0)
            return;
        colors.resize(colors_count);

        for (unsigned int i = 0; i < colors_count; ++i) {
            const std::string& txt_color = config->opt_string("filament_colour", i);
            if (Slic3r::GUI::BitmapCache::parse_color4(txt_color, rgba))
                colors[i].set(txt_color, rgba);
        }
    }

    for (GLVolume* volume : volumes) {
        if (volume == nullptr || volume->is_modifier || volume->is_wipe_tower || (volume->volume_idx() < 0))
            continue;

        int extruder_id = volume->extruder_id - 1;
        if (extruder_id < 0 || (int)colors.size() <= extruder_id)
            extruder_id = 0;

        const Color& color = colors[extruder_id];
        if (!color.text.empty()) {
            for (int i = 0; i < 4; ++i) {
                if (is_update_alpha == false) {
                    if (i < 3) {
                        volume->color[i] = (float) color.rgba[i] * inv_255;
                    }
                    continue;
                }
                volume->color[i] = (float) color.rgba[i] * inv_255;
            }
        }
    }
}

std::vector<double> GLVolumeCollection::get_current_print_zs(bool active_only) const
{
    // Collect layer top positions of all volumes.
    std::vector<double> print_zs;
    for (GLVolume *vol : this->volumes)
    {
        if (!active_only || vol->is_active)
            append(print_zs, vol->print_zs);
    }
    std::sort(print_zs.begin(), print_zs.end());

    // Replace intervals of layers with similar top positions with their average value.
    int n = int(print_zs.size());
    int k = 0;
    for (int i = 0; i < n;) {
        int j = i + 1;
        coordf_t zmax = print_zs[i] + EPSILON;
        for (; j < n && print_zs[j] <= zmax; ++ j) ;
        print_zs[k ++] = (j > i + 1) ? (0.5 * (print_zs[i] + print_zs[j - 1])) : print_zs[i];
        i = j;
    }
    if (k < n)
        print_zs.erase(print_zs.begin() + k, print_zs.end());

    return print_zs;
}

size_t GLVolumeCollection::cpu_memory_used() const
{
	size_t memsize = sizeof(*this) + this->volumes.capacity() * sizeof(GLVolume);
	for (const GLVolume *volume : this->volumes)
		memsize += volume->cpu_memory_used();
	return memsize;
}

size_t GLVolumeCollection::gpu_memory_used() const
{
	size_t memsize = 0;
	for (const GLVolume *volume : this->volumes)
		memsize += volume->gpu_memory_used();
	return memsize;
}

std::string GLVolumeCollection::log_memory_info() const
{
	return " (GLVolumeCollection RAM: " + format_memsize_MB(this->cpu_memory_used()) + " GPU: " + format_memsize_MB(this->gpu_memory_used()) + " Both: " + format_memsize_MB(this->gpu_memory_used()) + ")";
}

// caller is responsible for supplying NO lines with zero length
static void thick_lines_to_indexed_vertex_array(
    const Lines                 &lines,
    const std::vector<double>   &widths,
    const std::vector<double>   &heights,
    bool                         closed,
    double                       top_z,
    GLIndexedVertexArray        &volume)
{
    assert(! lines.empty());
    if (lines.empty())
        return;

#define LEFT    0
#define RIGHT   1
#define TOP     2
#define BOTTOM  3

    // right, left, top, bottom
    int     idx_prev[4]      = { -1, -1, -1, -1 };
    double  bottom_z_prev    = 0.;
    Vec2d   b1_prev(Vec2d::Zero());
    Vec2d   v_prev(Vec2d::Zero());
    int     idx_initial[4]   = { -1, -1, -1, -1 };
    double  width_initial    = 0.;
    double  bottom_z_initial = 0.0;
    double  len_prev = 0.0;

    // loop once more in case of closed loops
    size_t lines_end = closed ? (lines.size() + 1) : lines.size();
    for (size_t ii = 0; ii < lines_end; ++ ii) {
        size_t i = (ii == lines.size()) ? 0 : ii;
        const Line &line = lines[i];
        double bottom_z = top_z - heights[i];
        double middle_z = 0.5 * (top_z + bottom_z);
        double width = widths[i];

        bool is_first = (ii == 0);
        bool is_last = (ii == lines_end - 1);
        bool is_closing = closed && is_last;

        Vec2d v = unscale(line.vector()).normalized();
        double len = unscale<double>(line.length());

        Vec2d a = unscale(line.a);
        Vec2d b = unscale(line.b);
        Vec2d a1 = a;
        Vec2d a2 = a;
        Vec2d b1 = b;
        Vec2d b2 = b;
        {
            double dist = 0.5 * width;  // scaled
            double dx = dist * v(0);
            double dy = dist * v(1);
            a1 += Vec2d(+dy, -dx);
            a2 += Vec2d(-dy, +dx);
            b1 += Vec2d(+dy, -dx);
            b2 += Vec2d(-dy, +dx);
        }

        // calculate new XY normals
        Vec2d xy_right_normal = unscale(line.normal()).normalized();

        int idx_a[4] = { 0, 0, 0, 0 }; // initialized to avoid warnings
        int idx_b[4] = { 0, 0, 0, 0 }; // initialized to avoid warnings
        int idx_last = int(volume.vertices_and_normals_interleaved.size() / 6);

        bool bottom_z_different = bottom_z_prev != bottom_z;
        bottom_z_prev = bottom_z;

        if (!is_first && bottom_z_different)
        {
            // Found a change of the layer thickness -> Add a cap at the end of the previous segment.
            volume.push_quad(idx_b[BOTTOM], idx_b[LEFT], idx_b[TOP], idx_b[RIGHT]);
        }

        // Share top / bottom vertices if possible.
        if (is_first) {
            idx_a[TOP] = idx_last++;
            volume.push_geometry(a(0), a(1), top_z   , 0., 0.,  1.);
        } else {
            idx_a[TOP] = idx_prev[TOP];
        }

        if (is_first || bottom_z_different) {
            // Start of the 1st line segment or a change of the layer thickness while maintaining the print_z.
            idx_a[BOTTOM] = idx_last ++;
            volume.push_geometry(a(0), a(1), bottom_z, 0., 0., -1.);
            idx_a[LEFT ] = idx_last ++;
            volume.push_geometry(a2(0), a2(1), middle_z, -xy_right_normal(0), -xy_right_normal(1), 0.0);
            idx_a[RIGHT] = idx_last ++;
            volume.push_geometry(a1(0), a1(1), middle_z, xy_right_normal(0), xy_right_normal(1), 0.0);
        }
        else {
            idx_a[BOTTOM] = idx_prev[BOTTOM];
        }

        if (is_first) {
            // Start of the 1st line segment.
            width_initial    = width;
            bottom_z_initial = bottom_z;
            memcpy(idx_initial, idx_a, sizeof(int) * 4);
        } else {
            // Continuing a previous segment.
            // Share left / right vertices if possible.
			double v_dot    = v_prev.dot(v);
            // To reduce gpu memory usage, we try to reuse vertices
            // To reduce the visual artifacts, due to averaged normals, we allow to reuse vertices only when any of two adjacent edges
            // is longer than a fixed threshold.
            // The following value is arbitrary, it comes from tests made on a bunch of models showing the visual artifacts
            double len_threshold = 2.5;

            // Generate new vertices if the angle between adjacent edges is greater than 45 degrees or thresholds conditions are met
            bool sharp = (v_dot < 0.707) || (len_prev > len_threshold) || (len > len_threshold);
            if (sharp) {
                if (!bottom_z_different)
                {
                    // Allocate new left / right points for the start of this segment as these points will receive their own normals to indicate a sharp turn.
                    idx_a[RIGHT] = idx_last++;
                    volume.push_geometry(a1(0), a1(1), middle_z, xy_right_normal(0), xy_right_normal(1), 0.0);
                    idx_a[LEFT] = idx_last++;
                    volume.push_geometry(a2(0), a2(1), middle_z, -xy_right_normal(0), -xy_right_normal(1), 0.0);
                    if (cross2(v_prev, v) > 0.) {
                        // Right turn. Fill in the right turn wedge.
                        volume.push_triangle(idx_prev[RIGHT], idx_a[RIGHT], idx_prev[TOP]);
                        volume.push_triangle(idx_prev[RIGHT], idx_prev[BOTTOM], idx_a[RIGHT]);
                    }
                    else {
                        // Left turn. Fill in the left turn wedge.
                        volume.push_triangle(idx_prev[LEFT], idx_prev[TOP], idx_a[LEFT]);
                        volume.push_triangle(idx_prev[LEFT], idx_a[LEFT], idx_prev[BOTTOM]);
                    }
                }
            }
            else
            {
                if (!bottom_z_different)
                {
                    // The two successive segments are nearly collinear.
                    idx_a[LEFT ] = idx_prev[LEFT];
                    idx_a[RIGHT] = idx_prev[RIGHT];
                }
            }
            if (is_closing) {
                if (!sharp) {
                    if (!bottom_z_different)
                    {
                        // Closing a loop with smooth transition. Unify the closing left / right vertices.
                        memcpy(volume.vertices_and_normals_interleaved.data() + idx_initial[LEFT ] * 6, volume.vertices_and_normals_interleaved.data() + idx_prev[LEFT ] * 6, sizeof(float) * 6);
                        memcpy(volume.vertices_and_normals_interleaved.data() + idx_initial[RIGHT] * 6, volume.vertices_and_normals_interleaved.data() + idx_prev[RIGHT] * 6, sizeof(float) * 6);
                        volume.vertices_and_normals_interleaved.erase(volume.vertices_and_normals_interleaved.end() - 12, volume.vertices_and_normals_interleaved.end());
                        // Replace the left / right vertex indices to point to the start of the loop.
                        for (size_t u = volume.quad_indices.size() - 16; u < volume.quad_indices.size(); ++ u) {
                            if (volume.quad_indices[u] == idx_prev[LEFT])
                                volume.quad_indices[u] = idx_initial[LEFT];
                            else if (volume.quad_indices[u] == idx_prev[RIGHT])
                                volume.quad_indices[u] = idx_initial[RIGHT];
                        }
                    }
                }
                // This is the last iteration, only required to solve the transition.
                break;
            }
        }

        // Only new allocate top / bottom vertices, if not closing a loop.
        if (is_closing) {
            idx_b[TOP] = idx_initial[TOP];
        } else {
            idx_b[TOP] = idx_last ++;
            volume.push_geometry(b(0), b(1), top_z   , 0., 0.,  1.);
        }

        if (is_closing && (width == width_initial) && (bottom_z == bottom_z_initial)) {
            idx_b[BOTTOM] = idx_initial[BOTTOM];
        } else {
            idx_b[BOTTOM] = idx_last ++;
            volume.push_geometry(b(0), b(1), bottom_z, 0., 0., -1.);
        }
        // Generate new vertices for the end of this line segment.
        idx_b[LEFT  ] = idx_last ++;
        volume.push_geometry(b2(0), b2(1), middle_z, -xy_right_normal(0), -xy_right_normal(1), 0.0);
        idx_b[RIGHT ] = idx_last ++;
        volume.push_geometry(b1(0), b1(1), middle_z, xy_right_normal(0), xy_right_normal(1), 0.0);

        memcpy(idx_prev, idx_b, 4 * sizeof(int));
        bottom_z_prev = bottom_z;
        b1_prev = b1;
        v_prev = v;
        len_prev = len;

        if (bottom_z_different && (closed || (!is_first && !is_last)))
        {
            // Found a change of the layer thickness -> Add a cap at the beginning of this segment.
            volume.push_quad(idx_a[BOTTOM], idx_a[RIGHT], idx_a[TOP], idx_a[LEFT]);
        }

        if (! closed) {
            // Terminate open paths with caps.
            if (is_first)
                volume.push_quad(idx_a[BOTTOM], idx_a[RIGHT], idx_a[TOP], idx_a[LEFT]);
            // We don't use 'else' because both cases are true if we have only one line.
            if (is_last)
                volume.push_quad(idx_b[BOTTOM], idx_b[LEFT], idx_b[TOP], idx_b[RIGHT]);
        }

        // Add quads for a straight hollow tube-like segment.
        // bottom-right face
        volume.push_quad(idx_a[BOTTOM], idx_b[BOTTOM], idx_b[RIGHT], idx_a[RIGHT]);
        // top-right face
        volume.push_quad(idx_a[RIGHT], idx_b[RIGHT], idx_b[TOP], idx_a[TOP]);
        // top-left face
        volume.push_quad(idx_a[TOP], idx_b[TOP], idx_b[LEFT], idx_a[LEFT]);
        // bottom-left face
        volume.push_quad(idx_a[LEFT], idx_b[LEFT], idx_b[BOTTOM], idx_a[BOTTOM]);
    }

#undef LEFT
#undef RIGHT
#undef TOP
#undef BOTTOM
}

// caller is responsible for supplying NO lines with zero length
static void thick_lines_to_indexed_vertex_array(const Lines3& lines,
    const std::vector<double>& widths,
    const std::vector<double>& heights,
    bool closed,
    GLIndexedVertexArray& volume)
{
    assert(!lines.empty());
    if (lines.empty())
        return;

#define LEFT    0
#define RIGHT   1
#define TOP     2
#define BOTTOM  3

    // left, right, top, bottom
    int      idx_initial[4] = { -1, -1, -1, -1 };
    int      idx_prev[4] = { -1, -1, -1, -1 };
    double   z_prev = 0.0;
    double   len_prev = 0.0;
    Vec3d    n_right_prev = Vec3d::Zero();
    Vec3d    n_top_prev = Vec3d::Zero();
    Vec3d    unit_v_prev = Vec3d::Zero();
    double   width_initial = 0.0;

    // new vertices around the line endpoints
    // left, right, top, bottom
    Vec3d a[4] = { Vec3d::Zero(), Vec3d::Zero(), Vec3d::Zero(), Vec3d::Zero() };
    Vec3d b[4] = { Vec3d::Zero(), Vec3d::Zero(), Vec3d::Zero(), Vec3d::Zero() };

    // loop once more in case of closed loops
    size_t lines_end = closed ? (lines.size() + 1) : lines.size();
    for (size_t ii = 0; ii < lines_end; ++ii)
    {
        size_t i = (ii == lines.size()) ? 0 : ii;

        const Line3& line = lines[i];
        double height = heights[i];
        double width = widths[i];

        Vec3d unit_v = unscale(line.vector()).normalized();
        double len = unscale<double>(line.length());

        Vec3d n_top = Vec3d::Zero();
        Vec3d n_right = Vec3d::Zero();

        if ((line.a(0) == line.b(0)) && (line.a(1) == line.b(1)))
        {
            // vertical segment
            n_top = Vec3d::UnitY();
            n_right = Vec3d::UnitX();
            if (line.a(2) < line.b(2))
                n_right = -n_right;
        }
        else
        {
            // horizontal segment
            n_right = unit_v.cross(Vec3d::UnitZ()).normalized();
            n_top = n_right.cross(unit_v).normalized();
        }

        Vec3d rl_displacement = 0.5 * width * n_right;
        Vec3d tb_displacement = 0.5 * height * n_top;
        Vec3d l_a = unscale(line.a);
        Vec3d l_b = unscale(line.b);

        a[RIGHT] = l_a + rl_displacement;
        a[LEFT] = l_a - rl_displacement;
        a[TOP] = l_a + tb_displacement;
        a[BOTTOM] = l_a - tb_displacement;
        b[RIGHT] = l_b + rl_displacement;
        b[LEFT] = l_b - rl_displacement;
        b[TOP] = l_b + tb_displacement;
        b[BOTTOM] = l_b - tb_displacement;

        Vec3d n_bottom = -n_top;
        Vec3d n_left = -n_right;

        int idx_a[4];
        int idx_b[4];
        int idx_last = int(volume.vertices_and_normals_interleaved.size() / 6);

        bool z_different = (z_prev != l_a(2));
        z_prev = l_b(2);

        // Share top / bottom vertices if possible.
        if (ii == 0)
        {
            idx_a[TOP] = idx_last++;
            volume.push_geometry(a[TOP], n_top);
        }
        else
            idx_a[TOP] = idx_prev[TOP];

        if ((ii == 0) || z_different)
        {
            // Start of the 1st line segment or a change of the layer thickness while maintaining the print_z.
            idx_a[BOTTOM] = idx_last++;
            volume.push_geometry(a[BOTTOM], n_bottom);
            idx_a[LEFT] = idx_last++;
            volume.push_geometry(a[LEFT], n_left);
            idx_a[RIGHT] = idx_last++;
            volume.push_geometry(a[RIGHT], n_right);
        }
        else
            idx_a[BOTTOM] = idx_prev[BOTTOM];

        if (ii == 0)
        {
            // Start of the 1st line segment.
            width_initial = width;
            ::memcpy(idx_initial, idx_a, sizeof(int) * 4);
        }
        else
        {
            // Continuing a previous segment.
            // Share left / right vertices if possible.
            double v_dot = unit_v_prev.dot(unit_v);
            bool is_right_turn = n_top_prev.dot(unit_v_prev.cross(unit_v)) > 0.0;

            // To reduce gpu memory usage, we try to reuse vertices
            // To reduce the visual artifacts, due to averaged normals, we allow to reuse vertices only when any of two adjacent edges
            // is longer than a fixed threshold.
            // The following value is arbitrary, it comes from tests made on a bunch of models showing the visual artifacts
            double len_threshold = 2.5;

            // Generate new vertices if the angle between adjacent edges is greater than 45 degrees or thresholds conditions are met
            bool is_sharp = (v_dot < 0.707) || (len_prev > len_threshold) || (len > len_threshold);
            if (is_sharp)
            {
                // Allocate new left / right points for the start of this segment as these points will receive their own normals to indicate a sharp turn.
                idx_a[RIGHT] = idx_last++;
                volume.push_geometry(a[RIGHT], n_right);
                idx_a[LEFT] = idx_last++;
                volume.push_geometry(a[LEFT], n_left);

                if (is_right_turn)
                {
                    // Right turn. Fill in the right turn wedge.
                    volume.push_triangle(idx_prev[RIGHT], idx_a[RIGHT], idx_prev[TOP]);
                    volume.push_triangle(idx_prev[RIGHT], idx_prev[BOTTOM], idx_a[RIGHT]);
                }
                else
                {
                    // Left turn. Fill in the left turn wedge.
                    volume.push_triangle(idx_prev[LEFT], idx_prev[TOP], idx_a[LEFT]);
                    volume.push_triangle(idx_prev[LEFT], idx_a[LEFT], idx_prev[BOTTOM]);
                }
            }
            else
            {
                // The two successive segments are nearly collinear.
                idx_a[LEFT] = idx_prev[LEFT];
                idx_a[RIGHT] = idx_prev[RIGHT];
            }

            if (ii == lines.size())
            {
                if (!is_sharp)
                {
                    // Closing a loop with smooth transition. Unify the closing left / right vertices.
                    ::memcpy(volume.vertices_and_normals_interleaved.data() + idx_initial[LEFT] * 6, volume.vertices_and_normals_interleaved.data() + idx_prev[LEFT] * 6, sizeof(float) * 6);
                    ::memcpy(volume.vertices_and_normals_interleaved.data() + idx_initial[RIGHT] * 6, volume.vertices_and_normals_interleaved.data() + idx_prev[RIGHT] * 6, sizeof(float) * 6);
                    volume.vertices_and_normals_interleaved.erase(volume.vertices_and_normals_interleaved.end() - 12, volume.vertices_and_normals_interleaved.end());
                    // Replace the left / right vertex indices to point to the start of the loop.
                    for (size_t u = volume.quad_indices.size() - 16; u < volume.quad_indices.size(); ++u)
                    {
                        if (volume.quad_indices[u] == idx_prev[LEFT])
                            volume.quad_indices[u] = idx_initial[LEFT];
                        else if (volume.quad_indices[u] == idx_prev[RIGHT])
                            volume.quad_indices[u] = idx_initial[RIGHT];
                    }
                }

                // This is the last iteration, only required to solve the transition.
                break;
            }
        }

        // Only new allocate top / bottom vertices, if not closing a loop.
        if (closed && (ii + 1 == lines.size()))
            idx_b[TOP] = idx_initial[TOP];
        else
        {
            idx_b[TOP] = idx_last++;
            volume.push_geometry(b[TOP], n_top);
        }

        if (closed && (ii + 1 == lines.size()) && (width == width_initial))
            idx_b[BOTTOM] = idx_initial[BOTTOM];
        else
        {
            idx_b[BOTTOM] = idx_last++;
            volume.push_geometry(b[BOTTOM], n_bottom);
        }

        // Generate new vertices for the end of this line segment.
        idx_b[LEFT] = idx_last++;
        volume.push_geometry(b[LEFT], n_left);
        idx_b[RIGHT] = idx_last++;
        volume.push_geometry(b[RIGHT], n_right);

        ::memcpy(idx_prev, idx_b, 4 * sizeof(int));
        n_right_prev = n_right;
        n_top_prev = n_top;
        unit_v_prev = unit_v;
        len_prev = len;

        if (!closed)
        {
            // Terminate open paths with caps.
            if (i == 0)
                volume.push_quad(idx_a[BOTTOM], idx_a[RIGHT], idx_a[TOP], idx_a[LEFT]);

            // We don't use 'else' because both cases are true if we have only one line.
            if (i + 1 == lines.size())
                volume.push_quad(idx_b[BOTTOM], idx_b[LEFT], idx_b[TOP], idx_b[RIGHT]);
        }

        // Add quads for a straight hollow tube-like segment.
        // bottom-right face
        volume.push_quad(idx_a[BOTTOM], idx_b[BOTTOM], idx_b[RIGHT], idx_a[RIGHT]);
        // top-right face
        volume.push_quad(idx_a[RIGHT], idx_b[RIGHT], idx_b[TOP], idx_a[TOP]);
        // top-left face
        volume.push_quad(idx_a[TOP], idx_b[TOP], idx_b[LEFT], idx_a[LEFT]);
        // bottom-left face
        volume.push_quad(idx_a[LEFT], idx_b[LEFT], idx_b[BOTTOM], idx_a[BOTTOM]);
    }

#undef LEFT
#undef RIGHT
#undef TOP
#undef BOTTOM
}

static void point_to_indexed_vertex_array(const Vec3crd& point,
    double width,
    double height,
    GLIndexedVertexArray& volume)
{
    // builds a double piramid, with vertices on the local axes, around the point

    Vec3d center = unscale(point);

    double scale_factor = 1.0;
    double w = scale_factor * width;
    double h = scale_factor * height;

    // new vertices ids
    int idx_last = int(volume.vertices_and_normals_interleaved.size() / 6);
    int idxs[6];
    for (int i = 0; i < 6; ++i)
    {
        idxs[i] = idx_last + i;
    }

    Vec3d displacement_x(w, 0.0, 0.0);
    Vec3d displacement_y(0.0, w, 0.0);
    Vec3d displacement_z(0.0, 0.0, h);

    Vec3d unit_x(1.0, 0.0, 0.0);
    Vec3d unit_y(0.0, 1.0, 0.0);
    Vec3d unit_z(0.0, 0.0, 1.0);

    // vertices
    volume.push_geometry(center - displacement_x, -unit_x); // idxs[0]
    volume.push_geometry(center + displacement_x, unit_x);  // idxs[1]
    volume.push_geometry(center - displacement_y, -unit_y); // idxs[2]
    volume.push_geometry(center + displacement_y, unit_y);  // idxs[3]
    volume.push_geometry(center - displacement_z, -unit_z); // idxs[4]
    volume.push_geometry(center + displacement_z, unit_z);  // idxs[5]

    // top piramid faces
    volume.push_triangle(idxs[0], idxs[2], idxs[5]);
    volume.push_triangle(idxs[2], idxs[1], idxs[5]);
    volume.push_triangle(idxs[1], idxs[3], idxs[5]);
    volume.push_triangle(idxs[3], idxs[0], idxs[5]);

    // bottom piramid faces
    volume.push_triangle(idxs[2], idxs[0], idxs[4]);
    volume.push_triangle(idxs[1], idxs[2], idxs[4]);
    volume.push_triangle(idxs[3], idxs[1], idxs[4]);
    volume.push_triangle(idxs[0], idxs[3], idxs[4]);
}

void _3DScene::thick_lines_to_verts(
    const Lines                 &lines,
    const std::vector<double>   &widths,
    const std::vector<double>   &heights,
    bool                         closed,
    double                       top_z,
    GLVolume                    &volume)
{
    thick_lines_to_indexed_vertex_array(lines, widths, heights, closed, top_z, *(volume.indexed_vertex_array));
}

void _3DScene::thick_lines_to_verts(const Lines3& lines,
    const std::vector<double>& widths,
    const std::vector<double>& heights,
    bool closed,
    GLVolume& volume)
{
    thick_lines_to_indexed_vertex_array(lines, widths, heights, closed, *(volume.indexed_vertex_array));
}

static void thick_point_to_verts(const Vec3crd& point,
    double width,
    double height,
    GLVolume& volume)
{
    point_to_indexed_vertex_array(point, width, height, *(volume.indexed_vertex_array));
}

void _3DScene::extrusionentity_to_verts(const Polyline &polyline, float width, float height, float print_z, GLVolume& volume)
{
	if (polyline.size() >= 2) {
		size_t num_segments = polyline.size() - 1;
		thick_lines_to_verts(polyline.lines(), std::vector<double>(num_segments, width), std::vector<double>(num_segments, height), false, print_z, volume);
	}
}

// Fill in the qverts and tverts with quads and triangles for the extrusion_path.
void _3DScene::extrusionentity_to_verts(const ExtrusionPath &extrusion_path, float print_z, GLVolume &volume)
{
	extrusionentity_to_verts(extrusion_path.polyline, extrusion_path.width, extrusion_path.height, print_z, volume);
}

// Fill in the qverts and tverts with quads and triangles for the extrusion_path.
void _3DScene::extrusionentity_to_verts(const ExtrusionPath &extrusion_path, float print_z, const Point &copy, GLVolume &volume)
{
    Polyline            polyline = extrusion_path.polyline;
    polyline.remove_duplicate_points();
    polyline.translate(copy);
    Lines               lines = polyline.lines();
    std::vector<double> widths(lines.size(), extrusion_path.width);
    std::vector<double> heights(lines.size(), extrusion_path.height);
    thick_lines_to_verts(lines, widths, heights, false, print_z, volume);
}

// Fill in the qverts and tverts with quads and triangles for the extrusion_loop.
void _3DScene::extrusionentity_to_verts(const ExtrusionLoop &extrusion_loop, float print_z, const Point &copy, GLVolume &volume)
{
    Lines               lines;
    std::vector<double> widths;
    std::vector<double> heights;
    for (const ExtrusionPath &extrusion_path : extrusion_loop.paths) {
        Polyline            polyline = extrusion_path.polyline;
        polyline.remove_duplicate_points();
        polyline.translate(copy);
        Lines lines_this = polyline.lines();
        append(lines, lines_this);
        widths.insert(widths.end(), lines_this.size(), extrusion_path.width);
        heights.insert(heights.end(), lines_this.size(), extrusion_path.height);
    }
    thick_lines_to_verts(lines, widths, heights, true, print_z, volume);
}

// Fill in the qverts and tverts with quads and triangles for the extrusion_multi_path.
void _3DScene::extrusionentity_to_verts(const ExtrusionMultiPath &extrusion_multi_path, float print_z, const Point &copy, GLVolume &volume)
{
    Lines               lines;
    std::vector<double> widths;
    std::vector<double> heights;
    for (const ExtrusionPath &extrusion_path : extrusion_multi_path.paths) {
        Polyline            polyline = extrusion_path.polyline;
        polyline.remove_duplicate_points();
        polyline.translate(copy);
        Lines lines_this = polyline.lines();
        append(lines, lines_this);
        widths.insert(widths.end(), lines_this.size(), extrusion_path.width);
        heights.insert(heights.end(), lines_this.size(), extrusion_path.height);
    }
    thick_lines_to_verts(lines, widths, heights, false, print_z, volume);
}

void _3DScene::extrusionentity_to_verts(const ExtrusionEntityCollection &extrusion_entity_collection, float print_z, const Point &copy, GLVolume &volume)
{
    for (const ExtrusionEntity *extrusion_entity : extrusion_entity_collection.entities)
        extrusionentity_to_verts(extrusion_entity, print_z, copy, volume);
}

void _3DScene::extrusionentity_to_verts(const ExtrusionEntity *extrusion_entity, float print_z, const Point &copy, GLVolume &volume)
{
    if (extrusion_entity != nullptr) {
        auto *extrusion_path = dynamic_cast<const ExtrusionPath*>(extrusion_entity);
        if (extrusion_path != nullptr)
            extrusionentity_to_verts(*extrusion_path, print_z, copy, volume);
        else {
            auto *extrusion_loop = dynamic_cast<const ExtrusionLoop*>(extrusion_entity);
            if (extrusion_loop != nullptr)
                extrusionentity_to_verts(*extrusion_loop, print_z, copy, volume);
            else {
                auto *extrusion_multi_path = dynamic_cast<const ExtrusionMultiPath*>(extrusion_entity);
                if (extrusion_multi_path != nullptr)
                    extrusionentity_to_verts(*extrusion_multi_path, print_z, copy, volume);
                else {
                    auto *extrusion_entity_collection = dynamic_cast<const ExtrusionEntityCollection*>(extrusion_entity);
                    if (extrusion_entity_collection != nullptr)
                        extrusionentity_to_verts(*extrusion_entity_collection, print_z, copy, volume);
                    else {
                        throw Slic3r::RuntimeError("Unexpected extrusion_entity type in to_verts()");
                    }
                }
            }
        }
    }
}

void _3DScene::polyline3_to_verts(const Polyline3& polyline, double width, double height, GLVolume& volume)
{
    Lines3 lines = polyline.lines();
    std::vector<double> widths(lines.size(), width);
    std::vector<double> heights(lines.size(), height);
    thick_lines_to_verts(lines, widths, heights, false, volume);
}

void _3DScene::point3_to_verts(const Vec3crd& point, double width, double height, GLVolume& volume)
{
    thick_point_to_verts(point, width, height, volume);
}

} // namespace Slic3r
