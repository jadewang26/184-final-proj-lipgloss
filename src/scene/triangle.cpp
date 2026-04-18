#include "triangle.h"

#include "CGL/CGL.h"
#include "GL/glew.h"

namespace CGL {
namespace SceneObjects {

Triangle::Triangle(const Mesh *mesh, size_t v1, size_t v2, size_t v3) {
  p1 = mesh->positions[v1];
  p2 = mesh->positions[v2];
  p3 = mesh->positions[v3];
  n1 = mesh->normals[v1];
  n2 = mesh->normals[v2];
  n3 = mesh->normals[v3];
  bbox = BBox(p1);
  bbox.expand(p2);
  bbox.expand(p3);

  bsdf = mesh->get_bsdf();
}

BBox Triangle::get_bbox() const { return bbox; }

bool Triangle::has_intersection(const Ray &r) const {
  // Part 1, Task 3: implement ray-triangle intersection
  // The difference between this function and the next function is that the next
  // function records the "intersection" while this function only tests whether
  // there is a intersection.
  
  Matrix3x3 system(-r.d.x, p2.x - p1.x, p3.x - p1.x,
                    -r.d.y, p2.y - p1.y, p3.y - p1.y,
                    -r.d.z, p2.z - p1.z, p3.z - p1.z);
  
  Vector3D intersection_params = system.inv() * (r.o - p1);
  
  double distance = intersection_params.x;
  double u_bary = intersection_params.y;
  double v_bary = intersection_params.z;
  
  bool in_ray_range = (distance >= r.min_t) && (distance <= r.max_t);
  bool valid_bary = (u_bary >= 0) && (v_bary >= 0) && ((u_bary + v_bary) <= 1) && (u_bary <= 1) && (v_bary <= 1);
  
  if (!in_ray_range || !valid_bary) {
    return false;
  }
  
  r.max_t = distance;
  return true;

}

bool Triangle::intersect(const Ray &r, Intersection *isect) const {
  // Part 1, Task 3:
  // implement ray-triangle intersection. When an intersection takes
  // place, the Intersection data should be updated accordingly
  
  Matrix3x3 coeff_matrix(-r.d.x, p2.x - p1.x, p3.x - p1.x,
                         -r.d.y, p2.y - p1.y, p3.y - p1.y,
                         -r.d.z, p2.z - p1.z, p3.z - p1.z);
  
  Vector3D params = coeff_matrix.inv() * (r.o - p1);
  
  double ray_t = params.x;
  double bary_u = params.y;
  double bary_v = params.z;
  double bary_w = 1.0 - bary_u - bary_v;
  
  if (ray_t < r.min_t || ray_t > r.max_t) {
    return false;
  }
  
  if (bary_u < 0 || bary_u > 1 || bary_v < 0 || bary_v > 1 || bary_w < 0 || bary_w > 1) {
    return false;
  }
  
  r.max_t = ray_t;
  isect->t = ray_t;
  isect->n = bary_w * n1 + bary_u * n2 + bary_v * n3;
  isect->primitive = this;
  isect->bsdf = get_bsdf();
  
  return true;
}

void Triangle::draw(const Color &c, float alpha) const {
  glColor4f(c.r, c.g, c.b, alpha);
  glBegin(GL_TRIANGLES);
  glVertex3d(p1.x, p1.y, p1.z);
  glVertex3d(p2.x, p2.y, p2.z);
  glVertex3d(p3.x, p3.y, p3.z);
  glEnd();
}

void Triangle::drawOutline(const Color &c, float alpha) const {
  glColor4f(c.r, c.g, c.b, alpha);
  glBegin(GL_LINE_LOOP);
  glVertex3d(p1.x, p1.y, p1.z);
  glVertex3d(p2.x, p2.y, p2.z);
  glVertex3d(p3.x, p3.y, p3.z);
  glEnd();
}

} // namespace SceneObjects
} // namespace CGL
