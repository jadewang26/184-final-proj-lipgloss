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
  
  const Vector3D E1 = p2 - p1;
  const Vector3D E2 = p3 - p1;
  const Vector3D S = r.o - p1;
  const Vector3D S1 = cross(r.d, E2);
  const Vector3D S2 = cross(S, E1);
  const double denom = dot(E1, S1);
  if (fabs(denom) < EPS_D) return false;

  const double b_1_num = dot(S, S1);
  const double b_2_num = dot(r.d, S2);
  const double t_num = dot(E2, S2);
  const double edge_eps = EPS_D * fabs(denom);

  if (denom > 0.0) {
    if (b_1_num < -edge_eps || b_1_num > denom + edge_eps) return false;
    if (b_2_num < -edge_eps || b_1_num + b_2_num > denom + edge_eps) return false;
    if (t_num < r.min_t * denom || t_num > r.max_t * denom) return false;
  } else {
    if (b_1_num > edge_eps || b_1_num < denom - edge_eps) return false;
    if (b_2_num > edge_eps || b_1_num + b_2_num < denom - edge_eps) return false;
    if (t_num > r.min_t * denom || t_num < r.max_t * denom) return false;
  }

  return true;

}

bool Triangle::intersect(const Ray &r, Intersection *isect) const {
  // Part 1, Task 3:
  // implement ray-triangle intersection. When an intersection takes
  // place, the Intersection data should be updated accordingly
  
  const Vector3D E1 = p2 - p1;
  const Vector3D E2 = p3 - p1;
  const Vector3D S = r.o - p1;
  const Vector3D S1 = cross(r.d, E2);
  const Vector3D S2 = cross(S, E1);
  const double denom = dot(E1, S1);
  if (fabs(denom) < EPS_D) return false;

  // switched to using doubles because I thought that was causing the precision error
  // not sure if it was this or just the division problem, 
  // but performance seems fine with doubles so I figure we keep them
  const double b_1_num = dot(S, S1);
  const double b_2_num = dot(r.d, S2);
  const double t_num = dot(E2, S2);
  const double edge_eps = EPS_D * fabs(denom);

  // weird artifacts showed up on bigger .dae files with small/dense triangle mesheswhen I divided by denom before comparisons
  // that suggested some kind of precision error so we perform the same comparison without division
  if (denom > 0.0) {
    if (b_1_num < -edge_eps || b_1_num > denom + edge_eps) return false;
    if (b_2_num < -edge_eps || b_1_num + b_2_num > denom + edge_eps) return false;
    if (t_num < r.min_t * denom || t_num > r.max_t * denom) return false;
  } else {
    if (b_1_num > edge_eps || b_1_num < denom - edge_eps) return false;
    if (b_2_num > edge_eps || b_1_num + b_2_num < denom - edge_eps) return false;
    if (t_num > r.min_t * denom || t_num < r.max_t * denom) return false;
  }

  const double inv_denom = 1.0 / denom;
  const double t = t_num * inv_denom;
  const double b_1 = b_1_num * inv_denom;
  const double b_2 = b_2_num * inv_denom;
  const Vector3D n = ((1.0 - b_1 - b_2) * n1 + b_1 * n2 + b_2 * n3).unit();

  isect->t = t;
  isect->n = n;
  isect->primitive = this;
  isect->bsdf = this->get_bsdf();
  
  r.max_t = isect->t;

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
