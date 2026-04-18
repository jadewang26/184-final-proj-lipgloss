#include "bbox.h"

#include "GL/glew.h"

#include <algorithm>
#include <iostream>

namespace CGL {

bool BBox::intersect(const Ray& r, double& t0, double& t1) const {

  // TODO (Part 2.2):
  // Implement ray - bounding box intersection test
  // If the ray intersected the bouding box within the range given by
  // t0, t1, update t0 and t1 with the new intersection times.
  
  double x_slab_near = (min.x - r.o.x) / r.d.x;
  double x_slab_far = (max.x - r.o.x) / r.d.x;
  double y_slab_near = (min.y - r.o.y) / r.d.y;
  double y_slab_far = (max.y - r.o.y) / r.d.y;
  double z_slab_near = (min.z - r.o.z) / r.d.z;
  double z_slab_far = (max.z - r.o.z) / r.d.z;
  
  if (x_slab_near > x_slab_far) std::swap(x_slab_near, x_slab_far);
  if (y_slab_near > y_slab_far) std::swap(y_slab_near, y_slab_far);
  if (z_slab_near > z_slab_far) std::swap(z_slab_near, z_slab_far);
  
  double box_entry_time = std::max({x_slab_near, y_slab_near, z_slab_near});
  double box_exit_time = std::min({x_slab_far, y_slab_far, z_slab_far});
  
  if (box_entry_time > box_exit_time || box_exit_time < t0 || t1 < box_entry_time) {
    return false;
  }
  
  t0 = box_entry_time;
  t1 = box_exit_time;

  return true;

}

void BBox::draw(Color c, float alpha) const {

  glColor4f(c.r, c.g, c.b, alpha);

  // top
  glBegin(GL_LINE_STRIP);
  glVertex3d(max.x, max.y, max.z);
  glVertex3d(max.x, max.y, min.z);
  glVertex3d(min.x, max.y, min.z);
  glVertex3d(min.x, max.y, max.z);
  glVertex3d(max.x, max.y, max.z);
  glEnd();

  // bottom
  glBegin(GL_LINE_STRIP);
  glVertex3d(min.x, min.y, min.z);
  glVertex3d(min.x, min.y, max.z);
  glVertex3d(max.x, min.y, max.z);
  glVertex3d(max.x, min.y, min.z);
  glVertex3d(min.x, min.y, min.z);
  glEnd();

  // side
  glBegin(GL_LINES);
  glVertex3d(max.x, max.y, max.z);
  glVertex3d(max.x, min.y, max.z);
  glVertex3d(max.x, max.y, min.z);
  glVertex3d(max.x, min.y, min.z);
  glVertex3d(min.x, max.y, min.z);
  glVertex3d(min.x, min.y, min.z);
  glVertex3d(min.x, max.y, max.z);
  glVertex3d(min.x, min.y, max.z);
  glEnd();

}

std::ostream& operator<<(std::ostream& os, const BBox& b) {
  return os << "BBOX(" << b.min << ", " << b.max << ")";
}

} // namespace CGL
