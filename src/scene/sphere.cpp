#include "sphere.h"

#include <cmath>

#include "pathtracer/bsdf.h"
#include "util/sphere_drawing.h"

namespace CGL {
namespace SceneObjects {

bool Sphere::test(const Ray &r, double &t1, double &t2) const {

  // TODO (Part 1.4):
  // Implement ray - sphere intersection test.
  // Return true if there are intersections and writing the
  // smaller of the two intersection times in t1 and the larger in t2.
  
  // Vector from ray origin to sphere center
  Vector3D origin_to_center = r.o - o;
  
  // Solve quadratic equation: a*t^2 + b*t + c = 0
  double ray_direction_magnitude_sq = dot(r.d, r.d);
  double linear_coefficient = 2.0 * dot(origin_to_center, r.d);
  double constant_coefficient = dot(origin_to_center, origin_to_center) - r2;
  
  // Compute discriminant to determine if solutions exist
  double discriminant_value = linear_coefficient * linear_coefficient - 
                              4.0 * ray_direction_magnitude_sq * constant_coefficient;
  
  if (discriminant_value < 0.0) {
    return false;
  }
  
  // Calculate both intersection times using quadratic formula
  double sqrt_discriminant = sqrt(discriminant_value);
  double near_intersection_time = (-linear_coefficient - sqrt_discriminant) / (2.0 * ray_direction_magnitude_sq);
  double far_intersection_time = (-linear_coefficient + sqrt_discriminant) / (2.0 * ray_direction_magnitude_sq);
  
  // Check if at least one intersection falls within valid ray parameter range
  bool near_within_range = (near_intersection_time >= r.min_t) && (near_intersection_time <= r.max_t);
  bool far_within_range = (far_intersection_time >= r.min_t) && (far_intersection_time <= r.max_t);
  
  if (!near_within_range && !far_within_range) {
    return false;
  }
  
  t1 = near_intersection_time;
  t2 = far_intersection_time;
  return true;

}

bool Sphere::has_intersection(const Ray &r) const {

  // TODO (Part 1.4):
  // Implement ray - sphere intersection.
  // Note that you might want to use the the Sphere::test helper here.
  double near_hit_time, far_hit_time;
  if (!test(r, near_hit_time, far_hit_time)) {
    return false;
  }
  double closest_valid_time = far_hit_time;
  bool near_hit_valid = (near_hit_time >= r.min_t) && (near_hit_time <= r.max_t);
  if (near_hit_valid) {
    closest_valid_time = near_hit_time;
  }
  r.max_t = closest_valid_time;
  return true;
}

bool Sphere::intersect(const Ray &r, Intersection *i) const {

  // TODO (Part 1.4):
  // Implement ray - sphere intersection.
  // Note again that you might want to use the the Sphere::test helper here.
  // When an intersection takes place, the Intersection data should be updated
  // correspondingly.
  
  if (!has_intersection(r)) {
    return false;
  }
  
  Vector3D intersection_point = r.o + r.max_t * r.d;
  Vector3D surface_normal = (intersection_point - o).unit();
  i->bsdf = get_bsdf();
  i->primitive = this;
  i->n = surface_normal;
  i->t = r.max_t;
  return true;
  
}

void Sphere::draw(const Color &c, float alpha) const {
  Misc::draw_sphere_opengl(o, r, c);
}

void Sphere::drawOutline(const Color &c, float alpha) const {
  // Misc::draw_sphere_opengl(o, r, c);
}

} // namespace SceneObjects
} // namespace CGL
