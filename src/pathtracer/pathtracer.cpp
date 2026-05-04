#include "pathtracer.h"

#include "scene/light.h"
#include "scene/sphere.h"
#include "scene/triangle.h"

#include <algorithm>
#include <cmath>


using namespace CGL::SceneObjects;

namespace CGL {

using std::max;
using std::min;
using std::swap;

namespace {

const int kMaxSubsurfaceEvents = 16;

Vector3D clamp_vector_min(const Vector3D& value, double min_value) {
  return Vector3D(max(value.x, min_value),
                  max(value.y, min_value),
                  max(value.z, min_value));
}

Vector3D clamp_color01(const Vector3D& value) {
  return Vector3D(clamp(value.x, 0.0, 1.0),
                  clamp(value.y, 0.0, 1.0),
                  clamp(value.z, 0.0, 1.0));
}

bool is_zero_vector(const Vector3D& value) {
  return value.x == 0.0 && value.y == 0.0 && value.z == 0.0;
}

Vector3D random_walk_base_tint(const RandomWalkSSSBSDF* bsdf) {
  if (is_zero_vector(bsdf->get_base_color())) {
    return Vector3D(1.0);
  }
  return clamp_color01(clamp_vector_min(bsdf->get_base_color(), 0.0) *
                       clamp(bsdf->get_saturation(), 0.0, 1.5));
}

Vector3D exp_vector(const Vector3D& value) {
  return Vector3D(exp(value.x), exp(value.y), exp(value.z));
}

double mean_channel(const Vector3D& value) {
  return (value.x + value.y + value.z) / 3.0;
}

double schlick_fresnel(double cos_theta, double eta_i, double eta_t) {
  cos_theta = clamp(cos_theta, 0.0, 1.0);
  double r0 = (eta_i - eta_t) / (eta_i + eta_t);
  r0 *= r0;
  double m = 1.0 - cos_theta;
  return r0 + (1.0 - r0) * m * m * m * m * m;
}

double ggx_ndf(double n_dot_h, double alpha2) {
  if (n_dot_h <= 0.0) return 0.0;
  double cos2_theta_h = n_dot_h * n_dot_h;
  double denom = PI * pow(cos2_theta_h * (alpha2 - 1.0) + 1.0, 2.0);
  return denom > 0.0 ? alpha2 / denom : 0.0;
}

double smith_ggx_g1(double n_dot_w, double alpha2) {
  if (n_dot_w <= 0.0) return 0.0;
  double cos2_theta = n_dot_w * n_dot_w;
  return (2.0 * n_dot_w) /
         (n_dot_w + sqrt(alpha2 + (1.0 - alpha2) * cos2_theta));
}

double roughness_to_alpha2(double roughness) {
  double alpha = max(roughness * roughness, 0.02);
  return max(alpha * alpha, 0.0004);
}

double power_heuristic(double pdf_a, double pdf_b) {
  double a2 = pdf_a * pdf_a;
  double b2 = pdf_b * pdf_b;
  return a2 + b2 > 0.0 ? a2 / (a2 + b2) : 0.0;
}

double rough_dielectric_reflection_pdf(const Vector3D& wo,
                                       const Vector3D& wi,
                                       const Vector3D& normal,
                                       double roughness) {
  double n_dot_v = dot(normal, wo);
  double n_dot_l = dot(normal, wi);
  if (n_dot_v <= 0.0 || n_dot_l <= 0.0) return 0.0;

  Vector3D h = wo + wi;
  if (h.norm2() == 0.0) return 0.0;
  h.normalize();

  double v_dot_h = dot(wo, h);
  double n_dot_h = dot(normal, h);
  if (v_dot_h <= 0.0 || n_dot_h <= 0.0) return 0.0;

  double D = ggx_ndf(n_dot_h, roughness_to_alpha2(roughness));
  return D * n_dot_h / (4.0 * v_dot_h);
}

double rough_dielectric_brdf(const Vector3D& wo,
                             const Vector3D& wi,
                             const Vector3D& normal,
                             double roughness,
                             double eta) {
  double n_dot_v = dot(normal, wo);
  double n_dot_l = dot(normal, wi);
  if (n_dot_v <= 0.0 || n_dot_l <= 0.0) return 0.0;

  Vector3D h = wo + wi;
  if (h.norm2() == 0.0) return 0.0;
  h.normalize();

  double v_dot_h = dot(wo, h);
  double n_dot_h = dot(normal, h);
  if (v_dot_h <= 0.0 || n_dot_h <= 0.0) return 0.0;

  double alpha2 = roughness_to_alpha2(roughness);
  double D = ggx_ndf(n_dot_h, alpha2);
  double G = smith_ggx_g1(n_dot_v, alpha2) * smith_ggx_g1(n_dot_l, alpha2);
  double F = schlick_fresnel(v_dot_h, 1.0, eta);
  return D * F * G / max(4.0 * n_dot_v * n_dot_l, 1e-6);
}

bool sample_rough_dielectric_reflection(const Vector3D& wo,
                                        const Vector3D& normal,
                                        double roughness,
                                        double eta,
                                        Vector3D* wi,
                                        double* weight,
                                        double* pdf) {
  double n_dot_v = dot(normal, wo);
  if (n_dot_v <= 0.0) return false;

  double alpha2 = roughness_to_alpha2(roughness);
  Vector2D u(random_uniform(), random_uniform());
  double phi = 2.0 * PI * u.y;
  double tan_theta2 = alpha2 * u.x / max(1.0 - u.x, 1e-6);
  double cos_theta = 1.0 / sqrt(1.0 + tan_theta2);
  double sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
  Vector3D h_local(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);

  Matrix3x3 o2w;
  make_coord_space(o2w, normal);
  Vector3D h = o2w * h_local;
  h.normalize();

  double v_dot_h = dot(wo, h);
  if (v_dot_h <= 0.0) return false;

  *wi = 2.0 * v_dot_h * h - wo;
  wi->normalize();

  double n_dot_l = dot(normal, *wi);
  if (n_dot_l <= 0.0) return false;

  double n_dot_h = max(dot(normal, h), 0.0);
  double D = ggx_ndf(n_dot_h, alpha2);
  if (D <= 0.0) return false;

  double G = smith_ggx_g1(n_dot_v, alpha2) * smith_ggx_g1(n_dot_l, alpha2);
  double F = schlick_fresnel(v_dot_h, 1.0, eta);
  double sample_weight = F * G * v_dot_h / max(n_dot_v * n_dot_h, 1e-6);
  *pdf = D * n_dot_h / (4.0 * v_dot_h);
  *weight = min(sample_weight, 5.0);
  return *weight > 0.0 && *pdf > 0.0;
}

double area_light_pdf_for_direction(const AreaLight* light,
                                    const Vector3D& p,
                                    const Vector3D& wi) {
  double denom = dot(wi, light->direction);
  if (denom >= -1e-8) return 0.0;

  double t = dot(light->position - p, light->direction) / denom;
  if (t <= EPS_F) return 0.0;

  Vector3D q = p + wi * t;
  Vector3D rel = q - light->position;
  double dim_x_len2 = light->dim_x.norm2();
  double dim_y_len2 = light->dim_y.norm2();
  if (dim_x_len2 <= 0.0 || dim_y_len2 <= 0.0) return 0.0;

  double u = dot(rel, light->dim_x) / dim_x_len2;
  double v = dot(rel, light->dim_y) / dim_y_len2;
  if (fabs(u) > 0.5 || fabs(v) > 0.5) return 0.0;

  return t / (light->area * fabs(denom));
}

double light_pdf_for_direction(const std::vector<SceneLight*>& lights,
                               const Vector3D& p,
                               const Vector3D& wi) {
  double pdf = 0.0;
  for (SceneLight* light : lights) {
    if (AreaLight* area = dynamic_cast<AreaLight*>(light)) {
      pdf += area_light_pdf_for_direction(area, p, wi);
    }
  }
  return pdf;
}

Vector3D evaluate_bsdf_at_hit(const Intersection& isect,
                              const Vector3D& wo,
                              const Vector3D& wi) {
  return isect.has_uv ? isect.bsdf->f(wo, wi, isect.uv)
                      : isect.bsdf->f(wo, wi);
}

Vector3D sample_bsdf_at_hit(const Intersection& isect,
                            const Vector3D& wo,
                            Vector3D* wi,
                            double* pdf) {
  return isect.has_uv ? isect.bsdf->sample_f(wo, wi, pdf, isect.uv)
                      : isect.bsdf->sample_f(wo, wi, pdf);
}

Vector3D reflect_across_normal(const Vector3D& direction,
                               const Vector3D& normal) {
  return (direction - 2.0 * dot(direction, normal) * normal).unit();
}

Vector3D sample_henyey_greenstein(const Vector3D& direction, double g) {
  g = clamp(g, -0.95, 0.95);

  double u1 = random_uniform();
  double u2 = random_uniform();
  double cos_theta;
  if (fabs(g) < 1e-3) {
    cos_theta = 1.0 - 2.0 * u1;
  } else {
    double sqr_term = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
    cos_theta = (1.0 + g * g - sqr_term * sqr_term) / (2.0 * g);
    cos_theta = clamp(cos_theta, -1.0, 1.0);
  }

  double sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
  double phi = 2.0 * PI * u2;
  Vector3D local(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);

  Matrix3x3 o2w;
  make_coord_space(o2w, direction.unit());
  return (o2w * local).unit();
}

bool sphere_boundary_distance(const Sphere* sphere, const Vector3D& origin,
                              const Vector3D& direction, double* t_exit) {
  Vector3D oc = origin - sphere->o;
  double a = dot(direction, direction);
  double b = 2.0 * dot(oc, direction);
  double c = dot(oc, oc) - sphere->r2;
  double discriminant = b * b - 4.0 * a * c;
  if (discriminant < 0.0) return false;

  double root = sqrt(discriminant);
  double t0 = (-b - root) / (2.0 * a);
  double t1 = (-b + root) / (2.0 * a);
  if (t0 > t1) swap(t0, t1);

  if (t1 <= EPS_F) return false;
  *t_exit = t0 > EPS_F ? t0 : t1;
  return true;
}

bool same_subsurface_volume(const Primitive* volume_primitive,
                            const Primitive* hit_primitive) {
  if (volume_primitive == hit_primitive) return true;

  const Triangle* volume_triangle =
      dynamic_cast<const Triangle*>(volume_primitive);
  const Triangle* hit_triangle = dynamic_cast<const Triangle*>(hit_primitive);
  return volume_triangle && hit_triangle && volume_triangle->mesh &&
         volume_triangle->mesh == hit_triangle->mesh;
}

bool find_subsurface_boundary(const BVHAccel* bvh,
                              const Primitive* volume_primitive,
                              const Sphere* sphere,
                              const Vector3D& origin,
                              const Vector3D& direction,
                              double* t_exit,
                              Vector3D* exit_normal) {
  if (sphere) {
    if (!sphere_boundary_distance(sphere, origin, direction, t_exit)) {
      return false;
    }
    *exit_normal = sphere->normal(origin + direction * *t_exit).unit();
    return true;
  }

  Ray boundary_ray(origin, direction);
  boundary_ray.min_t = EPS_F;
  Intersection boundary_isect;
  if (!bvh->intersect(boundary_ray, &boundary_isect)) return false;
  if (!same_subsurface_volume(volume_primitive, boundary_isect.primitive)) {
    return false;
  }

  *t_exit = boundary_isect.t;
  *exit_normal = boundary_isect.n.unit();
  if (dot(direction, *exit_normal) < 0.0) {
    *exit_normal = -*exit_normal;
  }
  return *t_exit > EPS_F;
}

} // namespace

PathTracer::PathTracer() {
  gridSampler = new UniformGridSampler2D();
  hemisphereSampler = new UniformHemisphereSampler3D();

  tm_gamma = 2.2f;
  tm_level = 1.0f;
  tm_key = 0.18;
  tm_wht = 5.0f;
}

PathTracer::~PathTracer() {
  delete gridSampler;
  delete hemisphereSampler;
}

void PathTracer::set_frame_size(size_t width, size_t height) {
  sampleBuffer.resize(width, height);
  sampleCountBuffer.resize(width * height);
}

void PathTracer::clear() {
  bvh = NULL;
  scene = NULL;
  camera = NULL;
  sampleBuffer.clear();
  sampleCountBuffer.clear();
  sampleBuffer.resize(0, 0);
  sampleCountBuffer.resize(0, 0);
}

void PathTracer::write_to_framebuffer(ImageBuffer &framebuffer, size_t x0,
                                      size_t y0, size_t x1, size_t y1) {
  sampleBuffer.toColor(framebuffer, x0, y0, x1, y1);
}

Vector3D
PathTracer::estimate_direct_lighting_hemisphere(const Ray &r,
                                                const Intersection &isect) {
  // Estimate the lighting from this intersection coming directly from a light.
  // For this function, sample uniformly in a hemisphere.

  // Note: When comparing Cornel Box (CBxxx.dae) results to importance sampling, you may find the "glow" around the light source is gone.
  // This is totally fine: the area lights in importance sampling has directionality, however in hemisphere sampling we don't model this behaviour.

  // make a coordinate system for a hit point
  // with N aligned with the Z direction.
  Matrix3x3 o2w;
  make_coord_space(o2w, isect.n);
  Matrix3x3 w2o = o2w.T();

  // w_out points towards the source of the ray (e.g.,
  // toward the camera if this is a primary ray)
  const Vector3D hit_p = r.o + r.d * isect.t;
  const Vector3D w_out = w2o * (-r.d);

  // This is the same number of total samples as
  // estimate_direct_lighting_importance (outside of delta lights). We keep the
  // same number of samples for clarity of comparison.
  int num_samples = scene->lights.size() * ns_area_light;
  Vector3D L_out;

  // (Part 3): Write your sampling loop here
  // UPDATE `est_radiance_global_illumination` to return direct lighting instead of normal shading 
  
  for (int i = 0; i < num_samples; i++) {
    Vector3D incident_direction = hemisphereSampler->get_sample();
    Vector3D world_space_direction = o2w * incident_direction;
    double cosine_coefficient = dot(world_space_direction, isect.n);
    Ray secondary_ray = Ray(hit_p, world_space_direction);
    secondary_ray.min_t = EPS_F;
    Intersection next_surface_intersection;

    if (bvh->intersect(secondary_ray, &next_surface_intersection)) {
      Vector3D material_response =
          evaluate_bsdf_at_hit(isect, w_out, incident_direction);
      Vector3D emitted_light = next_surface_intersection.bsdf->get_emission();
      double uniform_hemisphere_pdf = 1.0 / (2.0 * PI);
      L_out += material_response * emitted_light * cosine_coefficient / uniform_hemisphere_pdf;
    }
  }                                               
  return L_out / num_samples;

}

Vector3D
PathTracer::estimate_direct_lighting_importance(const Ray &r,
                                                const Intersection &isect) {
  // Estimate the lighting from this intersection coming directly from a light.
  // To implement importance sampling, sample only from lights, not uniformly in
  // a hemisphere.

  // make a coordinate system for a hit point
  // with N aligned with the Z direction.
  Matrix3x3 o2w;
  make_coord_space(o2w, isect.n);
  Matrix3x3 w2o = o2w.T();

  // w_out points towards the source of the ray (e.g.,
  // toward the camera if this is a primary ray)
  const Vector3D hit_p = r.o + r.d * isect.t;
  const Vector3D w_out = w2o * (-r.d);
  Vector3D L_out;
  for (SceneLight* current_light : scene->lights) {
    bool is_point_light = current_light->is_delta_light();
    size_t samples_per_light = is_point_light ? 1 : ns_area_light;
    
    for (size_t sample_idx = 0; sample_idx < samples_per_light; sample_idx++) {
      Vector3D incident_light_direction;
      double distance_to_light_source;
      double sampling_probability;
      Vector3D light_radiance = current_light->sample_L(hit_p, &incident_light_direction, &distance_to_light_source, &sampling_probability);

      double cosine_term = dot(incident_light_direction, isect.n);
      if (cosine_term < 0) continue; // Skip this sample
      
      Ray visibility_ray = Ray(hit_p, incident_light_direction);
      visibility_ray.min_t = EPS_F;
      visibility_ray.max_t = distance_to_light_source - EPS_F;
      
      bool light_is_visible = !bvh->has_intersection(visibility_ray);
      
      if (light_is_visible) {
        Vector3D bsdf_value =
            evaluate_bsdf_at_hit(isect, w_out, w2o * incident_light_direction);
        double contribution_weight = is_point_light ? ns_area_light : 1.0;
        L_out += contribution_weight * bsdf_value * light_radiance * cosine_term / sampling_probability;
      }
    }
  }
  return L_out / ns_area_light;

}

Vector3D PathTracer::zero_bounce_radiance(const Ray &r,
                                          const Intersection &isect) {
  // Part 3, Task 2
  // Returns the light that results from no bounces of light

  if (isect.bsdf == nullptr) return Vector3D();
  return isect.bsdf->get_emission();

}

Vector3D PathTracer::one_bounce_radiance(const Ray &r,
                                         const Intersection &isect) {
  // Part 3, Task 3
  // Returns either the direct illumination by hemisphere or importance sampling
  // depending on `direct_hemisphere_sample`

  if (!direct_hemisphere_sample) {
    return estimate_direct_lighting_importance(r, isect);
  }
  return estimate_direct_lighting_hemisphere(r, isect);

}

Vector3D PathTracer::random_walk_subsurface_radiance(
    const Ray &r, const Intersection &isect, const RandomWalkSSSBSDF* bsdf) {
  const Primitive* volume_primitive = isect.primitive;
  const Sphere* sphere = dynamic_cast<const Sphere*>(volume_primitive);
  bool mesh_volume = dynamic_cast<const Triangle*>(volume_primitive) != NULL;
  if ((!sphere && !mesh_volume) || !bsdf) return Vector3D();

  Vector3D hit_p = r.o + r.d * isect.t;
  Vector3D normal = isect.n.unit();
  if (dot(r.d, normal) > 0.0) {
    normal = -normal;
  }
  double cos_entry = -dot(r.d, normal);
  if (cos_entry <= 0.0) return Vector3D();

  auto trace_continuation = [this](const Ray& ray) {
    Intersection next_isect;
    if (bvh->intersect(ray, &next_isect)) {
      return zero_bounce_radiance(ray, next_isect) +
             at_least_one_bounce_radiance(ray, next_isect);
    }
    return envLight ? envLight->sample_dir(ray) : Vector3D();
  };

  double eta = max(bsdf->get_ior(), 1.0001);
  double entry_fresnel = schlick_fresnel(cos_entry, 1.0, eta);
  double specular_weight = clamp(bsdf->get_specular_weight(), 0.0, 1.0);
  bool proposal_layered = bsdf->get_preset_type() == BSDF_PRESET_RANDOM_WALK_LAYERED;
  double surface_roughness = clamp(bsdf->get_surface_roughness(), 0.02, 1.0);
  double entry_surface_reflectance = specular_weight * entry_fresnel;
  double base_layer_weight = proposal_layered
      ? 1.0 - specular_weight
      : 1.0 - entry_surface_reflectance;
  Vector3D base_tint = random_walk_base_tint(bsdf);
  Vector3D view_dir = (-r.d).unit();
  Vector3D gloss_origin = hit_p + normal * EPS_F;
  Vector3D L_out;

  if (specular_weight > 0.0) {
    Vector3D direct_gloss;
    for (SceneLight* current_light : scene->lights) {
      bool is_point_light = current_light->is_delta_light();
      size_t samples_per_light = is_point_light ? 1 : ns_area_light;

      for (size_t sample_idx = 0; sample_idx < samples_per_light; sample_idx++) {
        Vector3D light_direction;
        double distance_to_light;
        double light_pdf;
        Vector3D light_radiance =
            current_light->sample_L(gloss_origin, &light_direction,
                                    &distance_to_light, &light_pdf);
        if (light_pdf <= 0.0 || light_radiance.illum() <= 0.0) continue;

        double n_dot_l = dot(normal, light_direction);
        if (n_dot_l <= 0.0) continue;

        Ray shadow_ray(gloss_origin, light_direction);
        shadow_ray.min_t = EPS_F;
        shadow_ray.max_t = distance_to_light - EPS_F;
        if (bvh->has_intersection(shadow_ray)) continue;

        double gloss_f = rough_dielectric_brdf(view_dir, light_direction, normal,
                                               surface_roughness, eta);
        if (gloss_f <= 0.0) continue;

        double bsdf_pdf = rough_dielectric_reflection_pdf(view_dir, light_direction,
                                                          normal, surface_roughness);
        double mis_weight = is_point_light ? 1.0 : power_heuristic(light_pdf, bsdf_pdf);
        double contribution_weight = is_point_light ? ns_area_light : 1.0;
        direct_gloss += contribution_weight * specular_weight * gloss_f *
                        light_radiance * n_dot_l * mis_weight / light_pdf;
      }
    }
    if (ns_area_light > 0) {
      direct_gloss /= ns_area_light;
    }
    L_out += direct_gloss;
  }

  if (r.depth > 1 && specular_weight > 0.0) {
    Vector3D reflected;
    double reflection_weight = 0.0;
    double bsdf_pdf = 0.0;
    if (sample_rough_dielectric_reflection(view_dir, normal, surface_roughness, eta,
                                           &reflected, &reflection_weight,
                                           &bsdf_pdf)) {
      Ray reflected_ray(gloss_origin, reflected);
      reflected_ray.min_t = EPS_F;
      reflected_ray.depth = r.depth - 1;

      Vector3D reflected_radiance;
      Intersection reflected_isect;
      if (bvh->intersect(reflected_ray, &reflected_isect)) {
        Vector3D emitted = zero_bounce_radiance(reflected_ray, reflected_isect);
        if (emitted.illum() > 0.0) {
          double light_pdf = light_pdf_for_direction(scene->lights, gloss_origin, reflected);
          double mis_weight = light_pdf > 0.0 ? power_heuristic(bsdf_pdf, light_pdf) : 1.0;
          reflected_radiance += emitted * mis_weight;
        }
        reflected_radiance += at_least_one_bounce_radiance(reflected_ray,
                                                           reflected_isect);
      } else if (envLight) {
        reflected_radiance += envLight->sample_dir(reflected_ray);
      }

      L_out += reflected_radiance * (specular_weight * reflection_weight);
    }
  }

  if (base_layer_weight <= 0.0) {
    return L_out;
  }

  Vector3D sigma_a = clamp_vector_min(bsdf->get_sigma_a() * bsdf->get_scale(), 0.0);
  Vector3D sigma_s = clamp_vector_min(bsdf->get_sigma_s() * bsdf->get_scale(), 0.0);
  Vector3D sigma_t = sigma_a + sigma_s;
  double sigma_t_mean = max(mean_channel(sigma_t), 1e-6);
  Vector3D albedo = sigma_s / (sigma_t + Vector3D(1e-8));

  Vector3D throughput(base_layer_weight);
  Vector3D direction = r.d.unit();
  Vector3D position = hit_p + direction * EPS_F;
  bool scattered = false;

  for (int event = 0; event < kMaxSubsurfaceEvents; event++) {
    double boundary_distance;
    Vector3D exit_normal;
    if (!find_subsurface_boundary(bvh, volume_primitive, sphere, position,
                                  direction, &boundary_distance,
                                  &exit_normal)) {
      return L_out;
    }

    double u = max(1.0 - random_uniform(), 1e-8);
    double scatter_distance = -log(u) / sigma_t_mean;

    if (scatter_distance < boundary_distance) {
      throughput = throughput * exp_vector(-sigma_a * scatter_distance) * albedo;
      position += direction * scatter_distance;
      direction = sample_henyey_greenstein(direction, bsdf->get_anisotropy_g());
      scattered = true;

      if (event >= 3) {
        double survival = clamp(throughput.illum(), 0.1, 0.95);
        if (!coin_flip(survival)) return L_out;
        throughput /= survival;
      }
      continue;
    }

    throughput = throughput * exp_vector(-sigma_a * boundary_distance);
    Vector3D exit_p = position + direction * boundary_distance;
    double cos_exit = dot(direction, exit_normal);
    if (cos_exit <= 0.0) {
      exit_normal = -exit_normal;
      cos_exit = dot(direction, exit_normal);
      if (cos_exit <= 0.0) return L_out;
    }

    double exit_fresnel = schlick_fresnel(cos_exit, eta, 1.0);
    double exit_reflectance = clamp(specular_weight * exit_fresnel, 0.0, 0.95);
    if (exit_reflectance > 0.0 && coin_flip(exit_reflectance)) {
      direction = reflect_across_normal(direction, exit_normal);
      position = exit_p + direction * EPS_F;
      continue;
    }

    if (!scattered) {
      Ray ballistic_ray(exit_p + direction * EPS_F, direction);
      ballistic_ray.min_t = EPS_F;
      ballistic_ray.depth = r.depth > 0 ? r.depth - 1 : 0;
      L_out += throughput * base_tint * trace_continuation(ballistic_ray);
      return L_out;
    }

    Vector3D exit_origin = exit_p + exit_normal * EPS_F;
    Vector3D exit_direct;
    for (SceneLight* current_light : scene->lights) {
      bool is_point_light = current_light->is_delta_light();
      size_t samples_per_light = is_point_light ? 1 : ns_area_light;

      for (size_t sample_idx = 0; sample_idx < samples_per_light; sample_idx++) {
        Vector3D light_direction;
        double distance_to_light;
        double light_pdf;
        Vector3D light_radiance =
            current_light->sample_L(exit_origin, &light_direction,
                                    &distance_to_light, &light_pdf);
        if (light_pdf <= 0.0) continue;

        double cos_light = dot(light_direction, exit_normal);
        if (cos_light <= 0.0) continue;

        Ray shadow_ray(exit_origin, light_direction);
        shadow_ray.min_t = EPS_F;
        shadow_ray.max_t = distance_to_light - EPS_F;
        if (!bvh->has_intersection(shadow_ray)) {
          double light_fresnel = schlick_fresnel(cos_light, 1.0, eta);
          double contribution_weight = is_point_light ? ns_area_light : 1.0;
          exit_direct += contribution_weight * light_radiance *
                         (1.0 - light_fresnel) * cos_light / light_pdf;
        }
      }
    }
    if (ns_area_light > 0) {
      exit_direct /= ns_area_light;
    }

    // This is a BSSRDF-style exit event, not glass transmission. Once the
    // random walk exits, gather illumination incident at that exit point; do
    // not continue the original camera path through the object into the scene.
    L_out += throughput * base_tint * exit_direct / PI;
    return L_out;
  }

  return L_out;
}

Vector3D PathTracer::at_least_one_bounce_radiance(const Ray &r,
                                                   const Intersection &isect) {
  if (RandomWalkSSSBSDF* sss = dynamic_cast<RandomWalkSSSBSDF*>(isect.bsdf)) {
    bool supports_random_walk =
        dynamic_cast<const Sphere*>(isect.primitive) ||
        dynamic_cast<const Triangle*>(isect.primitive);
    if (supports_random_walk && dot(r.d, isect.n) < 0.0) {
      return random_walk_subsurface_radiance(r, isect, sss);
    }
  }

  Matrix3x3 o2w;
  make_coord_space(o2w, isect.n);
  Matrix3x3 w2o = o2w.T();

  Vector3D hit_p = r.o + r.d * isect.t;
  Vector3D w_out = w2o * (-r.d);
  Vector3D L_out(0, 0, 0);

  if (isAccumBounces || r.depth == 1) {
    L_out += one_bounce_radiance(r, isect);
  }
  if (r.depth <= 1) return L_out;

  const double survival_probability = 0.7;
  if (!coin_flip(survival_probability)) return L_out;

  Vector3D wi;
  double pdf = 0.0;
  Vector3D f = sample_bsdf_at_hit(isect, w_out, &wi, &pdf);
  if (pdf <= 0.0) return L_out;

  Ray rn = Ray(hit_p, o2w * wi);
  rn.min_t = EPS_F;
  rn.depth = r.depth - 1;

  Intersection isectn;
  if (bvh->intersect(rn, &isectn)) {
    Vector3D incoming = zero_bounce_radiance(rn, isectn);
    incoming += at_least_one_bounce_radiance(rn, isectn);
    L_out += f * incoming * abs_cos_theta(wi) / (pdf * survival_probability);
  } else if (envLight) {
    Vector3D incoming = envLight->sample_dir(rn);
    L_out += f * incoming * abs_cos_theta(wi) / (pdf * survival_probability);
  }

  return L_out;
}

Vector3D PathTracer::est_radiance_global_illumination(const Ray &r) {
  Intersection isect;
  Vector3D L_out;

  // You will extend this in assignment 3-2.
  // If no intersection occurs, we simply return black.
  // This changes if you implement hemispherical lighting for extra credit.

  // The following line of code returns a debug color depending
  // on whether ray intersection with triangles or spheres has
  // been implemented.
  //
  
  // detects hit
  if (!bvh->intersect(r, &isect)) // no hit
    return envLight ? envLight->sample_dir(r) : L_out;

  L_out = zero_bounce_radiance(r, isect);
  if (max_ray_depth > 0) {
    L_out += at_least_one_bounce_radiance(r, isect);
  }
  return L_out;
}

void PathTracer::raytrace_pixel(size_t x, size_t y) {
  // Part 1.2, Part 5
  // Make a loop that generates num_samples camera rays and traces them
  // through the scene. Return the average Vector3D.
  // You should call est_radiance_global_illumination in this function.
  int num_samples = 0;
  Vector3D result = Vector3D(0, 0, 0);
  
  double s1 = 0.0, s2 = 0.0;
  for (int sample_idx = 0; sample_idx < ns_aa; sample_idx++) {
    Vector2D jitter = gridSampler->get_sample();
    Vector2D pixel_location = {(x + jitter.x) / sampleBuffer.w, (y + jitter.y) / sampleBuffer.h};
    
    Ray camera_ray = camera->generate_ray(pixel_location.x, pixel_location.y);
    camera_ray.depth = max_ray_depth;
    
    Vector3D sample_color = est_radiance_global_illumination(camera_ray);
    result += sample_color;
    num_samples++;

    double xk = sample_color.illum();
    s1 += xk;
    s2 += xk * xk;

    // check if bigI is within tolerance
    if (num_samples % samplesPerBatch == 0) {
      double mean = s1 / num_samples;
      double vari = (1.0 / (num_samples - 1)) * (s2 - s1 * s1 / num_samples);
      double bigI = 1.96 * sqrt(vari / num_samples);
      if (bigI <= maxTolerance * mean) { // tolerable noise
        break;
      }
    }
  }

  result /= num_samples;
  sampleBuffer.update_pixel(result, x, y);
  sampleCountBuffer[x + y * sampleBuffer.w] = num_samples;

}

void PathTracer::autofocus(Vector2D loc) {
  Ray r = camera->generate_ray(loc.x / sampleBuffer.w, loc.y / sampleBuffer.h);
  Intersection isect;

  bvh->intersect(r, &isect);

  camera->focalDistance = isect.t;
}

} // namespace CGL
