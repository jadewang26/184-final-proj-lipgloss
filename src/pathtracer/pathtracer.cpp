#include "pathtracer.h"

#include "scene/light.h"
#include "scene/sphere.h"
#include "scene/triangle.h"


using namespace CGL::SceneObjects;

namespace CGL {

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
      Vector3D material_response = isect.bsdf->f(w_out, incident_direction);
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
      
      Vector3D bsdf_value = isect.bsdf->f(w_out, w2o * incident_light_direction);
      
      Ray visibility_ray = Ray(hit_p, incident_light_direction);
      visibility_ray.min_t = EPS_F;
      visibility_ray.max_t = distance_to_light_source - EPS_F;
      
      Intersection occluding_intersection;
      bool light_is_visible = !bvh->intersect(visibility_ray, &occluding_intersection);
      
      if (light_is_visible) {
        double cosine_term = dot(incident_light_direction, isect.n);
        if (cosine_term < 0) continue; // Skip this sample
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

Vector3D PathTracer::at_least_one_bounce_radiance(const Ray &r,
                                                  const Intersection &isect) {
  Matrix3x3 o2w;
  make_coord_space(o2w, isect.n);
  Matrix3x3 w2o = o2w.T();

  Vector3D hit_p = r.o + r.d * isect.t;
  Vector3D w_out = w2o * (-r.d);
  Vector3D L_out(0, 0, 0);

  // Part 4.2
  // Returns the one bounce radiance + radiance from extra bounces at this point
  // Should be called recursively to simulate extra bounces

  if (r.depth == 1) { // base case
    return one_bounce_radiance(r, isect);
  }
  if (isAccumBounces) { // add direct lighting at this hit point
    L_out += one_bounce_radiance(r, isect);
  }

    /* for ONLY direct and ONLY indirect
  if (r.depth == 1) { // base case
    // indirect only would set this to black
    if (r.depth == max_ray_depth) {
        return Vector3D(0, 0, 0); 
    }
    return one_bounce_radiance(r, isect);
  }
  
  if (isAccumBounces) { // add direct lighting at this hit point
    // adds light if it isn't the direct light
    if (r.depth != max_ray_depth) {
        L_out += one_bounce_radiance(r, isect);
    }
  }
    */

  Vector3D wi;
  double pdf;
  Vector3D f = isect.bsdf->sample_f(w_out, &wi, &pdf); // samples next bounce

  // recursive ray
  Ray rn = Ray(hit_p, o2w *wi);
  rn.min_t = EPS_F;
  rn.depth = r.depth - 1;

  Intersection isectn;
  double prr = 0.35; // probability of termination, 0.35
  if (bvh->intersect(rn, &isectn)) {
    if (isAccumBounces) {
      if ((max_ray_depth > 1 && r.depth == max_ray_depth) || !coin_flip(prr)) {
        L_out += at_least_one_bounce_radiance(rn, isectn) * f * dot(o2w * wi, isect.n) / pdf;
      }
    } else {
      return at_least_one_bounce_radiance(rn, isectn) * f * dot(o2w * wi, isect.n) / pdf;
    }
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


  // L_out = (isect.t == INF_D) ? debug_shading(r.d) : normal_shading(isect.n);

  // (Part 3): Return the direct illumination.
  //Vector3D zero_bounce_component = zero_bounce_radiance(r, isect);
  //Vector3D one_bounce_component = one_bounce_radiance(r, isect);
  // return zero_bounce_component + one_bounce_component;

  
  // Part 4: Accumulate the "direct" and "indirect"
  // parts of global illumination into L_out rather than just direct

  if (isAccumBounces) { // gather all light
    // just light from object it hit if == 0, else add at_least_one_bounce_radiance
    return r.depth == 0 ? zero_bounce_radiance(r, isect) : zero_bounce_radiance(r, isect) + at_least_one_bounce_radiance(r, isect);
  } else {
    return r.depth == 0 ? zero_bounce_radiance(r, isect) : at_least_one_bounce_radiance(r, isect); // not accumulate
  }

  // return L_out;
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
