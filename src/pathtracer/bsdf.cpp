#include "bsdf.h"

#include "application/visual_debugger.h"

#include <algorithm>
#include <iostream>
#include <utility>


using std::max;
using std::min;
using std::swap;

namespace CGL {

namespace {

const BSDFPresetType kEditableBSDFPresetTypes[] = {
    BSDF_PRESET_DIFFUSE,
    BSDF_PRESET_MICROFACET,
    BSDF_PRESET_MIRROR,
    BSDF_PRESET_REFRACTION,
    BSDF_PRESET_GLASS,
    BSDF_PRESET_EMISSION,
    BSDF_PRESET_APPROXIMATE_BSSRDF,
    BSDF_PRESET_RANDOM_WALK_SSS,
    BSDF_PRESET_RANDOM_WALK_LAYERED,
    BSDF_PRESET_LAYERED,
    BSDF_PRESET_FAST_LAYERED,
    BSDF_PRESET_DISNEY_LAYERED,
};

void sync_base_layer(ApproximateBSSRDF*& base_layer, const Vector3D& base_color,
                     double roughness) {
  delete base_layer;
  base_layer = new ApproximateBSSRDF(base_color, roughness);
}

bool is_zero_vector(const Vector3D& value) {
  return value.x == 0.0 && value.y == 0.0 && value.z == 0.0;
}

Vector3D default_surface_color() {
  return Vector3D(0.8, 0.8, 0.8);
}

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

Vector3D sigma_albedo(const Vector3D& sigma_a, const Vector3D& sigma_s) {
  Vector3D absorption = clamp_vector_min(sigma_a, 0.0);
  Vector3D scattering = clamp_vector_min(sigma_s, 0.0);
  Vector3D sigma_t = absorption + scattering + Vector3D(1e-8);
  return clamp_color01(scattering / sigma_t);
}

Vector3D saturated_base_tint(const Vector3D& base_color, double saturation) {
  return clamp_color01(clamp_vector_min(base_color, 0.0) *
                       clamp(saturation, 0.0, 1.5));
}

double smoothstep(double edge0, double edge1, double x) {
  double t = clamp((x - edge0) / max(edge1 - edge0, 1e-8), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

double gaussian(double x, double width) {
  return exp(-(x * x) / max(width * width, 1e-8));
}

double radial_lip_wrinkle_u(double u, double local_v, bool upper) {
  double s = max(0.0, 1.0 - u * u);
  double corner = smoothstep(0.20, 0.92, fabs(u));
  double outward_v = upper ? local_v : 1.0 - local_v;
  double signed_side = tanh(4.0 * u);
  double shear = (0.012 + 0.120 * corner) * outward_v;
  double bow = sin(PI * outward_v) * sin(PI * u) * pow(s, 0.30);
  double curve = -signed_side * shear + (upper ? -0.020 : 0.018) * bow;
  return u + curve;
}

double lip_pooling_factor(const Vector2D& uv) {
  double u = 2.0 * clamp(uv.x, 0.0, 1.0) - 1.0;
  double v = clamp(uv.y, 0.0, 1.0);
  bool upper = v >= 0.5;
  double local_v = upper ? 2.0 * (v - 0.5) : 2.0 * v;
  double s = max(0.0, 1.0 - u * u);
  double edge_fade = 1.0 - smoothstep(0.72, 1.0, fabs(u));

  double center = gaussian(u, 0.42);
  double lower_wet_band = center * gaussian(v - 0.30, 0.13);
  double mouth_crease = (0.65 + 0.35 * center) * gaussian(v - 0.50, 0.045);
  double cupid_pool = gaussian(u, 0.16) * gaussian(v - 0.72, 0.10);

  double curved_u = radial_lip_wrinkle_u(u, local_v, upper);
  double curved_x = 0.5 * (curved_u + 1.0);
  double wave = 0.5 + 0.5 * cos(24.0 * PI * curved_x +
                                 0.55 * sin(9.0 * PI * curved_x));
  double corner_relief = 1.0 - 0.45 * smoothstep(0.55, 0.98, fabs(u));
  double wrinkle_valleys = pow(wave, 7.0) * gaussian(v - 0.43, 0.34) *
                           edge_fade * corner_relief;

  return clamp(0.45 * lower_wet_band + 0.35 * mouth_crease +
               0.25 * wrinkle_valleys + 0.12 * cupid_pool,
               0.0, 1.0);
}

void pooled_layer_params(const Vector2D& uv, double pooling_strength,
                         double base_roughness, double base_thickness,
                         double base_saturation, double* roughness_out,
                         double* thickness_out, double* saturation_out) {
  double p = clamp(pooling_strength, 0.0, 1.0) * lip_pooling_factor(uv);
  double wet_roughness = max(0.025, base_roughness * 0.35);
  *roughness_out = clamp(base_roughness * (1.0 - p) + wet_roughness * p,
                         0.02, 1.0);
  *thickness_out = clamp(base_thickness + (1.0 - base_thickness) * 0.75 * p,
                         0.0, 1.0);
  *saturation_out = clamp(base_saturation + 0.15 * p, 0.0, 1.5);
}

Vector3D infer_surface_color(const BSDFPreset& preset) {
  switch (preset.type) {
    case BSDF_PRESET_DIFFUSE:
    case BSDF_PRESET_MIRROR:
    case BSDF_PRESET_EMISSION:
    case BSDF_PRESET_APPROXIMATE_BSSRDF:
    case BSDF_PRESET_LAYERED:
    case BSDF_PRESET_FAST_LAYERED:
    case BSDF_PRESET_DISNEY_LAYERED:
    case BSDF_PRESET_REFRACTION:
      return preset.vector_a;
    case BSDF_PRESET_RANDOM_WALK_SSS:
      return !is_zero_vector(preset.vector_c)
          ? preset.vector_c
          : sigma_albedo(preset.vector_a, preset.vector_b);
    case BSDF_PRESET_RANDOM_WALK_LAYERED:
      return !is_zero_vector(preset.vector_c)
          ? preset.vector_c
          : sigma_albedo(preset.vector_a, preset.vector_b);
    case BSDF_PRESET_GLASS:
      return !is_zero_vector(preset.vector_b) ? preset.vector_b : preset.vector_a;
    case BSDF_PRESET_MICROFACET:
      return default_surface_color();
    default:
      return default_surface_color();
  }
}

Vector3D infer_subsurface_color(const BSDFPreset& preset) {
  if ((preset.type == BSDF_PRESET_LAYERED ||
       preset.type == BSDF_PRESET_FAST_LAYERED ||
       preset.type == BSDF_PRESET_DISNEY_LAYERED) &&
      !is_zero_vector(preset.vector_b)) {
    return preset.vector_b;
  }
  if (preset.type == BSDF_PRESET_APPROXIMATE_BSSRDF) {
    return preset.vector_a;
  }
  if (preset.type == BSDF_PRESET_RANDOM_WALK_LAYERED &&
      !is_zero_vector(preset.vector_c)) {
    return preset.vector_c;
  }
  if (preset.type == BSDF_PRESET_RANDOM_WALK_SSS ||
      preset.type == BSDF_PRESET_RANDOM_WALK_LAYERED) {
    return sigma_albedo(preset.vector_a, preset.vector_b);
  }
  return infer_surface_color(preset);
}

double infer_roughness(const BSDFPreset& preset) {
  switch (preset.type) {
    case BSDF_PRESET_MICROFACET:
    case BSDF_PRESET_REFRACTION:
    case BSDF_PRESET_GLASS:
    case BSDF_PRESET_APPROXIMATE_BSSRDF:
    case BSDF_PRESET_LAYERED:
    case BSDF_PRESET_FAST_LAYERED:
    case BSDF_PRESET_DISNEY_LAYERED:
      return preset.scalar_a;
    case BSDF_PRESET_RANDOM_WALK_SSS:
    case BSDF_PRESET_RANDOM_WALK_LAYERED:
      return preset.scalar_d;
    default:
      return 0.2;
  }
}

double infer_subsurface_roughness(const BSDFPreset& preset) {
  if (preset.type == BSDF_PRESET_LAYERED ||
      preset.type == BSDF_PRESET_FAST_LAYERED ||
      preset.type == BSDF_PRESET_DISNEY_LAYERED) {
    return preset.scalar_e;
  }
  return infer_roughness(preset);
}

double infer_ior(const BSDFPreset& preset) {
  switch (preset.type) {
    case BSDF_PRESET_REFRACTION:
    case BSDF_PRESET_GLASS:
      return preset.scalar_b;
    case BSDF_PRESET_RANDOM_WALK_SSS:
    case BSDF_PRESET_RANDOM_WALK_LAYERED:
      return preset.scalar_b;
    case BSDF_PRESET_LAYERED:
    case BSDF_PRESET_FAST_LAYERED:
    case BSDF_PRESET_DISNEY_LAYERED:
      return preset.scalar_d;
    default:
      return 1.5;
  }
}

bool preset_carries_ior(const BSDFPreset& preset) {
  switch (preset.type) {
    case BSDF_PRESET_REFRACTION:
    case BSDF_PRESET_GLASS:
    case BSDF_PRESET_RANDOM_WALK_SSS:
    case BSDF_PRESET_RANDOM_WALK_LAYERED:
    case BSDF_PRESET_LAYERED:
    case BSDF_PRESET_FAST_LAYERED:
    case BSDF_PRESET_DISNEY_LAYERED:
      return true;
    default:
      return false;
  }
}

double infer_thickness(const BSDFPreset& preset) {
  switch (preset.type) {
    case BSDF_PRESET_RANDOM_WALK_SSS:
    case BSDF_PRESET_RANDOM_WALK_LAYERED:
      return preset.scalar_e;
    case BSDF_PRESET_LAYERED:
    case BSDF_PRESET_FAST_LAYERED:
    case BSDF_PRESET_DISNEY_LAYERED:
      return preset.scalar_b;
    default:
      return 0.5;
  }
}

double infer_saturation(const BSDFPreset& preset) {
  switch (preset.type) {
    case BSDF_PRESET_RANDOM_WALK_LAYERED:
      return preset.scalar_f;
    case BSDF_PRESET_LAYERED:
    case BSDF_PRESET_FAST_LAYERED:
    case BSDF_PRESET_DISNEY_LAYERED:
      return preset.scalar_c;
    default:
      return 1.0;
  }
}

// Cosine-weighted hemisphere density used by the flesh/base layer.
// We recompute this even after a gloss sample so the final pdf can be the
// full two-lobe mixture instead of only the selected branch density.
double cosine_hemisphere_pdf(const Vector3D& wi) {
  return wi.z > 0.0 ? wi.z / PI : 0.0;
}

// Keep the gloss roughness remap identical to the evaluator. If the sampler
// and evaluator use different roughness values, recursive f / pdf weighting
// drifts and the lips darken or brighten incorrectly at higher ray depths.
double disney_gloss_roughness(double roughness) {
  return max(roughness * 0.8, 0.02);
}

// GGX normal distribution function used by the Disney gloss lobe.
// This gives the density of half-vectors around the surface normal.
double ggx_ndf(double cos_theta_h, double alpha2) {
  if (cos_theta_h <= 0.0) return 0.0;
  double cos2_theta_h = cos_theta_h * cos_theta_h;
  double denom = PI * pow(cos2_theta_h * (alpha2 - 1.0) + 1.0, 2.0);
  return denom > 0.0 ? alpha2 / denom : 0.0;
}

// Convert the GGX half-vector density into a direction-space pdf for wi.
// This is the gloss branch density that matches the Cook-Torrance-style
// gloss term evaluated in DisneyLayeredBSDF::f().
double disney_gloss_pdf(const Vector3D& wo, const Vector3D& wi, double roughness) {
  if (wo.z <= 0.0 || wi.z <= 0.0) return 0.0;

  Vector3D h = wo + wi;
  if (h.norm2() == 0.0) return 0.0;
  h.normalize();

  double wo_dot_h = dot(wo, h);
  if (wo_dot_h <= 0.0) return 0.0;

  double alpha = disney_gloss_roughness(roughness);
  double alpha2 = max(alpha * alpha, 0.0004);
  double D = ggx_ndf(max(h.z, 0.0), alpha2);
  return D * max(h.z, 0.0) / (4.0 * wo_dot_h);
}

// Sample the same GGX gloss lobe that DisneyLayeredBSDF::f() evaluates.
// The old code used a reflected direction plus a cosine perturbation, which
// did not match the lobe being evaluated and caused recursive bias.
bool sample_disney_gloss_lobe(const Vector3D& wo, double roughness,
                              Vector3D* wi, double* pdf) {
  if (wo.z <= 0.0) {
    *pdf = 0.0;
    return false;
  }

  double alpha = disney_gloss_roughness(roughness);
  double alpha2 = max(alpha * alpha, 0.0004);
  Vector2D u(random_uniform(), random_uniform());
  double phi = 2.0 * PI * u.y;
  double tan_theta2 = alpha2 * u.x / max(1.0 - u.x, 1e-6);
  double cos_theta = 1.0 / sqrt(1.0 + tan_theta2);
  double sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
  Vector3D h(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);

  double wo_dot_h = dot(wo, h);
  if (wo_dot_h <= 0.0) {
    *pdf = 0.0;
    return false;
  }

  *wi = 2.0 * wo_dot_h * h - wo;
  if (wi->z <= 0.0) {
    *pdf = 0.0;
    return false;
  }

  *pdf = disney_gloss_pdf(wo, *wi, roughness);
  return *pdf > 0.0;
}

double layered_gloss_alpha2(double roughness) {
  double alpha = max(roughness * roughness, 0.001);
  return alpha * alpha;
}

// Convert the GGX half-vector density used by LayeredBSDF::f() into a
// direction-space pdf for wi.
double layered_gloss_pdf(const Vector3D& wo, const Vector3D& wi,
                         double roughness) {
  if (wo.z <= 0.0 || wi.z <= 0.0) return 0.0;

  Vector3D h = wo + wi;
  if (h.norm2() == 0.0) return 0.0;
  h.normalize();

  double wo_dot_h = dot(wo, h);
  if (wo_dot_h <= 0.0) return 0.0;

  double alpha2 = layered_gloss_alpha2(roughness);
  double D = ggx_ndf(max(h.z, 0.0), alpha2);
  return D * max(h.z, 0.0) / (4.0 * wo_dot_h);
}

// Sample the same GGX gloss lobe used by LayeredBSDF::f() so recursive
// transport uses a consistent gloss distribution and pdf.
bool sample_layered_gloss_lobe(const Vector3D& wo, double roughness,
                               Vector3D* wi, double* pdf) {
  if (wo.z <= 0.0) {
    *pdf = 0.0;
    return false;
  }

  double alpha2 = layered_gloss_alpha2(roughness);
  Vector2D u(random_uniform(), random_uniform());
  double phi = 2.0 * PI * u.y;
  double tan_theta2 = alpha2 * u.x / max(1.0 - u.x, 1e-6);
  double cos_theta = 1.0 / sqrt(1.0 + tan_theta2);
  double sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
  Vector3D h(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);

  double wo_dot_h = dot(wo, h);
  if (wo_dot_h <= 0.0) {
    *pdf = 0.0;
    return false;
  }

  *wi = 2.0 * wo_dot_h * h - wo;
  if (wi->z <= 0.0) {
    *pdf = 0.0;
    return false;
  }

  *pdf = layered_gloss_pdf(wo, *wi, roughness);
  return *pdf > 0.0;
}

double fast_layered_gloss_exponent(double roughness) {
  double r4 = max(pow(roughness, 4.0), 0.0001);
  return max((2.0 / r4) - 2.0, 0.0);
}

// Convert the normalized Blinn-Phong half-vector density used by the fast
// layered gloss term into a direction-space pdf for wi.
double fast_layered_gloss_pdf(const Vector3D& wo, const Vector3D& wi,
                              double roughness) {
  if (wo.z <= 0.0 || wi.z <= 0.0) return 0.0;

  Vector3D h = wo + wi;
  if (h.norm2() == 0.0) return 0.0;
  h.normalize();

  double wo_dot_h = dot(wo, h);
  if (wo_dot_h <= 0.0) return 0.0;

  double exponent = fast_layered_gloss_exponent(roughness);
  double cos_theta_h = max(h.z, 0.0);
  double half_vector_pdf =
      ((exponent + 2.0) / (2.0 * PI)) * pow(cos_theta_h, exponent + 1.0);
  return half_vector_pdf / (4.0 * wo_dot_h);
}

// Sample the same normalized Blinn-Phong gloss lobe used by
// FastLayeredBSDF::f(). This keeps the fast layered evaluator and sampler
// aligned so recursive bounces use a consistent f / pdf ratio.
bool sample_fast_layered_gloss_lobe(const Vector3D& wo, double roughness,
                                    Vector3D* wi, double* pdf) {
  if (wo.z <= 0.0) {
    *pdf = 0.0;
    return false;
  }

  double exponent = fast_layered_gloss_exponent(roughness);
  Vector2D u(random_uniform(), random_uniform());
  double phi = 2.0 * PI * u.y;
  double cos_theta = pow(1.0 - u.x, 1.0 / (exponent + 2.0));
  double sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
  Vector3D h(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);

  double wo_dot_h = dot(wo, h);
  if (wo_dot_h <= 0.0) {
    *pdf = 0.0;
    return false;
  }

  *wi = 2.0 * wo_dot_h * h - wo;
  if (wi->z <= 0.0) {
    *pdf = 0.0;
    return false;
  }

  *pdf = fast_layered_gloss_pdf(wo, *wi, roughness);
  return *pdf > 0.0;
}

BSDFPreset make_default_bsdf_preset(BSDFPresetType type) {
  BSDFPreset preset;
  preset.type = type;

  switch (type) {
    case BSDF_PRESET_DIFFUSE:
    case BSDF_PRESET_MIRROR:
      preset.vector_a = default_surface_color();
      break;
    case BSDF_PRESET_MICROFACET:
      preset.vector_a = Vector3D(1.5, 1.5, 1.5);
      preset.vector_b = Vector3D(1.0, 1.0, 1.0);
      preset.scalar_a = 0.2;
      break;
    case BSDF_PRESET_REFRACTION:
      preset.vector_a = Vector3D(0.95, 0.95, 0.95);
      preset.scalar_a = 0.0;
      preset.scalar_b = 1.5;
      break;
    case BSDF_PRESET_GLASS:
      preset.vector_a = Vector3D(0.95, 0.95, 0.95);
      preset.vector_b = default_surface_color();
      preset.scalar_a = 0.0;
      preset.scalar_b = 1.5;
      break;
    case BSDF_PRESET_EMISSION:
      preset.vector_a = Vector3D(1.0, 1.0, 1.0);
      break;
    case BSDF_PRESET_APPROXIMATE_BSSRDF:
      preset.vector_a = Vector3D(0.8, 0.6, 0.6);
      preset.scalar_a = 0.3;
      break;
    case BSDF_PRESET_RANDOM_WALK_SSS:
      preset.vector_a = Vector3D(0.013, 0.070, 0.145);
      preset.vector_b = Vector3D(1.09, 1.59, 1.79);
      preset.vector_c = Vector3D(1.0);
      preset.scalar_a = 0.0;
      preset.scalar_b = 1.3;
      preset.scalar_c = 5.0;
      preset.scalar_d = 0.55;
      preset.scalar_e = 0.3;
      preset.scalar_f = 1.0;
      break;
    case BSDF_PRESET_RANDOM_WALK_LAYERED:
      preset.vector_a = Vector3D(0.013, 0.070, 0.145);
      preset.vector_b = Vector3D(1.09, 1.59, 1.79);
      preset.vector_c = Vector3D(0.8, 0.2, 0.2);
      preset.scalar_a = 0.0;
      preset.scalar_b = 1.3;
      preset.scalar_c = 5.0;
      preset.scalar_d = 0.15;
      preset.scalar_e = 0.5;
      preset.scalar_f = 1.0;
      break;
    case BSDF_PRESET_LAYERED:
    case BSDF_PRESET_FAST_LAYERED:
    case BSDF_PRESET_DISNEY_LAYERED:
      preset.vector_a = Vector3D(0.8, 0.2, 0.2);
      preset.vector_b = preset.vector_a;
      preset.scalar_a = 0.15;
      preset.scalar_b = 0.5;
      preset.scalar_c = 1.0;
      preset.scalar_d = 1.5;
      preset.scalar_e = 0.3;
      preset.scalar_f = 0.0;
      break;
    default:
      break;
  }

  return preset;
}

BSDFPreset convert_bsdf_preset_type(const BSDFPreset& source,
                                    BSDFPresetType target_type) {
  BSDFPreset converted = make_default_bsdf_preset(target_type);
  converted.material_id = source.material_id;
  converted.material_name = source.material_name;

  Vector3D surface_color = infer_surface_color(source);
  Vector3D subsurface_color = infer_subsurface_color(source);
  double roughness = infer_roughness(source);
  double subsurface_roughness = infer_subsurface_roughness(source);
  double ior = infer_ior(source);
  double thickness = infer_thickness(source);
  double saturation = infer_saturation(source);

  switch (target_type) {
    case BSDF_PRESET_DIFFUSE:
    case BSDF_PRESET_MIRROR:
    case BSDF_PRESET_EMISSION:
      converted.vector_a = surface_color;
      break;
    case BSDF_PRESET_MICROFACET:
      converted.scalar_a = roughness;
      break;
    case BSDF_PRESET_REFRACTION:
      converted.vector_a = surface_color;
      converted.scalar_a = roughness;
      converted.scalar_b = ior;
      break;
    case BSDF_PRESET_GLASS:
      converted.vector_a = surface_color;
      converted.vector_b = surface_color;
      converted.scalar_a = roughness;
      converted.scalar_b = ior;
      break;
    case BSDF_PRESET_APPROXIMATE_BSSRDF:
      converted.vector_a = subsurface_color;
      converted.scalar_a = subsurface_roughness;
      break;
    case BSDF_PRESET_RANDOM_WALK_SSS:
      // Keep the physically meaningful PBRT skin coefficients instead of
      // deriving medium parameters from the previous material's display color.
      converted.vector_c = surface_color;
      converted.scalar_d = roughness;
      converted.scalar_e = thickness;
      converted.scalar_f = saturation;
      if (preset_carries_ior(source)) {
        converted.scalar_b = ior;
      }
      if (source.type == BSDF_PRESET_RANDOM_WALK_SSS ||
          source.type == BSDF_PRESET_RANDOM_WALK_LAYERED) {
        converted.vector_a = source.vector_a;
        converted.vector_b = source.vector_b;
        converted.scalar_a = source.scalar_a;
        converted.scalar_c = source.scalar_c;
        converted.scalar_d = source.scalar_d;
        converted.scalar_e = source.scalar_e;
        converted.vector_c = !is_zero_vector(source.vector_c)
            ? source.vector_c
            : surface_color;
        converted.scalar_f = source.scalar_f;
      }
      break;
    case BSDF_PRESET_RANDOM_WALK_LAYERED:
      // Keep the physically meaningful PBRT skin coefficients instead of
      // deriving medium parameters from the previous material's display color.
      converted.vector_c = surface_color;
      converted.scalar_d = roughness;
      converted.scalar_e = thickness;
      converted.scalar_f = saturation;
      if (preset_carries_ior(source)) {
        converted.scalar_b = ior;
      }
      if (source.type == BSDF_PRESET_RANDOM_WALK_SSS ||
          source.type == BSDF_PRESET_RANDOM_WALK_LAYERED) {
        converted.vector_a = source.vector_a;
        converted.vector_b = source.vector_b;
        converted.scalar_a = source.scalar_a;
        converted.scalar_c = source.scalar_c;
        converted.scalar_d = source.scalar_d;
        converted.scalar_e = source.scalar_e;
        converted.vector_c = source.type == BSDF_PRESET_RANDOM_WALK_LAYERED &&
                             !is_zero_vector(source.vector_c)
            ? source.vector_c
            : surface_color;
        converted.scalar_f = source.type == BSDF_PRESET_RANDOM_WALK_LAYERED
            ? source.scalar_f
            : saturation;
      } else if (source.type == BSDF_PRESET_LAYERED ||
                 source.type == BSDF_PRESET_FAST_LAYERED ||
                 source.type == BSDF_PRESET_DISNEY_LAYERED) {
        converted.scalar_d = roughness;
        converted.scalar_e = thickness;
      }
      break;
    case BSDF_PRESET_LAYERED:
    case BSDF_PRESET_FAST_LAYERED:
    case BSDF_PRESET_DISNEY_LAYERED:
      converted.vector_a = surface_color;
      converted.vector_b = subsurface_color;
      converted.scalar_a = roughness;
      converted.scalar_b = thickness;
      converted.scalar_c = saturation;
      converted.scalar_d = ior;
      converted.scalar_e = subsurface_roughness;
      if (source.type == BSDF_PRESET_LAYERED ||
          source.type == BSDF_PRESET_FAST_LAYERED ||
          source.type == BSDF_PRESET_DISNEY_LAYERED) {
        converted.scalar_f = source.scalar_f;
      }
      break;
    default:
      break;
  }

  return converted;
}

int editable_bsdf_preset_type_index(BSDFPresetType type) {
  const int count = sizeof(kEditableBSDFPresetTypes) / sizeof(kEditableBSDFPresetTypes[0]);
  for (int i = 0; i < count; i++) {
    if (kEditableBSDFPresetTypes[i] == type) return i;
  }
  return 0;
}

} // namespace

/**
 * This function creates a object space (basis vectors) from the normal vector
 */
void make_coord_space(Matrix3x3 &o2w, const Vector3D n) {

  Vector3D z = Vector3D(n.x, n.y, n.z);
  Vector3D h = z;
  if (fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z))
    h.x = 1.0;
  else if (fabs(h.y) <= fabs(h.x) && fabs(h.y) <= fabs(h.z))
    h.y = 1.0;
  else
    h.z = 1.0;

  z.normalize();
  Vector3D y = cross(h, z);
  y.normalize();
  Vector3D x = cross(z, y);
  x.normalize();

  o2w[0] = x;
  o2w[1] = y;
  o2w[2] = z;
}

BSDFPreset BSDF::get_preset() const {
  return BSDFPreset();
}

void BSDF::apply_preset(const BSDFPreset& preset) {
}

const char* bsdf_preset_type_name(BSDFPresetType type) {
  switch (type) {
    case BSDF_PRESET_DIFFUSE: return "Diffuse";
    case BSDF_PRESET_MICROFACET: return "Microfacet";
    case BSDF_PRESET_MIRROR: return "Mirror";
    case BSDF_PRESET_REFRACTION: return "Refraction";
    case BSDF_PRESET_GLASS: return "Glass";
    case BSDF_PRESET_EMISSION: return "Emission";
    case BSDF_PRESET_APPROXIMATE_BSSRDF: return "Approximate BSSRDF";
    case BSDF_PRESET_RANDOM_WALK_SSS: return "Random Walk SSS";
    case BSDF_PRESET_RANDOM_WALK_LAYERED: return "Random Walk Layered";
    case BSDF_PRESET_LAYERED: return "Layered";
    case BSDF_PRESET_FAST_LAYERED: return "Fast Layered";
    case BSDF_PRESET_DISNEY_LAYERED: return "Disney Layered";
    default: return "Unsupported";
  }
}

bool render_bsdf_preset_controls(BSDFPreset& preset) {
  bool changed = false;

  int current_type_index = editable_bsdf_preset_type_index(preset.type);
  if (ImGui::BeginCombo("Material Type",
                        bsdf_preset_type_name(kEditableBSDFPresetTypes[current_type_index]))) {
    const int count = sizeof(kEditableBSDFPresetTypes) / sizeof(kEditableBSDFPresetTypes[0]);
    for (int i = 0; i < count; i++) {
      const BSDFPresetType type = kEditableBSDFPresetTypes[i];
      const bool selected = type == preset.type;
      if (ImGui::Selectable(bsdf_preset_type_name(type), selected)) {
        if (type != preset.type) {
          preset = convert_bsdf_preset_type(preset, type);
          changed = true;
        }
      }
      if (selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }

  switch (preset.type) {
    case BSDF_PRESET_DIFFUSE:
    case BSDF_PRESET_MIRROR:
      changed |= DragDouble3("Reflectance", &preset.vector_a[0], 0.005f);
      break;
    case BSDF_PRESET_EMISSION:
      changed |= DragDouble3("Radiance", &preset.vector_a[0], 0.005f);
      break;
    case BSDF_PRESET_MICROFACET:
      changed |= DragDouble3("Eta", &preset.vector_a[0], 0.005f);
      changed |= DragDouble3("K", &preset.vector_b[0], 0.005f);
      changed |= DragDouble("Alpha", &preset.scalar_a, 0.005f);
      break;
    case BSDF_PRESET_REFRACTION:
      changed |= DragDouble3("Transmittance", &preset.vector_a[0], 0.005f);
      changed |= DragDouble("Roughness", &preset.scalar_a, 0.005f);
      changed |= DragDouble("IOR", &preset.scalar_b, 0.005f);
      break;
    case BSDF_PRESET_GLASS:
      changed |= DragDouble3("Transmittance", &preset.vector_a[0], 0.005f);
      changed |= DragDouble3("Reflectance", &preset.vector_b[0], 0.005f);
      changed |= DragDouble("Roughness", &preset.scalar_a, 0.005f);
      changed |= DragDouble("IOR", &preset.scalar_b, 0.005f);
      break;
    case BSDF_PRESET_APPROXIMATE_BSSRDF:
      changed |= DragDouble3("Skin Color", &preset.vector_a[0], 0.005f);
      changed |= DragDouble("Roughness", &preset.scalar_a, 0.005f);
      break;
    case BSDF_PRESET_RANDOM_WALK_SSS:
      ImGui::Text("Medium Controls");
      changed |= DragDouble3("Sigma A", &preset.vector_a[0], 0.005f);
      changed |= DragDouble3("Sigma S", &preset.vector_b[0], 0.005f);
      changed |= DragDouble("Anisotropy g", &preset.scalar_a, 0.005f);
      changed |= DragDouble("IOR", &preset.scalar_b, 0.005f);
      changed |= DragDouble("Scale", &preset.scalar_c, 0.005f);
      changed |= DragDouble3("Subsurface Tint", &preset.vector_c[0], 0.005f);
      changed |= DragDouble("Saturation", &preset.scalar_f, 0.005f);
      ImGui::Spacing();
      ImGui::Text("Boundary Controls");
      changed |= DragDouble("Surface Roughness", &preset.scalar_d, 0.005f);
      changed |= DragDouble("Specular Weight", &preset.scalar_e, 0.005f);
      break;
    case BSDF_PRESET_RANDOM_WALK_LAYERED:
      ImGui::Text("Layer Controls");
      changed |= DragDouble("Gloss Roughness", &preset.scalar_d, 0.005f);
      changed |= DragDouble("Thickness", &preset.scalar_e, 0.005f);
      changed |= DragDouble3("Base Color", &preset.vector_c[0], 0.005f);
      changed |= DragDouble("Saturation", &preset.scalar_f, 0.005f);
      changed |= DragDouble("IOR", &preset.scalar_b, 0.005f);
      ImGui::Spacing();
      ImGui::Text("Random-Walk BSSRDF Controls");
      changed |= DragDouble3("Sigma A", &preset.vector_a[0], 0.005f);
      changed |= DragDouble3("Sigma S", &preset.vector_b[0], 0.005f);
      changed |= DragDouble("Anisotropy g", &preset.scalar_a, 0.005f);
      changed |= DragDouble("Scale", &preset.scalar_c, 0.005f);
      break;
    case BSDF_PRESET_LAYERED:
    case BSDF_PRESET_FAST_LAYERED:
    case BSDF_PRESET_DISNEY_LAYERED:
      ImGui::Text("Layer Controls");
      changed |= DragDouble("Gloss Roughness", &preset.scalar_a, 0.005f);
      changed |= DragDouble("Thickness", &preset.scalar_b, 0.005f);
      changed |= DragDouble3("Base Color", &preset.vector_a[0], 0.005f);
      changed |= DragDouble("Saturation", &preset.scalar_c, 0.005f);
      changed |= DragDouble("IOR", &preset.scalar_d, 0.005f);
      changed |= DragDouble("Pooling Strength", &preset.scalar_f, 0.005f);
      ImGui::Spacing();
      ImGui::Text("Base BSSRDF Controls");
      changed |= DragDouble3("Skin Color", &preset.vector_b[0], 0.005f);
      changed |= DragDouble("Base Roughness", &preset.scalar_e, 0.005f);
      break;
    default:
      ImGui::TextDisabled("This BSDF type is not editable here.");
      break;
  }

  return changed;
}

BSDF* create_bsdf_from_preset(const BSDFPreset& preset) {
  switch (preset.type) {
    case BSDF_PRESET_DIFFUSE:
      return new DiffuseBSDF(preset.vector_a);
    case BSDF_PRESET_MICROFACET:
      return new MicrofacetBSDF(preset.vector_a, preset.vector_b, preset.scalar_a);
    case BSDF_PRESET_MIRROR:
      return new MirrorBSDF(preset.vector_a);
    case BSDF_PRESET_REFRACTION:
      return new RefractionBSDF(preset.vector_a, preset.scalar_a, preset.scalar_b);
    case BSDF_PRESET_GLASS:
      return new GlassBSDF(preset.vector_a, preset.vector_b, preset.scalar_a,
                           preset.scalar_b);
    case BSDF_PRESET_EMISSION:
      return new EmissionBSDF(preset.vector_a);
    case BSDF_PRESET_APPROXIMATE_BSSRDF:
      return new ApproximateBSSRDF(preset.vector_a, preset.scalar_a);
    case BSDF_PRESET_RANDOM_WALK_SSS:
      return new RandomWalkSSSBSDF(preset.vector_a, preset.vector_b,
                                   preset.scalar_a, preset.scalar_b,
                                   preset.scalar_c, preset.scalar_d,
                                   preset.scalar_e, preset.type,
                                   preset.vector_c, preset.scalar_f);
    case BSDF_PRESET_RANDOM_WALK_LAYERED:
      return new RandomWalkSSSBSDF(preset.vector_a, preset.vector_b,
                                   preset.scalar_a, preset.scalar_b,
                                   preset.scalar_c, preset.scalar_d,
                                   preset.scalar_e, preset.type,
                                   preset.vector_c, preset.scalar_f);
    case BSDF_PRESET_LAYERED: {
      LayeredBSDF* bsdf = new LayeredBSDF(preset.scalar_a, preset.scalar_b,
                                          preset.vector_a, preset.scalar_c,
                                          preset.scalar_d, preset.scalar_f);
      bsdf->apply_preset(preset);
      return bsdf;
    }
    case BSDF_PRESET_FAST_LAYERED: {
      FastLayeredBSDF* bsdf = new FastLayeredBSDF(preset.scalar_a, preset.scalar_b,
                                                  preset.vector_a, preset.scalar_c,
                                                  preset.scalar_d, preset.scalar_f);
      bsdf->apply_preset(preset);
      return bsdf;
    }
    case BSDF_PRESET_DISNEY_LAYERED: {
      DisneyLayeredBSDF* bsdf =
          new DisneyLayeredBSDF(preset.scalar_a, preset.scalar_b, preset.vector_a,
                                preset.scalar_c, preset.scalar_d, preset.scalar_f);
      bsdf->apply_preset(preset);
      return bsdf;
    }
    default:
      return nullptr;
  }
}

/**
 * Evaluate diffuse lambertian BSDF.
 * Given incident light direction wi and outgoing light direction wo. Note
 * that both wi and wo are defined in the local coordinate system at the
 * point of intersection.
 * \param wo outgoing light direction in local space of point of intersection
 * \param wi incident light direction in local space of point of intersection
 * \return reflectance in the given incident/outgoing directions
 */
Vector3D DiffuseBSDF::f(const Vector3D wo, const Vector3D wi) {
  // (Part 3.1):
  // This function takes in both wo and wi and returns the evaluation of
  // the BSDF for those two directions.

  return reflectance / PI;

}

/**
 * Evalutate diffuse lambertian BSDF.
 */
Vector3D DiffuseBSDF::sample_f(const Vector3D wo, Vector3D *wi, double *pdf) {
  // Part 4.1
  // This function takes in only wo and provides pointers for wi and pdf,
  // which should be assigned by this function.
  // After sampling a value for wi, it returns the evaluation of the BSDF
  // at (wo, *wi).
  // You can use the `f` function. The reference solution only takes two lines.

  *wi = sampler.get_sample(pdf);
  return f(wo, *wi);

}

void DiffuseBSDF::render_debugger_node()
{
  if (ImGui::TreeNode(this, "Diffuse BSDF"))
  {
    DragDouble3("Reflectance", &reflectance[0], 0.005);
    ImGui::TreePop();
  }
}

BSDFPreset DiffuseBSDF::get_preset() const {
  BSDFPreset preset;
  preset.type = BSDF_PRESET_DIFFUSE;
  preset.vector_a = reflectance;
  return preset;
}

void DiffuseBSDF::apply_preset(const BSDFPreset& preset) {
  if (preset.type != BSDF_PRESET_DIFFUSE) return;
  reflectance = preset.vector_a;
}

/**
 * Evalutate Emission BSDF (Light Source)
 */
Vector3D EmissionBSDF::f(const Vector3D wo, const Vector3D wi) {
  return Vector3D();
}

/**
 * Evalutate Emission BSDF (Light Source)
 */
Vector3D EmissionBSDF::sample_f(const Vector3D wo, Vector3D *wi, double *pdf) {
  *pdf = 1.0 / PI;
  *wi = sampler.get_sample(pdf);
  return Vector3D();
}

void EmissionBSDF::render_debugger_node()
{
  if (ImGui::TreeNode(this, "Emission BSDF"))
  {
    DragDouble3("Radiance", &radiance[0], 0.005);
    ImGui::TreePop();
  }
}

BSDFPreset EmissionBSDF::get_preset() const {
  BSDFPreset preset;
  preset.type = BSDF_PRESET_EMISSION;
  preset.vector_a = radiance;
  return preset;
}

void EmissionBSDF::apply_preset(const BSDFPreset& preset) {
  if (preset.type != BSDF_PRESET_EMISSION) return;
  radiance = preset.vector_a;
}

/**
 * Evaluate Approximate BSSRDF.
 * Uses a diffuse-like model with color to approximate subsurface scattering.
 */
Vector3D ApproximateBSSRDF::f(const Vector3D wo, const Vector3D wi) {
  // Treat this as a one-sided diffuse-like lobe. Letting it scatter when
  // either direction is below the surface causes bright leaks and noisy
  // artifacts around thin or open mesh boundaries.
  if (wo.z <= 0.0 || wi.z <= 0.0) return Vector3D(0, 0, 0);

  // Keep the slight angular tint from the older implementation, but put it in
  // f() itself so direct and recursive paths evaluate the same material.
  double cosine_factor = cos_theta(wi);
  Vector3D modulated_color = skin_color * (0.8 + 0.2 * cosine_factor);
  return modulated_color / PI;
}

/**
 * Sample Approximate BSSRDF.
 * Uses cosine-weighted hemisphere sampling.
 */
Vector3D ApproximateBSSRDF::sample_f(const Vector3D wo, Vector3D *wi, double *pdf) {
  if (wo.z <= 0.0) {
    *pdf = 0.0;
    return Vector3D(0, 0, 0);
  }

  // Sample using cosine-weighted hemisphere distribution
  *wi = sampler.get_sample(pdf);

  // Return the same diffuse-like BSSRDF that f() evaluates.
  return f(wo, *wi);
}

void ApproximateBSSRDF::render_debugger_node()
{
  if (ImGui::TreeNode(this, "Approximate BSSRDF"))
  {
    DragDouble3("Skin Color", &skin_color[0], 0.005);
    DragDouble("Roughness", &roughness, 0.005);
    ImGui::TreePop();
  }
}

BSDFPreset ApproximateBSSRDF::get_preset() const {
  BSDFPreset preset;
  preset.type = BSDF_PRESET_APPROXIMATE_BSSRDF;
  preset.vector_a = skin_color;
  preset.scalar_a = roughness;
  return preset;
}

void ApproximateBSSRDF::apply_preset(const BSDFPreset& preset) {
  if (preset.type != BSDF_PRESET_APPROXIMATE_BSSRDF) return;
  skin_color = preset.vector_a;
  roughness = preset.scalar_a;
}

Vector3D RandomWalkSSSBSDF::diffuse_fallback_color() const {
  Vector3D base = sigma_albedo(sigma_a, sigma_s);
  if (!is_zero_vector(base_color)) {
    base = base * saturated_base_tint(base_color, saturation);
  }
  return clamp_color01(base);
}

Vector3D RandomWalkSSSBSDF::f(const Vector3D wo, const Vector3D wi) {
  if (wo.z <= 0.0 || wi.z <= 0.0) return Vector3D(0, 0, 0);
  return diffuse_fallback_color() / PI;
}

Vector3D RandomWalkSSSBSDF::sample_f(const Vector3D wo, Vector3D *wi, double *pdf) {
  if (wo.z <= 0.0) {
    *pdf = 0.0;
    return Vector3D(0, 0, 0);
  }

  *wi = sampler.get_sample(pdf);
  return f(wo, *wi);
}

void RandomWalkSSSBSDF::render_debugger_node()
{
  const char* node_name = preset_type == BSDF_PRESET_RANDOM_WALK_LAYERED
      ? "Random Walk Layered BSDF"
      : "Random Walk SSS BSDF";
  if (ImGui::TreeNode(this, "%s", node_name))
  {
    if (preset_type == BSDF_PRESET_RANDOM_WALK_LAYERED) {
      ImGui::Text("Layer Controls");
      DragDouble("Gloss Roughness", &surface_roughness, 0.005);
      DragDouble("Thickness", &specular_weight, 0.005);
      DragDouble3("Base Color", &base_color[0], 0.005);
      DragDouble("Saturation", &saturation, 0.005);
      DragDouble("IOR", &ior, 0.005);
      ImGui::Spacing();
      ImGui::Text("Random-Walk BSSRDF Controls");
    }
    DragDouble3("Sigma A", &sigma_a[0], 0.005);
    DragDouble3("Sigma S", &sigma_s[0], 0.005);
    DragDouble("Anisotropy g", &anisotropy_g, 0.005);
    DragDouble("Scale", &scale, 0.005);
    if (preset_type != BSDF_PRESET_RANDOM_WALK_LAYERED) {
      DragDouble3("Subsurface Tint", &base_color[0], 0.005);
      DragDouble("Saturation", &saturation, 0.005);
      DragDouble("IOR", &ior, 0.005);
      DragDouble("Surface Roughness", &surface_roughness, 0.005);
      DragDouble("Specular Weight", &specular_weight, 0.005);
    }
    sigma_a = clamp_vector_min(sigma_a, 0.0);
    sigma_s = clamp_vector_min(sigma_s, 0.0);
    base_color = clamp_color01(base_color);
    anisotropy_g = clamp(anisotropy_g, -0.95, 0.95);
    ior = max(ior, 1.0001);
    scale = max(scale, 1e-6);
    surface_roughness = clamp(surface_roughness, 0.02, 1.0);
    specular_weight = clamp(specular_weight, 0.0, 1.0);
    saturation = max(saturation, 0.0);
    ImGui::TreePop();
  }
}

BSDFPreset RandomWalkSSSBSDF::get_preset() const {
  BSDFPreset preset;
  preset.type = preset_type;
  preset.vector_a = sigma_a;
  preset.vector_b = sigma_s;
  preset.scalar_a = anisotropy_g;
  preset.scalar_b = ior;
  preset.scalar_c = scale;
  preset.scalar_d = surface_roughness;
  preset.scalar_e = specular_weight;
  preset.vector_c = base_color;
  preset.scalar_f = saturation;
  return preset;
}

void RandomWalkSSSBSDF::apply_preset(const BSDFPreset& preset) {
  if (preset.type != BSDF_PRESET_RANDOM_WALK_SSS &&
      preset.type != BSDF_PRESET_RANDOM_WALK_LAYERED) {
    return;
  }
  preset_type = preset.type;
  sigma_a = clamp_vector_min(preset.vector_a, 0.0);
  sigma_s = clamp_vector_min(preset.vector_b, 0.0);
  anisotropy_g = clamp(preset.scalar_a, -0.95, 0.95);
  ior = max(preset.scalar_b, 1.0001);
  scale = max(preset.scalar_c, 1e-6);
  surface_roughness = clamp(preset.scalar_d, 0.02, 1.0);
  specular_weight = clamp(preset.scalar_e, 0.0, 1.0);
  base_color = !is_zero_vector(preset.vector_c)
      ? clamp_color01(preset.vector_c)
      : (preset_type == BSDF_PRESET_RANDOM_WALK_LAYERED
          ? Vector3D(0.8, 0.2, 0.2)
          : Vector3D(1.0));
  saturation = max(preset.scalar_f, 0.0);
}

// Uncomment this version for iteration 2
/**
 * Evaluate Layered BSDF.
 * Blends between base (diffuse) and gloss (specular) contributions based on thickness.
 */
/*
Vector3D LayeredBSDF::f(const Vector3D wo, const Vector3D wi) {
  // Get base layer contribution
  Vector3D base_contrib = base_layer->f(wo, wi);
  
  // Apply saturation to base color (clamp to preserve energy)
  Vector3D saturated_base = base_contrib * clamp(saturation, 0.0, 1.5);
  
  // Calculate Fresnel effect for gloss layer
  // Fresnel reflection increases at grazing angles
  double cos_i = abs_cos_theta(wi);
  double fresnel = 0.04 + (1.0 - 0.04) * pow(1.0 - cos_i, 5.0);
  
  // Compute microfacet distribution: compare wi to perfect reflection direction
  Vector3D reflected_dir;
  reflect(wo, &reflected_dir);
  
  // Measure alignment with reflection direction
  double alignment = dot(wi, reflected_dir);
  alignment = clamp(alignment, 0.0, 1.0);
  
  // Roughness controls the width of the specular lobe
  double alpha = roughness * roughness;
  double exponent = max(1.0, 2.0 / (alpha * alpha + 1e-5) - 2.0);
  double spec_distribution = pow(alignment, exponent);
  
  // Simplified geometry term: normalize by cosine factors
  double geometry = 1.0 / max(abs_cos_theta(wo), 0.1);
  
  // Combine Fresnel, distribution, and geometry
  double gloss_value = fresnel * spec_distribution * geometry;
  
  Vector3D gloss = Vector3D(gloss_value, gloss_value, gloss_value);
  
  // Blend based on thickness: 0 = all base, 1 = all gloss
  return saturated_base * (1.0 - thickness) + gloss * thickness;
}
*/

/**
 * Evaluate Layered BSDF.
 * Blends between base (diffuse) and gloss (Cook-Torrance specular) contributions.
 */
Vector3D LayeredBSDF::f(const Vector3D wo, const Vector3D wi) {
  // If either raywis below the surface, no light is reflected
  if (wo.z <= 0.0 || wi.z <= 0.0) return Vector3D(0, 0, 0);

  // --- Base Layer Evaluation ---
  Vector3D base_contrib = base_layer->f(wo, wi);
  Vector3D saturated_base = base_contrib * clamp(saturation, 0.0, 1.5);

  // --- Gloss Layer (Cook-Torrance Microfacet) Evaluation ---
  
  // Calculate the half-vector (h)
  Vector3D h = wo + wi;
  if (h.norm2() == 0.0) return Vector3D(0, 0, 0);
  h.normalize();

  // Clamp roughness to avoid division by zero artifacts
  double alpha = max(roughness * roughness, 0.001);
  double alpha2 = alpha * alpha;

  // 1. Fresnel (Schlick's Approximation using IOR)
  double actual_ior = max(ior, 1.0001); // Prevent div by 0
  double R0 = pow((1.0 - actual_ior) / (1.0 + actual_ior), 2.0);
  double cos_theta_d = max(dot(wi, h), 0.0);
  double F = R0 + (1.0 - R0) * pow(1.0 - cos_theta_d, 5.0);

  // 2. Normal Distribution Function (D) - GGX
  double cos_theta_h = max(h.z, 0.0);
  double cos2_theta_h = cos_theta_h * cos_theta_h;
  double D_denom = PI * pow(cos2_theta_h * (alpha2 - 1.0) + 1.0, 2.0);
  double D = (D_denom > 0.0) ? (alpha2 / D_denom) : 0.0;

  // 3. Geometry Term (G) - Smith's method for GGX
  auto G1 = [](double cos_theta, double a2) {
    double cos2_theta = cos_theta * cos_theta;
    return (2.0 * cos_theta) / (cos_theta + sqrt(a2 + (1.0 - a2) * cos2_theta));
  };
  double G = G1(wo.z, alpha2) * G1(wi.z, alpha2);

  // Combine Cook-Torrance components
  double cos_theta_i = wi.z;
  double cos_theta_o = wo.z;
  double ct_val = (D * F * G) / (4.0 * cos_theta_i * cos_theta_o);
  
  // Clamp specular values to prevent fireflies while preserving energy conservation
  ct_val = min(ct_val, 1.0);
  
  Vector3D gloss(ct_val, ct_val, ct_val);
  Vector3D gloss_tinted = gloss + (base_color * 0.15) * ct_val;

  // Blend based on thickness parameter
  return saturated_base * (1.0 - thickness) + gloss_tinted * thickness;
}

Vector3D LayeredBSDF::f(const Vector3D wo, const Vector3D wi,
                        const Vector2D uv) {
  if (pooling_strength <= 0.0) return f(wo, wi);
  if (wo.z <= 0.0 || wi.z <= 0.0) return Vector3D(0, 0, 0);

  double local_roughness, local_thickness, local_saturation;
  pooled_layer_params(uv, pooling_strength, roughness, thickness, saturation,
                      &local_roughness, &local_thickness, &local_saturation);

  Vector3D base_contrib = base_layer->f(wo, wi);
  Vector3D saturated_base = base_contrib * local_saturation;

  Vector3D h = wo + wi;
  if (h.norm2() == 0.0) return Vector3D(0, 0, 0);
  h.normalize();

  double alpha = max(local_roughness * local_roughness, 0.001);
  double alpha2 = alpha * alpha;

  double actual_ior = max(ior, 1.0001);
  double R0 = pow((1.0 - actual_ior) / (1.0 + actual_ior), 2.0);
  double cos_theta_d = max(dot(wi, h), 0.0);
  double F = R0 + (1.0 - R0) * pow(1.0 - cos_theta_d, 5.0);

  double cos_theta_h = max(h.z, 0.0);
  double cos2_theta_h = cos_theta_h * cos_theta_h;
  double D_denom = PI * pow(cos2_theta_h * (alpha2 - 1.0) + 1.0, 2.0);
  double D = (D_denom > 0.0) ? (alpha2 / D_denom) : 0.0;

  auto G1 = [](double cos_theta, double a2) {
    double cos2_theta = cos_theta * cos_theta;
    return (2.0 * cos_theta) / (cos_theta + sqrt(a2 + (1.0 - a2) * cos2_theta));
  };
  double G = G1(wo.z, alpha2) * G1(wi.z, alpha2);

  double ct_val = (D * F * G) / (4.0 * wi.z * wo.z);
  ct_val = min(ct_val, 1.0);

  Vector3D gloss(ct_val, ct_val, ct_val);
  Vector3D gloss_tinted = gloss + (base_color * 0.15) * ct_val;
  return saturated_base * (1.0 - local_thickness) +
         gloss_tinted * local_thickness;
}

/**
 * Sample Layered BSDF.
 * Probabilistically samples from base or gloss layer based on thickness.
 */

Vector3D LayeredBSDF::sample_f(const Vector3D wo, Vector3D* wi, double* pdf) {
  if (wo.z <= 0.0) {
    *pdf = 0.0;
    return Vector3D(0, 0, 0);
  }

  // The layered material is a base-plus-gloss mixture. We sample one lobe at
  // a time, but because sample_f returns the full BSDF evaluation, the
  // denominator must be the full mixture pdf.
  double gloss_weight = clamp(thickness, 0.0, 1.0);
  double base_weight = 1.0 - gloss_weight;

  if (random_uniform() < gloss_weight) {
    double sampled_gloss_pdf = 0.0;
    if (!sample_layered_gloss_lobe(wo, roughness, wi, &sampled_gloss_pdf)) {
      *pdf = 0.0;
      return Vector3D(0, 0, 0);
    }
  } else {
    double base_pdf;
    base_layer->sample_f(wo, wi, &base_pdf);
    if (base_pdf <= 0.0 || wi->z <= 0.0) {
      *pdf = 0.0;
      return Vector3D(0, 0, 0);
    }
  }

  double base_pdf = cosine_hemisphere_pdf(*wi);
  double gloss_pdf = layered_gloss_pdf(wo, *wi, roughness);
  *pdf = base_weight * base_pdf + gloss_weight * gloss_pdf;

  if (*pdf <= 1e-10) {
    return Vector3D(0, 0, 0);
  }

  return f(wo, *wi);
}

Vector3D LayeredBSDF::sample_f(const Vector3D wo, Vector3D* wi, double* pdf,
                               const Vector2D uv) {
  if (pooling_strength <= 0.0) return sample_f(wo, wi, pdf);
  if (wo.z <= 0.0) {
    *pdf = 0.0;
    return Vector3D(0, 0, 0);
  }

  double local_roughness, local_thickness, local_saturation;
  pooled_layer_params(uv, pooling_strength, roughness, thickness, saturation,
                      &local_roughness, &local_thickness, &local_saturation);
  (void)local_saturation;
  double gloss_weight = clamp(local_thickness, 0.0, 1.0);
  double base_weight = 1.0 - gloss_weight;

  if (random_uniform() < gloss_weight) {
    double sampled_gloss_pdf = 0.0;
    if (!sample_layered_gloss_lobe(wo, local_roughness, wi, &sampled_gloss_pdf)) {
      *pdf = 0.0;
      return Vector3D(0, 0, 0);
    }
  } else {
    double base_pdf;
    base_layer->sample_f(wo, wi, &base_pdf);
    if (base_pdf <= 0.0 || wi->z <= 0.0) {
      *pdf = 0.0;
      return Vector3D(0, 0, 0);
    }
  }

  double base_pdf = cosine_hemisphere_pdf(*wi);
  double gloss_pdf = layered_gloss_pdf(wo, *wi, local_roughness);
  *pdf = base_weight * base_pdf + gloss_weight * gloss_pdf;
  if (*pdf <= 1e-10) return Vector3D(0, 0, 0);
  return f(wo, *wi, uv);
}

void LayeredBSDF::render_debugger_node()
{
  if (ImGui::TreeNode(this, "Layered BSDF"))
  {
    bool changed = false;
    changed |= DragDouble("Gloss Roughness", &roughness, 0.005);
    changed |= DragDouble("Thickness", &thickness, 0.005);
    changed |= DragDouble3("Base Color", &base_color[0], 0.005);
    changed |= DragDouble("Saturation", &saturation, 0.005);
    changed |= DragDouble("IOR", &ior, 0.005);
    changed |= DragDouble("Pooling Strength", &pooling_strength, 0.005);
    ImGui::Spacing();
    ImGui::Text("Base BSSRDF");
    changed |= DragDouble3("Skin Color", &subsurface_color[0], 0.005);
    changed |= DragDouble("Base Roughness", &subsurface_roughness, 0.005);
    if (changed) {
      sync_base_layer(base_layer, subsurface_color, subsurface_roughness);
    }
    ImGui::TreePop();
  }
}

BSDFPreset LayeredBSDF::get_preset() const {
  BSDFPreset preset;
  preset.type = BSDF_PRESET_LAYERED;
  preset.vector_a = base_color;
  preset.vector_b = subsurface_color;
  preset.scalar_a = roughness;
  preset.scalar_b = thickness;
  preset.scalar_c = saturation;
  preset.scalar_d = ior;
  preset.scalar_e = subsurface_roughness;
  preset.scalar_f = pooling_strength;
  return preset;
}

void LayeredBSDF::apply_preset(const BSDFPreset& preset) {
  if (preset.type != BSDF_PRESET_LAYERED) return;
  roughness = preset.scalar_a;
  thickness = preset.scalar_b;
  base_color = preset.vector_a;
  saturation = preset.scalar_c;
  ior = preset.scalar_d;
  subsurface_color = preset.vector_b;
  subsurface_roughness = preset.scalar_e;
  pooling_strength = clamp(preset.scalar_f, 0.0, 1.0);
  sync_base_layer(base_layer, subsurface_color, subsurface_roughness);
}

/**
 * Fast Layered BSDF.
 * Blends between base (diffuse) and gloss using Normalized Blinn-Phong 
 * with the highly performant Kelemen geometry approximation.
 */
Vector3D FastLayeredBSDF::f(const Vector3D wo, const Vector3D wi) {
  // if either ray is below the surface, no light is reflected
  if (wo.z <= 0.0 || wi.z <= 0.0) return Vector3D(0, 0, 0);

  // base layer
  Vector3D base_contrib = base_layer->f(wo, wi);
  Vector3D saturated_base = base_contrib * clamp(saturation, 0.0, 1.5);

  // gloss layer (Blinn-Phong / Kelemen)
  
  // calculate half-vector (h)
  Vector3D h = wo + wi;
  if (h.norm2() == 0.0) return Vector3D(0, 0, 0);
  h.normalize();

  // convert roughness to Blinn-Phong exponent (shininess)
  // roughness^4 perceptually matches the GGX roughness scale 
  double r4 = max(pow(roughness, 4.0), 0.0001);
  double exponent = (2.0 / r4) - 2.0;
  exponent = max(exponent, 0.0); // no negatives

  // Fresnel (Schlick's Approximation using IOR)
  double actual_ior = max(ior, 1.0001);
  double R0 = pow((1.0 - actual_ior) / (1.0 + actual_ior), 2.0);
  double cos_theta_d = max(dot(wi, h), 0.0);
  double F = R0 + (1.0 - R0) * pow(1.0 - cos_theta_d, 5.0);

  // Normal Distribution Function (D) - Normalized Blinn-Phong
  double cos_theta_h = max(h.z, 0.0); // h.z is dot(n, h) in local space
  double D = ((exponent + 2.0) / (2.0 * PI)) * pow(cos_theta_h, exponent);

  // The Kelemen Approximation
  // standard microfacet model = (D * F * G) / (4 * wi.z * wo.z).
  // Kelemen approximates G / (4 * wi.z * wo.z) as 1 / (4 * dot(wo, h)^2).
  // much faster than Smith G1/G2 functions.
  double cos_wo_h = max(dot(wo, h), 0.001); // no division by zero
  double ct_val = (D * F) / (4.0 * cos_wo_h * cos_wo_h);
  
  // clamp specular values to prevent fireflies
  ct_val = min(ct_val, 1.0);
  
  Vector3D gloss(ct_val, ct_val, ct_val);
  Vector3D gloss_tinted = gloss + (base_color * 0.15) * ct_val;

  // blend based on thickness parameter
  return saturated_base * (1.0 - thickness) + gloss_tinted * thickness;
}

Vector3D FastLayeredBSDF::f(const Vector3D wo, const Vector3D wi,
                            const Vector2D uv) {
  if (pooling_strength <= 0.0) return f(wo, wi);
  if (wo.z <= 0.0 || wi.z <= 0.0) return Vector3D(0, 0, 0);

  double local_roughness, local_thickness, local_saturation;
  pooled_layer_params(uv, pooling_strength, roughness, thickness, saturation,
                      &local_roughness, &local_thickness, &local_saturation);

  Vector3D base_contrib = base_layer->f(wo, wi);
  Vector3D saturated_base = base_contrib * local_saturation;

  Vector3D h = wo + wi;
  if (h.norm2() == 0.0) return Vector3D(0, 0, 0);
  h.normalize();

  double r4 = max(pow(local_roughness, 4.0), 0.0001);
  double exponent = max((2.0 / r4) - 2.0, 0.0);

  double actual_ior = max(ior, 1.0001);
  double R0 = pow((1.0 - actual_ior) / (1.0 + actual_ior), 2.0);
  double cos_theta_d = max(dot(wi, h), 0.0);
  double F = R0 + (1.0 - R0) * pow(1.0 - cos_theta_d, 5.0);

  double cos_theta_h = max(h.z, 0.0);
  double D = ((exponent + 2.0) / (2.0 * PI)) * pow(cos_theta_h, exponent);
  double cos_wo_h = max(dot(wo, h), 0.001);
  double ct_val = (D * F) / (4.0 * cos_wo_h * cos_wo_h);
  ct_val = min(ct_val, 1.0);

  Vector3D gloss(ct_val, ct_val, ct_val);
  Vector3D gloss_tinted = gloss + (base_color * 0.15) * ct_val;
  return saturated_base * (1.0 - local_thickness) +
         gloss_tinted * local_thickness;
}

/**
 * sample fast layered BSDF.
 */
Vector3D FastLayeredBSDF::sample_f(const Vector3D wo, Vector3D* wi, double* pdf) {
  if (wo.z <= 0.0) {
    *pdf = 0.0;
    return Vector3D(0, 0, 0);
  }

  // The fast layered model is still a two-lobe mixture: base flesh plus a
  // fast gloss approximation. We sample one branch, then report the pdf of
  // the full mixture because sample_f returns the full BSDF value f(wo, wi).
  double gloss_weight = clamp(thickness, 0.0, 1.0);
  double base_weight = 1.0 - gloss_weight;

  if (random_uniform() < gloss_weight) {
    double sampled_gloss_pdf = 0.0;
    if (!sample_fast_layered_gloss_lobe(wo, roughness, wi, &sampled_gloss_pdf)) {
      *pdf = 0.0;
      return Vector3D(0, 0, 0);
    }
  } else {
    double base_pdf;
    base_layer->sample_f(wo, wi, &base_pdf);
    if (base_pdf <= 0.0 || wi->z <= 0.0) {
      *pdf = 0.0;
      return Vector3D(0, 0, 0);
    }
  }

  double base_pdf = cosine_hemisphere_pdf(*wi);
  double gloss_pdf = fast_layered_gloss_pdf(wo, *wi, roughness);
  *pdf = base_weight * base_pdf + gloss_weight * gloss_pdf;

  if (*pdf <= 1e-10) {
    return Vector3D(0, 0, 0);
  }

  return f(wo, *wi);
}

Vector3D FastLayeredBSDF::sample_f(const Vector3D wo, Vector3D* wi, double* pdf,
                                   const Vector2D uv) {
  if (pooling_strength <= 0.0) return sample_f(wo, wi, pdf);
  if (wo.z <= 0.0) {
    *pdf = 0.0;
    return Vector3D(0, 0, 0);
  }

  double local_roughness, local_thickness, local_saturation;
  pooled_layer_params(uv, pooling_strength, roughness, thickness, saturation,
                      &local_roughness, &local_thickness, &local_saturation);
  (void)local_saturation;
  double gloss_weight = clamp(local_thickness, 0.0, 1.0);
  double base_weight = 1.0 - gloss_weight;

  if (random_uniform() < gloss_weight) {
    double sampled_gloss_pdf = 0.0;
    if (!sample_fast_layered_gloss_lobe(wo, local_roughness, wi, &sampled_gloss_pdf)) {
      *pdf = 0.0;
      return Vector3D(0, 0, 0);
    }
  } else {
    double base_pdf;
    base_layer->sample_f(wo, wi, &base_pdf);
    if (base_pdf <= 0.0 || wi->z <= 0.0) {
      *pdf = 0.0;
      return Vector3D(0, 0, 0);
    }
  }

  double base_pdf = cosine_hemisphere_pdf(*wi);
  double gloss_pdf = fast_layered_gloss_pdf(wo, *wi, local_roughness);
  *pdf = base_weight * base_pdf + gloss_weight * gloss_pdf;
  if (*pdf <= 1e-10) return Vector3D(0, 0, 0);
  return f(wo, *wi, uv);
}

void FastLayeredBSDF::render_debugger_node()
{
  if (ImGui::TreeNode(this, "Fast Layered BSDF"))
  {
    bool changed = false;
    changed |= DragDouble("Gloss Roughness", &roughness, 0.005);
    changed |= DragDouble("Thickness", &thickness, 0.005);
    changed |= DragDouble3("Base Color", &base_color[0], 0.005);
    changed |= DragDouble("Saturation", &saturation, 0.005);
    changed |= DragDouble("IOR", &ior, 0.005);
    changed |= DragDouble("Pooling Strength", &pooling_strength, 0.005);
    ImGui::Spacing();
    ImGui::Text("Base BSSRDF");
    changed |= DragDouble3("Skin Color", &subsurface_color[0], 0.005);
    changed |= DragDouble("Base Roughness", &subsurface_roughness, 0.005);
    if (changed) {
      sync_base_layer(base_layer, subsurface_color, subsurface_roughness);
    }
    ImGui::TreePop();
  }
}

BSDFPreset FastLayeredBSDF::get_preset() const {
  BSDFPreset preset;
  preset.type = BSDF_PRESET_FAST_LAYERED;
  preset.vector_a = base_color;
  preset.vector_b = subsurface_color;
  preset.scalar_a = roughness;
  preset.scalar_b = thickness;
  preset.scalar_c = saturation;
  preset.scalar_d = ior;
  preset.scalar_e = subsurface_roughness;
  preset.scalar_f = pooling_strength;
  return preset;
}

void FastLayeredBSDF::apply_preset(const BSDFPreset& preset) {
  if (preset.type != BSDF_PRESET_FAST_LAYERED) return;
  roughness = preset.scalar_a;
  thickness = preset.scalar_b;
  base_color = preset.vector_a;
  saturation = preset.scalar_c;
  ior = preset.scalar_d;
  subsurface_color = preset.vector_b;
  subsurface_roughness = preset.scalar_e;
  pooling_strength = clamp(preset.scalar_f, 0.0, 1.0);
  sync_base_layer(base_layer, subsurface_color, subsurface_roughness);
}

/**
 * Disney Realistic Lip BSDF
 * Uses Disney's Subsurface approximation for flesh + a water-IOR clearcoat.
 */
Vector3D DisneyLayeredBSDF::f(const Vector3D wo, const Vector3D wi) {
  if (wo.z <= 0.0 || wi.z <= 0.0) return Vector3D(0, 0, 0);

  Vector3D h = wo + wi;
  if (h.norm2() == 0.0) return Vector3D(0, 0, 0);
  h.normalize();

  double cos_theta_i = wi.z;
  double cos_theta_o = wo.z;
  double cos_theta_d = max(dot(wi, h), 0.0); // angle btwn incident and half-vector

  // THE FLESH LAYER (Disney Subsurface Approximation)
  
  // this math flattens the diffuse shape at glancing angles to simulate 
  // light scattering through the edges of the lips (translucency).
  double fss90 = subsurface_roughness * cos_theta_d * cos_theta_d;
  
  auto schlick_weight = [](double cos_theta) {
    double m = clamp(1.0 - cos_theta, 0.0, 1.0);
    return pow(m, 5.0);
  };

  double fss_in = 1.0 + (fss90 - 1.0) * schlick_weight(cos_theta_i);
  double fss_out = 1.0 + (fss90 - 1.0) * schlick_weight(cos_theta_o);
  
  // the subsurface blend
  double ss_factor = 1.25 * (fss_in * fss_out * (1.0 / (cos_theta_i + cos_theta_o + 0.05) - 0.5) + 0.5);
  
  // mix in some dark red/pink for the subsurface bleed color
  Vector3D subsurface_bleed = subsurface_color * 0.8 + Vector3D(0.8, 0.1, 0.1) * 0.2;
  Vector3D flesh_contrib = (subsurface_bleed / PI) * ss_factor * clamp(saturation, 0.0, 1.5);

  // THE GLOSS LAYER (Realistic Lip Gloss)
  
  // Use input roughness directly for realistic gloss control (lower = shinier, more gloss-like)
  double gloss_roughness = max(roughness * 0.8, 0.02); // scale down to make gloss smoother
  double alpha2_gloss = max(gloss_roughness * gloss_roughness, 0.0004);
  
  // Use input IOR for physically accurate reflectance
  double R0 = pow((1.0 - ior) / (1.0 + ior), 2.0);
  double F_gloss = R0 + (1.0 - R0) * schlick_weight(cos_theta_d);

  double cos_theta_h = max(h.z, 0.0);
  double cos2_theta_h = cos_theta_h * cos_theta_h;
  
  // GGX Normal Distribution for smooth gloss layer
  double D_denom = PI * pow(cos2_theta_h * (alpha2_gloss - 1.0) + 1.0, 2.0);
  double D = (D_denom > 0.0) ? (alpha2_gloss / D_denom) : 0.0;

  auto G1 = [](double cos_theta, double a2) {
    double cos2_theta = cos_theta * cos_theta;
    return (2.0 * cos_theta) / (cos_theta + sqrt(a2 + (1.0 - a2) * cos2_theta));
  };
  double G = G1(cos_theta_o, alpha2_gloss) * G1(cos_theta_i, alpha2_gloss);

  double gloss_val = (D * F_gloss * G) / (4.0 * max(cos_theta_i * cos_theta_o, 0.001));
  
  // Clamp specular to prevent fireflies while keeping gloss realistic
  gloss_val = min(gloss_val, 5.0);
  Vector3D gloss_contrib(gloss_val, gloss_val, gloss_val);

  // ENERGY CONSERVATION & BLEND
  
  // Blend between flesh (diffuse) and gloss (specular) layers
  // The gloss layer tints slightly with the base color for realism
  Vector3D gloss_tinted = gloss_contrib + (base_color * 0.15) * gloss_val;
  
  // Blend: flesh layer is dimmed where gloss reflects, but not as aggressively
  // This preserves more of the red/pink undertone even with high thickness
  Vector3D final_flesh = flesh_contrib * (1.0 - F_gloss * 0.6);

  return final_flesh * (1.0 - thickness) + gloss_tinted * thickness;

  /*
  // THE WET LAYER (GGX Clearcoat for Saliva/Gloss)
  
  // force a low roughness for the wet look, regardless of the fleshy skin roughness
  double wet_roughness = 0.15; 
  double alpha2 = max(wet_roughness * wet_roughness, 0.001);
  
  // Saliva/Water IOR is around 1.33
  double water_ior = 1.48;
  double R0 = pow((1.0 - water_ior) / (1.0 + water_ior), 2.0);
  double F = R0 + (1.0 - R0) * schlick_weight(cos_theta_d);

  // GGX Normal Distribution
  double cos_theta_h = max(h.z, 0.0);
  double cos2_theta_h = cos_theta_h * cos_theta_h;
  double D_denom = PI * pow(cos2_theta_h * (alpha2 - 1.0) + 1.0, 2.0);
  double D = (D_denom > 0.0) ? (alpha2 / D_denom) : 0.0;

  // Smith Geometry
  auto G1 = [](double cos_theta, double a2) {
    double cos2_theta = cos_theta * cos_theta;
    return (2.0 * cos_theta) / (cos_theta + sqrt(a2 + (1.0 - a2) * cos2_theta));
  };
  double G = G1(cos_theta_o, alpha2) * G1(cos_theta_i, alpha2);

  double ct_val = (D * F * G) / (4.0 * max(cos_theta_i * cos_theta_o, 0.001));
  ct_val = min(ct_val, 1.0); // anti-firefly
  Vector3D wet_contrib(ct_val, ct_val, ct_val);

  // ENERGY CONSERVATION & BLEND
  
  // if wet layer reflects light, that light NEVER reaches the flesh below
  // subtract the Fresnel reflectance (F) from the flesh layer
  Vector3D final_flesh = flesh_contrib * (1.0 - F);

  // blend based on how much "gloss" (thickness) we want applied
  return final_flesh * (1.0 - thickness) + (final_flesh + wet_contrib) * thickness;
  */
}

Vector3D DisneyLayeredBSDF::f(const Vector3D wo, const Vector3D wi,
                              const Vector2D uv) {
  if (pooling_strength <= 0.0) return f(wo, wi);
  if (wo.z <= 0.0 || wi.z <= 0.0) return Vector3D(0, 0, 0);

  double local_roughness, local_thickness, local_saturation;
  pooled_layer_params(uv, pooling_strength, roughness, thickness, saturation,
                      &local_roughness, &local_thickness, &local_saturation);

  Vector3D h = wo + wi;
  if (h.norm2() == 0.0) return Vector3D(0, 0, 0);
  h.normalize();

  double cos_theta_i = wi.z;
  double cos_theta_o = wo.z;
  double cos_theta_d = max(dot(wi, h), 0.0);

  auto schlick_weight = [](double cos_theta) {
    double m = clamp(1.0 - cos_theta, 0.0, 1.0);
    return pow(m, 5.0);
  };

  double fss90 = subsurface_roughness * cos_theta_d * cos_theta_d;
  double fss_in = 1.0 + (fss90 - 1.0) * schlick_weight(cos_theta_i);
  double fss_out = 1.0 + (fss90 - 1.0) * schlick_weight(cos_theta_o);
  double ss_factor = 1.25 * (fss_in * fss_out *
      (1.0 / (cos_theta_i + cos_theta_o + 0.05) - 0.5) + 0.5);

  Vector3D subsurface_bleed = subsurface_color * 0.8 +
                              Vector3D(0.8, 0.1, 0.1) * 0.2;
  Vector3D flesh_contrib =
      (subsurface_bleed / PI) * ss_factor * local_saturation;

  double gloss_roughness = max(local_roughness * 0.8, 0.02);
  double alpha2_gloss = max(gloss_roughness * gloss_roughness, 0.0004);

  double R0 = pow((1.0 - ior) / (1.0 + ior), 2.0);
  double F_gloss = R0 + (1.0 - R0) * schlick_weight(cos_theta_d);

  double cos_theta_h = max(h.z, 0.0);
  double cos2_theta_h = cos_theta_h * cos_theta_h;
  double D_denom = PI * pow(cos2_theta_h * (alpha2_gloss - 1.0) + 1.0, 2.0);
  double D = (D_denom > 0.0) ? (alpha2_gloss / D_denom) : 0.0;

  auto G1 = [](double cos_theta, double a2) {
    double cos2_theta = cos_theta * cos_theta;
    return (2.0 * cos_theta) / (cos_theta + sqrt(a2 + (1.0 - a2) * cos2_theta));
  };
  double G = G1(cos_theta_o, alpha2_gloss) * G1(cos_theta_i, alpha2_gloss);

  double gloss_val = (D * F_gloss * G) /
                     (4.0 * max(cos_theta_i * cos_theta_o, 0.001));
  gloss_val = min(gloss_val, 5.0);
  Vector3D gloss_contrib(gloss_val, gloss_val, gloss_val);

  Vector3D gloss_tinted = gloss_contrib + (base_color * 0.15) * gloss_val;
  Vector3D final_flesh = flesh_contrib * (1.0 - F_gloss * 0.6);
  return final_flesh * (1.0 - local_thickness) +
         gloss_tinted * local_thickness;
}

Vector3D DisneyLayeredBSDF::sample_f(const Vector3D wo, Vector3D* wi, double* pdf) {
  if (wo.z <= 0.0) {
    *pdf = 0.0;
    return Vector3D(0, 0, 0);
  }

  // --- Mixture Weights ---
  // The layered lip model is evaluated as a blend of flesh/base and gloss.
  // We still sample one branch at a time for efficiency, but the final pdf
  // must be the density of the full mixture because we return the full BSDF
  // value f(wo, wi), not only the chosen branch contribution.
  double gloss_weight = clamp(thickness, 0.0, 1.0);
  double base_weight = 1.0 - gloss_weight;

  // --- Sample One Branch ---
  // Pick a branch using thickness, then generate wi from that branch's sampler.
  // The gloss branch now uses GGX half-vector sampling so it matches the lobe
  // computed in DisneyLayeredBSDF::f().
  if (random_uniform() < gloss_weight) {
    double sampled_gloss_pdf = 0.0;
    if (!sample_disney_gloss_lobe(wo, roughness, wi, &sampled_gloss_pdf)) {
      *pdf = 0.0;
      return Vector3D(0, 0, 0);
    }
  } else {
    double base_pdf;
    base_layer->sample_f(wo, wi, &base_pdf);
    if (base_pdf <= 0.0 || wi->z <= 0.0) {
      *pdf = 0.0;
      return Vector3D(0, 0, 0);
    }
  }

  // --- Recompute the Full Mixture PDF ---
  // Even if the sample came from only one branch, the same wi could have been
  // produced by either lobe. The correct denominator for Monte Carlo transport
  // is therefore
  //   p(wi) = (1 - t) * p_base(wi) + t * p_gloss(wi)
  // instead of only the selected branch pdf.
  double base_pdf = cosine_hemisphere_pdf(*wi);
  double gloss_pdf = disney_gloss_pdf(wo, *wi, roughness);
  *pdf = base_weight * base_pdf + gloss_weight * gloss_pdf;

  if (*pdf <= 1e-10) {
    return Vector3D(0, 0, 0);
  }

  // Return the full two-lobe BSDF value. The path integrator handles the
  // abs_cos_theta(wi) / pdf weighting, so sample_f should only ensure that
  // wi, f(wo, wi), and pdf all refer to the same sampling model.
  return f(wo, *wi);
}

Vector3D DisneyLayeredBSDF::sample_f(const Vector3D wo, Vector3D* wi,
                                     double* pdf, const Vector2D uv) {
  if (pooling_strength <= 0.0) return sample_f(wo, wi, pdf);
  if (wo.z <= 0.0) {
    *pdf = 0.0;
    return Vector3D(0, 0, 0);
  }

  double local_roughness, local_thickness, local_saturation;
  pooled_layer_params(uv, pooling_strength, roughness, thickness, saturation,
                      &local_roughness, &local_thickness, &local_saturation);
  (void)local_saturation;
  double gloss_weight = clamp(local_thickness, 0.0, 1.0);
  double base_weight = 1.0 - gloss_weight;

  if (random_uniform() < gloss_weight) {
    double sampled_gloss_pdf = 0.0;
    if (!sample_disney_gloss_lobe(wo, local_roughness, wi, &sampled_gloss_pdf)) {
      *pdf = 0.0;
      return Vector3D(0, 0, 0);
    }
  } else {
    double base_pdf;
    base_layer->sample_f(wo, wi, &base_pdf);
    if (base_pdf <= 0.0 || wi->z <= 0.0) {
      *pdf = 0.0;
      return Vector3D(0, 0, 0);
    }
  }

  double base_pdf = cosine_hemisphere_pdf(*wi);
  double gloss_pdf = disney_gloss_pdf(wo, *wi, local_roughness);
  *pdf = base_weight * base_pdf + gloss_weight * gloss_pdf;
  if (*pdf <= 1e-10) return Vector3D(0, 0, 0);
  return f(wo, *wi, uv);
}

void DisneyLayeredBSDF::render_debugger_node()
{
  if (ImGui::TreeNode(this, "Disney Layered BSDF"))
  {
    bool changed = false;
    changed |= DragDouble("Gloss Roughness", &roughness, 0.005);
    changed |= DragDouble("Thickness", &thickness, 0.005);
    changed |= DragDouble3("Base Color", &base_color[0], 0.005);
    changed |= DragDouble("Saturation", &saturation, 0.005);
    changed |= DragDouble("IOR", &ior, 0.005);
    changed |= DragDouble("Pooling Strength", &pooling_strength, 0.005);
    ImGui::Spacing();
    ImGui::Text("Base BSSRDF");
    changed |= DragDouble3("Skin Color", &subsurface_color[0], 0.005);
    changed |= DragDouble("Base Roughness", &subsurface_roughness, 0.005);
    if (changed) {
      sync_base_layer(base_layer, subsurface_color, subsurface_roughness);
    }
    ImGui::TreePop();
  }
}

BSDFPreset DisneyLayeredBSDF::get_preset() const {
  BSDFPreset preset;
  preset.type = BSDF_PRESET_DISNEY_LAYERED;
  preset.vector_a = base_color;
  preset.vector_b = subsurface_color;
  preset.scalar_a = roughness;
  preset.scalar_b = thickness;
  preset.scalar_c = saturation;
  preset.scalar_d = ior;
  preset.scalar_e = subsurface_roughness;
  preset.scalar_f = pooling_strength;
  return preset;
}

void DisneyLayeredBSDF::apply_preset(const BSDFPreset& preset) {
  if (preset.type != BSDF_PRESET_DISNEY_LAYERED) return;
  roughness = preset.scalar_a;
  thickness = preset.scalar_b;
  base_color = preset.vector_a;
  saturation = preset.scalar_c;
  ior = preset.scalar_d;
  subsurface_color = preset.vector_b;
  subsurface_roughness = preset.scalar_e;
  pooling_strength = clamp(preset.scalar_f, 0.0, 1.0);
  sync_base_layer(base_layer, subsurface_color, subsurface_roughness);
}

} // namespace CGL
