#ifndef CGL_STATICSCENE_BSDF_H
#define CGL_STATICSCENE_BSDF_H

#include "CGL/CGL.h"
#include "CGL/vector2D.h"
#include "CGL/vector3D.h"
#include "CGL/matrix3x3.h"

#include "pathtracer/sampler.h"
#include "util/image.h"

#include <algorithm>
#include <string>

namespace CGL {

// Helper math functions. Assume all vectors are in unit hemisphere //

inline double clamp (double n, double lower, double upper) {
  return std::max(lower, std::min(n, upper));
}

inline double cos_theta(const Vector3D w) {
  return w.z;
}

inline double abs_cos_theta(const Vector3D w) {
  return fabs(w.z);
}

inline double sin_theta2(const Vector3D w) {
  return fmax(0.0, 1.0 - cos_theta(w) * cos_theta(w));
}

inline double sin_theta(const Vector3D w) {
  return sqrt(sin_theta2(w));
}

inline double cos_phi(const Vector3D w) {
  double sinTheta = sin_theta(w);
  if (sinTheta == 0.0) return 1.0;
  return clamp(w.x / sinTheta, -1.0, 1.0);
}

inline double sin_phi(const Vector3D w) {
  double sinTheta = sin_theta(w);
  if (sinTheta) return 0.0;
  return clamp(w.y / sinTheta, -1.0, 1.0);
}

void make_coord_space(Matrix3x3& o2w, const Vector3D n);

enum BSDFPresetType {
  BSDF_PRESET_UNKNOWN,
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

struct BSDFPreset {
  BSDFPresetType type;
  std::string material_id;
  std::string material_name;
  Vector3D vector_a;
  Vector3D vector_b;
  Vector3D vector_c;
  double scalar_a;
  double scalar_b;
  double scalar_c;
  double scalar_d;
  double scalar_e;
  double scalar_f;

  BSDFPreset()
      : type(BSDF_PRESET_UNKNOWN),
        vector_a(),
        vector_b(),
        vector_c(),
        scalar_a(0.0),
        scalar_b(0.0),
        scalar_c(0.0),
        scalar_d(0.0),
        scalar_e(0.0),
        scalar_f(0.0) {}
};

class BSDF;

const char* bsdf_preset_type_name(BSDFPresetType type);
bool render_bsdf_preset_controls(BSDFPreset& preset);
BSDF* create_bsdf_from_preset(const BSDFPreset& preset);

/**
 * Interface for BSDFs.
 * BSDFs (Bidirectional Scattering Distribution Functions)
 * describe the ratio of incoming light scattered from
 * incident direction to outgoing direction.
 * Scene objects are initialized with a BSDF subclass, used
 * to represent the object's material and associated properties.
 */
class BSDF {
 public:
  virtual ~BSDF() {}

  /**
   * Evaluate BSDF.
   * Given incident light direction wi and outgoing light direction wo. Note
   * that both wi and wo are defined in the local coordinate system at the
   * point of intersection.
   * \param wo outgoing light direction in local space of point of intersection
   * \param wi incident light direction in local space of point of intersection
   * \return reflectance in the given incident/outgoing directions
   */
  virtual Vector3D f (const Vector3D wo, const Vector3D wi) = 0;
  virtual Vector3D f (const Vector3D wo, const Vector3D wi,
                      const Vector2D uv) {
    return f(wo, wi);
  }

  /**
   * Evaluate BSDF.
   * Given the outgoing light direction wo, samplea incident light
   * direction and store it in wi. Store the pdf of the sampled direction in pdf.
   * Again, note that wo and wi should both be defined in the local coordinate
   * system at the point of intersection.
   * \param wo outgoing light direction in local space of point of intersection
   * \param wi address to store incident light direction
   * \param pdf address to store the pdf of the sampled incident direction
   * \return reflectance in the output incident and given outgoing directions
   */
  virtual Vector3D sample_f (const Vector3D wo, Vector3D* wi, double* pdf) = 0;
  virtual Vector3D sample_f (const Vector3D wo, Vector3D* wi, double* pdf,
                             const Vector2D uv) {
    return sample_f(wo, wi, pdf);
  }

  /**
   * Get the emission value of the surface material. For non-emitting surfaces
   * this would be a zero energy Vector3D.
   * \return emission Vector3D of the surface material
   */
  virtual Vector3D get_emission () const = 0;

  /**
   * If the BSDF is a delta distribution. Materials that are perfectly specular,
   * (e.g. water, glass, mirror) only scatter light from a single incident angle
   * to a single outgoing angle. These BSDFs are best described with alpha
   * distributions that are zero except for the single direction where light is
   * scattered.
   */
  virtual bool is_delta() const = 0;

  virtual void render_debugger_node() {};
  virtual BSDFPreset get_preset() const;
  virtual void apply_preset(const BSDFPreset& preset);

  /**
   * Reflection helper
   */
  virtual void reflect(const Vector3D wo, Vector3D* wi);

  /**
   * Refraction helper
   */
  virtual bool refract(const Vector3D wo, Vector3D* wi, double ior);

  const HDRImageBuffer* reflectanceMap;
  const HDRImageBuffer* normalMap;

}; // class BSDF

/**
 * Diffuse BSDF.
 */
class DiffuseBSDF : public BSDF {
 public:

  /**
   * DiffuseBSDFs are constructed with a Vector3D as input,
   * which is stored into the member variable `reflectance`.
   */
  DiffuseBSDF(const Vector3D a) : reflectance(a) { }

  Vector3D f(const Vector3D wo, const Vector3D wi);
  Vector3D sample_f(const Vector3D wo, Vector3D* wi, double* pdf);
  Vector3D get_emission() const { return Vector3D(); }
  bool is_delta() const { return false; }

  void render_debugger_node();
  BSDFPreset get_preset() const;
  void apply_preset(const BSDFPreset& preset);

private:
  /*
   * Reflectance is also commonly called the "albedo" of a surface,
   * which ranges from [0,1] in RGB, representing a range of
   * total absorption(0) vs. total reflection(1) per color channel.
   */
  Vector3D reflectance;
  /*
   * A sampler object that can be used to obtain
   * a random Vector3D sampled according to a 
   * cosine-weighted hemisphere distribution.
   * See pathtracer/sampler.cpp.
   */
  CosineWeightedHemisphereSampler3D sampler;

}; // class DiffuseBSDF

/**
 * Microfacet BSDF.
 */

class MicrofacetBSDF : public BSDF {
public:

  MicrofacetBSDF(const Vector3D eta, const Vector3D k, double alpha)
    : eta(eta), k(k), alpha(alpha) { }

  double getTheta(const Vector3D w) {
    return acos(clamp(w.z, -1.0 + 1e-5, 1.0 - 1e-5));
  }

  double Lambda(const Vector3D w) {
    double theta = getTheta(w);
    double a = 1.0 / (alpha * tan(theta));
    return 0.5 * (erf(a) - 1.0 + exp(-a * a) / (a * PI));
  }

  Vector3D F(double cosThetaI);

  double G(const Vector3D wo, const Vector3D wi);

  double D(const Vector3D h);

  Vector3D f(const Vector3D wo, const Vector3D wi);
  Vector3D sample_f(const Vector3D wo, Vector3D* wi, double* pdf);
  Vector3D get_emission() const { return Vector3D(); }
  bool is_delta() const { return false; }

  void render_debugger_node();
  BSDFPreset get_preset() const;
  void apply_preset(const BSDFPreset& preset);

private:
  Vector3D eta, k;
  double alpha;
  UniformGridSampler2D sampler;
  CosineWeightedHemisphereSampler3D cosineHemisphereSampler;
}; // class MicrofacetBSDF

/**
 * Mirror BSDF
 */
class MirrorBSDF : public BSDF {
 public:

  MirrorBSDF(const Vector3D reflectance) : reflectance(reflectance) { }

  Vector3D f(const Vector3D wo, const Vector3D wi);
  Vector3D sample_f(const Vector3D wo, Vector3D* wi, double* pdf);
  Vector3D get_emission() const { return Vector3D(); }
  bool is_delta() const { return true; }

  void render_debugger_node();
  BSDFPreset get_preset() const;
  void apply_preset(const BSDFPreset& preset);

private:

  double roughness;
  Vector3D reflectance;

}; // class MirrorBSDF*/

/**
 * Refraction BSDF.
 */
class RefractionBSDF : public BSDF {
 public:

  RefractionBSDF(const Vector3D transmittance, double roughness, double ior)
    : transmittance(transmittance), roughness(roughness), ior(ior) { }

  Vector3D f(const Vector3D wo, const Vector3D wi);
  Vector3D sample_f(const Vector3D wo, Vector3D* wi, double* pdf);
  Vector3D get_emission() const { return Vector3D(); }
  bool is_delta() const { return true; }

  void render_debugger_node();
  BSDFPreset get_preset() const;
  void apply_preset(const BSDFPreset& preset);

 private:

  double ior;
  double roughness;
  Vector3D transmittance;

}; // class RefractionBSDF

/**
 * Glass BSDF.
 */
class GlassBSDF : public BSDF {
 public:

  GlassBSDF(const Vector3D transmittance, const Vector3D reflectance,
            double roughness, double ior) :
    transmittance(transmittance), reflectance(reflectance),
    roughness(roughness), ior(ior) { }

  Vector3D f(const Vector3D wo, const Vector3D wi);
  Vector3D sample_f(const Vector3D wo, Vector3D* wi, double* pdf);
  Vector3D get_emission() const { return Vector3D(); }
  bool is_delta() const { return true; }

  void render_debugger_node();
  BSDFPreset get_preset() const;
  void apply_preset(const BSDFPreset& preset);

 private:

  double ior;
  double roughness;
  Vector3D reflectance;
  Vector3D transmittance;

}; // class GlassBSDF

/**
 * Emission BSDF.
 */
class EmissionBSDF : public BSDF {
 public:

  EmissionBSDF(const Vector3D radiance) : radiance(radiance) { }

  Vector3D f(const Vector3D wo, const Vector3D wi);
  Vector3D sample_f(const Vector3D wo, Vector3D* wi, double* pdf);
  Vector3D get_emission() const { return radiance; }
  bool is_delta() const { return false; }

  void render_debugger_node();
  BSDFPreset get_preset() const;
  void apply_preset(const BSDFPreset& preset);

 private:

  Vector3D radiance;
  CosineWeightedHemisphereSampler3D sampler;

}; // class EmissionBSDF

/**
 * Approximate BSSRDF (Bidirectional Scattering Surface Reflectance Distribution Function).
 * Simulates subsurface scattering in skin/biological materials using an approximate diffuse-like model.
 * This is the base layer for layered materials like lips with gloss.
 */
class ApproximateBSSRDF : public BSDF {
 public:

  /**
   * ApproximateBSSRDF is constructed with a skin color and surface roughness.
   * \param skin_color The base color of the skin/material
   * \param roughness Surface micro-roughness for additional scattering detail
   */
  ApproximateBSSRDF(const Vector3D skin_color, double roughness = 0.3)
    : skin_color(skin_color), roughness(roughness) { }

  Vector3D f(const Vector3D wo, const Vector3D wi);
  Vector3D sample_f(const Vector3D wo, Vector3D* wi, double* pdf);
  Vector3D get_emission() const { return Vector3D(); }
  bool is_delta() const { return false; }

  void render_debugger_node();
  BSDFPreset get_preset() const;
  void apply_preset(const BSDFPreset& preset);

 private:

  Vector3D skin_color;
  double roughness;
  CosineWeightedHemisphereSampler3D sampler;

}; // class ApproximateBSSRDF

/**
 * Random-walk subsurface material.
 *
 * The path integrator treats this BSDF specially for supported closed primitives:
 * camera/light paths enter the medium, perform volumetric random walks using
 * sigma_a/sigma_s, and re-emerge from another surface point. The local f() and
 * sample_f() methods are a diffuse fallback for unsupported geometry.
 */
class RandomWalkSSSBSDF : public BSDF {
 public:

  RandomWalkSSSBSDF(const Vector3D sigma_a, const Vector3D sigma_s,
                   double anisotropy_g = 0.0, double ior = 1.3,
                   double scale = 5.0, double surface_roughness = 0.55,
                   double specular_weight = 0.3,
                   BSDFPresetType preset_type = BSDF_PRESET_RANDOM_WALK_SSS,
                   const Vector3D base_color = Vector3D(1.0),
                   double saturation = 1.0)
    : sigma_a(sigma_a), sigma_s(sigma_s), anisotropy_g(anisotropy_g),
      ior(ior), scale(scale), surface_roughness(surface_roughness),
      specular_weight(specular_weight), preset_type(preset_type),
      base_color(base_color), saturation(saturation) { }

  Vector3D f(const Vector3D wo, const Vector3D wi);
  Vector3D sample_f(const Vector3D wo, Vector3D* wi, double* pdf);
  Vector3D get_emission() const { return Vector3D(); }
  bool is_delta() const { return false; }

  void render_debugger_node();
  BSDFPreset get_preset() const;
  void apply_preset(const BSDFPreset& preset);

  Vector3D get_sigma_a() const { return sigma_a; }
  Vector3D get_sigma_s() const { return sigma_s; }
  double get_anisotropy_g() const { return anisotropy_g; }
  double get_ior() const { return ior; }
  double get_scale() const { return scale; }
  double get_surface_roughness() const { return surface_roughness; }
  double get_specular_weight() const { return specular_weight; }
  BSDFPresetType get_preset_type() const { return preset_type; }
  Vector3D get_base_color() const { return base_color; }
  double get_saturation() const { return saturation; }
  Vector3D diffuse_fallback_color() const;

 private:

  Vector3D sigma_a;
  Vector3D sigma_s;
  double anisotropy_g;
  double ior;
  double scale;
  double surface_roughness;
  double specular_weight;
  BSDFPresetType preset_type;
  Vector3D base_color;
  double saturation;
  CosineWeightedHemisphereSampler3D sampler;

}; // class RandomWalkSSSBSDF

/**
 * Layered BSDF.
 * Combines a base BSSRDF layer (e.g., skin) with a glossy specular layer (e.g., gloss/lipstick).
 * The thickness parameter controls the blend between base and gloss layers.
 */
class LayeredBSDF : public BSDF {
 public:

  /**
   * LayeredBSDF is constructed with roughness, thickness, base color, saturation, and IOR.
   * \param roughness Dielectric layer roughness (controls shine)
   * \param thickness Opacity of gloss layer (0 = all base, 1 = all gloss)
   * \param base_color Skin/lip color
   * \param saturation Color saturation multiplier
   * \param ior Index of refraction for the gloss layer (default 1.5)
   */
  LayeredBSDF(double roughness, double thickness, const Vector3D base_color,
              double saturation, double ior = 1.5,
              double pooling_strength = 0.0)
    : roughness(roughness), thickness(thickness), base_color(base_color),
      saturation(saturation), ior(ior),
      subsurface_color(base_color), subsurface_roughness(roughness),
      pooling_strength(pooling_strength),
      base_layer(new ApproximateBSSRDF(subsurface_color, subsurface_roughness)) { }

  ~LayeredBSDF() {
    delete base_layer;
  }

  Vector3D f(const Vector3D wo, const Vector3D wi);
  Vector3D f(const Vector3D wo, const Vector3D wi, const Vector2D uv);
  Vector3D sample_f(const Vector3D wo, Vector3D* wi, double* pdf);
  Vector3D sample_f(const Vector3D wo, Vector3D* wi, double* pdf,
                    const Vector2D uv);
  Vector3D get_emission() const { return Vector3D(); }
  bool is_delta() const { return false; }

  void render_debugger_node();
  BSDFPreset get_preset() const;
  void apply_preset(const BSDFPreset& preset);

 private:

  double roughness;
  double thickness;
  Vector3D base_color;
  double saturation;
  double ior;
  Vector3D subsurface_color;
  double subsurface_roughness;
  double pooling_strength;
  ApproximateBSSRDF* base_layer;
  CosineWeightedHemisphereSampler3D sampler;

}; // class LayeredBSDF

/**
 * Fast Layered BSDF.
 */
class FastLayeredBSDF : public BSDF {
 public:

  /**
   * \param roughness Dielectric layer roughness (controls shine)
   * \param thickness Opacity of gloss layer (0 = all base, 1 = all gloss)
   * \param base_color Skin/lip color
   * \param saturation Color saturation multiplier
   * \param ior Index of refraction for the gloss layer (default 1.5)
   */
  FastLayeredBSDF(double roughness, double thickness, const Vector3D base_color,
              double saturation, double ior = 1.5,
              double pooling_strength = 0.0)
    : roughness(roughness), thickness(thickness), base_color(base_color),
      saturation(saturation), ior(ior),
      subsurface_color(base_color), subsurface_roughness(roughness),
      pooling_strength(pooling_strength),
      base_layer(new ApproximateBSSRDF(subsurface_color, subsurface_roughness)) { }

  ~FastLayeredBSDF() {
    delete base_layer;
  }

  Vector3D f(const Vector3D wo, const Vector3D wi);
  Vector3D f(const Vector3D wo, const Vector3D wi, const Vector2D uv);
  Vector3D sample_f(const Vector3D wo, Vector3D* wi, double* pdf);
  Vector3D sample_f(const Vector3D wo, Vector3D* wi, double* pdf,
                    const Vector2D uv);
  Vector3D get_emission() const { return Vector3D(); }
  bool is_delta() const { return false; }

  void render_debugger_node();
  BSDFPreset get_preset() const;
  void apply_preset(const BSDFPreset& preset);

 private:

  double roughness;
  double thickness;
  Vector3D base_color;
  double saturation;
  double ior;
  Vector3D subsurface_color;
  double subsurface_roughness;
  double pooling_strength;
  ApproximateBSSRDF* base_layer;
  CosineWeightedHemisphereSampler3D sampler;

}; // class FastLayeredBSDF


/**
 * Disney Layered BSDF.
 */
class DisneyLayeredBSDF : public BSDF {
 public:

  /**
   * \param roughness Dielectric layer roughness (controls shine)
   * \param thickness Opacity of gloss layer (0 = all base, 1 = all gloss)
   * \param base_color Skin/lip color
   * \param saturation Color saturation multiplier
   * \param ior Index of refraction for the gloss layer (default 1.5)
   */
  DisneyLayeredBSDF(double roughness, double thickness, const Vector3D base_color,
              double saturation, double ior = 1.5,
              double pooling_strength = 0.0)
    : roughness(roughness), thickness(thickness), base_color(base_color),
      saturation(saturation), ior(ior),
      subsurface_color(base_color), subsurface_roughness(roughness),
      pooling_strength(pooling_strength),
      base_layer(new ApproximateBSSRDF(subsurface_color, subsurface_roughness)) { }

  ~DisneyLayeredBSDF() {
    delete base_layer;
  }

  Vector3D f(const Vector3D wo, const Vector3D wi);
  Vector3D f(const Vector3D wo, const Vector3D wi, const Vector2D uv);
  Vector3D sample_f(const Vector3D wo, Vector3D* wi, double* pdf);
  Vector3D sample_f(const Vector3D wo, Vector3D* wi, double* pdf,
                    const Vector2D uv);
  Vector3D get_emission() const { return Vector3D(); }
  bool is_delta() const { return false; }

  void render_debugger_node();
  BSDFPreset get_preset() const;
  void apply_preset(const BSDFPreset& preset);

 private:

  double roughness;
  double thickness;
  Vector3D base_color;
  double saturation;
  double ior;
  Vector3D subsurface_color;
  double subsurface_roughness;
  double pooling_strength;
  ApproximateBSSRDF* base_layer;
  CosineWeightedHemisphereSampler3D sampler;

}; // class DisneyLayeredBSDF

}  // namespace CGL

#endif  // CGL_STATICSCENE_BSDF_H
