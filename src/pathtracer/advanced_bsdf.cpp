#include "bsdf.h"

#include <algorithm>
#include <iostream>
#include <utility>

#include "application/visual_debugger.h"

using std::max;
using std::min;
using std::swap;

namespace CGL {

// Mirror BSDF //

Vector3D MirrorBSDF::f(const Vector3D wo, const Vector3D wi) {
  return Vector3D();
}

Vector3D MirrorBSDF::sample_f(const Vector3D wo, Vector3D* wi, double* pdf) {
  reflect(wo, wi);
  *pdf = 1.0;
  return reflectance / abs_cos_theta(*wi);
}

void MirrorBSDF::render_debugger_node()
{
  if (ImGui::TreeNode(this, "Mirror BSDF"))
  {
    DragDouble3("Reflectance", &reflectance[0], 0.005);
    ImGui::TreePop();
  }
}

BSDFPreset MirrorBSDF::get_preset() const {
  BSDFPreset preset;
  preset.type = BSDF_PRESET_MIRROR;
  preset.vector_a = reflectance;
  return preset;
}

void MirrorBSDF::apply_preset(const BSDFPreset& preset) {
  if (preset.type != BSDF_PRESET_MIRROR) return;
  reflectance = preset.vector_a;
}

// Microfacet BSDF //

double MicrofacetBSDF::G(const Vector3D wo, const Vector3D wi) {
  return 1.0 / (1.0 + Lambda(wi) + Lambda(wo));
}

double MicrofacetBSDF::D(const Vector3D h) {
  const double cosTheta = h.z;
  if (cosTheta <= 0.0) return 0.0;

  const double cosTheta2 = cosTheta * cosTheta;
  const double alpha2 = alpha * alpha;
  const double tanTheta2 = sin_theta2(h) / cosTheta2;

  return exp(-tanTheta2 / alpha2) / (PI * alpha2 * cosTheta2 * cosTheta2);
}

Vector3D MicrofacetBSDF::F(double cosThetaI) {
  const double cosTheta = clamp(fabs(cosThetaI), 0.0, 1.0);
  const double cosTheta2 = cosTheta * cosTheta;
  const Vector3D eta2 = eta * eta;
  const Vector3D k2 = k * k;
  const Vector3D t0 = eta2 + k2;
  const Vector3D twoEtaCos = eta * (2.0 * cosTheta);
  const Vector3D cosTheta2Vec(cosTheta2);

  const Vector3D Rs =
      (t0 - twoEtaCos + cosTheta2Vec) / (t0 + twoEtaCos + cosTheta2Vec);
  const Vector3D Rp =
      (t0 * cosTheta2 - twoEtaCos + Vector3D(1.0)) /
      (t0 * cosTheta2 + twoEtaCos + Vector3D(1.0));

  return 0.5 * (Rs + Rp);
}

Vector3D MicrofacetBSDF::f(const Vector3D wo, const Vector3D wi) {
  const double cosThetaO = wo.z;
  const double cosThetaI = wi.z;
  if (cosThetaO <= 0.0 || cosThetaI <= 0.0) return Vector3D();

  const Vector3D h = (wo + wi).unit();
  const double wiDotH = dot(wi, h);
  if (wiDotH <= 0.0) return Vector3D();

  const double denom = 4.0 * cosThetaO * cosThetaI;
  return F(wiDotH) * (G(wo, wi) * D(h) / denom);
}

Vector3D MicrofacetBSDF::sample_f(const Vector3D wo, Vector3D* wi, double* pdf) {
  if (wo.z <= 0.0) {
    *pdf = 0.0;
    return Vector3D();
  }

  const Vector2D u = sampler.get_sample();
  const double phi = 2.0 * PI * u.y;
  const double tanTheta2 = -alpha * alpha * log(1.0 - u.x);
  const double cosTheta = 1.0 / sqrt(1.0 + tanTheta2);
  const double sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
  const Vector3D h(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

  const double woDotH = dot(wo, h);
  if (woDotH <= 0.0) {
    *pdf = 0.0;
    return Vector3D();
  }

  *wi = 2.0 * woDotH * h - wo;
  if (wi->z <= 0.0) {
    *pdf = 0.0;
    return Vector3D();
  }

  const double D_h = D(h);
  *pdf = D_h * h.z / (4.0 * woDotH);

  return MicrofacetBSDF::f(wo, *wi);
}

void MicrofacetBSDF::render_debugger_node()
{
  if (ImGui::TreeNode(this, "Micofacet BSDF"))
  {
    DragDouble3("eta", &eta[0], 0.005);
    DragDouble3("K", &k[0], 0.005);
    DragDouble("alpha", &alpha, 0.005);
    ImGui::TreePop();
  }
}

BSDFPreset MicrofacetBSDF::get_preset() const {
  BSDFPreset preset;
  preset.type = BSDF_PRESET_MICROFACET;
  preset.vector_a = eta;
  preset.vector_b = k;
  preset.scalar_a = alpha;
  return preset;
}

void MicrofacetBSDF::apply_preset(const BSDFPreset& preset) {
  if (preset.type != BSDF_PRESET_MICROFACET) return;
  eta = preset.vector_a;
  k = preset.vector_b;
  alpha = preset.scalar_a;
}

// Refraction BSDF //

Vector3D RefractionBSDF::f(const Vector3D wo, const Vector3D wi) {
  return Vector3D();
}

Vector3D RefractionBSDF::sample_f(const Vector3D wo, Vector3D* wi, double* pdf) {
  *pdf = 1.0;
  if (!refract(wo, wi, ior)) {
    *pdf = 0.0;
    return Vector3D();
  }

  const double eta = wo.z > 0.0 ? 1.0 / ior : ior;
  return transmittance / (abs_cos_theta(*wi) * eta * eta);
}

void RefractionBSDF::render_debugger_node()
{
  if (ImGui::TreeNode(this, "Refraction BSDF"))
  {
    DragDouble3("Transmittance", &transmittance[0], 0.005);
    DragDouble("ior", &ior, 0.005);
    ImGui::TreePop();
  }
}

BSDFPreset RefractionBSDF::get_preset() const {
  BSDFPreset preset;
  preset.type = BSDF_PRESET_REFRACTION;
  preset.vector_a = transmittance;
  preset.scalar_a = roughness;
  preset.scalar_b = ior;
  return preset;
}

void RefractionBSDF::apply_preset(const BSDFPreset& preset) {
  if (preset.type != BSDF_PRESET_REFRACTION) return;
  transmittance = preset.vector_a;
  roughness = preset.scalar_a;
  ior = preset.scalar_b;
}

// Glass BSDF //

Vector3D GlassBSDF::f(const Vector3D wo, const Vector3D wi) {
  return Vector3D();
}

Vector3D GlassBSDF::sample_f(const Vector3D wo, Vector3D* wi, double* pdf) {
  const double cos_theta_i = abs_cos_theta(wo);
  const double eta_i = wo.z > 0.0 ? 1.0 : ior;
  const double eta_t = wo.z > 0.0 ? ior : 1.0;
  const double eta = eta_i / eta_t;
  const double sin2_theta_t = eta * eta * sin_theta2(wo);

  if (sin2_theta_t >= 1.0) {
    reflect(wo, wi);
    *pdf = 1.0;
    return reflectance / abs_cos_theta(*wi);
  }

  const double cos_theta_t = sqrt(1.0 - sin2_theta_t);
  const double eta_t_cos_i = eta_t * cos_theta_i;
  const double eta_i_cos_t = eta_i * cos_theta_t;
  const double rs = (eta_t_cos_i - eta_i_cos_t) / (eta_t_cos_i + eta_i_cos_t);
  const double rp = (eta_i * cos_theta_i - eta_t * cos_theta_t) /
                    (eta_i * cos_theta_i + eta_t * cos_theta_t);
  const double fr = 0.5 * (rs * rs + rp * rp);

  if (coin_flip(fr)) {
    reflect(wo, wi);
    *pdf = fr;
    return reflectance * fr / abs_cos_theta(*wi);
  }

  *wi = Vector3D(-eta * wo.x, -eta * wo.y, wo.z > 0.0 ? -cos_theta_t : cos_theta_t);
  *pdf = 1.0 - fr;
  return transmittance * (1.0 - fr) / (abs_cos_theta(*wi) * eta * eta);
}

void GlassBSDF::render_debugger_node()
{
  if (ImGui::TreeNode(this, "Refraction BSDF"))
  {
    DragDouble3("Reflectance", &reflectance[0], 0.005);
    DragDouble3("Transmittance", &transmittance[0], 0.005);
    DragDouble("ior", &ior, 0.005);
    ImGui::TreePop();
  }
}

BSDFPreset GlassBSDF::get_preset() const {
  BSDFPreset preset;
  preset.type = BSDF_PRESET_GLASS;
  preset.vector_a = transmittance;
  preset.vector_b = reflectance;
  preset.scalar_a = roughness;
  preset.scalar_b = ior;
  return preset;
}

void GlassBSDF::apply_preset(const BSDFPreset& preset) {
  if (preset.type != BSDF_PRESET_GLASS) return;
  transmittance = preset.vector_a;
  reflectance = preset.vector_b;
  roughness = preset.scalar_a;
  ior = preset.scalar_b;
}

void BSDF::reflect(const Vector3D wo, Vector3D* wi) {
  *wi = Vector3D(-wo.x, -wo.y, wo.z);
}

bool BSDF::refract(const Vector3D wo, Vector3D* wi, double ior) {
  const double eta = wo.z > 0.0 ? 1.0 / ior : ior;
  const double sin2_theta_t = eta * eta * sin_theta2(wo);
  if (sin2_theta_t >= 1.0) return false;

  const double cos_theta_t = sqrt(1.0 - sin2_theta_t);
  *wi = Vector3D(-eta * wo.x, -eta * wo.y, wo.z > 0.0 ? -cos_theta_t : cos_theta_t);
  return true;
}

} // namespace CGL
