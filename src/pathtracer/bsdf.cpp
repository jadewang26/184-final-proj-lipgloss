#include "bsdf.h"
#include "bsdf.h"
#include "bsdf.h"

#include "application/visual_debugger.h"

#include <algorithm>
#include <iostream>
#include <utility>


using std::max;
using std::min;
using std::swap;

namespace CGL {

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

/**
 * Evaluate Approximate BSSRDF.
 * Uses a diffuse-like model with color to approximate subsurface scattering.
 */
Vector3D ApproximateBSSRDF::f(const Vector3D wo, const Vector3D wi) {
  // Approximate BSSRDF using a diffuse Lambertian model
  // The color-based reflectance simulates subsurface scattering
  return skin_color / PI;
}

/**
 * Sample Approximate BSSRDF.
 * Uses cosine-weighted hemisphere sampling with slight color-based modulation.
 */
Vector3D ApproximateBSSRDF::sample_f(const Vector3D wo, Vector3D *wi, double *pdf) {
  // Sample using cosine-weighted hemisphere distribution
  *wi = sampler.get_sample(pdf);
  
  // Return BSDF value at sampled direction
  // Apply slight angle-dependent coloring to simulate subsurface scattering
  double cosine_factor = cos_theta(*wi);
  Vector3D modulated_color = skin_color * (0.8 + 0.2 * cosine_factor);
  
  return modulated_color / PI;
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
  // If either ray is below the surface, no light is reflected
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
  
  Vector3D gloss(ct_val, ct_val, ct_val);

  // Blend based on thickness parameter
  return saturated_base * (1.0 - thickness) + gloss * thickness;
}

/**
 * Sample Layered BSDF.
 * Probabilistically samples from base or gloss layer based on thickness.
 */

Vector3D LayeredBSDF::sample_f(const Vector3D wo, Vector3D* wi, double* pdf) {
  // Randomly choose between base and gloss layer based on thickness
  double random_sample = random_uniform();
  
  if (random_sample < thickness) {
    // Sample from gloss (specular) layer
    Vector3D reflected_dir;
    reflect(wo, &reflected_dir);
    
    if (roughness > 1e-5) {
      // For rough gloss: perturb reflection direction with cosine-weighted hemisphere
      double perturb_pdf;
      Vector3D perturbation = sampler.get_sample(&perturb_pdf);
      
      // Interpolate between pure reflection and diffuse based on roughness
      double alpha = roughness;
      *wi = (reflected_dir * (1.0 - alpha) + perturbation * alpha);
      wi->normalize();
    } else {
      // Perfect mirror reflection
      *wi = reflected_dir;
    }
    
    // PDF: thickness probability * directional probability for gloss layer
    // For simplicity, assume uniform angular distribution around reflection
    double angular_pdf = 1.0 / (2.0 * PI * max(roughness * roughness, 0.01));
    *pdf = thickness * angular_pdf;
    
  } else {
    // Sample from base (diffuse) layer using cosine-weighted hemisphere
    double base_pdf;
    base_layer->sample_f(wo, wi, &base_pdf);
    
    // PDF: (1.0 - thickness) probability * base layer's directional PDF
    *pdf = (1.0 - thickness) * base_pdf;
  }
  
  // Return the blended BSDF value properly normalized
  Vector3D result = f(wo, *wi);
  
  // Avoid division by very small PDF
  if (*pdf < 1e-10) {
    return Vector3D(0, 0, 0);
  }
  
  return result / *pdf;
}

void LayeredBSDF::render_debugger_node()
{
  if (ImGui::TreeNode(this, "Layered BSDF"))
  {
    DragDouble("Roughness", &roughness, 0.005);
    DragDouble("Thickness", &thickness, 0.005);
    DragDouble3("Base Color", &base_color[0], 0.005);
    DragDouble("Saturation", &saturation, 0.005);
    DragDouble("IOR", &ior, 0.005);
    ImGui::TreePop();
  }
}

} // namespace CGL
