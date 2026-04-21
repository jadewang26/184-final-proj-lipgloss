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
    // Sample from gloss (specular) layer using cosine-weighted perturbation around reflection
    Vector3D reflected_dir;
    reflect(wo, &reflected_dir);
    
    if (roughness > 1e-5) {
      // Sample perturbation from cosine-weighted hemisphere
      double perturb_pdf;
      Vector3D perturbation = sampler.get_sample(&perturb_pdf);
      
      // Blend between reflection and perturbed direction based on roughness
      // Higher roughness = more diffuse, lower roughness = more mirror-like
      double blend_factor = min(roughness, 1.0);
      *wi = (reflected_dir * (1.0 - blend_factor) + perturbation * blend_factor);
      wi->normalize();
      
      // PDF: (thickness) × (blended cosine-weighted distribution)
      // When blend_factor = 0 (mirror), PDF approaches delta; when = 1 (diffuse), PDF = cos/PI
      *pdf = thickness * (perturb_pdf * blend_factor + (1.0 - blend_factor) * 0.25);
    } else {
      // Perfect mirror reflection - use a small but reasonable PDF
      *wi = reflected_dir;
      *pdf = thickness * 0.25; // Smooth minimum PDF to avoid singularities
    }
    
  } else {
    // Sample from base (diffuse) layer using cosine-weighted hemisphere
    double base_pdf;
    base_layer->sample_f(wo, wi, &base_pdf);
    
    // PDF: (1.0 - thickness) × base_layer_pdf
    *pdf = (1.0 - thickness) * base_pdf;
  }
  
  // Evaluate BSDF at sampled direction
  Vector3D result = f(wo, *wi);
  
  // Importance sampling: return result / pdf
  // Clamp the final result to prevent fireflies
  if (*pdf > 1e-10) {
    result = result / *pdf;
    // Clamp each channel to prevent extreme values
    result.x = min(result.x, 10.0);
    result.y = min(result.y, 10.0);
    result.z = min(result.z, 10.0);
    return result;
  } else {
    return Vector3D(0, 0, 0);
  }
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

  // blend based on thickness parameter
  return saturated_base * (1.0 - thickness) + gloss * thickness;
}

/**
 * sample fast layered BSDF.
 */
Vector3D FastLayeredBSDF::sample_f(const Vector3D wo, Vector3D* wi, double* pdf) {
  // randomly choose between base and gloss layer based on thickness
  double random_sample = random_uniform();
  
  if (random_sample < thickness) {
    // simpler sampling strategy for performance using a direct function of roughness.
    Vector3D reflected_dir;
    reflect(wo, &reflected_dir);
    
    if (roughness > 1e-5) {
      double perturb_pdf;
      Vector3D perturbation = sampler.get_sample(&perturb_pdf);
      
      // blend factor scaled by roughness
      double blend_factor = min(roughness, 1.0);
      *wi = (reflected_dir * (1.0 - blend_factor) + perturbation * blend_factor);
      wi->normalize();
      
      *pdf = thickness * (perturb_pdf * blend_factor + (1.0 - blend_factor) * 0.25);
    } else {
      *wi = reflected_dir;
      *pdf = thickness * 0.25; 
    }
    
  } else {
    // sample from base (diffuse) layer
    double base_pdf;
    base_layer->sample_f(wo, wi, &base_pdf);
    *pdf = (1.0 - thickness) * base_pdf;
  }
  
  // evaluate BSDF at sampled direction
  Vector3D result = f(wo, *wi);
  
  // importance sampling: return result / pdf
  if (*pdf > 1e-10) {
    result = result / *pdf;
    result.x = min(result.x, 10.0);
    result.y = min(result.y, 10.0);
    result.z = min(result.z, 10.0);
    return result;
  } else {
    return Vector3D(0, 0, 0);
  }
}

void FastLayeredBSDF::render_debugger_node()
{
  if (ImGui::TreeNode(this, "Fast Layered BSDF"))
  {
    DragDouble("Roughness", &roughness, 0.005);
    DragDouble("Thickness", &thickness, 0.005);
    DragDouble3("Base Color", &base_color[0], 0.005);
    DragDouble("Saturation", &saturation, 0.005);
    DragDouble("IOR", &ior, 0.005);
    ImGui::TreePop();
  }
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
  double fss90 = roughness * cos_theta_d * cos_theta_d;
  
  auto schlick_weight = [](double cos_theta) {
    double m = clamp(1.0 - cos_theta, 0.0, 1.0);
    return pow(m, 5.0);
  };

  double fss_in = 1.0 + (fss90 - 1.0) * schlick_weight(cos_theta_i);
  double fss_out = 1.0 + (fss90 - 1.0) * schlick_weight(cos_theta_o);
  
  // the subsurface blend
  double ss_factor = 1.25 * (fss_in * fss_out * (1.0 / (cos_theta_i + cos_theta_o + 0.05) - 0.5) + 0.5);
  
  // mix in some dark red/pink for the subsurface bleed color
  Vector3D subsurface_bleed = base_color * 0.8 + Vector3D(0.8, 0.1, 0.1) * 0.2;
  Vector3D flesh_contrib = (subsurface_bleed / PI) * ss_factor * clamp(saturation, 0.0, 1.5);

  // THE WET LAYER (Broad Gloss)
  
  double wet_roughness = 0.15; // broad, smooth moisture
  double alpha2_wet = max(wet_roughness * wet_roughness, 0.001);
  
  double water_ior = 1.48; // cosmetic lip gloss IOR
  double R0_wet = pow((1.0 - water_ior) / (1.0 + water_ior), 2.0);
  double F_wet = R0_wet + (1.0 - R0_wet) * schlick_weight(cos_theta_d);

  double cos_theta_h = max(h.z, 0.0);
  double cos2_theta_h = cos_theta_h * cos_theta_h;
  
  // GGX Math for broad wet layer
  double D_denom_wet = PI * pow(cos2_theta_h * (alpha2_wet - 1.0) + 1.0, 2.0);
  double D_wet = (D_denom_wet > 0.0) ? (alpha2_wet / D_denom_wet) : 0.0;

  auto G1 = [](double cos_theta, double a2) {
    double cos2_theta = cos_theta * cos_theta;
    return (2.0 * cos_theta) / (cos_theta + sqrt(a2 + (1.0 - a2) * cos2_theta));
  };
  double G_wet = G1(cos_theta_o, alpha2_wet) * G1(cos_theta_i, alpha2_wet);

  double wet_val = (D_wet * F_wet * G_wet) / (4.0 * max(cos_theta_i * cos_theta_o, 0.001));

  // THE SPARKLE LAYER (Micro-glitter flakes)
  
  double sparkle_roughness = 0.04; // extremely sharp to simulate tiny glitter flakes
  double alpha2_sparkle = max(sparkle_roughness * sparkle_roughness, 0.0001);
  
  // glitter reflects almost all light (acting more like a metal)
  double F_sparkle = 0.8 + (1.0 - 0.8) * schlick_weight(cos_theta_d); 
  
  double D_denom_spk = PI * pow(cos2_theta_h * (alpha2_sparkle - 1.0) + 1.0, 2.0);
  double D_sparkle = (D_denom_spk > 0.0) ? (alpha2_sparkle / D_denom_spk) : 0.0;
  double G_sparkle = G1(cos_theta_o, alpha2_sparkle) * G1(cos_theta_i, alpha2_sparkle);

  double sparkle_val = (D_sparkle * F_sparkle * G_sparkle) / (4.0 * max(cos_theta_i * cos_theta_o, 0.001));


  // ENERGY CONSERVATION & BLEND
  
  // clamp to a high HDR value (like 50.0) instead of 1.0. 
  // this allows the center of the highlight to be blindingly bright (sparkly)
  // without breaking the renderer with literal infinity
  double combined_specular = min(wet_val + (sparkle_val * 0.5), 20.0);
  Vector3D wet_contrib(combined_specular, combined_specular, combined_specular);

  // subtract the broad Fresnel from the flesh so energy is conserved
  Vector3D final_flesh = flesh_contrib * (1.0 - F_wet);

  return final_flesh * (1.0 - thickness) + (final_flesh + wet_contrib) * thickness;

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

Vector3D DisneyLayeredBSDF::sample_f(const Vector3D wo, Vector3D* wi, double* pdf) {
  // if the ray is coming from inside the geometry, catch it early
  if (wo.z <= 0.0) return Vector3D(0, 0, 0);

  /* double random_sample = random_uniform();
  
  // lock the wet roughness to match what we hardcoded in the f() function
  double wet_roughness = 0.15; 

  if (random_sample < thickness) {
    // SAMPLE THE WET LAYER (Specular)
    
    // find perfect mirror reflection direction
    Vector3D reflected_dir;
    reflect(wo, &reflected_dir);
    
    // get random perturbation based on a cosine-weighted hemisphere
    double perturb_pdf;
    Vector3D perturbation = sampler.get_sample(&perturb_pdf);
    
    // blend perfect reflection with the perturbation based on our wet roughness
    // this creates a tight specular lobe perfect for saliva/gloss
    *wi = (reflected_dir * (1.0 - wet_roughness) + perturbation * wet_roughness);
    wi->normalize();
    
    // calculate the probability density of picking this exact direction
    *pdf = thickness * (perturb_pdf * wet_roughness + (1.0 - wet_roughness) * 0.25);
    
  } else {
    // SAMPLE THE FLESH LAYER (Diffuse/Subsurface)
    
    // the flesh layer spreads light widely, so standard cosine-weighted 
    // hemisphere sampling is the mathematically correct choice here
    double base_pdf;
    *wi = sampler.get_sample(&base_pdf);
    
    // the probability is scaled by the chance we picked the flesh layer
    *pdf = (1.0 - thickness) * base_pdf;
  }
  */
  *wi = sampler.get_sample(pdf);

  // EVALUATE AND RETURN
  
  // using our direction (*wi), evaluate our fancy f() function 
  // (which calculates BOTH layers) at this specific angle
  Vector3D result = f(wo, *wi);
  
  // importance sampling: divide the result by the probability of taking this path
  if (*pdf > 1e-10) {
    result = result / *pdf;
    
    // anti-firefly clamping: prevents single rays from exploding in brightness 
    // if they hit a weird microfacet angle with a tiny PDF
    result.x = min(result.x, 10.0);
    result.y = min(result.y, 10.0);
    result.z = min(result.z, 10.0);
    
    return result;
  } else {
    return Vector3D(0, 0, 0);
  }
}

void DisneyLayeredBSDF::render_debugger_node()
{
  if (ImGui::TreeNode(this, "Fast Layered BSDF"))
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
