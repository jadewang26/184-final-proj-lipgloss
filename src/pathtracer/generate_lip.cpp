
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <string>

// vector math, just what we need for the parametric surface and normals
struct Vec3 {
  double x, y, z;
  Vec3(double x = 0, double y = 0, double z = 0) : x(x), y(y), z(z) {}
  Vec3 operator+(const Vec3& o) const { return Vec3(x+o.x, y+o.y, z+o.z); }
  Vec3 operator-(const Vec3& o) const { return Vec3(x-o.x, y-o.y, z-o.z); }
  Vec3 operator*(double t)      const { return Vec3(x*t, y*t, z*t); }
};

static Vec3 cross(Vec3 a, Vec3 b) {
  return Vec3(a.y*b.z - a.z*b.y,
              a.z*b.x - a.x*b.z,
              a.x*b.y - a.y*b.x);
}
static Vec3 normalize(Vec3 v) {
  double l = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
  if (l < 1e-12) return Vec3(0, 0, 1);
  return Vec3(v.x/l, v.y/l, v.z/l);
}
static double clamp01(double x) {
  return std::max(0.0, std::min(1.0, x));
}

static double mix(double a, double b, double t) {
  return a + (b - a) * t;
}

static double gaussian(double x, double sigma) {
  return std::exp(-(x * x) / (2.0 * sigma * sigma));
}

static double smootherstep(double x) {
  x = clamp01(x);
  return x * x * x * (x * (x * 6.0 - 15.0) + 10.0);
}

// The procedural lip surface is built as three patches: upper lip, lower lip,
// and a narrow recessed mouth crease. World axes match the CS184 Collada
// scenes: +X right, +Y up, +Z toward the camera.

typedef Vec3 (*LipFunc)(double, double);

static double taper(double u) {
  return std::max(0.0, 1.0 - u * u);
}

static double mouth_line_y(double u) {
  double s = taper(u);
  double corner_drop = -0.055 * std::pow(std::abs(u), 4.0);
  double soft_smile = -0.020 * std::pow(s, 0.85);
  return soft_smile + corner_drop - 0.008 * u;
}

static double lip_x(double u) {
  double corner_pinching = 0.018 * std::sin(2.0 * M_PI * u);
  return 0.86 * (u + corner_pinching);
}

static double vertical_folds(double u, double v, double amount) {
  double s = taper(u);
  double border_fade = smootherstep(1.0 - v);
  double envelope = std::pow(s, 0.95) * std::pow(std::sin(M_PI * clamp01(v)), 0.70) * border_fade;
  double fine = std::sin(21.0 * M_PI * (u + 0.013));
  double mid = std::sin(13.0 * M_PI * (u - 0.071));
  double broad = std::sin(7.0 * M_PI * (u + 0.11));
  double grooves = -0.5 * std::abs(fine) - 0.32 * std::abs(mid) + 0.18 * broad;
  return amount * envelope * grooves;
}

// u in [-1, 1] : left corner -> right corner of the mouth
// v in [ 0, 1] : mouth line  -> upper vermilion border
static Vec3 upper_lip(double u, double v) {
  double s = taper(u);
  double side = std::abs(u);
  double t = smootherstep(v);

  double y_mouth = mouth_line_y(u);
  double cupid_dip = 0.070 * gaussian(u, 0.120);
  double bow_peaks = 0.045 * (gaussian(u - 0.28, 0.16) + gaussian(u + 0.28, 0.16));
  double arch = 0.335 * std::pow(s, 0.62);
  double y_border = arch + bow_peaks - cupid_dip - 0.055 * std::pow(side, 4.0) - 0.010 * u;
  double y = mix(y_mouth, y_border, t);

  double roll = std::pow(std::sin(M_PI * clamp01(v)), 0.78);
  double central_swell = 0.160 * std::pow(s, 0.52) * roll;
  double bow_indent = -0.035 * gaussian(u, 0.15) * gaussian(v - 0.83, 0.16);
  double seam_recess = -0.052 * std::pow(s, 0.70) * gaussian(v, 0.10);
  double border_recess = -0.030 * std::pow(s, 0.85) * gaussian(v - 1.0, 0.18);
  double corner_recess = -0.060 * std::pow(side, 3.0);
  double z = central_swell + bow_indent + seam_recess + border_recess + corner_recess;

  // Tiny anatomical asymmetry helps the surface avoid a CG-perfect mirror look.
  z += 0.010 * std::sin(5.0 * M_PI * (u + 0.08)) * std::pow(s, 1.2) * roll;
  z += vertical_folds(u, v, 0.006);

  return Vec3(lip_x(u), y, z);
}

// u in [-1, 1] : left corner -> right corner of the mouth
// v in [ 0, 1] : mouth line  -> lower vermilion border
static Vec3 lower_lip(double u, double v) {
  double s = taper(u);
  double side = std::abs(u);
  double t = smootherstep(v);

  double y_mouth = mouth_line_y(u);
  double lower_drop = -0.405 * std::pow(s, 0.72) - 0.030 * gaussian(u, 0.34);
  double y_border = lower_drop - 0.052 * std::pow(side, 4.0) - 0.004 * u;
  double y = mix(y_mouth, y_border, t);

  double roll = std::pow(std::sin(M_PI * clamp01(v)), 0.60);
  double central_pillow = 0.265 * std::pow(s, 0.43) * roll;
  double lower_lobe = 0.065 * gaussian(u, 0.42) * gaussian(v - 0.58, 0.22);
  double seam_recess = -0.064 * std::pow(s, 0.70) * gaussian(v, 0.095);
  double border_recess = -0.040 * std::pow(s, 0.90) * gaussian(v - 1.0, 0.20);
  double corner_recess = -0.068 * std::pow(side, 3.0);
  double z = central_pillow + lower_lobe + seam_recess + border_recess + corner_recess;

  z += 0.012 * std::sin(4.0 * M_PI * (u - 0.04)) * std::pow(s, 1.1) * roll;
  z += vertical_folds(u, v, 0.008);

  return Vec3(lip_x(u), y, z);
}

// A narrow, recessed crease makes the closed mouth read as depth instead of a
// bent lower-lip patch.
static Vec3 mouth_crease(double u, double v) {
  double s = taper(u);
  double center_shadow = gaussian(u, 0.46);
  double y = mouth_line_y(u) + (v - 0.5) * 0.026 * std::pow(s, 0.70);
  double z = -0.042 - 0.072 * std::pow(s, 0.58) - 0.020 * std::pow(std::abs(u), 2.0)
             - 0.030 * center_shadow;
  return Vec3(lip_x(u), y, z);
}

static Vec3 surface_normal(LipFunc f, double u, double v) {
  const double eps = 1e-3;
  double up = std::min( 0.999, u + eps);
  double um = std::max(-0.999, u - eps);
  double vp = std::min( 1.0,   v + eps);
  double vm = std::max( 0.0,   v - eps);

  Vec3 du = f(up, v) - f(um, v);
  Vec3 dv = f(u, vp) - f(u, vm);
  Vec3 n  = cross(du, dv);

  if (n.z < 0) n = n * -1.0;   
  return normalize(n);
}

//  tessellation 
// Samples (nu+1) x (nv+1) vertices and emits 2 * nu * nv triangles.
// CCW winding order when viewed from +Z, so the visible side is the
// camera-facing side.
static void tesselate(LipFunc f,
                      std::vector<Vec3>& verts,
                      std::vector<Vec3>& norms,
                      std::vector<int>&  tris,
                      int nu, int nv,
                      bool flip_winding = false) {
  int base = (int)verts.size();

  const double u_min = -0.99, u_max = 0.99;

  for (int i = 0; i <= nu; i++) {
    double u = u_min + (u_max - u_min) * double(i) / nu;
    for (int j = 0; j <= nv; j++) {
      double v = double(j) / nv;
      verts.push_back(f(u, v));
      norms.push_back(surface_normal(f, u, v));
    }
  }

  const int stride = nv + 1;
  for (int i = 0; i < nu; i++) {
    for (int j = 0; j < nv; j++) {
      int p00 = base + (i  ) * stride + (j  );
      int p01 = base + (i  ) * stride + (j+1);
      int p10 = base + (i+1) * stride + (j  );
      int p11 = base + (i+1) * stride + (j+1);
      if (flip_winding) {
        tris.push_back(p00); tris.push_back(p01); tris.push_back(p10);
        tris.push_back(p01); tris.push_back(p11); tris.push_back(p10);
      } else {
        tris.push_back(p00); tris.push_back(p10); tris.push_back(p01);
        tris.push_back(p01); tris.push_back(p10); tris.push_back(p11);
      }
    }
  }
}

static void write_lip_geometry(std::ofstream& out,
                               const std::string& id,
                               const std::string& name,
                               const std::string& material_id,
                               const std::vector<Vec3>& verts,
                               const std::vector<Vec3>& norms,
                               const std::vector<int>& tris) {
  out << "    <geometry id=\"" << id << "\" name=\"" << name << "\">\n"
      << "      <mesh>\n";

  out << "        <source id=\"" << id << "-positions\">\n"
      << "          <float_array id=\"" << id << "-positions-array\" count=\""
      << verts.size() * 3 << "\">";
  for (const Vec3& v : verts) out << " " << v.x << " " << v.y << " " << v.z;
  out << "</float_array>\n"
      << "          <technique_common><accessor source=\"#" << id
      << "-positions-array\" count=\"" << verts.size() << "\" stride=\"3\">\n"
      << "            <param name=\"X\" type=\"float\"/>"
         "<param name=\"Y\" type=\"float\"/>"
         "<param name=\"Z\" type=\"float\"/>\n"
      << "          </accessor></technique_common>\n"
      << "        </source>\n";

  out << "        <source id=\"" << id << "-normals\">\n"
      << "          <float_array id=\"" << id << "-normals-array\" count=\""
      << norms.size() * 3 << "\">";
  for (const Vec3& n : norms) out << " " << n.x << " " << n.y << " " << n.z;
  out << "</float_array>\n"
      << "          <technique_common><accessor source=\"#" << id
      << "-normals-array\" count=\"" << norms.size() << "\" stride=\"3\">\n"
      << "            <param name=\"X\" type=\"float\"/>"
         "<param name=\"Y\" type=\"float\"/>"
         "<param name=\"Z\" type=\"float\"/>\n"
      << "          </accessor></technique_common>\n"
      << "        </source>\n";

  out << "        <vertices id=\"" << id << "-vertices\">\n"
      << "          <input semantic=\"POSITION\" source=\"#" << id << "-positions\"/>\n"
      << "        </vertices>\n";

  out << "        <triangles count=\"" << tris.size() / 3
      << "\" material=\"" << material_id << "\">\n"
      << "          <input semantic=\"VERTEX\" source=\"#" << id
      << "-vertices\" offset=\"0\"/>\n"
      << "          <input semantic=\"NORMAL\" source=\"#" << id
      << "-normals\"  offset=\"1\"/>\n"
      << "          <p>";
  for (int idx : tris) out << " " << idx << " " << idx;
  out << "</p>\n"
      << "        </triangles>\n"
      << "      </mesh>\n"
      << "    </geometry>\n";
}

// the  main 
int main(int argc, char** argv) {
  std::string outpath = "dae/final-project/lips1.dae";
  if (argc > 1) outpath = argv[1];

  std::vector<Vec3> verts, norms;
  std::vector<int>  tris;

  const int NU = 96;   // samples across the width
  const int NV = 36;   // samples from mouth line to vermilion border
  tesselate(upper_lip, verts, norms, tris, NU, NV);
  tesselate(lower_lip, verts, norms, tris, NU, NV, true);
  tesselate(mouth_crease, verts, norms, tris, NU, 4);

  std::ofstream out(outpath);
  if (!out) {
    std::cerr << "ERROR: could not open " << outpath << " for writing\n";
    return 1;
  }
  out << std::fixed << std::setprecision(6);

  // scene header, camera, two lights, lip-gloss material 
  out <<
R"(<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
    <unit name="meter" meter="1"/>
    <up_axis>Y_UP</up_axis>
  </asset>
  <library_cameras>
    <camera id="Camera-camera" name="Camera">
      <optics><technique_common><perspective>
        <xfov sid="xfov">66</xfov>
        <aspect_ratio>1.333333</aspect_ratio>
        <znear sid="znear">0.1</znear>
        <zfar  sid="zfar">100</zfar>
      </perspective></technique_common></optics>
    </camera>
  </library_cameras>
  <library_lights>
    <light id="KeyLight-light" name="KeyLight">
      <technique_common><point>
        <color sid="color">18 18 18</color>
        <constant_attenuation>1</constant_attenuation>
        <linear_attenuation>0</linear_attenuation>
        <quadratic_attenuation>0.25</quadratic_attenuation>
      </point></technique_common>
    </light>
    <light id="FillLight-light" name="FillLight">
      <technique_common><point>
        <color sid="color">6 6 7</color>
        <constant_attenuation>1</constant_attenuation>
        <linear_attenuation>0</linear_attenuation>
        <quadratic_attenuation>0.25</quadratic_attenuation>
      </point></technique_common>
    </light>
  </library_lights>
  <library_effects>
    <effect id="lips-effect">
      <profile_COMMON>
        <technique sid="common">
          <phong>
            <emission><color sid="emission">0 0 0 1</color></emission>
            <ambient ><color sid="ambient" >0 0 0 1</color></ambient>
            <diffuse ><color sid="diffuse" >0.58 0.075 0.105 1</color></diffuse>
            <specular><color sid="specular">0.16 0.13 0.13 1</color></specular>
            <shininess><float sid="shininess">3</float></shininess>
            <index_of_refraction><float sid="index_of_refraction">1</float></index_of_refraction>
          </phong>
        </technique>
      </profile_COMMON>
      <extra>
        <technique profile="CGL">
          <layered>
            <roughness>0.055</roughness>
            <thickness>0.50</thickness>
            <base_color>0.62 0.08 0.12</base_color>
            <saturation>1.05</saturation>
            <ior>1.48</ior>
          </layered>
        </technique>
      </extra>
    </effect>
    <effect id="glossy-lips-effect">
      <profile_COMMON>
        <technique sid="common">
          <phong>
            <emission><color sid="emission">0 0 0 1</color></emission>
            <ambient ><color sid="ambient" >0 0 0 1</color></ambient>
            <diffuse ><color sid="diffuse" >0.66 0.095 0.135 1</color></diffuse>
            <specular><color sid="specular">0.28 0.24 0.24 1</color></specular>
            <shininess><float sid="shininess">7</float></shininess>
            <index_of_refraction><float sid="index_of_refraction">1</float></index_of_refraction>
          </phong>
        </technique>
      </profile_COMMON>
      <extra>
        <technique profile="CGL">
          <layered>
            <roughness>0.018</roughness>
            <thickness>0.82</thickness>
            <base_color>0.70 0.10 0.15</base_color>
            <saturation>1.18</saturation>
            <ior>1.52</ior>
          </layered>
        </technique>
      </extra>
    </effect>
    <effect id="matte-lips-effect">
      <profile_COMMON>
        <technique sid="common">
          <phong>
            <emission><color sid="emission">0 0 0 1</color></emission>
            <ambient ><color sid="ambient" >0 0 0 1</color></ambient>
            <diffuse ><color sid="diffuse" >0.46 0.060 0.085 1</color></diffuse>
            <specular><color sid="specular">0.025 0.018 0.018 1</color></specular>
            <shininess><float sid="shininess">0.35</float></shininess>
            <index_of_refraction><float sid="index_of_refraction">1</float></index_of_refraction>
          </phong>
        </technique>
      </profile_COMMON>
      <extra>
        <technique profile="CGL">
          <layered>
            <roughness>0.65</roughness>
            <thickness>0.02</thickness>
            <base_color>0.48 0.065 0.090</base_color>
            <saturation>0.92</saturation>
            <ior>1.35</ior>
          </layered>
        </technique>
      </extra>
    </effect>
  </library_effects>
  <library_materials>
    <material id="lips-material" name="lips_material">
      <instance_effect url="#lips-effect"/>
    </material>
    <material id="glossy-lips-material" name="glossy_lips_material">
      <instance_effect url="#glossy-lips-effect"/>
    </material>
    <material id="matte-lips-material" name="matte_lips_material">
      <instance_effect url="#matte-lips-effect"/>
    </material>
  </library_materials>
  <library_geometries>
)";

  write_lip_geometry(out, "lips-mesh", "lips", "lips-material", verts, norms, tris);
  write_lip_geometry(out, "glossy-lips-mesh", "glossy_lips",
                     "glossy-lips-material", verts, norms, tris);
  write_lip_geometry(out, "matte-lips-mesh", "matte_lips",
                     "matte-lips-material", verts, norms, tris);

  //  scene graph 
  out <<
R"(  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene" name="Scene">
      <node id="Camera" name="Camera" type="NODE">
        <matrix sid="transform">1 0 0 0  0 1 0 -0.04  0 0 1 3.2  0 0 0 1</matrix>
        <instance_camera url="#Camera-camera"/>
      </node>
      <node id="KeyLight" name="KeyLight" type="NODE">
        <matrix sid="transform">1 0 0 0.8  0 1 0 1.5  0 0 1 2.4  0 0 0 1</matrix>
        <instance_light url="#KeyLight-light"/>
      </node>
      <node id="FillLight" name="FillLight" type="NODE">
        <matrix sid="transform">1 0 0 -2.0  0 1 0 0.3  0 0 1 2.2  0 0 0 1</matrix>
        <instance_light url="#FillLight-light"/>
      </node>
      <node id="Lips" name="Lips" type="NODE">
        <matrix sid="transform">1 0 0 -1.75  0 1 0 0  0 0 1 0  0 0 0 1</matrix>
        <instance_geometry url="#lips-mesh">
          <bind_material><technique_common>
            <instance_material symbol="lips-material" target="#lips-material"/>
          </technique_common></bind_material>
        </instance_geometry>
      </node>
      <node id="GlossyLips" name="GlossyLips" type="NODE">
        <matrix sid="transform">1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1</matrix>
        <instance_geometry url="#glossy-lips-mesh">
          <bind_material><technique_common>
            <instance_material symbol="glossy-lips-material" target="#glossy-lips-material"/>
          </technique_common></bind_material>
        </instance_geometry>
      </node>
      <node id="MatteLips" name="MatteLips" type="NODE">
        <matrix sid="transform">1 0 0 1.75  0 1 0 0  0 0 1 0  0 0 0 1</matrix>
        <instance_geometry url="#matte-lips-mesh">
          <bind_material><technique_common>
            <instance_material symbol="matte-lips-material" target="#matte-lips-material"/>
          </technique_common></bind_material>
        </instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>
)";

  out.close();
  std::cout << "Wrote " << outpath << ": "
            << verts.size() * 3 << " verts across 3 lip meshes, "
            << (tris.size()/3) * 3 << " triangles\n";
  return 0;
}
