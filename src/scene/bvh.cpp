#include "bvh.h"

#include "CGL/CGL.h"
#include "triangle.h"

#include <iostream>
#include <stack>

using namespace std;

namespace CGL {
namespace SceneObjects {

BVHAccel::BVHAccel(const std::vector<Primitive *> &_primitives,
                   size_t max_leaf_size) {

  primitives = std::vector<Primitive *>(_primitives);
  root = construct_bvh(primitives.begin(), primitives.end(), max_leaf_size);
}

BVHAccel::~BVHAccel() {
  if (root)
    delete root;
  primitives.clear();
}

BBox BVHAccel::get_bbox() const { return root->bb; }

void BVHAccel::draw(BVHNode *node, const Color &c, float alpha) const {
  if (node->isLeaf()) {
    for (auto p = node->start; p != node->end; p++) {
      (*p)->draw(c, alpha);
    }
  } else {
    draw(node->l, c, alpha);
    draw(node->r, c, alpha);
  }
}

void BVHAccel::drawOutline(BVHNode *node, const Color &c, float alpha) const {
  if (node->isLeaf()) {
    for (auto p = node->start; p != node->end; p++) {
      (*p)->drawOutline(c, alpha);
    }
  } else {
    drawOutline(node->l, c, alpha);
    drawOutline(node->r, c, alpha);
  }
}

BVHNode *BVHAccel::construct_bvh(std::vector<Primitive *>::iterator start,
                                 std::vector<Primitive *>::iterator end,
                                 size_t max_leaf_size) {

  // TODO (Part 2.1):
  // Construct a BVH from the given vector of primitives and maximum leaf
  // size configuration. The starter code build a BVH aggregate with a
  // single leaf node (which is also the root) that encloses all the
  // primitives.

  BBox bounding_volume;
  for (auto current_primitive = start; current_primitive != end; current_primitive++) {
    bounding_volume.expand((*current_primitive)->get_bbox());
  }
  BVHNode *tree_node = new BVHNode(bounding_volume);
  int primitive_count = end - start;

  if (primitive_count <= max_leaf_size) {
    std::vector<Primitive *> *leaf_primitives = new std::vector<Primitive *>();
    for (auto current_primitive = start; current_primitive != end; current_primitive++) {
      leaf_primitives->push_back(*current_primitive);
    }

    tree_node->start = leaf_primitives->begin();
    tree_node->end = leaf_primitives->end();
    tree_node->l = NULL;
    tree_node->r = NULL;
    tree_node->bb = bounding_volume;
    return tree_node;
  }
  
  Vector3D volume_extents = bounding_volume.extent;
  int split_dimension = 0;
  double max_extent_value = volume_extents.x;
  
  if (volume_extents.y > max_extent_value) {
    split_dimension = 1;
    max_extent_value = volume_extents.y;
  }
  if (volume_extents.z > max_extent_value) {
    split_dimension = 2;
  }
  
  Vector3D volume_centroid = bounding_volume.centroid();
  std::vector<Primitive *> left_partition;
  std::vector<Primitive *> right_partition;
  
  for (auto current_primitive = start; current_primitive != end; current_primitive++) {
    BBox primitive_bounding_box = (*current_primitive)->get_bbox();
    Vector3D primitive_center = primitive_bounding_box.centroid();
    
    double primitive_coordinate = (split_dimension == 0) ? primitive_center.x :
                                   (split_dimension == 1) ? primitive_center.y : primitive_center.z;
    double split_coordinate = (split_dimension == 0) ? volume_centroid.x :
                              (split_dimension == 1) ? volume_centroid.y : volume_centroid.z;
    
    if (primitive_coordinate < split_coordinate) {
      left_partition.push_back(*current_primitive);
    } else {
      right_partition.push_back(*current_primitive);
    }
  }

  if (left_partition.empty() || right_partition.empty()) {
    left_partition.clear();
    right_partition.clear();
    int midpoint_index = primitive_count / 2;
    for (auto current_primitive = start; current_primitive != end; current_primitive++) {
      if (current_primitive - start < midpoint_index) {
        left_partition.push_back(*current_primitive);
      } else {
        right_partition.push_back(*current_primitive);
      }
    }
  }
  
  tree_node->l = construct_bvh(left_partition.begin(), left_partition.end(), max_leaf_size);
  tree_node->r = construct_bvh(right_partition.begin(), right_partition.end(), max_leaf_size);
  tree_node->bb = bounding_volume;

  return tree_node;

}

bool BVHAccel::has_intersection(const Ray &ray, BVHNode *node) const {
  // TODO (Part 2.3):
  // Fill in the intersect function.
  // Take note that this function has a short-circuit that the
  // Intersection version cannot, since it returns as soon as it finds
  // a hit, it doesn't actually have to find the closest hit.

  double slab_entry_time = ray.min_t;
  double slab_exit_time = ray.max_t;
  
  if (!node->bb.intersect(ray, slab_entry_time, slab_exit_time)) {
    return false;
  }
  if (node->isLeaf()) {
    for (auto primitive_ptr = node->start; primitive_ptr != node->end; primitive_ptr++) {
      if ((*primitive_ptr)->has_intersection(ray)) {
        return true;
      }
    }
    return false;
  }
  if (has_intersection(ray, node->l)) {
    return true;
  }
  return has_intersection(ray, node->r);


}

bool BVHAccel::intersect(const Ray &ray, Intersection *i, BVHNode *node) const {
  // TODO (Part 2.3):
  // Fill in the intersect function.

  double intersection_t_min = ray.min_t;
  double intersection_t_max = ray.max_t;
  
  if (!node->bb.intersect(ray, intersection_t_min, intersection_t_max)) {
    return false;
  }
  
  Intersection current_intersection;
  bool any_hit_found = false;

  if (node->isLeaf()) {
    for (auto primitive_iterator = node->start; primitive_iterator != node->end; primitive_iterator++) {
      if ((*primitive_iterator)->intersect(ray, &current_intersection)) {
        if (current_intersection.t < i->t) {
          *i = current_intersection;
          any_hit_found = true;
        }
      }
    }
    return any_hit_found;
  }
  if (intersect(ray, &current_intersection, node->l)) {
    if (current_intersection.t < i->t) {
      *i = current_intersection;
      any_hit_found = true;
    }
  }
  if (intersect(ray, &current_intersection, node->r)) {
    if (current_intersection.t < i->t) {
      *i = current_intersection;
      any_hit_found = true;
    }
  }
  return any_hit_found;
}

} // namespace SceneObjects
} // namespace CGL
