#ifndef UTILS_H
#define UTILS_H

#include <vector_types.h>
#include <string>
#include <unordered_set>
namespace utils {
inline bool dim3_compare(const dim3& a, const dim3& b) {
  if (a.x != b.x) return a.x < b.x;
  if (a.y != b.y) return a.y < b.y;
  return a.z < b.z;
}

inline bool dim3_equal(const dim3& a, const dim3& b) {
  return !dim3_compare(a, b) && !dim3_compare(b, a);
}

inline std::string dim3_to_string(const dim3& d) {
  return std::to_string(d.x) + " " + std::to_string(d.y) + " " +
         std::to_string(d.z);
}

struct Dim3Equal {
  bool operator()(const dim3& a, const dim3& b) const {
    return dim3_equal(a, b);
  }
};

struct Dim3Hash {
  size_t operator()(const dim3& d) const {
    size_t h1 = std::hash<unsigned>{}(d.x);
    size_t h2 = std::hash<unsigned>{}(d.y);
    size_t h3 = std::hash<unsigned>{}(d.z);
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

struct Dim3Compare {
  bool operator()(const dim3& a, const dim3& b) const {
    return dim3_compare(a, b);
  }
};

using Dim3Set = std::unordered_set<dim3, Dim3Hash, Dim3Equal>;
}  // namespace utils
#endif  // !UTILS_H