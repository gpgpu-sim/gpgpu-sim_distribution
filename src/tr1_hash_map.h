#pragma once 

// detection and fallback for unordered_map in C++0x
#ifdef __cplusplus
   #ifdef __GNUC__
   #if __GNUC__ >= 4 && __GNUC_MINOR__ >= 3
      #include <unordered_map>
      #define tr1_hash_map std::unordered_map
      #define tr1_hash_map_ismap 0
   #else
      #include <map>
      #define tr1_hash_map std::map
      #define tr1_hash_map_ismap 1
   #endif
   #else
      #include <map>
      #define tr1_hash_map std::map
      #define tr1_hash_map_ismap 1
   #endif
#endif


