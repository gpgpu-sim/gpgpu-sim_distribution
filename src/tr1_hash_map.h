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

    // detect gcc 4.3 and use unordered map (part of c++0x)
    // unordered map doesn't play nice with _GLIBCXX_DEBUG, just use a map if its enabled.
    #if  defined( __GNUC__ ) and not defined( _GLIBCXX_DEBUG )
        #if __GNUC__ >= 4 && __GNUC_MINOR__ >= 3
           #include <unordered_map>
           #define my_hash_map std::unordered_map
        #else
           #include <ext/hash_map>
           namespace std {
              using namespace __gnu_cxx;
           }
           #define my_hash_map std::hash_map
        #endif
    #else
       #include <map>
       #define my_hash_map std::map
       #define USE_MAP
    #endif

#endif
