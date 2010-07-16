     class Float4 {
     public:
       float x, y, z, w;

       __device__ const Float4
       operator+(const Float4& iv) const {
          Float4 rv ;
          rv.x = x + iv.x ;
          rv.y = y + iv.y ;
          rv.z = z + iv.z ;
          rv.w = w + iv.w ;
          return Float4( rv ) ;
       }
       __device__ const Float4
       operator*(const Float4& iv) const {
          Float4 rv ;
          rv.x = x * iv.x ;
          rv.y = y * iv.y ;
          rv.z = z * iv.z ;
          rv.w = w * iv.w ;
          return Float4( rv ) ;
       }
       __device__ const Float4
       operator/(const Float4& iv) const {
          Float4 rv ;
          rv.x = x / iv.x ;
          rv.y = y / iv.y ;
          rv.z = z / iv.z ;
          rv.w = w / iv.w ;
          return Float4( rv ) ;
       }
       __device__ const Float4
       operator-(const Float4& iv) const {
          Float4 rv ;
          rv.x = x - iv.x ;
          rv.y = y - iv.y ;
          rv.z = z - iv.z ;
          rv.w = w - iv.w ;
          return Float4( rv ) ;
       }

       __device__ const Float4
       operator+(const float iv) const {
          Float4 rv ;
          rv.x = x + iv ;
          rv.y = y + iv ;
          rv.z = z + iv ;
          rv.w = w + iv ;
          return Float4( rv ) ;
       }
       __device__ const Float4
       operator*(const float iv) const {
          Float4 rv ;
          rv.x = x * iv ;
          rv.y = y * iv ;
          rv.z = z * iv ;
          rv.w = w * iv ;
          return Float4( rv ) ;
       }
       __device__ const Float4
       operator/(const float iv) const {
          Float4 rv ;
          rv.x = x / iv ;
          rv.y = y / iv ;
          rv.z = z / iv ;
          rv.w = w / iv ;
          return Float4( rv ) ;
       }
       __device__ const Float4
       operator-(const float iv) const {
          Float4 rv ;
          rv.x = x - iv ;
          rv.y = y - iv ;
          rv.z = z - iv ;
          rv.w = w - iv ;
          return Float4( rv ) ;
       }
       __device__ const Float4
       operator-() const {
          Float4 rv ;
          rv.x = -x  ;
          rv.y = -y  ;
          rv.z = -z  ;
          rv.w = -w  ;
          return Float4( rv ) ;
       }

       __device__ void operator=(const float iv) {
           x = iv ;
           y = iv ;
           z = iv ;
           w = iv ;
       }

       __device__ void operator+=(const Float4 iv) {
           x += iv.x ;
           y += iv.y ;
           z += iv.z ;
           w += iv.w ;
       }
       __device__ void operator-=(const Float4 iv) {
           x -= iv.x ;
           y -= iv.y ;
           z -= iv.z ;
           w -= iv.w ;
       }

     };

     __device__ const Float4
     operator+( const float iv1, const Float4 iv2 ) {
          Float4 rv ;
          rv.x = iv1 + iv2.x ;
          rv.y = iv1 + iv2.y ;
          rv.z = iv1 + iv2.z ;
          rv.w = iv1 + iv2.w ;
          return Float4( rv ) ;
     }
     __device__ const Float4
     operator*( const float iv1, const Float4 iv2 ) {
          Float4 rv ;
          rv.x = iv1 * iv2.x ;
          rv.y = iv1 * iv2.y ;
          rv.z = iv1 * iv2.z ;
          rv.w = iv1 * iv2.w ;
          return Float4( rv ) ;
     }
     __device__ const Float4
     operator/( const float iv1, const Float4 iv2 ) {
          Float4 rv ;
          rv.x = iv1 / iv2.x ;
          rv.y = iv1 / iv2.y ;
          rv.z = iv1 / iv2.z ;
          rv.w = iv1 / iv2.w ;
          return Float4( rv ) ;
     }
     __device__ const Float4
     operator-( const float iv1, const Float4 iv2 ) {
          Float4 rv ;
          rv.x = iv1 - iv2.x ;
          rv.y = iv1 - iv2.y ;
          rv.z = iv1 - iv2.z ;
          rv.w = iv1 - iv2.w ;
          return Float4( rv ) ;
     }

__device__ Float4 max ( const Float4 a , const Float4 b )
{
    Float4 c  ;
    c.x = (a.x>b.x)?a.x:b.x;
    c.y = (a.y>b.y)?a.y:b.y;
    c.z = (a.z>b.z)?a.z:b.z;
    c.w = (a.w>b.w)?a.w:b.w;
    return(c) ;
}
__device__ Float4 max ( const float a , const Float4 b )
{
    Float4 c  ;
    c.x = (a>b.x)?a:b.x;
    c.y = (a>b.y)?a:b.y;
    c.z = (a>b.z)?a:b.z;
    c.w = (a>b.w)?a:b.w;
    return(c) ;
}
__device__ Float4 max ( const Float4 a , const float b )
{
    Float4 c  ;
    c.x = (a.x>b)?a.x:b;
    c.y = (a.y>b)?a.y:b;
    c.z = (a.z>b)?a.z:b;
    c.w = (a.w>b)?a.w:b;
    return(c) ;
}
//__device__ float max ( const float a , const float b )
//{
//    return(a>b)?a:b) ;
//}

__device__ Float4 min ( const Float4 a , const Float4 b )
{
    Float4 c  ;
    c.x = (a.x<b.x)?a.x:b.x;
    c.y = (a.y<b.y)?a.y:b.y;
    c.z = (a.z<b.z)?a.z:b.z;
    c.w = (a.w<b.w)?a.w:b.w;
    return(c) ;
}
__device__ Float4 min ( const float a , const Float4 b )
{
    Float4 c  ;
    c.x = (a<b.x)?a:b.x;
    c.y = (a<b.y)?a:b.y;
    c.z = (a<b.z)?a:b.z;
    c.w = (a<b.w)?a:b.w;
    return(c) ;
}
__device__ Float4 min ( const Float4 a , const float b )
{
    Float4 c  ;
    c.x = (a.x<b)?a.x:b;
    c.y = (a.y<b)?a.y:b;
    c.z = (a.z<b)?a.z:b;
    c.w = (a.w<b)?a.w:b;
    return(c) ;
}

__device__ Float4 trunc ( const Float4 a )
{
    Float4 c  ;
    c.x = trunc(a.x) ;
    c.y = trunc(a.y) ;
    c.z = trunc(a.z) ;
    c.w = trunc(a.w) ;
    return(c) ;
}

__device__ Float4 log ( const Float4 a )
{
    Float4 c  ;
    c.x = log(a.x) ; c.y = log(a.y) ; c.z = log(a.z) ; c.w = log(a.w) ;
    return(c) ;
}

__device__ Float4 exp ( const Float4 a )
{
    Float4 c  ;
    c.x = exp(a.x) ; c.y = exp(a.y) ; c.z = exp(a.z) ; c.w = exp(a.w) ;
    return(c) ;
}

__device__ Float4 sqrt ( const Float4 a )
{
    Float4 c  ;
    c.x = sqrt(a.x) ; c.y = sqrt(a.y) ; c.z = sqrt(a.z) ; c.w = sqrt(a.w) ;
    return(c) ;
}

#if 0
    int main() {

      Float4 a, b, c ;

      a.x = 0. ; a.y = 1. ; a.z = 2. ; a.w = 3. ;
      b.x = 0. ; b.y = 1. ; b.z = 2. ; b.w = 3. ;

      c = 2. + a ;
      fprintf(stderr,"%f %f %f %f\n",a.x,a.y,a.z,a.w) ;
      fprintf(stderr,"%f %f %f %f\n",b.x,b.y,b.z,b.w) ;
      fprintf(stderr,"%f %f %f %f\n",c.x,c.y,c.z,c.w) ;
      c = 2. * b ;
      fprintf(stderr,"%f %f %f %f\n",a.x,a.y,a.z,a.w) ;
      fprintf(stderr,"%f %f %f %f\n",b.x,b.y,b.z,b.w) ;
      fprintf(stderr,"%f %f %f %f\n",c.x,c.y,c.z,c.w) ;
      c = 2. - b ;
      fprintf(stderr,"%f %f %f %f\n",a.x,a.y,a.z,a.w) ;
      fprintf(stderr,"%f %f %f %f\n",b.x,b.y,b.z,b.w) ;
      fprintf(stderr,"%f %f %f %f\n",c.x,c.y,c.z,c.w) ;
      c = 2. / b ;
      fprintf(stderr,"%f %f %f %f\n",a.x,a.y,a.z,a.w) ;
      fprintf(stderr,"%f %f %f %f\n",b.x,b.y,b.z,b.w) ;
      fprintf(stderr,"%f %f %f %f\n",c.x,c.y,c.z,c.w) ;

    }
#endif

