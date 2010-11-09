# This script downloads the latest version of decuda from http://github.com/laanwj/decuda, untars it and patches it.
# Only works with decuda 0.4.2.

# decuda.patch must be in the same folder as getDecuda.sh

cd ./getDecuda
wget --no-check-certificate  http://github.com/laanwj/decuda/tarball/master
tar -xf ./laanwj-decuda-c30bd17.tar.gz
mv laanwj-decuda-c30bd17 decuda
patch -d ./decuda < ./decuda.patch
rm ./laanwj-decuda-c30bd17.tar.gz
mv decuda ../decuda
cd ..
