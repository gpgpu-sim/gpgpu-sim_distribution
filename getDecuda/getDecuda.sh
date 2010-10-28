# This script downloads the latest version of decuda from http://github.com/laanwj/decuda, untars it and patches it.
# Currently hardcoded to work with only decuda 0.4.2.

# decuda.patch must be in the same folder as getDecuda.sh

wget -q  http://github.com/laanwj/decuda/tarball/master
tar -xf ./laanwj-decuda-c30bd17.tar.gz
mv laanwj-decuda-c30bd17 ../decuda
patch -s -d ../decuda < ./decuda.patch
rm ./laanwj-decuda-c30bd17.tar.gz
