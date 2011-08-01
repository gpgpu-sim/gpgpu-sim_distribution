# This script downloads the latest version of decuda from http://github.com/laanwj/decuda, untars it and patches it.
# Only works with decuda 0.4.2.

# decuda.patch must be in the same folder as getDecuda.sh

cd ./getDecuda
wget -O decuda.tgz --no-check-certificate http://github.com/laanwj/decuda/tarball/master
tar -xf ./decuda.tgz
mv laanwj-decuda-c30bd17 decuda  # if this fails, we need to check what has changed in decuda, or get a perminant link to decuda 0.4.2
patch -d ./decuda < ./decuda.patch
rm ./decuda.tgz
mv decuda ../decuda
cd ..
