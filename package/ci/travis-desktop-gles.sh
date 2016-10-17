#!/bin/bash
set -ev

# Corrade
git clone --depth 1 git://github.com/mosra/corrade.git
cd corrade
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$HOME/deps \
    -DCMAKE_INSTALL_RPATH=$HOME/deps/lib \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_INTERCONNECT=OFF
make -j install
cd ../..

# Magnum
git clone --depth 1 git://github.com/mosra/magnum.git
cd magnum
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$HOME/deps \
    -DCMAKE_INSTALL_RPATH=$HOME/deps/lib \
    -DCMAKE_BUILD_TYPE=Release \
    -DTARGET_GLES=ON \
    -DTARGET_GLES2=$TARGET_GLES2 \
    -DTARGET_DESKTOP_GLES=ON \
    -DWITH_AUDIO=ON \
    -DWITH_DEBUGTOOLS=OFF \
    -DWITH_MESHTOOLS=ON \
    -DWITH_PRIMITIVES=OFF \
    -DWITH_SCENEGRAPH=OFF \
    -DWITH_SHADERS=OFF \
    -DWITH_SHAPES=OFF \
    -DWITH_TEXT=ON \
    -DWITH_TEXTURETOOLS=ON \
    -DWITH_WINDOWLESS${PLATFORM_GL_API}APPLICATION=ON \
    -DWITH_OBJIMPORTER=ON \
    -DWITH_TGAIMPORTER=ON \
    -DWITH_WAVAUDIOIMPORTER=ON # for Any*Importer tests
make -j install
cd ../..

mkdir build && cd build
cmake .. \
    -DCMAKE_CXX_FLAGS=$COVERAGE \
    -DCMAKE_PREFIX_PATH="$HOME/deps;$HOME/harfbuzz" \
    -DCMAKE_INSTALL_PREFIX=$HOME/deps \
    -DCMAKE_INSTALL_RPATH=$HOME/deps/lib \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_ANYAUDIOIMPORTER=ON \
    -DWITH_ANYIMAGECONVERTER=ON \
    -DWITH_ANYIMAGEIMPORTER=ON \
    -DWITH_ANYSCENEIMPORTER=ON \
    -DWITH_COLLADAIMPORTER=$WITH_COLLADAIMPORTER \
    -DWITH_DDSIMPORTER=ON \
    -DWITH_DRFLACAUDIOIMPORTER=ON \
    -DWITH_DRWAVAUDIOIMPORTER=ON \
    -DWITH_FREETYPEFONT=ON \
    -DWITH_HARFBUZZFONT=ON \
    -DWITH_JPEGIMAGECONVERTER=ON \
    -DWITH_JPEGIMPORTER=ON \
    -DWITH_MINIEXRIMAGECONVERTER=ON \
    -DWITH_OPENGEXIMPORTER=ON \
    -DWITH_PNGIMAGECONVERTER=ON \
    -DWITH_PNGIMPORTER=ON \
    -DWITH_STANFORDIMPORTER=ON \
    -DWITH_STBIMAGEIMPORTER=ON \
    -DWITH_STBPNGIMAGECONVERTER=ON \
    -DWITH_STBTRUETYPEFONT=ON \
    -DWITH_STBVORBISAUDIOIMPORTER=ON \
    -DBUILD_TESTS=ON \
    -DBUILD_GL_TESTS=ON
make -j${JOBS_LIMIT}
make install # for Any*Importer tests
CORRADE_TEST_COLOR=ON ctest -V -E GLTest
