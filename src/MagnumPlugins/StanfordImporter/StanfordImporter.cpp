/*
    This file is part of Magnum.

    Copyright © 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019
              Vladimír Vondruš <mosra@centrum.cz>

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*/

#include "StanfordImporter.h"

#include <Corrade/Containers/Array.h>
#include <Corrade/Containers/Optional.h>
#include <Corrade/Utility/DebugStl.h>
#include <Corrade/Utility/Endianness.h>
#include <Corrade/Utility/String.h>
#include <Magnum/Array.h>
#include <Magnum/Mesh.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Trade/MeshData3D.h>

namespace Magnum { namespace Trade {

StanfordImporter::StanfordImporter() = default;

StanfordImporter::StanfordImporter(PluginManager::AbstractManager& manager, const std::string& plugin): AbstractImporter{manager, plugin} {}

StanfordImporter::~StanfordImporter() = default;

auto StanfordImporter::doFeatures() const -> Features { return Feature::OpenData; }

bool StanfordImporter::doIsOpened() const { return !!_in; }

void StanfordImporter::doClose() { _in = nullptr; }

void StanfordImporter::doOpenData(const Containers::ArrayView<const char> data) {
    /* Because here we're copying the data and using the _in to check if file
       is opened, having them nullptr would mean openData() would fail without
       any error message. It's not possible to do this check on the importer
       side, because empty file is valid in some formats (OBJ or glTF). We also
       can't do the full import here because then doImage2D() would need to
       copy the imported data instead anyway (and the uncompressed size is much
       larger). This way it'll also work nicely with a future openMemory(). */
    if(data.empty()) {
        Error{} << "Trade::StanfordImporter::openData(): the file is empty";
        return;
    }

    _in = Containers::Array<char>{Containers::NoInit, data.size()};
    std::memcpy(_in, data, data.size());
}

UnsignedInt StanfordImporter::doMesh3DCount() const { return 1; }

namespace {

enum class FileFormat {
    LittleEndian = 1,
    BigEndian = 2
};

enum class Type {
    UnsignedByte = 1,
    Byte,
    UnsignedShort,
    Short,
    UnsignedInt,
    Int,
    Float,
    Double
};

enum class PropertyType {
    Vertex = 1,
    Face,
    Ignored
};

Type parseType(const std::string& type) {
    if(type == "uchar"  || type == "uint8")     return Type::UnsignedByte;
    if(type == "char"   || type == "int8")      return Type::Byte;
    if(type == "ushort" || type == "uint16")    return Type::UnsignedShort;
    if(type == "short"  || type == "int16")     return Type::Short;
    if(type == "uint"   || type == "uint32")    return Type::UnsignedInt;
    if(type == "int"    || type == "int32")     return Type::Int;
    if(type == "float"  || type == "float32")   return Type::Float;
    if(type == "double" || type == "float64")   return Type::Double;

    return {};
}

std::size_t sizeOf(Type type) {
    switch(type) {
        /* LCOV_EXCL_START */
        #define _c(type) case Type::type: return sizeof(type);
        _c(UnsignedByte)
        _c(Byte)
        _c(UnsignedShort)
        _c(Short)
        _c(UnsignedInt)
        _c(Int)
        _c(Float)
        _c(Double)
        #undef _c
        /* LCOV_EXCL_STOP */
    }

    CORRADE_ASSERT_UNREACHABLE(); /* LCOV_EXCL_LINE */
}

template<FileFormat format, class T> struct EndianSwap;
template<class T> struct EndianSwap<FileFormat::LittleEndian, T> {
    constexpr T operator()(T value) const { return Utility::Endianness::littleEndian<T>(value); }
};
template<class T> struct EndianSwap<FileFormat::BigEndian, T> {
    constexpr T operator()(T value) const { return Utility::Endianness::bigEndian<T>(value); }
};

template<class T, FileFormat format, class U> inline T extractAndSkip(const char*& buffer) {
    /* do a memcpy() instead of *reinterpret_cast, as that'll correctly handle
       unaligned loads as well */
    U dest;
    std::memcpy(&dest, buffer, sizeof(U));
    const auto result = T(EndianSwap<format, U>{}(dest));
    buffer += sizeof(U);
    return result;
}

template<class T, FileFormat format> T extractAndSkip(const char*& buffer, const Type type) {
    switch(type) {
        /* LCOV_EXCL_START */
        #define _c(type) case Type::type: return extractAndSkip<T, format, type>(buffer);
        _c(UnsignedByte)
        _c(Byte)
        _c(UnsignedShort)
        _c(Short)
        _c(UnsignedInt)
        _c(Int)
        _c(Float)
        _c(Double)
        #undef _c
        /* LCOV_EXCL_STOP */
    }

    CORRADE_ASSERT_UNREACHABLE(); /* LCOV_EXCL_LINE */
}

template<class T> inline T extractAndSkip(const char*& buffer, const FileFormat fileFormat, const Type type) {
    return fileFormat == FileFormat::LittleEndian ?
        extractAndSkip<T, FileFormat::LittleEndian>(buffer, type) :
        extractAndSkip<T, FileFormat::BigEndian>(buffer, type);
}

template<class T> inline T extract(const char* const buffer, const FileFormat fileFormat, const Type type) {
    const char* mutableBuffer = buffer;
    return extractAndSkip<T>(mutableBuffer, fileFormat, type);
}

template<class T, FileFormat format, class U> inline T extractNormalizedAndSkip(const char*& buffer) {
    return Math::unpack<T, U>(extractAndSkip<T, format, U>(buffer));
}

template<class T, FileFormat format> T extractNormalizedAndSkip(const char*& buffer, const Type type) {
    switch(type) {
        /* Floats are not denormalized. For coverage ensure at least one of the
           integer variants and at least one of the floating-point variants is
           covered */
        #define _c(type) case Type::type: return extractNormalizedAndSkip<T, format, type>(buffer);
        _c(UnsignedByte)        /* LCOV_EXCL_LINE */
        _c(Byte)                /* LCOV_EXCL_LINE */
        _c(UnsignedShort)
        _c(Short)               /* LCOV_EXCL_LINE */
        _c(UnsignedInt)         /* LCOV_EXCL_LINE */
        _c(Int)                 /* LCOV_EXCL_LINE */
        #undef _c
        #define _c(type) case Type::type: return extractAndSkip<T, format, type>(buffer);
        _c(Float)
        _c(Double)              /* LCOV_EXCL_LINE */
        #undef c
    }

    CORRADE_ASSERT_UNREACHABLE(); /* LCOV_EXCL_LINE */
}

template<class T> inline T extractNormalizedAndSkip(const char*& buffer, const FileFormat fileFormat, const Type type) {
    return fileFormat == FileFormat::LittleEndian ?
        extractNormalizedAndSkip<T, FileFormat::LittleEndian>(buffer, type) :
        extractNormalizedAndSkip<T, FileFormat::BigEndian>(buffer, type);
}

template<class T> inline T extractNormalized(const char* const buffer, const FileFormat fileFormat, const Type type) {
    const char* mutableBuffer = buffer;
    return extractNormalizedAndSkip<T>(mutableBuffer, fileFormat, type);
}

inline void extractTriangle(std::vector<UnsignedInt>& indices, const char* const buffer, const FileFormat fileFormat, const Type indexType) {
    const char* position = buffer;

    indices.insert(indices.end(), {
        extractAndSkip<UnsignedInt>(position, fileFormat, indexType),
        extractAndSkip<UnsignedInt>(position, fileFormat, indexType),
        extractAndSkip<UnsignedInt>(position, fileFormat, indexType)
    });
}

inline void extractQuad(std::vector<UnsignedInt>& indices, const char* const buffer, const FileFormat fileFormat, const Type indexType) {
    const char* position = buffer;

    /* GCC <=4.8 doesn't properly sequence the operations in list-initializer
       (e.g. Vector4ui{extractAndSkip(), extractAndSkip(), ...}, so I need to
       make the order really explicit. From what I understood from the specs,
       this should be defined when using {}. Am I right? */
    Vector4ui quad{Math::NoInit};
    quad[0] = extractAndSkip<UnsignedInt>(position, fileFormat, indexType);
    quad[1] = extractAndSkip<UnsignedInt>(position, fileFormat, indexType);
    quad[2] = extractAndSkip<UnsignedInt>(position, fileFormat, indexType);
    quad[3] = extractAndSkip<UnsignedInt>(position, fileFormat, indexType);

    /* 0 0---3
       |\ \  |
       | \ \ |
       |  \ \|
       1---2 2 */
    indices.insert(indices.end(), {
        quad[0],
        quad[1],
        quad[2],
        quad[0],
        quad[2],
        quad[3]
    });
}

std::string extractLine(Containers::ArrayView<const char>& in) {
    for(const char& i: in) if(i == '\n') {
        std::size_t end = &i - in;
        auto out = in.prefix(end);
        in = in.suffix(end + 1);
        return {out.begin(), out.end()};
    }

    auto out = in;
    in = {};
    return {out.begin(), out.end()};
}

}

Containers::Optional<MeshData3D> StanfordImporter::doMesh3D(UnsignedInt) {
    Containers::ArrayView<const char> in = _in;

    /* Check file signature */
    {
        std::string header = Utility::String::rtrim(extractLine(in));
        if(header != "ply") {
            Error() << "Trade::StanfordImporter::mesh3D(): invalid file signature" << header;
            return Containers::NullOpt;
        }
    }

    /* Parse format line */
    FileFormat fileFormat{};
    {
        while(in) {
            const std::string line = extractLine(in);
            std::vector<std::string> tokens = Utility::String::splitWithoutEmptyParts(line);

            /* Skip empty lines and comments */
            if(tokens.empty() || tokens.front() == "comment")
                continue;

            if(tokens[0] != "format") {
                Error{} << "Trade::StanfordImporter::mesh3D(): expected format line, got" << line;
                return Containers::NullOpt;
            }

            if(tokens.size() != 3) {
                Error() << "Trade::StanfordImporter::mesh3D(): invalid format line" << line;
                return Containers::NullOpt;
            }

            if(tokens[2] == "1.0") {
                if(tokens[1] == "binary_little_endian") {
                    fileFormat = FileFormat::LittleEndian;
                    break;
                } else if(tokens[1] == "binary_big_endian") {
                    fileFormat = FileFormat::BigEndian;
                    break;
                }
            }

            Error() << "Trade::StanfordImporter::mesh3D(): unsupported file format" << tokens[1] << tokens[2];
            return Containers::NullOpt;
        }
    }

    /* Check format line consistency */
    if(fileFormat == FileFormat{}) {
        Error() << "Trade::StanfordImporter::mesh3D(): missing format line";
        return Containers::NullOpt;
    }

    /* Parse rest of the header */
    UnsignedInt vertexStride{}, vertexCount{}, faceIndicesOffset{}, faceSkip{}, faceCount{};
    Array3D<Type> positionTypes, colorTypes;
    Type faceSizeType{}, faceIndexType{};
    Vector3i positionOffsets{-1}, colorOffsets{-1};
    {
        std::size_t vertexComponentOffset{};
        PropertyType propertyType{};
        while(in) {
            const std::string line = extractLine(in);
            std::vector<std::string> tokens = Utility::String::splitWithoutEmptyParts(line);

            /* Skip empty lines and comments */
            if(tokens.empty() || tokens.front() == "comment")
                continue;

            /* Elements */
            if(tokens[0] == "element") {
                /* Vertex elements */
                if(tokens.size() == 3 && tokens[1] == "vertex") {
                    vertexCount = std::stoi(tokens[2]);
                    propertyType = PropertyType::Vertex;

                /* Face elements */
                } else if(tokens.size() == 3 &&tokens[1] == "face") {
                    faceCount = std::stoi(tokens[2]);
                    propertyType = PropertyType::Face;

                /* Something else */
                } else {
                    Error() << "Trade::StanfordImporter::mesh3D(): unknown element" << tokens[1];
                    return Containers::NullOpt;
                }

            /* Element properties */
            } else if(tokens[0] == "property") {
                /* Vertex element properties */
                if(propertyType == PropertyType::Vertex) {
                    if(tokens.size() != 3) {
                        Error() << "Trade::StanfordImporter::mesh3D(): invalid vertex property line" << line;
                        return Containers::NullOpt;
                    }

                    /* Component type */
                    const Type componentType = parseType(tokens[1]);
                    if(componentType == Type{}) {
                        Error() << "Trade::StanfordImporter::mesh3D(): invalid vertex component type" << tokens[1];
                        return Containers::NullOpt;
                    }

                    /* Component */
                    if(tokens[2] == "x") {
                        positionOffsets.x() = vertexComponentOffset;
                        positionTypes.x() = componentType;
                    } else if(tokens[2] == "y") {
                        positionOffsets.y() = vertexComponentOffset;
                        positionTypes.y() = componentType;
                    } else if(tokens[2] == "z") {
                        positionOffsets.z() = vertexComponentOffset;
                        positionTypes.z() = componentType;
                    } else if(tokens[2] == "red") {
                        colorOffsets.x() = vertexComponentOffset;
                        colorTypes.x() = componentType;
                    } else if(tokens[2] == "green") {
                        colorOffsets.y() = vertexComponentOffset;
                        colorTypes.y() = componentType;
                    } else if(tokens[2] == "blue") {
                        colorOffsets.z() = vertexComponentOffset;
                        colorTypes.z() = componentType;
                    } else Debug{} << "Trade::StanfordImporter::mesh3D(): ignoring unknown vertex component" << tokens[2];

                    /* Add size of current component to total offset */
                    vertexComponentOffset += sizeOf(componentType);

                /* Face element properties */
                } else if(propertyType == PropertyType::Face) {
                    /* Face vertex indices */
                    if(tokens.size() == 5 && tokens[1] == "list" && tokens[4] == "vertex_indices") {
                        faceIndicesOffset = faceSkip;
                        faceSkip = 0;

                        /* Face size type */
                        if((faceSizeType = parseType(tokens[2])) == Type{}) {
                            Error() << "Trade::StanfordImporter::mesh3D(): invalid face size type" << tokens[2];
                            return Containers::NullOpt;
                        }

                        /* Face index type */
                        if((faceIndexType = parseType(tokens[3])) == Type{}) {
                            Error() << "Trade::StanfordImporter::mesh3D(): invalid face index type" << tokens[3];
                            return Containers::NullOpt;
                        }

                    /* Ignore unknown properties */
                    } else if(tokens.size() == 3) {
                        const Type faceType = parseType(tokens[1]);
                        if(faceType == Type{}) {
                            Error{} << "Trade::StanfordImporter::mesh3D(): invalid face component type" << tokens[1];
                            return Containers::NullOpt;
                        }

                        Debug{} << "Trade::StanfordImporter::mesh3D(): ignoring unknown face component" << tokens[2];
                        faceSkip += sizeOf(faceType);

                    /* Fail on unknwon lines */
                    } else {
                        Error() << "Trade::StanfordImporter::mesh3D(): invalid face property line" << line;
                        return Containers::NullOpt;
                    }

                /* Unexpected property line */
                } else if(propertyType != PropertyType::Ignored) {
                    Error() << "Trade::StanfordImporter::mesh3D(): unexpected property line";
                    return Containers::NullOpt;
                }

            /* Header end */
            } else if(tokens[0] == "end_header") {
                break;

            /* Something else */
            } else {
                Error() << "Trade::StanfordImporter::mesh3D(): unknown line" << line;
                return Containers::NullOpt;
            }
        }

        vertexStride = vertexComponentOffset;
    }

    /* Check header consistency */
    if((positionOffsets < Vector3i{0}).any()) {
        Error() << "Trade::StanfordImporter::mesh3D(): incomplete vertex specification";
        return Containers::NullOpt;
    }
    if(faceSizeType == Type{} || faceIndexType == Type{}) {
        Error() << "Trade::StanfordImporter::mesh3D(): incomplete face specification";
        return Containers::NullOpt;
    }

    /* Parse vertices */
    std::vector<Vector3> positions;
    std::vector<std::vector<Color4>> colors;
    positions.reserve(vertexCount);
    if((colorOffsets > Vector3i{0}).any()) {
        colors.emplace_back();
        colors.back().reserve(vertexCount);
    }
    {
        for(std::size_t i = 0; i != vertexCount; ++i) {
            if(in.size() < vertexStride) {
                Error() << "Trade::StanfordImporter::mesh3D(): incomplete vertex data";
                return Containers::NullOpt;
            }

            Containers::ArrayView<const char> buffer = in.prefix(vertexStride);
            in = in.suffix(vertexStride);

            positions.emplace_back(
                extract<Float>(buffer + positionOffsets.x(), fileFormat, positionTypes.x()),
                extract<Float>(buffer + positionOffsets.y(), fileFormat, positionTypes.y()),
                extract<Float>(buffer + positionOffsets.z(), fileFormat, positionTypes.z())
            );
            if((colorOffsets > Vector3i{0}).any()) colors.back().emplace_back(
                colorOffsets.x() != -1 ? extractNormalized<Float>(buffer + colorOffsets.x(), fileFormat, colorTypes.x()) : 0.0f,
                colorOffsets.y() != -1 ? extractNormalized<Float>(buffer + colorOffsets.y(), fileFormat, colorTypes.y()) : 0.0f,
                colorOffsets.z() != -1 ? extractNormalized<Float>(buffer + colorOffsets.z(), fileFormat, colorTypes.z()) : 0.0f
            );
        }
    }

    /* Parse faces, reserve optimistically amount for all-triangle faces */
    std::vector<UnsignedInt> indices;
    indices.reserve(faceCount*3);
    {
        const UnsignedInt faceSizeTypeSize = sizeOf(faceSizeType);
        const UnsignedInt faceIndexTypeSize = sizeOf(faceIndexType);
        for(std::size_t i = 0; i != faceCount; ++i) {
            if(in.size() < faceIndicesOffset + faceSizeTypeSize) {
                Error() << "Trade::StanfordImporter::mesh3D(): incomplete index data";
                return Containers::NullOpt;
            }

            in = in.suffix(faceIndicesOffset);

            /* Get face size */
            Containers::ArrayView<const char> buffer = in.prefix(faceSizeTypeSize);
            in = in.suffix(faceSizeTypeSize);
            const UnsignedInt faceSize = extract<UnsignedInt>(buffer, fileFormat, faceSizeType);
            if(faceSize < 3 || faceSize > 4) {
                Error() << "Trade::StanfordImporter::mesh3D(): unsupported face size" << faceSize;
                return Containers::NullOpt;
            }

            /* Parse face indices */
            if(in.size() < faceIndexTypeSize*faceSize + faceSkip) {
                Error() << "Trade::StanfordImporter::mesh3D(): incomplete face data";
                return Containers::NullOpt;
            }

            buffer = in.prefix(faceIndexTypeSize*faceSize);
            in = in.suffix(faceIndexTypeSize*faceSize + faceSkip);

            faceSize == 3 ?
                extractTriangle(indices, buffer, fileFormat, faceIndexType) :
                extractQuad(indices, buffer, fileFormat, faceIndexType);
        }
    }

    return MeshData3D{MeshPrimitive::Triangles, std::move(indices), {std::move(positions)}, {}, {}, colors, nullptr};
}

}}

CORRADE_PLUGIN_REGISTER(StanfordImporter, Magnum::Trade::StanfordImporter,
    "cz.mosra.magnum.Trade.AbstractImporter/0.3")
