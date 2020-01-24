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
#include <Corrade/Containers/GrowableArray.h>
#include <Corrade/Containers/Optional.h>
#include <Corrade/Utility/Algorithms.h>
#include <Corrade/Utility/DebugStl.h>
#include <Corrade/Utility/EndiannessBatch.h>
#include <Corrade/Utility/String.h>
#include <Magnum/Array.h>
#include <Magnum/Mesh.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Trade/ArrayAllocator.h>
#include <Magnum/Trade/MeshData.h>

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
    Utility::copy(data, _in);
}

UnsignedInt StanfordImporter::doMeshCount() const { return 1; }

namespace {

enum class PropertyType {
    Vertex = 1,
    Face
};

MeshIndexType parseIndexType(const std::string& type) {
    if(type == "uchar"  || type == "uint8" ||
       type == "char"   || type == "int8")
        return MeshIndexType::UnsignedByte;
    if(type == "ushort" || type == "uint16" ||
       type == "short"  || type == "int16")
        return MeshIndexType::UnsignedShort;
    if(type == "uint"   || type == "uint32" ||
       type == "int"    || type == "int32")
        return MeshIndexType::UnsignedInt;

    return {};
}

MeshAttributeType parseAttributeType(const std::string& type) {
    if(type == "uchar"  || type == "uint8")
        return MeshAttributeType::UnsignedByte;
    if(type == "char"   || type == "int8")
        return MeshAttributeType::Byte;
    if(type == "ushort" || type == "uint16")
        return MeshAttributeType::UnsignedShort;
    if(type == "short"  || type == "int16")
        return MeshAttributeType::Short;
    if(type == "uint"   || type == "uint32")
        return MeshAttributeType::UnsignedInt;
    if(type == "int"    || type == "int32")
        return MeshAttributeType::Int;
    if(type == "float"  || type == "float32")
        return MeshAttributeType::Float;
    if(type == "double" || type == "float64")
        return MeshAttributeType::Double;

    return {};
}

template<class T, class U> inline T extractValue(const char* buffer, const bool endianSwap) {
    /* do a memcpy() instead of *reinterpret_cast, as that'll correctly handle
       unaligned loads as well */
    U dest;
    std::memcpy(&dest, buffer, sizeof(U));
    if(endianSwap) Utility::Endianness::swapInPlace(dest);
    return T(dest);
}

template<class T> T extractIndexValue(const char* buffer, const MeshIndexType type, const bool endianSwap) {
    switch(type) {
        /* LCOV_EXCL_START */
        #define _c(type) case MeshIndexType::type: return extractValue<T, type>(buffer, endianSwap);
        _c(UnsignedByte)
        _c(UnsignedShort)
        _c(UnsignedInt)
        #undef _c
        /* LCOV_EXCL_STOP */

        default: CORRADE_ASSERT_UNREACHABLE(); /* LCOV_EXCL_LINE */
    }
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

Containers::Optional<MeshData> StanfordImporter::doMesh(UnsignedInt) {
    Containers::ArrayView<const char> in = _in;

    /* Check file signature */
    {
        std::string header = Utility::String::rtrim(extractLine(in));
        if(header != "ply") {
            Error() << "Trade::StanfordImporter::mesh(): invalid file signature" << header;
            return Containers::NullOpt;
        }
    }

    /* Parse format line */
    Containers::Optional<bool> fileFormatNeedsEndianSwapping{};
    {
        while(in) {
            const std::string line = extractLine(in);
            std::vector<std::string> tokens = Utility::String::splitWithoutEmptyParts(line);

            /* Skip empty lines and comments */
            if(tokens.empty() || tokens.front() == "comment")
                continue;

            if(tokens[0] != "format") {
                Error{} << "Trade::StanfordImporter::mesh(): expected format line, got" << line;
                return Containers::NullOpt;
            }

            if(tokens.size() != 3) {
                Error() << "Trade::StanfordImporter::mesh(): invalid format line" << line;
                return Containers::NullOpt;
            }

            if(tokens[2] == "1.0") {
                if(tokens[1] == "binary_little_endian") {
                    fileFormatNeedsEndianSwapping = Utility::Endianness::isBigEndian();
                    break;
                } else if(tokens[1] == "binary_big_endian") {
                    fileFormatNeedsEndianSwapping = !Utility::Endianness::isBigEndian();
                    break;
                }
            }

            Error() << "Trade::StanfordImporter::mesh(): unsupported file format" << tokens[1] << tokens[2];
            return Containers::NullOpt;
        }
    }

    /* Check format line consistency */
    if(!fileFormatNeedsEndianSwapping) {
        Error() << "Trade::StanfordImporter::mesh(): missing format line";
        return Containers::NullOpt;
    }

    /* Parse rest of the header */
    UnsignedInt vertexStride{}, vertexCount{}, faceIndicesOffset{}, faceSkip{}, faceCount{};
    Array3D<MeshAttributeType> positionTypes, colorTypes;
    MeshIndexType faceSizeType{}, faceIndexType{};
    Vector3ui positionOffsets{~UnsignedInt{}}, colorOffsets{~UnsignedInt{}};
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
                    Error() << "Trade::StanfordImporter::mesh(): unknown element" << tokens[1];
                    return Containers::NullOpt;
                }

            /* Element properties */
            } else if(tokens[0] == "property") {
                /* Vertex element properties */
                if(propertyType == PropertyType::Vertex) {
                    if(tokens.size() != 3) {
                        Error() << "Trade::StanfordImporter::mesh(): invalid vertex property line" << line;
                        return Containers::NullOpt;
                    }

                    /* Component type */
                    const MeshAttributeType componentType = parseAttributeType(tokens[1]);
                    if(componentType == MeshAttributeType{}) {
                        Error() << "Trade::StanfordImporter::mesh(): invalid vertex component type" << tokens[1];
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
                    } else Debug{} << "Trade::StanfordImporter::mesh(): ignoring unknown vertex component" << tokens[2];

                    /* Add size of current component to total offset */
                    vertexComponentOffset += meshAttributeTypeSize(componentType);

                /* Face element properties */
                } else if(propertyType == PropertyType::Face) {
                    /* Face vertex indices */
                    if(tokens.size() == 5 && tokens[1] == "list" && tokens[4] == "vertex_indices") {
                        faceIndicesOffset = faceSkip;
                        faceSkip = 0;

                        /* Face size type */
                        if((faceSizeType = parseIndexType(tokens[2])) == MeshIndexType{}) {
                            Error() << "Trade::StanfordImporter::mesh(): invalid face size type" << tokens[2];
                            return Containers::NullOpt;
                        }

                        /* Face index type */
                        if((faceIndexType = parseIndexType(tokens[3])) == MeshIndexType{}) {
                            Error() << "Trade::StanfordImporter::mesh(): invalid face index type" << tokens[3];
                            return Containers::NullOpt;
                        }

                    /* Ignore unknown properties */
                    } else if(tokens.size() == 3) {
                        const MeshAttributeType faceType = parseAttributeType(tokens[1]);
                        if(faceType == MeshAttributeType{}) {
                            Error{} << "Trade::StanfordImporter::mesh(): invalid face component type" << tokens[1];
                            return Containers::NullOpt;
                        }

                        Debug{} << "Trade::StanfordImporter::mesh(): ignoring unknown face component" << tokens[2];
                        faceSkip += meshAttributeTypeSize(faceType);

                    /* Fail on unknwon lines */
                    } else {
                        Error() << "Trade::StanfordImporter::mesh(): invalid face property line" << line;
                        return Containers::NullOpt;
                    }

                /* Unexpected property line */
                } else if(propertyType == PropertyType{}) {
                    Error() << "Trade::StanfordImporter::mesh(): unexpected property line";
                    return Containers::NullOpt;
                }

            /* Header end */
            } else if(tokens[0] == "end_header") {
                break;

            /* Something else */
            } else {
                Error() << "Trade::StanfordImporter::mesh(): unknown line" << line;
                return Containers::NullOpt;
            }
        }

        vertexStride = vertexComponentOffset;
    }

    /* Check header consistency */
    if((positionOffsets >= Vector3ui{~UnsignedInt{}}).any()) {
        Error() << "Trade::StanfordImporter::mesh(): incomplete vertex specification";
        return Containers::NullOpt;
    }
    if(faceSizeType == MeshIndexType{} || faceIndexType == MeshIndexType{}) {
        Error() << "Trade::StanfordImporter::mesh(): incomplete face specification";
        return Containers::NullOpt;
    }

    /* Copy all vertex data */
    if(in.size() < vertexStride*vertexCount) {
        Error() << "Trade::StanfordImporter::mesh(): incomplete vertex data";
        return Containers::NullOpt;
    }
    Containers::Array<char> vertexData{Containers::NoInit,
        vertexStride*vertexCount};
    Utility::copy(in.prefix(vertexStride*vertexCount), vertexData);
    in = in.suffix(vertexStride*vertexCount);

    /* Gather all attributes */
    std::size_t attributeCount = 1;
    if((colorOffsets < Vector3ui{~UnsignedInt{}}).any()) ++attributeCount;
    Containers::Array<MeshAttributeData> attributeData{attributeCount};
    std::size_t attributeOffset = 0;

    /* Wrap up positions */
    {
        /* Check that we have the same type for all position coordinates */
        if(positionTypes.x() != positionTypes.y() ||
           positionTypes.x() != positionTypes.z()) {
            Error{} << "Trade::StanfordImporter::mesh(): expecting all position coordinates to have the same type but got" << positionTypes;
            return Containers::NullOpt;
        }

        /* And that they are right after each other in correct order */
        const UnsignedInt positionTypeSize = meshAttributeTypeSize(positionTypes.x());
        if(positionOffsets.y() != positionOffsets.x() + positionTypeSize ||
           positionOffsets.z() != positionOffsets.y() + positionTypeSize) {
            Error{} << "Trade::StanfordImporter::mesh(): expecting position coordinates to be tightly packed, but got offsets" << positionOffsets << "for a" << positionTypeSize << Debug::nospace << "-byte type";
            return Containers::NullOpt;
        }

        /* Ensure the type is one of allowed */
        if(positionTypes.x() != MeshAttributeType::Float &&
           positionTypes.x() != MeshAttributeType::UnsignedByte &&
           positionTypes.x() != MeshAttributeType::Byte &&
           positionTypes.x() != MeshAttributeType::UnsignedShort &&
           positionTypes.x() != MeshAttributeType::Short) {
            Error{} << "Trade::StanfordImporter::mesh(): unsupported position component type" << positionTypes.x();
            return Containers::NullOpt;
        }

        /* Endian-swap them, if needed */
        if(*fileFormatNeedsEndianSwapping) {
            Containers::StridedArrayView3D<char> positionComponents{vertexData,
                vertexData + positionOffsets.x(),
                {3, vertexCount, positionTypeSize}, {std::ptrdiff_t(positionTypeSize), std::ptrdiff_t(vertexStride), 1}};
            for(std::size_t component = 0; component != 3; ++component) {
                if(positionTypeSize == 2)
                    Utility::Endianness::swapInPlace(Containers::arrayCast<1, UnsignedShort>(positionComponents[component]));
                else if(positionTypeSize == 4)
                    Utility::Endianness::swapInPlace(Containers::arrayCast<1, UnsignedInt>(positionComponents[component]));
                else CORRADE_INTERNAL_ASSERT(positionTypeSize == 1);
            }
        }

        /* Add the attribute */
        attributeData[attributeOffset++] = MeshAttributeData{
            MeshAttributeName::Position,
            meshAttributeType(positionTypes.x(), 3, false),
            Containers::StridedArrayView1D<void>{vertexData,
                vertexData + positionOffsets.x(),
                vertexCount, vertexStride}};
    }

    /* Wrap up colors, if any */
    if((colorOffsets < Vector3ui{~UnsignedInt{}}).any()) {
        /* Check that we have the same type for all position coordinates */
        if(colorTypes.x() != colorTypes.y() ||
           colorTypes.x() != colorTypes.z()) {
            Error{} << "Trade::StanfordImporter::mesh(): expecting all color channels to have the same type but got" << colorTypes;
            return Containers::NullOpt;
        }

        /* And that they are right after each other in correct order */
        const UnsignedInt colorTypeSize = meshAttributeTypeSize(colorTypes.x());
        if(colorOffsets.y() != colorOffsets.x() + colorTypeSize ||
           colorOffsets.z() != colorOffsets.y() + colorTypeSize) {
            Error{} << "Trade::StanfordImporter::mesh(): expecting color channels to be tightly packed, but got offsets" << colorOffsets << "for a" << colorTypeSize << Debug::nospace << "-byte type";
            return Containers::NullOpt;
        }

        /* Ensure the type is one of allowed */
        if(colorTypes.x() != MeshAttributeType::Float &&
           colorTypes.x() != MeshAttributeType::UnsignedByte &&
           colorTypes.x() != MeshAttributeType::UnsignedShort) {
            Error{} << "Trade::StanfordImporter::mesh(): unsupported color channel type" << colorTypes.x();
            return Containers::NullOpt;
        }

        /* Endian-swap them, if needed */
        if(*fileFormatNeedsEndianSwapping) {
            Containers::StridedArrayView3D<char> colorChannels{vertexData,
                vertexData + colorOffsets.x(),
                {3, vertexCount, colorTypeSize}, {std::ptrdiff_t(colorTypeSize), std::ptrdiff_t(vertexStride), 1}};
            for(std::size_t channel = 0; channel != 3; ++channel) {
                if(colorTypeSize == 2)
                    Utility::Endianness::swapInPlace(Containers::arrayCast<1, UnsignedShort>(colorChannels[channel]));
                else if(colorTypeSize== 4)
                    Utility::Endianness::swapInPlace(Containers::arrayCast<1, UnsignedInt>(colorChannels[channel]));
                else CORRADE_INTERNAL_ASSERT(colorTypeSize == 1);
            }
        }

        /* Add the attribute */
        attributeData[attributeOffset++] = MeshAttributeData{
            MeshAttributeName::Color,
            /* We want integer types normalized */
            meshAttributeType(colorTypes.x(), 3, colorTypes.x() != MeshAttributeType::Float),
            Containers::StridedArrayView1D<void>{vertexData,
                vertexData + colorOffsets.x(),
                vertexCount, vertexStride}};
    }

    /* Parse faces. Keep the original index type, reserve optimistically amount
       for all-triangle faces */
    Containers::Array<char> indexData;
    const UnsignedInt faceIndexTypeSize = meshIndexTypeSize(faceIndexType);
    Containers::arrayReserve<ArrayAllocator>(indexData,
        faceCount*3*faceIndexTypeSize);
    {
        const UnsignedInt faceSizeTypeSize = meshIndexTypeSize(faceSizeType);
        for(std::size_t i = 0; i != faceCount; ++i) {
            if(in.size() < faceIndicesOffset + faceSizeTypeSize) {
                Error() << "Trade::StanfordImporter::mesh(): incomplete index data";
                return Containers::NullOpt;
            }

            in = in.suffix(faceIndicesOffset);

            /* Get face size */
            Containers::ArrayView<const char> buffer = in.prefix(faceSizeTypeSize);
            in = in.suffix(faceSizeTypeSize);
            const UnsignedInt faceSize = extractIndexValue<UnsignedInt>(buffer, faceSizeType, *fileFormatNeedsEndianSwapping);
            if(faceSize < 3 || faceSize > 4) {
                Error() << "Trade::StanfordImporter::mesh(): unsupported face size" << faceSize;
                return Containers::NullOpt;
            }

            /* Parse face indices */
            if(in.size() < faceIndexTypeSize*faceSize + faceSkip) {
                Error() << "Trade::StanfordImporter::mesh(): incomplete face data";
                return Containers::NullOpt;
            }

            buffer = in.prefix(faceIndexTypeSize*faceSize);
            in = in.suffix(faceIndexTypeSize*faceSize + faceSkip);

            /* Append either the triangle or the first triangle of the quad */
            Containers::arrayAppend<ArrayAllocator>(indexData,
                buffer.prefix(3*faceIndexTypeSize));
            /* For a quad add the 0, 2 and 3 indices forming another triangle */
            if(faceSize == 4) {
                /* 0 0---3
                   |\ \  |
                   | \ \ |
                   |  \ \|
                   1---2 2 */
                Containers::arrayAppend<ArrayAllocator>(indexData,
                    buffer.slice(0*faceIndexTypeSize, 1*faceIndexTypeSize));
                Containers::arrayAppend<ArrayAllocator>(indexData,
                    buffer.slice(2*faceIndexTypeSize, 3*faceIndexTypeSize));
                Containers::arrayAppend<ArrayAllocator>(indexData,
                    buffer.slice(3*faceIndexTypeSize, 4*faceIndexTypeSize));
            }
        }

        /* Endian-swap the indices, if needed */
        if(*fileFormatNeedsEndianSwapping) {
            if(faceIndexTypeSize == 2)
                Utility::Endianness::swapInPlace(Containers::arrayCast<UnsignedShort>(indexData));
            else if(faceIndexTypeSize == 4)
                Utility::Endianness::swapInPlace(Containers::arrayCast<UnsignedInt>(indexData));
            else CORRADE_INTERNAL_ASSERT(faceIndexTypeSize == 1);
        }
    }

    MeshIndexData indices{faceIndexType, indexData};
    return MeshData{MeshPrimitive::Triangles,
        std::move(indexData), indices,
        std::move(vertexData), std::move(attributeData)};
}

}}

CORRADE_PLUGIN_REGISTER(StanfordImporter, Magnum::Trade::StanfordImporter,
    "cz.mosra.magnum.Trade.AbstractImporter/0.3")
