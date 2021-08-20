/*
 * Copyright (c) 2021, edgecraft. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "SharedBuffer.h"

// default.
st_SharedBuffer DefaultSharedBuffer = {
    .id = {
        .shm_header = "/my_shm_header",
        .shm_body   = "/my_shm_body",
        .sem        = "/my_sem",
    },
    .img_info = nullptr,
};

// type -> SharedBuffer::em_IMAGE_INFO_DATA_TYPE.
template<> SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer::imageInfoDataTypeFromType<uint8_t>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_8; }
template<> SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer::imageInfoDataTypeFromType<uint16_t>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_16; }
template<> SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer::imageInfoDataTypeFromType<uint32_t>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_32; }
template<> SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer::imageInfoDataTypeFromType<uint64_t>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_64; }
template<> SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer::imageInfoDataTypeFromType<int8_t>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::INT_TYPE_8; }
template<> SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer::imageInfoDataTypeFromType<int16_t>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::INT_TYPE_16; }
template<> SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer::imageInfoDataTypeFromType<int32_t>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::INT_TYPE_32; }
template<> SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer::imageInfoDataTypeFromType<int64_t>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::INT_TYPE_64; }
template<> SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer::imageInfoDataTypeFromType<float>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BINARY32; }
template<> SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer::imageInfoDataTypeFromType<double>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BINARY64; }
#ifdef __CUDACC__
template<> SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer::imageInfoDataTypeFromType<nv_bfloat16>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BFLOAT16; }
template<> SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer::imageInfoDataTypeFromType<half>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BINARY16; }
#endif

std::ostream &operator<<(std::ostream &ostr, const SharedBuffer::st_IMAGE_INFO_HEADER &header)
{
    ostr << "st_IMAGE_INFO_HEADER:" << std::endl;
    ostr << "  width       : " << header.width << std::endl;
    ostr << "  height      : " << header.height << std::endl;
    ostr << "  channel     : " << header.channel << std::endl;
    ostr << "  pixel_size  : " << header.pixel_size << std::endl;
    ostr << "  line_offset : " << header.line_offset << std::endl;
    ostr << "  data_type   : " << SharedBuffer::imageInfoDataTypeToString(header.data_type) << std::endl;

    return ostr;
}
