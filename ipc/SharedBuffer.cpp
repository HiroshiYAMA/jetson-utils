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
