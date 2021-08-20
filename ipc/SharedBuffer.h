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

#pragma once

#include <iostream>
#include <cstdint>
#include <memory>

#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <memory.h>
#include <semaphore.h>
#include <signal.h>

#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#endif

#include <Poco/DigestEngine.h>
#include <Poco/MD5Engine.h>

auto check_sum = [](const uint8_t *buf, size_t size) -> std::string {
    Poco::MD5Engine md5;

    md5.update(buf, size);
    auto digest = md5.digest();
    auto val = Poco::DigestEngine::digestToHex(digest);

    return val;
};

class SharedBuffer {
public:
    struct st_ID {
        const std::string shm_header;   // shared memory. header.
        const std::string shm_body;     // shared memory. body.
        const std::string sem;          // semaphore.
    };

    enum class em_IMAGE_INFO_DATA_TYPE : uint32_t {
        UINT_TYPE_8,
        UINT_TYPE_16,
        UINT_TYPE_32,
        UINT_TYPE_64,
        INT_TYPE_8,
        INT_TYPE_16,
        INT_TYPE_32,
        INT_TYPE_64,
        FLOAT_TYPE_BINARY16,	// IEEE 754 半精度. [s1, e5, f10].
        FLOAT_TYPE_BINARY32,	// IEEE 754 単精度. [s1, e8, f23].
        FLOAT_TYPE_BINARY64,	// IEEE 754 倍精度. [s1, e11, f52].
        FLOAT_TYPE_BFLOAT16,	// [s1, e8, f7].
    };

    struct st_IMAGE_INFO_HEADER {
        uint32_t width;			// 出力サイズ(横). ピクセル数.
        uint32_t height;		// 出力サイズ(縦). ピクセル数.
        uint32_t channel;		// チャンネル数.
        uint32_t pixel_size;	// 1チャンネル、1画素のバイト数.
        uint32_t line_offset;	// 次のラインの先頭までのオフセット. バイト数.
        em_IMAGE_INFO_DATA_TYPE data_type = em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_8;	// 整数、浮動小数点数のタイプ.

	    friend std::ostream &operator<<(std::ostream &ostr, const st_IMAGE_INFO_HEADER &header);
    };

    // SharedBuffer::em_IMAGE_INFO_DATA_TYPE -> type.
    template<em_IMAGE_INFO_DATA_TYPE T> struct imageInfoDataType { typedef uint8_t Type; };

    // SharedBuffer::em_IMAGE_INFO_DATA_TYPE -> String.
    static std::string imageInfoDataTypeToString(em_IMAGE_INFO_DATA_TYPE data_type)
    {
        std::string str;

        switch (data_type) {
        case em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_8:
            str = "UINT_TYPE_8";
            break;
        case em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_16:
            str = "UINT_TYPE_16";
            break;
        case em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_32:
            str = "UINT_TYPE_32";
            break;
        case em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_64:
            str = "UINT_TYPE_64";
            break;
        case em_IMAGE_INFO_DATA_TYPE::INT_TYPE_8:
            str = "INT_TYPE_8";
            break;
        case em_IMAGE_INFO_DATA_TYPE::INT_TYPE_16:
            str = "INT_TYPE_16";
            break;
        case em_IMAGE_INFO_DATA_TYPE::INT_TYPE_32:
            str = "INT_TYPE_32";
            break;
        case em_IMAGE_INFO_DATA_TYPE::INT_TYPE_64:
            str = "INT_TYPE_64";
            break;
        case em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BINARY16:
            str = "FLOAT_TYPE_BINARY16";
            break;
        case em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BINARY32:
            str = "FLOAT_TYPE_BINARY32";
            break;
        case em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BINARY64:
            str = "FLOAT_TYPE_BINARY64";
            break;
        case em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BFLOAT16:
            str = "FLOAT_TYPE_BFLOAT16";
            break;
        default:
            str = "invalid data type.";
        }

        return str;
    }

#ifdef USE_OPENCV
    // CV type -> type.
    template<int T> struct CVTpeyType { typedef uint8_t Type; };

    // CV type -> SharedBuffer::em_IMAGE_INFO_DATA_TYPE.
    static em_IMAGE_INFO_DATA_TYPE imageInfoDataTypeFromCVType(int CV_type)
    {
        em_IMAGE_INFO_DATA_TYPE data_type;

        switch (CV_type) {
        case CV_8U:
            data_type = em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_8;
            break;
        case CV_8S:
            data_type = em_IMAGE_INFO_DATA_TYPE::INT_TYPE_8;
            break;
        case CV_16U:
            data_type = em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_16;
            break;
        case CV_16S:
            data_type = em_IMAGE_INFO_DATA_TYPE::INT_TYPE_16;
            break;
        case CV_32S:
            data_type = em_IMAGE_INFO_DATA_TYPE::INT_TYPE_32;
            break;
        case CV_32F:
            data_type = em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BINARY32;
            break;
        case CV_64F:
            data_type = em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BINARY64;
            break;
#ifdef __CUDACC__
        case CV_16F:
            data_type = em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BINARY16;
            break;
#endif
        default:
            data_type = em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_8;
            std::cout << "invalid CV type - supported types are CV_8[US], CV_16[US], CV_32[SF], CV_64F." << std::endl;
        }

        return data_type;
    }

    // SharedBuffer::em_IMAGE_INFO_DATA_TYPE -> CV type.
    static int CVTypeFromimageInfoDataType(em_IMAGE_INFO_DATA_TYPE data_type)
    {
        int CV_type;

        switch (data_type) {
        case em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_8:
            CV_type = CV_8U;
            break;
        case em_IMAGE_INFO_DATA_TYPE::INT_TYPE_8:
            CV_type = CV_8S;
            break;
        case em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_16:
            CV_type = CV_16U;
            break;
        case em_IMAGE_INFO_DATA_TYPE::INT_TYPE_16:
            CV_type = CV_16S;
            break;
        case em_IMAGE_INFO_DATA_TYPE::INT_TYPE_32:
            CV_type = CV_32S;
            break;
        case em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BINARY32:
            CV_type = CV_32F;
            break;
        case em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BINARY64:
            CV_type = CV_64F;
            break;
#ifdef __CUDACC__
        case em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BINARY16:
            CV_type = CV_16F;
            break;
#endif
        default:
            CV_type = CV_32S;
            std::cout << "invalid data type - supported types are int{8|16|32}, uint{8|16}, float, double." << std::endl;
        }

        return CV_type;
    }
#endif

private:
    // ID name for shared memory and semaphore.
    const st_ID id;

    // file descripter for shared memory.
    int fd_header;
    int fd_body;

    // mapped address for shared memory.
    st_IMAGE_INFO_HEADER *img_info_header;
    uint8_t *img_info_body;

    // address for semaphore.
    sem_t *sem;

    bool is_ro;

    static bool SemWait(sem_t *sem)
    {
        if (sem_wait(sem) == -1) {
            std::cout << "ERROR: sem_wait" << std::endl;
            return false;
        }

        return true;
    }

    static bool SemPost(sem_t *sem)
    {
        if (sem_post(sem) == -1) {
            std::cout << "ERROR: sem_post" << std::endl;
            return false;
        }

        return true;
    }

public:
    static std::unique_ptr<SharedBuffer> Create_RW(const st_ID &id, const st_IMAGE_INFO_HEADER &header)
    {
        auto ptr = std::make_unique<SharedBuffer>(id);

        ptr->is_ro = false;

        // semaphore.
        ptr->sem = sem_open(id.sem.c_str(), O_CREAT, S_IRUSR | S_IWUSR, 1);
        if (ptr->sem == SEM_FAILED) {
            std::cout << "ERROR: sem_open" << std::endl;
            return nullptr;
        }

        auto create_shm = [](auto id, auto sz, auto &fd, auto *&img_info) -> auto {
            fd = shm_open(id, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
            if (fd == -1) {
                std::cout << "ERROR: shm_open" << std::endl;
                return -1;
            }

            if (ftruncate(fd, sz) == -1) {
                std::cout << "ERROR: ftruncate" << std::endl;
                return -1;
            }

            using img_info_type = std::remove_reference_t<decltype(img_info)>;
            img_info = static_cast<img_info_type>(mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
            if (img_info == MAP_FAILED) {
                std::cout << "ERROR: mmap" << std::endl;
                return -1;
            }

            return 0;
        };

        // shared memory. header.
        if (create_shm(id.shm_header.c_str(), sizeof(st_IMAGE_INFO_HEADER), ptr->fd_header, ptr->img_info_header) == -1) {
            std::cout << "ERROR: create_shm header" << std::endl;
            return nullptr;
        }

        // shared memory. body.
        if (create_shm(id.shm_body.c_str(), header.line_offset * header.height, ptr->fd_body, ptr->img_info_body) == -1) {
            std::cout << "ERROR: create_shm body" << std::endl;
            return nullptr;
        }

        return std::move(ptr);
    }

    static std::unique_ptr<SharedBuffer> Create_RO(const st_ID &id)
    {
        auto ptr = std::make_unique<SharedBuffer>(id);

        ptr->is_ro = true;

        // semaphore.
        ptr->sem = sem_open(id.sem.c_str(), O_RDWR);
        if (ptr->sem == SEM_FAILED) {
            std::cout << "ERROR: sem_open (RO)" << std::endl;
            return nullptr;
        }

        auto create_shm_ro = [](auto id, auto sz, auto &fd, auto *&img_info) -> auto {
            fd = shm_open(id, O_RDONLY, S_IRUSR | S_IWUSR);
            if (fd == -1) {
                std::cout << "ERROR: shm_open (RO)" << std::endl;
                return -1;
            }

            using img_info_type = std::remove_reference_t<decltype(img_info)>;
            img_info = static_cast<img_info_type>(mmap(NULL, sz, PROT_READ, MAP_SHARED, fd, 0));
            if (img_info == MAP_FAILED) {
                std::cout << "ERROR: mmap (RO)" << std::endl;
                return -1;
            }

            return 0;
        };

        // shared memory. header.
        if (create_shm_ro(id.shm_header.c_str(), sizeof(st_IMAGE_INFO_HEADER), ptr->fd_header, ptr->img_info_header) == -1) {
            std::cout << "ERROR: create_shm header (RO)" << std::endl;
            return nullptr;
        }

        st_IMAGE_INFO_HEADER header;
        {
            if (!SemWait(ptr->sem)) { return nullptr; }

            header = *(ptr->img_info_header);

            if (!SemPost(ptr->sem)) { return nullptr; }
        }

        // shared memory. body.
        if (create_shm_ro(id.shm_body.c_str(), header.line_offset * header.height, ptr->fd_body, ptr->img_info_body) == -1) {
            std::cout << "ERROR: create_shm body (RO)" << std::endl;
            return nullptr;
        }

        return std::move(ptr);
    }

    static std::unique_ptr<SharedBuffer> Create(const st_ID &id, const st_IMAGE_INFO_HEADER &header) { return std::move(Create_RW(id, header)); }
    static std::unique_ptr<SharedBuffer> Create(const st_ID &id) { return std::move(Create_RO(id)); }

    bool Destroy()
    {
        if (is_ro) return true;

        // mmap cleanup
        const size_t buf_size = img_info_header->line_offset * img_info_header->height;

        if (munmap(img_info_header, sizeof(st_IMAGE_INFO_HEADER)) == -1) {
            std::cout << "ERROR: munmap header" << std::endl;
            return false;
        }
        if (munmap(img_info_body, buf_size) == -1) {
            std::cout << "ERROR: munmap body" << std::endl;
            return false;
        }

        // shm_open cleanup
        fd_header = shm_unlink(id.shm_header.c_str());
        if (fd_header == -1) {
            std::cout << "ERROR: shm_unlink header" << std::endl;
            return false;
        }
        fd_body = shm_unlink(id.shm_body.c_str());
        if (fd_body == -1) {
            std::cout << "ERROR: shm_unlink body" << std::endl;
            return false;
        }

        // sem_open cleanup
        if (sem_close(sem) == -1) {
            std::cout << "ERROR: sem_close" << std::endl;
            return false;
        }
        if (sem_unlink(id.sem.c_str()) == -1) {
            std::cout << "ERROR: sem_unlink" << std::endl;
            return false;
        }

        return true;
    }

    // send to shm.
    bool SendHeader(const st_IMAGE_INFO_HEADER &header)
    {
        if (!SemWait(sem)) return false;

        // header.
        *img_info_header = header;

        if (!SemPost(sem)) return false;

        return true;
    }
    //
    bool SendBuf(const uint8_t *src_buf, size_t buf_size)
    {
        if (src_buf == nullptr) return false;

        if (!SemWait(sem)) return false;

        // buf.
        memcpy(img_info_body, src_buf, buf_size);

        if (!SemPost(sem)) return false;

        return true;
    }
    //
    bool Send(const st_IMAGE_INFO_HEADER &header, const uint8_t *src_buf)
    {
        if (src_buf == nullptr) return false;

        if (!SemWait(sem)) return false;

        // header.
        *img_info_header = header;

        // buf.
        const size_t buf_size = header.line_offset * header.height;
        memcpy(img_info_body, src_buf, buf_size);

        if (!SemPost(sem)) return false;

        return true;
    }

    // receive from shm.
    bool ReceiveHeader(st_IMAGE_INFO_HEADER &header)
    {
        if (!SemWait(sem)) return false;

        // header.
        header = *img_info_header;

        if (!SemPost(sem)) return false;

        return true;
    }
    //
    bool ReceiveBuf(uint8_t *dst_buf, size_t buf_size)
    {
        if (dst_buf == nullptr) return false;

        if (!SemWait(sem)) return false;

        // buf.
        memcpy(dst_buf, img_info_body, buf_size);

        if (!SemPost(sem)) return false;

        return true;
    }
    //
    bool Receive(st_IMAGE_INFO_HEADER &header, uint8_t *dst_buf)
    {
        if (dst_buf == nullptr) return false;

        if (!SemWait(sem)) return false;

        // header.
        header = *img_info_header;

        // buf.
        const size_t buf_size = header.line_offset * header.height;
        memcpy(dst_buf, img_info_body, buf_size);
        // sleep(3);    // for debug.

        if (!SemPost(sem)) return false;

        return true;
    }

    // send to shared buffer.
    static void Send(std::unique_ptr<SharedBuffer> &img_info, const SharedBuffer::st_IMAGE_INFO_HEADER &header, const void *img, const SharedBuffer::st_ID &sb_id)
    {
        if (img == nullptr) return;

        if (img_info == nullptr) {
            img_info = SharedBuffer::Create(sb_id, header);
            if (img_info == nullptr) return;
        }

        img_info->Send(header, static_cast<const uint8_t *>(img));
        // std::cout << "------------------- " << header.width << " x " << header.height << ", " << header.channel << ", " << header.pixel_size << " : ";
        // std::cout << "shm:check_sum(send): " << check_sum(src_buf, header.line_offset * header.height) << std::endl;
    }
    template<typename T> static void Send(std::unique_ptr<SharedBuffer> &img_info, const SharedBuffer::st_IMAGE_INFO_HEADER &header, const std::unique_ptr<T> img, const SharedBuffer::st_ID &sb_id)
    {
        if (img == nullptr) return;

        if (img_info == nullptr) {
            img_info = SharedBuffer::Create(sb_id, header);
            if (img_info == nullptr) return;
        }

        img_info->Send(header, static_cast<const uint8_t *>(img.get()));
        // std::cout << "------------------- " << header.width << " x " << header.height << ", " << header.channel << ", " << header.pixel_size << " : ";
        // std::cout << "shm:check_sum(send): " << check_sum(src_buf, header.line_offset * header.height) << std::endl;
    }

    // receice from shared buffer.
    static void Receive(std::unique_ptr<SharedBuffer> &img_info, SharedBuffer::st_IMAGE_INFO_HEADER &header, void *&img, const SharedBuffer::st_ID &sb_id)
    {
        if (img_info == nullptr) {
            img_info = SharedBuffer::Create(sb_id);
            if (img_info == nullptr) return;
        }

        img_info->ReceiveHeader(header);

        if (img == nullptr) {
            const size_t buf_size = header.line_offset * header.height;
            img = static_cast<uint8_t *>(malloc(buf_size));
            if (img == nullptr) {
                std::cout << "ERROR! malloc" << std::endl;
                return;
            }
        }

        img_info->Receive(header, static_cast<uint8_t *>(img));
        // std::cout << "------------------- " << header.width << " x " << header.height << ", " << header.channel << ", " << header.pixel_size << " : ";
        // std::cout << "shm:check_sum(receive): " << check_sum(dst_buf, header.line_offset * header.height) << std::endl;
    }
    template<typename T> static void Receive(std::unique_ptr<SharedBuffer> &img_info, SharedBuffer::st_IMAGE_INFO_HEADER &header, std::unique_ptr<T> &img, const SharedBuffer::st_ID &sb_id)
    {
        if (img_info == nullptr) {
            img_info = SharedBuffer::Create(sb_id);
            if (img_info == nullptr) return;
        }

        img_info->ReceiveHeader(header);

        if (img == nullptr) {
            const size_t buf_size = header.line_offset * header.height;
            img = std::make_unique<T>(buf_size);
            if (img == nullptr) {
                std::cout << "ERROR! make_unique" << std::endl;
                return;
            }
        }

        img_info->Receive(header, static_cast<uint8_t *>(img.get()));
        // std::cout << "------------------- " << header.width << " x " << header.height << ", " << header.channel << ", " << header.pixel_size << " : ";
        // std::cout << "shm:check_sum(receive): " << check_sum(dst_buf, header.line_offset * header.height) << std::endl;
    }

#ifdef USE_OPENCV
    // send to shared buffer.
    static void Send(std::unique_ptr<SharedBuffer> &img_info, const cv::Mat &img, const SharedBuffer::st_ID &sb_id)
    {
        if (img.empty()) return;

        const SharedBuffer::st_IMAGE_INFO_HEADER header = {
            .width = (uint32_t)img.cols,
            .height = (uint32_t)img.rows,
            .channel = (uint32_t)img.channels(),
            .pixel_size = (uint32_t)img.elemSize1(),
            .line_offset = (uint32_t)(img.cols * img.channels() * img.elemSize1()),
            .data_type = imageInfoDataTypeFromCVType(img.depth()),
        };

        const auto src_buf = static_cast<uint8_t *>(img.data);

        if (img_info == nullptr) {
            img_info = SharedBuffer::Create(sb_id, header);
            if (img_info == nullptr) return;
        }

        img_info->Send(header, src_buf);
        // std::cout << "------------------- " << header.width << " x " << header.height << ", " << header.channel << ", " << header.pixel_size << " : ";
        // std::cout << "shm:check_sum(send): " << check_sum(src_buf, header.line_offset * header.height) << std::endl;
    }

    // receice from shared buffer.
    static void Receive(std::unique_ptr<SharedBuffer> &img_info, cv::Mat &img, const SharedBuffer::st_ID &sb_id)
    {
        if (img_info == nullptr) {
            img_info = SharedBuffer::Create(sb_id);
            if (img_info == nullptr) return;
        }

        SharedBuffer::st_IMAGE_INFO_HEADER header;
        img_info->ReceiveHeader(header);

        std::cout << header << std::endl;

        auto cv_type = CV_MAKETYPE(CVTypeFromimageInfoDataType(header.data_type), header.channel);
        img = cv::Mat(header.height, header.width, cv_type);

        auto dst_buf = static_cast<uint8_t *>(img.data);

        img_info->Receive(header, dst_buf);
        // std::cout << "------------------- " << header.width << " x " << header.height << ", " << header.channel << ", " << header.pixel_size << " : ";
        // std::cout << "shm:check_sum(receive): " << check_sum(dst_buf, header.line_offset * header.height) << std::endl;
    }
#endif

    SharedBuffer(const st_ID &_id) : id(_id)
    {
        is_ro = true;
    }

    virtual ~SharedBuffer()
    {
        Destroy();
    }

protected:

private:
};

struct st_SharedBuffer {
    const SharedBuffer::st_ID id = {};
    // Image Info.
    std::unique_ptr<SharedBuffer> img_info = nullptr;
};
extern st_SharedBuffer DefaultSharedBuffer;

// SharedBuffer::em_IMAGE_INFO_DATA_TYPE -> type.
template<> struct SharedBuffer::imageInfoDataType<SharedBuffer::em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_8> { typedef uint8_t Type; };
template<> struct SharedBuffer::imageInfoDataType<SharedBuffer::em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_16> { typedef uint16_t Type; };
template<> struct SharedBuffer::imageInfoDataType<SharedBuffer::em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_32> { typedef uint32_t Type; };
template<> struct SharedBuffer::imageInfoDataType<SharedBuffer::em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_64> { typedef uint64_t Type; };
template<> struct SharedBuffer::imageInfoDataType<SharedBuffer::em_IMAGE_INFO_DATA_TYPE::INT_TYPE_8> { typedef int8_t Type; };
template<> struct SharedBuffer::imageInfoDataType<SharedBuffer::em_IMAGE_INFO_DATA_TYPE::INT_TYPE_16> { typedef int16_t Type; };
template<> struct SharedBuffer::imageInfoDataType<SharedBuffer::em_IMAGE_INFO_DATA_TYPE::INT_TYPE_32> { typedef int32_t Type; };
template<> struct SharedBuffer::imageInfoDataType<SharedBuffer::em_IMAGE_INFO_DATA_TYPE::INT_TYPE_64> { typedef int64_t Type; };
template<> struct SharedBuffer::imageInfoDataType<SharedBuffer::em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BINARY32> { typedef float Type; };
template<> struct SharedBuffer::imageInfoDataType<SharedBuffer::em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BINARY64> { typedef double Type; };
#ifdef __CUDACC__
template<> struct SharedBuffer::imageInfoDataType<SharedBuffer::em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BINARY16> { typedef half Type; };
template<> struct SharedBuffer::imageInfoDataType<SharedBuffer::em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BFLOAT16> { typedef nv_bfloat16 Type; };
#endif

#ifdef USE_OPENCV
// CV type -> type.
template<> struct SharedBuffer::CVTpeyType<CV_8U> { typedef uint8_t Type; };
template<> struct SharedBuffer::CVTpeyType<CV_8S> { typedef int8_t Type; };
template<> struct SharedBuffer::CVTpeyType<CV_16U> { typedef uint16_t Type; };
template<> struct SharedBuffer::CVTpeyType<CV_16S> { typedef int16_t Type; };
template<> struct SharedBuffer::CVTpeyType<CV_32S> { typedef int32_t Type; };
template<> struct SharedBuffer::CVTpeyType<CV_32F> { typedef float Type; };
template<> struct SharedBuffer::CVTpeyType<CV_64F> { typedef double Type; };
#ifdef __CUDACC__
template<> struct SharedBuffer::CVTpeyType<CV_16F> { typedef half Type; };
#endif
#endif

// type -> SharedBuffer::em_IMAGE_INFO_DATA_TYPE.
template<typename T> struct __data_type_assert_false : std::false_type { };
template<typename T> inline SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer_imageInfoDataTypeFromType()
{
    static_assert(__data_type_assert_false<T>::value, "invalid data type - supported types are int{8|16|32|64}, uint{8|16|32|64}, float, double.");
    return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_8;
}
template<> inline SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer_imageInfoDataTypeFromType<uint8_t>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_8; }
template<> inline SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer_imageInfoDataTypeFromType<uint16_t>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_16; }
template<> inline SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer_imageInfoDataTypeFromType<uint32_t>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_32; }
template<> inline SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer_imageInfoDataTypeFromType<uint64_t>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::UINT_TYPE_64; }
template<> inline SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer_imageInfoDataTypeFromType<int8_t>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::INT_TYPE_8; }
template<> inline SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer_imageInfoDataTypeFromType<int16_t>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::INT_TYPE_16; }
template<> inline SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer_imageInfoDataTypeFromType<int32_t>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::INT_TYPE_32; }
template<> inline SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer_imageInfoDataTypeFromType<int64_t>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::INT_TYPE_64; }
template<> inline SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer_imageInfoDataTypeFromType<float>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BINARY32; }
template<> inline SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer_imageInfoDataTypeFromType<double>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BINARY64; }
#ifdef __CUDACC__
template<> inline SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer_imageInfoDataTypeFromType<nv_bfloat16>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BFLOAT16; }
template<> inline SharedBuffer::em_IMAGE_INFO_DATA_TYPE SharedBuffer_imageInfoDataTypeFromType<half>() { return SharedBuffer::em_IMAGE_INFO_DATA_TYPE::FLOAT_TYPE_BINARY16; }
#endif
