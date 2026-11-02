"""
Unit tests for DirectBlockWriter's O_DIRECT mode. They run anywhere, including macOS, by replacing
the direct file descriptor with a plain one and recording what each write was given. The last test
uses real O_DIRECT where the platform and filesystem support it.
"""

import ctypes
import errno
import mmap
import os
import shutil
import tempfile
import unittest
from unittest import mock

from hypercache.datapath import ChunkLocation, DirectBlockWriter

BLOCK = 4096


def pattern(size, seed=7):
    return bytes((i * seed + i // 251) % 256 for i in range(size))


def aligned_buffer(data, shift=0):
    view = memoryview(mmap.mmap(-1, len(data) + shift + BLOCK))[shift : shift + len(data)]
    view[:] = data
    return view


class WriterTestCase(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp()
        os.ftruncate(handle, 64 * 1024 * 1024)
        os.close(handle)
        self.addCleanup(os.unlink, self.path)
        self.direct_fds = []
        self.direct_writes = []
        self.real_pwrite = os.pwrite

        def open_direct(path):
            fd = os.open(path, os.O_RDWR)
            self.direct_fds.append(fd)
            return fd

        for target, replacement in (
            ("hypercache.datapath.DirectBlockWriter._is_block_device", staticmethod(lambda path: True)),
            ("hypercache.datapath.DirectBlockWriter._open_direct", staticmethod(open_direct)),
            ("hypercache.datapath.logical_block_size", lambda path: BLOCK),
            ("hypercache.datapath.os.pwrite", self.spy_pwrite),
        ):
            patcher = mock.patch(target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        if not hasattr(os, "O_DIRECT"):
            patcher = mock.patch("hypercache.datapath.os.O_DIRECT", 0, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.writer = DirectBlockWriter(max_workers=4, direct_on_block_devices=True)
        self.addCleanup(self.writer.close)

    def spy_pwrite(self, fd, data, offset):
        if fd in self.direct_fds:
            address = ctypes.addressof(ctypes.c_char.from_buffer(data))
            self.direct_writes.append((address % BLOCK, len(data), offset))
        return self.real_pwrite(fd, data, offset)

    def layout(self, *lengths, start=BLOCK):
        chunks, offset = [], start
        for length in lengths:
            chunks.append(ChunkLocation(device=self.path, offset=offset, length=length))
            offset += (length + BLOCK - 1) // BLOCK * BLOCK
        return chunks

    def on_disk(self, offset, length):
        with open(self.path, "rb") as f:
            f.seek(offset)
            return f.read(length)

    def assert_direct_writes_are_aligned(self):
        self.assertTrue(self.direct_writes, "expected O_DIRECT to be used")
        for address_remainder, length, offset in self.direct_writes:
            self.assertEqual(address_remainder, 0)
            self.assertEqual(length % BLOCK, 0)
            self.assertEqual(offset % BLOCK, 0)

    def test_aligned_buffer_goes_direct_across_chunks(self):
        data = pattern(3 * 1024 * 1024)
        self.writer.write_chunks(self.layout(1024 * 1024, 1024 * 1024, 1024 * 1024), aligned_buffer(data))
        self.assertEqual(self.on_disk(BLOCK, len(data)), data)
        self.assert_direct_writes_are_aligned()
        self.assertEqual(sum(w[1] for w in self.direct_writes), len(data))

    def test_unaligned_buffer_is_copied_to_an_aligned_one(self):
        data = pattern(256 * 1024)
        self.writer.write_chunks(self.layout(len(data)), aligned_buffer(data, shift=1))
        self.assertEqual(self.on_disk(BLOCK, len(data)), data)
        self.assert_direct_writes_are_aligned()

    def test_read_only_bytes_are_supported(self):
        data = pattern(64 * 1024)
        self.writer.write_chunks(self.layout(len(data)), data)
        self.assertEqual(self.on_disk(BLOCK, len(data)), data)
        self.assert_direct_writes_are_aligned()

    def test_partial_last_block_does_not_touch_its_neighbour(self):
        neighbour_offset = BLOCK + 3 * BLOCK
        with open(self.path, "r+b") as f:
            f.seek(BLOCK + 2 * BLOCK + 100)
            f.write(b"\xAB" * (neighbour_offset - (BLOCK + 2 * BLOCK + 100) + BLOCK))
        data = pattern(2 * BLOCK + 100)
        self.writer.write_chunks(self.layout(len(data)), aligned_buffer(data))
        self.assertEqual(self.on_disk(BLOCK, len(data)), data)
        self.assertEqual(self.on_disk(BLOCK + len(data), 8), b"\xAB" * 8)
        self.assertEqual(self.on_disk(neighbour_offset, BLOCK), b"\xAB" * BLOCK)
        self.assertEqual(sum(w[1] for w in self.direct_writes), 2 * BLOCK)
        self.assert_direct_writes_are_aligned()

    def test_less_than_a_block_never_goes_direct(self):
        data = pattern(100)
        self.writer.write_chunks(self.layout(100), aligned_buffer(data))
        self.assertEqual(self.on_disk(BLOCK, 100), data)
        self.assertEqual(self.direct_writes, [])

    def test_unaligned_offset_never_goes_direct(self):
        data = pattern(2 * BLOCK)
        self.writer.write_chunks(self.layout(len(data), start=BLOCK + 512), aligned_buffer(data))
        self.assertEqual(self.on_disk(BLOCK + 512, len(data)), data)
        self.assertEqual(self.direct_writes, [])

    def test_refused_direct_write_falls_back_to_the_page_cache(self):
        data = pattern(16 * BLOCK)

        def refuse(fd, buffer, offset):
            if fd in self.direct_fds:
                raise OSError(errno.EINVAL, "refused")
            return self.real_pwrite(fd, buffer, offset)

        with mock.patch("hypercache.datapath.os.pwrite", refuse):
            self.writer.write_chunks(self.layout(len(data)), aligned_buffer(data))
        self.assertEqual(self.on_disk(BLOCK, len(data)), data)

    def test_short_direct_writes_complete_the_chunk(self):
        for short in (BLOCK, BLOCK + 17):
            with self.subTest(short=short):
                data = pattern(16 * BLOCK, seed=short % 13 + 1)
                calls = []

                def one_short_write(fd, buffer, offset, short=short, calls=calls):
                    if fd in self.direct_fds and not calls:
                        calls.append(1)
                        return self.real_pwrite(fd, buffer[:short], offset)
                    return self.real_pwrite(fd, buffer, offset)

                with mock.patch("hypercache.datapath.os.pwrite", one_short_write):
                    self.writer.write_chunks(self.layout(len(data)), aligned_buffer(data))
                self.assertEqual(self.on_disk(BLOCK, len(data)), data)

    def test_real_errors_are_reported(self):
        def full(fd, buffer, offset):
            raise OSError(errno.ENOSPC, "no space")

        with mock.patch("hypercache.datapath.os.pwrite", full):
            with self.assertRaises(OSError) as caught:
                self.writer.write_chunks(self.layout(4 * BLOCK), aligned_buffer(pattern(4 * BLOCK)))
        self.assertEqual(caught.exception.errno, errno.ENOSPC)

    def test_a_write_that_makes_no_progress_is_an_error(self):
        with mock.patch("hypercache.datapath.os.pwrite", lambda fd, buffer, offset: 0):
            with self.assertRaises(IOError):
                self.writer.write_chunks(self.layout(4 * BLOCK), aligned_buffer(pattern(4 * BLOCK)))

    def test_off_by_default(self):
        writer = DirectBlockWriter(max_workers=2)
        self.addCleanup(writer.close)
        data = pattern(4 * BLOCK)
        writer.write_chunks(self.layout(len(data)), aligned_buffer(data))
        self.assertEqual(self.on_disk(BLOCK, len(data)), data)
        self.assertEqual(self.direct_writes, [])

    def test_regular_files_are_not_direct(self):
        with mock.patch("hypercache.datapath.DirectBlockWriter._is_block_device", staticmethod(lambda path: False)):
            data = pattern(4 * BLOCK)
            self.writer.write_chunks(self.layout(len(data)), aligned_buffer(data))
        self.assertEqual(self.on_disk(BLOCK, len(data)), data)
        self.assertEqual(self.direct_writes, [])

    def test_failure_to_open_direct_uses_the_page_cache(self):
        def cannot_open(path):
            raise OSError(errno.EINVAL, "no O_DIRECT here")

        with mock.patch("hypercache.datapath.DirectBlockWriter._open_direct", staticmethod(cannot_open)):
            data = pattern(4 * BLOCK)
            self.writer.write_chunks(self.layout(len(data)), aligned_buffer(data))
        self.assertEqual(self.on_disk(BLOCK, len(data)), data)

    def test_ram_chunks_never_go_direct(self):
        target = self.path + ".ram"
        self.addCleanup(lambda: os.path.exists(target) and os.unlink(target))
        data = pattern(4 * BLOCK)
        chunk = ChunkLocation(device=target, offset=0, length=len(data), tier="ram")
        self.writer.write_chunks([chunk], aligned_buffer(data))
        with open(target, "rb") as f:
            self.assertEqual(f.read(), data)
        self.assertEqual(self.direct_writes, [])


class RealDirectIOTest(unittest.TestCase):
    @unittest.skipUnless(hasattr(os, "O_DIRECT"), "needs O_DIRECT")
    def test_real_o_direct_or_fallback_writes_correct_bytes(self):
        # Not every filesystem allows O_DIRECT. Either way, the bytes must be right.
        directory = tempfile.mkdtemp(dir=os.getcwd())
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        path = os.path.join(directory, "device")
        with open(path, "wb") as f:
            f.truncate(8 * 1024 * 1024)
        data = pattern(2 * 1024 * 1024 + 300)
        layout = [ChunkLocation(device=path, offset=4096, length=len(data))]
        with mock.patch("hypercache.datapath.DirectBlockWriter._is_block_device", staticmethod(lambda p: True)):
            writer = DirectBlockWriter(max_workers=2, direct_on_block_devices=True)
            writer.write_chunks(layout, aligned_buffer(data))
            writer.close()
        with open(path, "rb") as f:
            f.seek(4096)
            self.assertEqual(f.read(len(data)), data)


if __name__ == "__main__":
    unittest.main()
