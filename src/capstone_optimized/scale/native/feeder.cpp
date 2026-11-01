// Background file reader for the out-of-core Polya engine (scale/engine.py).
//
// Worker threads read file ranges into caller-supplied buffers (pinned host memory the GPU reads directly).
// Requests are grouped by "slot": the caller submits the reads for one block into one slot and later waits for
// that slot only, so the reads for the next blocks run while the GPU works on the current one.
// Each thread has its own handle per file (a synchronous handle serializes its reads), and files can be opened
// unbuffered (FILE_FLAG_NO_BUFFERING: no copy through the Windows file cache, so a graph larger than RAM does
// not evict everything else). Unbuffered offsets, lengths and buffers must be multiples of 4096.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Task {
    int file;
    long long off, len;
    char* dst;
    int slot;
};

struct Feeder {
    int nthreads = 0;
    long long chunk = 0;
    std::vector<std::vector<HANDLE>> h;  // h[thread][file]
    std::vector<std::thread> threads;
    std::mutex mu;
    std::condition_variable cv_work, cv_done;
    std::deque<Task> q;
    std::vector<long long> pending;
    std::vector<int> err;
    std::atomic<long long> bytes{0};
    bool stop = false;
};

void worker(Feeder* f, int t)
{
    for (;;) {
        Task task;
        HANDLE hf;
        {
            std::unique_lock<std::mutex> lk(f->mu);
            f->cv_work.wait(lk, [&] { return f->stop || !f->q.empty(); });
            if (f->q.empty()) return;
            task = f->q.front();
            f->q.pop_front();
            hf = task.file >= 0 ? f->h[t][task.file] : nullptr;
        }
        int e = 0;
        long long done = 0;
        if (task.file < 0) {  // a copy from host memory (the pageable RAM tier), source address in off
            memcpy(task.dst, (const char*)(intptr_t)task.off, (size_t)task.len);
            done = task.len;
        }
        while (done < task.len) {
            OVERLAPPED ov = {};
            long long o = task.off + done;
            ov.Offset = (DWORD)(o & 0xffffffffLL);
            ov.OffsetHigh = (DWORD)(o >> 32);
            DWORD want = (DWORD)std::min<long long>(task.len - done, 1LL << 30), got = 0;
            if (!ReadFile(hf, task.dst + done, want, &got, &ov)) {
                DWORD le = GetLastError();
                if (le != ERROR_HANDLE_EOF) e = (int)le;
                break;
            }
            done += got;
            if (got < want) break;  // end of file
        }
        f->bytes += done;
        std::lock_guard<std::mutex> lk(f->mu);
        if (e) f->err[task.slot] = e;
        if (--f->pending[task.slot] == 0) f->cv_done.notify_all();
    }
}

}  // namespace

extern "C" {

__declspec(dllexport) void* feeder_create(int nthreads, int nslots, long long chunk)
{
    Feeder* f = new Feeder();
    f->nthreads = nthreads;
    f->chunk = chunk;
    f->h.resize(nthreads);
    f->pending.assign(nslots, 0);
    f->err.assign(nslots, 0);
    for (int t = 0; t < nthreads; ++t) f->threads.emplace_back(worker, f, t);
    return f;
}

// Returns the file's index, or -(Windows error code).
__declspec(dllexport) int feeder_open(void* p, const wchar_t* path, int unbuffered)
{
    Feeder* f = (Feeder*)p;
    DWORD flags = FILE_FLAG_SEQUENTIAL_SCAN | (unbuffered ? FILE_FLAG_NO_BUFFERING : 0);
    std::vector<HANDLE> hs;
    for (int t = 0; t < f->nthreads; ++t) {
        HANDLE hf = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, flags, nullptr);
        if (hf == INVALID_HANDLE_VALUE) {
            int e = (int)GetLastError();
            for (HANDLE x : hs) CloseHandle(x);
            return -e;
        }
        hs.push_back(hf);
    }
    std::lock_guard<std::mutex> lk(f->mu);
    for (int t = 0; t < f->nthreads; ++t) f->h[t].push_back(hs[t]);
    return (int)f->h[0].size() - 1;
}

// Queue a read of [off, off + len) of a file into dst, as part of a slot. Split into chunk-sized tasks so
// several threads (and several NVMe queue entries) work on one large read.
__declspec(dllexport) int feeder_submit(void* p, int slot, int file, long long off, long long len, void* dst)
{
    Feeder* f = (Feeder*)p;
    std::lock_guard<std::mutex> lk(f->mu);
    for (long long d = 0; d < len; d += f->chunk) {
        f->q.push_back(Task{file, off + d, std::min(f->chunk, len - d), (char*)dst + d, slot});
        ++f->pending[slot];
    }
    f->cv_work.notify_all();
    return 0;
}

// Queue a copy of len bytes from host memory src into dst, as part of a slot (split like a read).
__declspec(dllexport) int feeder_submit_copy(void* p, int slot, const void* src, long long len, void* dst)
{
    return feeder_submit(p, slot, -1, (long long)(intptr_t)src, len, dst);
}

// Wait until every read of the slot is done; returns 0 or the first Windows error code.
__declspec(dllexport) int feeder_wait(void* p, int slot)
{
    Feeder* f = (Feeder*)p;
    std::unique_lock<std::mutex> lk(f->mu);
    f->cv_done.wait(lk, [&] { return f->pending[slot] == 0; });
    int e = f->err[slot];
    f->err[slot] = 0;
    return e;
}

__declspec(dllexport) long long feeder_bytes(void* p) { return ((Feeder*)p)->bytes.load(); }

__declspec(dllexport) void feeder_destroy(void* p)
{
    Feeder* f = (Feeder*)p;
    {
        std::lock_guard<std::mutex> lk(f->mu);
        f->stop = true;
    }
    f->cv_work.notify_all();
    for (auto& th : f->threads) th.join();
    for (auto& hs : f->h)
        for (HANDLE x : hs) CloseHandle(x);
    delete f;
}

}  // extern "C"
