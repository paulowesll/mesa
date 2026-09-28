#include <sync/sync.h>

#include <errno.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>

/* Termux builds load this library at runtime through the RUNPATH, so it must
 * implement the sync_file uapi instead of returning placeholder values.  This
 * mirrors the non-Android helpers in src/util/libsync.h. */

extern "C" {

/* timeout in msecs */
int sync_wait(int fd, int timeout)
{
   struct pollfd fds = {};
   fds.fd = fd;
   fds.events = POLLIN;

   for (;;) {
      struct timespec start, end;
      clock_gettime(CLOCK_MONOTONIC, &start);
      int ret = poll(&fds, 1, timeout);
      clock_gettime(CLOCK_MONOTONIC, &end);

      if (ret > 0) {
         if (fds.revents & (POLLERR | POLLNVAL)) {
            errno = EINVAL;
            return -1;
         }
         return 0;
      } else if (ret == 0) {
         errno = ETIME;
         return -1;
      } else if (errno != EINTR && errno != EAGAIN) {
         return ret;
      }

      if (timeout > 0) {
         const long elapsed_ms =
            (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
         timeout = elapsed_ms >= timeout ? 0 : timeout - elapsed_ms;
      }
   }
}

int sync_merge(const char *name, int fd1, int fd2)
{
   struct sync_merge_data data = {};
   int ret;

   data.fd2 = fd2;
   strncpy(data.name, name, sizeof(data.name) - 1);

   do {
      ret = ioctl(fd1, SYNC_IOC_MERGE, &data);
   } while (ret == -1 && (errno == EINTR || errno == EAGAIN));

   if (ret < 0)
      return ret;

   return data.fence;
}

struct sync_file_info* sync_file_info(int32_t fd)
{
   struct sync_file_info local_info = {};
   if (ioctl(fd, SYNC_IOC_FILE_INFO, &local_info) < 0)
      return nullptr;

   struct sync_file_info *info = static_cast<struct sync_file_info *>(
      calloc(1, sizeof(*info) + local_info.num_fences * sizeof(struct sync_fence_info)));
   if (!info)
      return nullptr;

   info->num_fences = local_info.num_fences;
   info->sync_fence_info = reinterpret_cast<uintptr_t>(info + 1);

   if (ioctl(fd, SYNC_IOC_FILE_INFO, info) < 0) {
      free(info);
      return nullptr;
   }

   return info;
}

void sync_file_info_free(struct sync_file_info* info)
{
   free(info);
}

}
