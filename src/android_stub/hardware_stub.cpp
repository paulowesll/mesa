#include <hardware/hardware.h>

#include <errno.h>

extern "C" {

/* There is no gralloc HAL module to hand out here.  Report it as missing
 * rather than returning success without writing *module: callers such as
 * u_gralloc's CrOS backend dereference the module as soon as we return 0,
 * which crashed every AHardwareBuffer import.
 */
int hw_get_module(const char *id, const struct hw_module_t **module)
{
   if (module)
      *module = NULL;
   return -ENOENT;
}

}
