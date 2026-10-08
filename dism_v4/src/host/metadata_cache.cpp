#include "host/metadata_upload.h"
#include "variant.cuh"
namespace DISM_VARIANT {
void clear_metadata_cache() { dism_metadata::clear_cache(); }
}
