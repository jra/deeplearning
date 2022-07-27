#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include <MagickWand/MagickWand.h>

#define ThrowWandException(wand) \
{ \
  char \
    *description; \
 \
  ExceptionType \
    severity; \
 \
  description=MagickGetException(wand,&severity); \
  (void) fprintf(stderr,"%s %s %lu %s\n",GetMagickModule(),description); \
  description=(char *) MagickRelinquishMemory(description); \
  exit(-1); \
}

void read_resize_image(char* path_image, uint32_t nrows, uint32_t ncols, double* data_image) {
    // Read an image.
    MagickWandGenesis();
    MagickWand *magick_wand = NULL;
    magick_wand = NewMagickWand();
    MagickBooleanType status;
    status = MagickReadImage(magick_wand, path_image);
    if (status == MagickFalse) {
        ThrowWandException(magick_wand);
    }
    MagickResizeImage(magick_wand, ncols, nrows, LanczosFilter);
    PixelIterator *iterator = NULL;
    iterator = NewPixelIterator(magick_wand);
    size_t y = 0;
    size_t x = 0;
    unsigned long width;
    for (y = 0; y < (size_t) MagickGetImageHeight(magick_wand); y++) {
        PixelWand **pixels;
        pixels = PixelGetNextIteratorRow(iterator, &width);
        if (pixels == (PixelWand **) NULL) {
            break;
        }
        for (x = 0; x < (size_t) width; x++) {
            data_image[y * (width * 3) + x * 3 + 0] = PixelGetRed(pixels[x]);
            data_image[y * (width * 3) + x * 3 + 1] = PixelGetGreen(pixels[x]);
            data_image[y * (width * 3) + x * 3 + 2] = PixelGetBlue(pixels[x]);
        }
    }
    //printf("final x, y, width = %zu, %zu, %lu\n", x, y, width);
    //printf("(y-1) * width + (x-1) * 3 = %lu\n", (y-1) * width + (x-1) * 3);
    if (y < (long) MagickGetImageHeight(magick_wand)) {
        ThrowWandException(magick_wand);
    }
    //PixelSyncIterator(iterator);
    iterator = DestroyPixelIterator(iterator);
    //status = MagickWriteImages(magick_wand, "tn2.ppm", MagickTrue);
    //if (status == MagickFalse) {
    //    ThrowWandException(magick_wand);
    //}
    magick_wand = DestroyMagickWand(magick_wand);
    MagickWandTerminus();
}
