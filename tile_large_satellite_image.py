"""Overlapping tile iterator for large RGB imagery."""
import numpy as np

def tile_starts(length, tile_size, stride):
    if length <= tile_size: return [0]
    starts = list(range(0, length-tile_size+1, stride))
    if starts[-1] != length-tile_size: starts.append(length-tile_size)
    return starts

def iter_tiles(image, tile_size=512, overlap=64):
    image = np.asarray(image)
    h,w = image.shape[:2]
    for top in tile_starts(h,tile_size,tile_size-overlap):
        for left in tile_starts(w,tile_size,tile_size-overlap):
            tile = image[top:min(top+tile_size,h), left:min(left+tile_size,w)]
            th,tw = tile.shape[:2]
            if th<tile_size or tw<tile_size:
                tile = np.pad(tile, ((0,tile_size-th),(0,tile_size-tw),(0,0)), mode="edge")
            yield top,left,th,tw,tile
