import os
import numpy as np
import h5py
from tqdm import tqdm
from .io import read_image, write_image, read_h5, get_h5_chunk2d

def get_tile_name(pattern, row=None, column=None):
    """
    Generate the tile name based on the pattern and indices.

    Args:
        pattern (str): The pattern for the tile name.
        row (int, optional): The row index. Defaults to None.
        column (int, optional): The column index. Defaults to None.

    Returns:
        str: The generated tile name.
    """    
    
    if "%" in pattern:
        return pattern % (row, column)
    elif "{" in pattern:
        return pattern.format(row=row, column=column)
    else:
        return pattern
    
    
def read_tile_image_by_bbox(filenames, z0p, z1p, y0p, y1p, x0p, x1p, tile_sz, tile_st=None, 
                     tile_dtype=np.uint8, tile_type="image", tile_ratio=1, 
                     tile_resize_mode=1, tile_border_padding="reflect", tile_blank="", 
                     volume_sz=None, zstep=1, no_tqdm=False, output_file=None, output_chunk=8192, relabel=None):
    """
    Read and assemble a volume from a set of tiled images.

    Args:
        filenames (list): The list of file names or patterns for the tiled images.
        z0p (int): The starting index of the z-axis.
        z1p (int): The ending index of the z-axis.
        y0p (int): The starting index of the y-axis.
        y1p (int): The ending index of the y-axis.
        x0p (int): The starting index of the x-axis.
        x1p (int): The ending index of the x-axis.
        tile_sz (int or list): The size of each tile in pixels. If an integer is provided, the same size is used for both dimensions.
        tile_st (list, optional): The starting index of the tiles. Defaults to [0, 0].
        tile_dtype (numpy.dtype, optional): The data type of the tiles. Defaults to np.uint8.
        tile_type (str, optional): "image" or "seg"
        tile_ratio (float or list, optional): The scaling factor for resizing the tiles. If a float is provided, the same ratio is used for both dimensions. Defaults to 1.
        tile_resize_mode (int, optional): The interpolation mode for resizing the tiles. Defaults to 1.
        tile_seg (bool, optional): Whether the tiles represent segmentation maps. Defaults to False.
        tile_border_padding (str, optional): The padding mode for tiles at the boundary. Defaults to "reflect".
        tile_blank (str, optional): The value or pattern to fill empty tiles. Defaults to "".
        volume_sz (list, optional): The size of the volume in each dimension. Defaults to None.
        zstep (int, optional): The step size for the z-axis. Defaults to 1.

    Returns:
        numpy.ndarray: The assembled volume.

    Notes:        
        - The tiles are specified by file names or patterns in the `filenames` parameter.
        - The volume is constructed by arranging the tiles according to their indices.
        - The size of each tile is specified by the `tile_sz` parameter.
        - The tiles can be resized using the `tile_ratio` parameter.
        - The tiles can be interpolated using the `tile_resize_mode` parameter.
        - The tiles can represent either grayscale images or segmentation maps.
        - The volume can be padded at the boundary using the `tile_bd` parameter.
        - Empty tiles can be filled with a value or pattern using the `tile_blank` parameter.
        - The size of the volume can be specified using the `volume_sz` parameter.
        - The step size for the z-axis can be adjusted using the `zstep` parameter.
    """
    if tile_st is None:
        tile_st = [0, 0]
    if not isinstance(tile_sz, (list,)):
        tile_sz = [tile_sz, tile_sz]
    if not isinstance(tile_ratio, (list,)):
        tile_ratio = [tile_ratio, tile_ratio]
    # [row,col]
    # no padding at the boundary
    # st: starting index 0 or 1

    bd = None
    if volume_sz is not None:
        bd = [
            max(-z0p, 0),
            max(0, z1p - volume_sz[0]),
            max(-y0p, 0),
            max(0, y1p - volume_sz[1]),
            max(-x0p, 0),
            max(0, x1p - volume_sz[2]),
        ]
        z0, y0, x0 = max(z0p, 0), max(y0p, 0), max(x0p, 0)
        z1, y1, x1 = (
            min(z1p, volume_sz[0]),
            min(y1p, volume_sz[1]),
            min(x1p, volume_sz[2]),
        )
    else:
        z0, y0, x0, z1, y1, x1 = z0p, y0p, x0p, z1p, y1p, x1p
    
    result_shape = ((z1 - z0 + zstep - 1) // zstep, y1 - y0, x1 - x0)
    if output_file is None:
        result = np.zeros(result_shape, tile_dtype)
    else:
        fid = h5py.File(output_file, 'w')
        chunk_sz = get_h5_chunk2d(output_chunk, result_shape[1:])
        chunk_sz = [min(chunk_sz[x],tile_sz[x]) for x in range(2)]
        result = fid.create_dataset('main', result_shape, dtype=tile_dtype, \
            compression="gzip", chunks=(1,chunk_sz[0],chunk_sz[1]))
    
    c0 = x0 // tile_sz[1]  # floor
    c1 = (x1 + tile_sz[1] - 1) // tile_sz[1]  # ceil
    r0 = y0 // tile_sz[0]
    r1 = (y1 + tile_sz[0] - 1) // tile_sz[0]
    z1 = min(len(filenames) - 1, z1)
    for i, z in tqdm(enumerate(range(z0, z1, zstep)), disable=no_tqdm):
        pattern = filenames[z]
        for row in range(r0, r1):
            for column in range(c0, c1):
                filename = get_tile_name(pattern, row + tile_st[0], column + tile_st[1])
                if os.path.exists(filename):
                    patch = read_image(filename, tile_type, tile_ratio, tile_resize_mode)
                    if relabel is not None:
                        patch = relabel[patch]
                    # exception: last tile may not have the right size
                    psz = patch.shape
                    xp0 = column * tile_sz[1]
                    xp1 = min(xp0 + psz[1], (column + 1) * tile_sz[1])
                    yp0 = row * tile_sz[0]
                    yp1 = min(yp0 + psz[0], (row + 1) * tile_sz[0])
                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    y0a = max(y0, yp0)
                    y1a = min(y1, yp1)
                    result[i, y0a - y0 : y1a - y0, x0a - x0 : x1a - x0] = (
                            patch[y0a - yp0 : y1a - yp0, x0a - xp0 : x1a - xp0]
                    )                    
                else:
                    print(f"Non-exist: {filename}")
    # blank case
    if tile_blank != "":
        blank_st = 0
        blank_lt = result.shape[0] - 1
        while blank_st <= blank_lt and not np.any(result[blank_st] > 0):
            blank_st += 1
        if blank_st == blank_lt + 1:
            print("!! This volume is all 0 !!")
        else:
            result[:blank_st] = result[blank_st : blank_st + 1]
            while blank_lt >= blank_st and not np.any(result[blank_lt] > 0):
                blank_lt -= 1
            result[blank_lt:] = result[blank_lt - 1 : blank_lt]
            for z in range(blank_st + 1, blank_lt):
                if not np.any(result[z] > 0):
                    result[z] = result[z - 1]
    
    if output_file is None:
        # boundary case
        if bd is not None and max(bd) > 0:
            result = np.pad(
                result, ((bd[0], bd[1]), (bd[2], bd[3]), (bd[4], bd[5])), tile_border_padding
            )
    else:
        fid.close()
    return result


def read_tile_h5_by_bbox(h5_name, z0, z1, y0, y1, x0, x1, zyx_sz, zyx0=[0,0,0], \
    cid=-1, dt=np.uint16, zz=[0,0,-1], tile_step=1, zstep=1, acc_id = False, \
        h5_key='main', no_tqdm=False, output_file=None, output_chunk=8192, tile_type='image', mask=None, h5_func=None):
    if not isinstance(tile_step, (list,)):
        tile_step = [tile_step, tile_step]
    # zz: extra number of slices in the first and last chunk 
    # zz[2]: last zid slice
    
    if tile_type == 'image':
        dt = np.uint8
    if output_file is None:
        # return the whole volume
        result = np.zeros((z1-z0, y1-y0, x1-x0), dt)
    else:
        if '.h5' in output_file:
            # output as h5
            fid_out = h5py.File(output_file, 'w')
            num_z = zyx_sz[0]
            chunk_sz = get_h5_chunk2d(output_chunk/num_z, zyx_sz[1:])
            result = fid_out.create_dataset('main', (z1-z0, y1-y0, x1-x0), \
                dtype=dt, compression="gzip", chunks=tuple([num_z,chunk_sz[0],chunk_sz[1]]))
        else:
            # output images for VAST import
            result = np.zeros((zyx_sz[0] + max(zz[:2]), y1-y0, x1-x0), dt) 
    
    c0 = max(0,x0-zyx0[2]) // zyx_sz[2] # floor
    c1 = (x1-zyx0[2]+zyx_sz[2]-1) // zyx_sz[2] # ceil
    r0 = max(0,y0-zyx0[1]) // zyx_sz[1]
    r1 = (y1-zyx0[1]+zyx_sz[1]-1) // zyx_sz[1]
    d0 = max(0,z0-zyx0[0]-zz[0]) // zyx_sz[0]
    d1 = (z1-zyx0[0]-zz[0]+zyx_sz[0]-1) // zyx_sz[0]
    if zz[2] > 0:
        d1 = min(zz[2], d1)

    mid = 0
    for zid in tqdm(range(d0, d1), disable=no_tqdm):
        zp0 = zid * zyx_sz[0] + zyx0[0]
        # chunk afterwards start with extra zz[0]
        if zid != 0:
            zp0 += zz[0]
        # all chunks end with extra zz[0]+zz[1]
        zp1 = (zid + 1) * zyx_sz[0]+ zyx0[0]                                        
        zp1 += zz[0]
        if zid == zz[2]:
            zp1 += zz[1] 
        z0a = max(z0, zp0)
        z1a = min(z1, zp1)        
        for yid in range(r0, r1):
            yp0 = yid * zyx_sz[1] + zyx0[1]
            yp1 = (yid + 1) * zyx_sz[1]+ zyx0[1]
            y0a = max(y0, yp0)
            y1a = min(y1, yp1)
            for xid in range(c0, c1):
                path = h5_name(zid,yid,xid)                
                if os.path.exists(path):
                    print(path)
                    if h5_func is None:
                        fid_file= h5py.File(path,'r')
                        fid = fid_file[h5_key]
                    else:                        
                        fid = read_h5(path, h5_key)
                        fid = h5_func(fid, zid, yid, xid)                        
                    
                    xp0 = xid * zyx_sz[2] + zyx0[2]
                    xp1 = (xid+1) * zyx_sz[2]+ zyx0[2]                    
                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    # cid: channel selection
                    # print(z0a, z1a, y0a, y1a, x0a, x1a)
                    # print(zp0, zp1, yp0, yp1, xp0, xp1)
                    if cid==-1:
                        if len(fid.shape)==3:                            
                            tmp = np.array(fid[(z0a - zp0)*zstep : (z1a - zp0)*zstep : zstep, \
                                               (y0a - yp0)*tile_step[0] : (y1a - yp0)*tile_step[0] : tile_step[0], \
                                               (x0a - xp0)*tile_step[1] : (x1a - xp0)*tile_step[1] : tile_step[1]])
                        else:
                            tmp = np.array(fid[(y0a - yp0)*tile_step[0] : (y1a - yp0)*tile_step[0] : tile_step[0], \
                                               (x0a - xp0)*tile_step[1] : (x1a - xp0)*tile_step[1] : tile_step[1]])
                    elif cid==-2:
                        tmp = np.array(fid[: , (z0a - zp0)*zstep : (z1a - zp0)*zstep : zstep, \
                                               (y0a - yp0)*tile_step[0] : (y1a - yp0)*tile_step[0] : tile_step[0], \
                                               (x0a - xp0)*tile_step[1] : (x1a - xp0)*tile_step[1] : tile_step[1]])
                    else:
                        tmp = np.array(fid[cid, (z0a - zp0)*zstep : (z1a - zp0)*zstep : zstep, \
                                               (y0a - yp0)*tile_step[0] : (y1a - yp0)*tile_step[0] : tile_step[0], \
                                               (x0a - xp0)*tile_step[1] : (x1a - xp0)*tile_step[1] : tile_step[1]])                   
                    if acc_id:
                        tmp_max = tmp.max()
                        tmp[tmp > 0] += mid
                        mid += tmp_max
                        print(mid)
                    if output_file is None or '.h5' in output_file: 
                        result[z0a-z0:z1a-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = tmp
                    else:
                        # save to top slices                        
                        result[:z1a-z0a, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = tmp 
                    if h5_func is None:
                        fid_file.close()
        # import pdb;pdb.set_trace()
        if mask is not None:
            if output_file is None or '.h5' in output_file: 
                result[z0a-z0 : z1a-z0] = result[z0a-z0 : z1a-z0] * np.array(mask[z0a-z0 : z1a-z0])
            else:
                result[:z1a-z0a] = result[:z1a-z0a] * np.array(mask[z0a-z0 : z1a-z0])                
        if output_file is not None and '.h5' not in output_file:
            for z in range(z0a, z1a):                        
                write_image(output_file % z, result[z-z0a], tile_type)
    if '.h5' in output_file:
        fid_out.close()
    return result
