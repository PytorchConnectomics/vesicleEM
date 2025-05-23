import os,sys
import shutil
from util import *
import numpy as np
from glob import glob

def neuron_to_id_name(conf, neuron):
    dict = read_yml(conf['mask_id'])
    if neuron.isnumeric():
        neuron_id = int(neuron)
        neuron_name = [x for x in dict if dict[x]==neuron_id][0]
    else:
        neuron_name = neuron
        neuron_id = dict[neuron_name]
    return neuron_id, neuron_name 

def neuron_name_to_id(conf, name):
    dict = read_yml(conf['mask_id'])
    if isinstance(name, str):
        name = [name]    
    return [dict[x] for x in name]            

def compute_bbox_tile(conf, job_id, job_num):
    fns = read_txt(conf['mask_filenames'])
    tsz = conf['mask_size'][1:]
    seg_relabel = vast_meta_relabel(conf['mask_meta'])
    bbox_folder = conf['mask_bbox_folder']
    mask_folder = conf['mask_folder'] 
    
    for fn in fns[job_id::job_num]:
        fn = fn[:-1]
        sn = f'{bbox_folder}/{fn}.txt'
        if not os.path.exists(sn):
            im = seg_relabel[rgb_to_seg(read_image(f'{mask_folder}/{fn}.png'))]
            bb = ''
            if im.max() > 0:
                # XY: 0-index
                zz = int(fn[fn.rfind('s')+1:fn.rfind('Y')])
                yy = int(fn[fn.rfind('Y')+1:fn.rfind('X')])
                xx = int(fn[fn.rfind('X')+1:])
                bb = compute_bbox_all(im)
                bb = np.hstack([bb[:,:1], np.ones([bb.shape[0], 2], int) * zz, bb[:,1:3] + yy*tsz, bb[:,3:5]+xx*tsz])
                np.savetxt(sn, bb, '%d')
            else:
                write_txt(sn, bb)

def merge_bbox_tile(conf, ratio=1, do_return=False):
    fns = read_txt(os.path.join(conf['mask_folder'], 'out.txt'))
    bbox_folder = conf['mask_bbox_folder']
    out = None
    for fn in fns:
        fn_bbox = f'{bbox_folder}/{fn[:-1]}.txt'
        if os.path.exists(fn_bbox) and os.stat(fn_bbox).st_size > 1:
            bbox = np.loadtxt(fn_bbox).astype(int)                        
            if len(bbox) > 0:
                if bbox.ndim == 1:
                    bbox = bbox.reshape(1,-1)
                out = merge_bbox_two_matrices(out, bbox.copy())                    
    if ratio != 1:
        # round it to the downsample ratio
        out[:,3::2] = out[:,3::2]//ratio * ratio
        out[:,4::2] = (out[:,4::2]+ratio-1)//ratio * ratio - 1
    if do_return:
        return out    
    else:
        np.savetxt(f'{bbox_folder[:-1]}_init.txt', out, '%d')

def merge_bbox_tile_outlier(conf, neuron_id, change_thres = 10000):
    fns = sorted(read_txt(os.path.join(conf['mask_folder'], 'out.txt')))
    bbox_folder = conf['mask_bbox_folder']
    out = None    
    for fn in fns:
        fn_bbox = f'{bbox_folder}/{fn[:-1]}.txt'
        if os.path.exists(fn_bbox) and os.stat(fn_bbox).st_size > 1:
            bbox = np.loadtxt(fn_bbox).astype(int)                        
            if len(bbox) > 0:
                if bbox.ndim == 1:
                    bbox = bbox.reshape(1,-1)                
                bbox = bbox[bbox[:,0] == neuron_id]
                if len(bbox) > 0:
                    pre_out = out.copy() if out is not None else None
                    out = merge_bbox_two_matrices(out, bbox.copy())                
                    if pre_out is not None and np.abs(np.array(pre_out) - out).max() > change_thres:
                        print(fn, pre_out, out)
                        import pdb;pdb.set_trace() 
                    zz = int(fn[fn.rfind('_s')+2:fn.rfind('_Y')])
                    # if zz >320 and zz < 325:
                    #     print(fn, bbox[0], np.array(out[0,2::2])-out[0,1::2])  
                    # if 's0322_Y3_X3' in fn:                            
                    #     import pdb;pdb.set_trace()
                    # make changes
                    print(fn, bbox[0], np.array(out[0,2::2])-out[0,1::2])  
                    if bbox[0,-1]<40000 and zz>600 and False:
                        print(fn, bbox[0], np.array(out[0,2::2])-out[0,1::2])  
                        shutil.copy(fn_bbox, fn_bbox.replace('.txt','_bk.txt'))
                        bbox = np.loadtxt(fn_bbox).astype(int)
                        np.savetxt(fn_bbox, bbox[bbox[:,0] != neuron_id], '%d')    
        
def neuron_id_to_bbox(conf, neuron_id, neuron_name=''):
    bbox = np.loadtxt(f"{conf['mask_bbox_folder'][:-1]}.txt").astype(int)
    # find the bounding box of the input id
    bb = bbox[bbox[:,0] == neuron_id, 1:7][0]
    #print(f'Neuron {neuron_id} bbox: {bb}')
    print(f'Neuron {neuron_id} bbox: {neuron_name} {bb[::2][::-1]}: {[2,2,1]*bb[::2][::-1]} - {[2,2,1]*bb[1::2][::-1]+[1,1,0]}')
    #print(f'Neuron {neuron_id} bbox: {bb[1::2]-bb[::2]+1}')
    return bb

def neuron_id_to_neuron(conf, neuron_id, ratio=[1,4,4], output_file=None):    
    if output_file is not None and os.path.exists(output_file):
        print('File exists:', output_file)
        return None
     
    # read in the bounding box
    bb = neuron_id_to_bbox(conf, neuron_id)
    
    # mip1: [30, 8, 8]
    filenames = [conf['mask_folder'] + conf['mask_template'].format(z) for z in range(conf['mask_size'][0])]
    tile_st = [0,0]   
    tile_sz = 8192
        
    # target: [30, 32, 32]
    tile_sz = [tile_sz // ratio[1], tile_sz// ratio[2]]
    bb[:2] = bb[:2] // ratio[0]
    bb[2:4] = bb[2:4] // ratio[1]
    bb[4:] = bb[4:] // ratio[2]
    # import pdb;pdb.set_trace()
    rl = vast_meta_relabel(conf['mask_meta']).astype(np.uint8)
    gid = rl==neuron_id
    rl[:] = 0
    rl[gid] = 1
    rl = rl.astype(np.uint8)
    out = read_tile_image_by_bbox(filenames, bb[0], bb[1]+1, bb[2], bb[3]+1, bb[4], bb[5]+1, \
                           tile_sz, tile_st, tile_type="seg", tile_ratio=[1./ratio[1], 1./ratio[2]], \
                               zstep=ratio[0], output_file=output_file, relabel=rl)        
    return out

if __name__ == "__main__":
    args = get_arguments()    
    conf = read_yml('conf/param.yml')
    
    # convert VAST export segments into bbox
    if args.task == 'tile-names':
        # find filenames
        get_filenames(conf['mask_folder'])
    elif args.task == 'tile-bbox':
        # compute bbox for each tile
        compute_bbox_tile(conf, args.job_id, args.job_num)
    elif args.task == 'neuron-bbox-all':
        # merge all bbbox
        merge_bbox_tile(conf, 4)
    elif args.task == 'neuron-bbox-outlier':
        # monitor the progress of merging each neuron
        for neuron in args.neuron:
            neuron_id, neuron_name = neuron_to_id_name(conf, neuron)
            merge_bbox_tile_outlier(conf, 4, neuron_id=neuron_id)
    elif args.task == 'neuron-bbox-print':
        # print bbox for the neuron
        # python neuron_mask.py -t neuron-bbox-print -n 11,1,2,5,6,17,18,26,62
        for neuron in args.neuron:
            neuron_id, neuron_name = neuron_to_id_name(conf, neuron)            
            neuron_id_to_bbox(conf, neuron_id, neuron_name)
    elif args.task == 'neuron-mask':
        # generate neuron mask from the neuron id or name
        # python neuron_mask.py -t neuron-mask -n 1 -r 1,4,4
        sn = arr_to_str(np.array(args.ratio) * conf['res'])
        for neuron in args.neuron[args.job_id::args.job_num]:
            neuron_id, neuron_name = neuron_to_id_name(conf, neuron)
            output_file = f'{conf["result_folder"]}/neuron_{neuron_name}_{sn}.h5'            
            neuron_id_to_neuron(conf, neuron_id, args.ratio, output_file)
