# Name: DinoSimilarities

import inviwopy as ivw
import numpy as np
import torch
import torch.nn.functional as F
import sys
from itertools import count
from pathlib import Path
from infer import sample_features3d, make_3d, make_4d, make_5d, norm_minmax
from bilateral_solver3d import apply_bilateral_solver3d, crop_pad, write_crop_into
import os
import subprocess
from contextlib import contextmanager

from scipy.ndimage import grey_closing, grey_opening

from cc_torch import connected_components_labeling

def log(name, t):
    # if isinstance(t, (np.ndarray, np.matrix)):
    if isinstance(t, np.ndarray):
        contig = f'C contiguous: {t.flags["C_CONTIGUOUS"]}     F contiguous: {t.flags["F_CONTIGUOUS"]}'
    elif torch.is_tensor(t):
        contig = f'C contiguous: {t.is_contiguous()}'
    else:
        contig = ''
    print(f'{name}: {tuple(t.shape)} ({t.dtype}) in range ({t.min():.2f}, {t.max():.2f}) {contig}')

def _reduce_max(acc, up, N): return torch.maximum(acc, up.max(dim=1).values)
def _reduce_avg(acc, up, N): return (acc * N + up.sum(dim=1)) / (N + up.size(1))

def enhance_contrast(data, value, factor=2.0):
    ''' Enhances contrast of `data` for values close to `value`

    Args:
        data (Tensor): Image/Volume to be contrast enhanced with value range [0,1]
        value (float): Value for which contrast shall be enhanced
        factor (float, optional): Strength of contrast enhancement. Defaults to 6.0.

    Returns:
        Tensor: Contrast enhanced image/volume with `value` shifted to 0.5, scaled back to [0,1]
    '''
    return torch.clamp((data - value) * factor + value, 0.0, 1.0)

def largest_connected_component(mask):
    ''' Returns largest connected component of `mask`
        Args:
            mask (Tensor): Binary mask

        Returns:
            Tensor: Largest connected component of `mask` as binary mask
    '''
    dims_even = (torch.Tensor([*mask.shape[-3:]]) % 2 == 0).all()
    if dims_even:
        em = slice(None)
    else:
        em = [slice(mask.size(d) - mask.size(d) % 2) for d in range(-3,0)]
    largest_island = torch.zeros_like(mask)
    labels = connected_components_labeling(mask[em].to(torch.uint8).cuda())
    uniq, sizes = labels.unique(sorted=True, return_counts=True)
    if len(uniq) == 1: return labels == uniq[0]
    largest_island_idx = sizes[1:].argmax() + 1
    largest_island[em] = (labels == uniq[largest_island_idx]).to(mask.device)
    return largest_island

reduce_fns = {
    'max': _reduce_max,
    'avg': _reduce_avg,
    'mean': _reduce_avg,
}


######### Inviwo Processor


def get_processor(id):
    net = ivw.getApp().network
    procs = [p for p in net.processors if p.identifier == id]
    if len(procs) == 0: raise Exception(f'Processor not found in network: {id}')
    elif len(procs) > 1:
        print('Something weird happened. Multiple processors have that id:')
        print(procs)
        print('Using first one')
        return procs[0]
    else: return procs[0]

def get_network(): return ivw.getApp().network

@contextmanager
def lockedNetwork(net=None):
    if net is None: net = get_network()
    net.lock()
    try:
        yield
    finally:
        net.unlock()

def get_annotations(proc_name, typ=torch.float32, dev=torch.device('cpu'), filter_dict=None, return_empty=False, return_requpdate=False):
    print('get_annotations()')
    THRESH = -1 if return_empty else 0
    dino_proc = get_processor(proc_name)
    if filter_dict is None:
        annotations = {
            ntf.identifier: np.array([a.array for a in ntf.getAnnotatedVoxels()])
            for ntf in dino_proc.tfs.properties if len(ntf.getAnnotatedVoxels()) > THRESH
        }
    else:
        annotations = {
            ntf.identifier: np.array([a.array for a in ntf.getAnnotatedVoxels()
                                        if a.identifier in filter_dict[ntf.identifier]])
                                    for ntf in dino_proc.tfs.properties
                                    if len(ntf.getAnnotatedVoxels()) > THRESH and ntf.identifier in filter_dict.keys()
        }
    annotations = { k: torch.from_numpy(v.astype(np.int64)).to(typ).to(dev) for k,v in annotations.items() }
    if return_requpdate:
        req_update = { ntf.identifier: ntf for ntf in dino_proc.tfs.properties if ntf.requiresUpdate }
        return annotations, req_update
    else:
        print({k: v.shape for k,v in annotations.items()})
        return annotations

def get_similarity_params(proc_name, return_empty=False):
    THRESH = -1 if return_empty else 0
    dino_proc = get_processor(proc_name)
    ret = {
        ntf.identifier: {
            'reduction': reduce_fns[ntf.similarityReduction],
            'modality': ntf.modality,
            'ramp': (ntf.ramp.x, ntf.ramp.y),
            'contrast': ntf.contrast,
            'connected_component': ntf.connectedComponent,
            'modalityWeight': torch.from_numpy(ntf.modalityWeight.array),
            'bls_sigma_spatial': ntf.blsSigmas[0],
            'bls_sigma_chroma': ntf.blsSigmas[1],
            'bls_sigma_luma': ntf.blsSigmas[2],
            'bls_enabled': ntf.blsEnabled
        }
        for ntf in dino_proc.tfs.properties
        if len(ntf.getAnnotatedVoxels()) > THRESH
    }
    print('get_similarity_params(): ', ret)
    return ret

def save_annotations(proc_name, path):
    annotations = get_annotations(proc_name, return_empty=True)
    np.save(path, annotations)
    print(f'Saved annotations {annotations.keys()} to {path}')

def save_similarity_params(proc_name, path):
    params = get_similarity_params(proc_name, return_empty=True)
    np.save(path, params)
    print(f'Saved similarity params {params.keys()} to {path}')

def save_nparray(array, path):
    np.save(path, array)
    if isinstance(array, dict):
        print(f'Saved array {array.keys()} to {path}')
    else:
        print(f'Saved array {array.shape} to {path}')

def is_path_creatable(pathname: str) -> bool:
    '''
    `True` if the current user has sufficient permissions to create the passed
    pathname; `False` otherwise.
    '''
    # Parent directory of the passed path. If empty, we substitute the current
    # working directory (CWD) instead.
    dirname = os.path.dirname(pathname) or os.getcwd()
    return os.access(dirname, os.W_OK)

class DinoSimilarities(ivw.Processor):
    def __init__(self, id, name):
        ivw.Processor.__init__(self, id, name)
        # Ports
        self.inport = ivw.data.VolumeInport("inport")
        self.addInport(self.inport, owner=False)
        # Properties
        self.dinoProcessorIdentifier = ivw.properties.StringProperty("dinoProcId", "DINO Volume Renderer ID", "DINOVolumeRenderer")
        self.cachePathOverride = ivw.properties.FileProperty("cahcePathOverride", "Override Cache Path", "")
        self.useCuda = ivw.properties.BoolProperty("useCuda", "Use CUDA", False)
        self.cleanupTemporaryVolume = ivw.properties.BoolProperty("cleanupTempVol", "Clean up volume that's temporarily created on disk to pass to infer.py", True)
        self.updateOnAnnotation = ivw.properties.BoolProperty("updateOnAnnotation", "Update Similarities on Annotation", True)
        self.similarityVolumeScalingFactor = ivw.properties.FloatProperty("simScaleFact", "Similarity Volume Downscale Factor", 4.0, 1.0, 8.0)
        self.similarityVolumeScalingFactor.invalidationLevel = ivw.properties.InvalidationLevel.Valid
        self.updatePorts = ivw.properties.ButtonProperty(
            "updateEverything", "Update Callbacks, Ports & Connections",
            self.addAndConnectOutports)
        self.clearSimilarityCache = ivw.properties.ButtonProperty("clearSimn", "Clear Similarity Cache", self.clearSimilarity)
        self.sliceAlong = ivw.properties.OptionPropertyString("sliceAlong", "DINO Slice along Axis", [
            ivw.properties.StringOption("alongALL", 'Slice along ALL', 'all'),
            ivw.properties.StringOption("alongX", 'Slice along X', 'x'),
            ivw.properties.StringOption("alongY", 'Slice along Y', 'y'),
            ivw.properties.StringOption("alongZ", 'Slice along Z', 'z')
        ])
        self.updateSims = ivw.properties.BoolProperty("updateSims", "Update Similarities", False, ivw.properties.InvalidationLevel.Valid)
        self.openCloseIterations = ivw.properties.IntProperty("openCloseIterations", "Open-Close Iterations", 0, 0, 8)
        self.addProperty(self.dinoProcessorIdentifier)
        self.addProperty(self.cachePathOverride)
        self.addProperty(self.useCuda)
        self.addProperty(self.cleanupTemporaryVolume)
        self.addProperty(self.updateOnAnnotation)
        self.addProperty(self.updateSims)
        self.addProperty(self.sliceAlong)
        self.addProperty(self.similarityVolumeScalingFactor)
        self.addProperty(self.updatePorts)
        self.addProperty(self.clearSimilarityCache)
        self.addProperty(self.openCloseIterations)
        # Callbacks
        self.inport.onChange(self.getVolumeDataPath)
        self.cachePathOverride.onChange(self.getVolumeDataPath)
        self.useCuda.onChange(self.updateFeatvolDevice)
        def invalidUpdateSims():
            if self.updateSims.value and self.updateOnAnnotation.value:
                print("Invalidating from updateSims.onChange()")
                self.invalidateOutput()
        self.updateSims.onChange(invalidUpdateSims)
        def temp():
            print("CALLING FROM dinoProcessorIdentifier.onChange():")
            self.addAndConnectOutports()
        self.dinoProcessorIdentifier.onChange(temp)
        # Init other variables
        self.outs = {}
        self.registeredCallbacks = {}
        self.similarities = {}
        self.feat_vol = None
        self.vol = None
        self.dims = None
        self.cache_path = None
        self.loaded_cache_path = None
        self.save_dir = ivw.properties.DirectoryProperty("saveDir", "Save Directory", "")
        self.save_btn = ivw.properties.ButtonProperty("saveBtn", "Save Similarities", self.save_similarities)
        self.load_btn = ivw.properties.ButtonProperty("loadBtn", "Load Similarities", self.load_similarities)
        self.addProperty(self.save_dir)
        self.addProperty(self.save_btn)
        self.addProperty(self.load_btn)

    def save_similarities(self):
        if self.save_dir.value != '' and is_path_creatable(self.save_dir.value):
            dir = Path(self.save_dir.value)
            dir.mkdir(parents=True, exist_ok=True)
            save_nparray(self.similarities, dir / 'similarities.npy')
            save_annotations(self.dinoProcessorIdentifier.value, dir / 'annotations.npy')
            save_similarity_params(self.dinoProcessorIdentifier.value, dir / 'similarity_params.npy')
            if self.inport.hasData():
                save_nparray(self.inport.getData().data, dir / 'volume.npy')
            if self.feat_vol is not None:
                save_nparray(self.feat_vol, dir / 'dino_features.npy')
        else:
            print("ERROR: Invalid save directory")

    def load_similarities(self):
        if self.save_dir.value != '' and Path(self.save_dir.value).exists():
            self.similarities = np.load(Path(self.save_dir.value) / 'similarities.npy', allow_pickle=True)[()]
            dino_proc = get_processor(self.dinoProcessorIdentifier.value)
            ntfs = dino_proc.tfs.properties
            annotations = np.load(Path(self.save_dir.value) / 'annotations.npy', allow_pickle=True)[()]
            for k,v in annotations.items():
                ntfs[k].setAnnotations(list(map(ivw.glm.size3_t, np.array(v).astype(np.int64).tolist())))

    @staticmethod
    def processorInfo():
        return ivw.ProcessorInfo(
            classIdentifier="org.inviwo.DinoSimilarities",
            displayName="DinoSimilarities",
            category="Python",
            codeState=ivw.CodeState.Stable,
            tags=ivw.Tags.PY
        )

    def getProcessorInfo(self):
        return DinoSimilarities.processorInfo()

    def getVolumeDataPath(self):
        if self.inport.isConnected():
            if self.cachePathOverride.value != '' and Path(self.cachePathOverride.value).exists():
                self.cache_path = Path(self.cachePathOverride.value)
                if self.cache_path != self.loaded_cache_path or self.loaded_cache_path is None:
                    self.feat_vol = None
                    self.similarities.clear()
                    self.loaded_cache_path = None
                    self.initializeResources()
            else:
                data_path = Path(self.inport.getConnectedOutport().processor.properties['filename'].value)
                clean_name = data_path.stem.replace(" ", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "")
                new_cache_path = data_path.parent/f'{clean_name}_DINOfeats_{self.sliceAlong.selectedValue}.npy'
                if self.cache_path != new_cache_path:
                    self.cache_path = new_cache_path
                    self.initializeResources()
        else:
            self.cache_path = None
        print('getVolumeDataPath(): ', self.cache_path)

    def registerCallbacks(self):
        print('registerCallbacks()')
        proc = get_processor(self.dinoProcessorIdentifier.value)
        # Register this function as callback on annotations list property
        if proc.annotationButtons.path not in self.registeredCallbacks.keys():
            def temp():
                print("CALLING FROM proc.annotationButtons.onChange():")
                self.addAndConnectOutports()
            cb = proc.annotationButtons.onChange(temp)
            self.registeredCallbacks[proc.annotationButtons.path] = cb

    def invalidateOutput(self, force=False):
        if force or self.updateOnAnnotation.value:
            print('invalidateOutput()')
            self.invalidate(ivw.properties.InvalidationLevel.InvalidOutput)

    def addAndConnectOutports(self):
        self.addVolumeOutports()
        self.connectVolumeOutports()
        # self.invalidateOutput()

    def addVolumeOutports(self):
        new_names = set(get_similarity_params(self.dinoProcessorIdentifier.value, return_empty=True).keys())
        cur_names = set(self.outs.keys())
        net = get_network()
        with lockedNetwork(net):
            for k in (cur_names - new_names):
                for inp in self.outs[k].getConnectedInports():
                    net.removeConnection(self.outs[k], inp)
                self.removeOutport(self.outs[k])
                del self.outs[k]
            for k in sorted(new_names - cur_names):
                self.outs[k] = ivw.data.VolumeOutport(k)
                self.addOutport(self.outs[k])

    def connectVolumeOutports(self):
        simInport = get_processor(self.dinoProcessorIdentifier.value).getInport('similarity')
        net = get_network()
        ports_with_data = get_similarity_params(self.dinoProcessorIdentifier.value, return_empty=False).keys()
        all_ports = set(get_similarity_params(self.dinoProcessorIdentifier.value, return_empty=True).keys())
        for k in all_ports - set(ports_with_data):
            self.outs[k].setData(ivw.data.Volume(np.zeros((4,4,4), dtype=np.uint8)))
        if len(all_ports) > 0:
            with lockedNetwork(net):
                for k in all_ports:
                    # simInport.connectTo(v) # does not fully connect the ports (visual link is missing, some things go wrong)
                    net.addConnection(self.outs[k], simInport)
                    # print(f'Connecting {simInport.identifier} to {self.outs[k].identifier}.')

    def updateFeatvolDevice(self):
        if self.feat_vol is not None:
            dev = torch.device("cuda" if torch.cuda.is_available and self.useCuda.value else "cpu")
            typ = torch.float16 if dev == torch.device("cuda") else torch.float32
            self.feat_vol = self.feat_vol.to(dev).to(typ)

    def loadCache(self, cache_path, attention_features='k'):
        print('loadCache() from ', cache_path)
        # if self.loaded_cache_path == cache_path: return
        if cache_path.suffix in ['.pt', '.pth']:
            data = torch.load(cache_path)
            if type(data) == dict:
                feat_vol = data[attention_features]
            else:
                feat_vol = data
        elif cache_path.suffix == '.npy':
            data = np.load(cache_path, allow_pickle=True)
            if data.dtype == "O":
                feat_vol = torch.from_numpy(data[()][attention_features])
            else:
                feat_vol = torch.from_numpy(data)
        else:
            raise Exception(f'Unsupported file extension: {cache_path.suffix}')
        if feat_vol.ndim == 4: feat_vol.unsqueeze_(0) # add empty M dimension -> (M, F, W, H, D)
        dev = torch.device("cuda" if torch.cuda.is_available and self.useCuda.value else "cpu")
        typ = torch.float16 if dev == torch.device("cuda") else torch.float32
        self.feat_vol = F.normalize(feat_vol.to(typ).to(dev), dim=1)
        # TODO: watch out me flippy
        # self.feat_vol = self.feat_vol.flip((-3, ))
        self.loaded_cache_path = cache_path
        log('Loaded self.feat_vol', self.feat_vol)

    def computeSimilarities(self, annotations):
        ''' Computes similarities for given `annotations` and combines with existing `self.similarities`

        Args:
            annotations (dict of torch.Tensor): Dictionary mapping NTF ID -> torch.Tensor(A, 3) A being number of annotations

        Returns:
            dict of torch.Tensor: Dictionary mapping NTF ID -> updated Similarity map (W, H, D) uint8 in [0, 255]
        '''
        print('computeSimilarities()')
        with torch.no_grad():
            dev, typ = self.feat_vol.device, self.feat_vol.dtype
            simparams = get_similarity_params(self.dinoProcessorIdentifier.value)
            print('got sims')
            inv_vol = self.inport.getData()
            in_dims = tuple(inv_vol.dimensions.array.tolist())
            sim_shape = tuple(map(lambda d: int(d // self.similarityVolumeScalingFactor.value), in_dims))
            vol_extent = torch.tensor([[[*in_dims]]], device=dev, dtype=typ)
            def split_into_classes(t):
                sims = {}
                idx = 0
                for k,v in annotations.items():
                    sims[k] = t[:, idx:idx+v.size(0)]
                    idx += v.size(0)
                return sims
            if len(annotations) == 0: return  # No NTFs
            abs_coords = torch.cat(list(annotations.values())).to(dev).to(typ)
            if abs_coords.numel() == 0: return # No annotation in any of the NTFs
            rel_coords = (abs_coords.float() + 0.5) / vol_extent * 2.0 - 1.0

            qf = sample_features3d(self.feat_vol, rel_coords, mode='bilinear')

            sims = torch.einsum('mfwhd,mcaf->mcawhd', (self.feat_vol, qf)).squeeze(1)
            # sims = resample_topk(self.feat_vol, sims, K=8, feature_sampling_mode='bilinear').squeeze(1)

            # bls_scale = self.similarityVolumeScalingFactor.value / 4.0
            lr_abs_coords = torch.round((rel_coords * 0.5 + 0.5) * (torch.tensor([*sims.shape[-3:]]).to(dev).to(typ) - 1.0)).long() # (A, 3)
            lr_abs_coords = split_into_classes(make_3d(lr_abs_coords)) # (1, A, 3) -> {NTF_ID: (1, a, 3)}
            sim_split = {}
            rel_coords_dict = split_into_classes(make_3d(rel_coords))
            if any([simparams[k]['bls_enabled'] for k in simparams.keys()]):
                in_vol = np.ascontiguousarray(inv_vol.data.astype(np.float32)) # TODO: refactor out of this loop
                if in_vol.ndim == 4: in_vol = np.transpose(in_vol, (3,0,1,2))
                vol = F.interpolate(make_5d(torch.from_numpy(in_vol.copy())).to(dev), sim_shape, mode='trilinear').squeeze(0)
                m = vol.size(0)
                vol = norm_minmax(vol)

            for k,sim in split_into_classes(sims).items():
                print('Reducing & Solving ', k, sim.shape)
                ramp_min = simparams[k]['ramp'][0]
                sim = torch.where(sim >= ramp_min, sim, torch.zeros(1, dtype=typ, device=dev)) ** 2.5 # Throw away low similarities & exponentiate
                sim = sim.mean(dim=1)
                if simparams[k]['connected_component']:
                    sim[~largest_connected_component(sim.squeeze() > ramp_min)[None]] = 0.0

                if simparams[k]['bls_enabled']:
                    bls_params = {
                        'sigma_spatial': simparams[k]['bls_sigma_spatial'],
                        'sigma_chroma': simparams[k]['bls_sigma_chroma'],
                        'sigma_luma': simparams[k]['bls_sigma_luma']
                    }
                    print(f'{k} has BLS {simparams[k]["bls_enabled"]} with params {bls_params}')
                    median_int = sample_features3d(vol, rel_coords_dict[k], mode='nearest').median()
                    svol = enhance_contrast(vol, value=median_int, factor=simparams[k]['contrast'])
                    svol = (255.0 * svol).to(torch.uint8)
                    if   svol.size(0) == 1: svol = svol[None].expand(1,3,-1,-1,-1)
                    elif svol.size(0) == 2: svol = svol[:,None].expand(-1,3,-1,-1,-1)
                    elif svol.size(0) == 3: svol = svol[None]
                    elif svol.size(0)  > 3: svol = svol[None, :3]
                    if tuple(sim.shape[-3:]) != sim_shape:
                        print(f'Resizing {k} similarity to', sim_shape)
                        sim = F.interpolate(make_5d(sim), sim_shape, mode='trilinear').squeeze(0)
                    # Apply Bilateral Solver
                    blsim = 0.0
                    for i, ssim, mvol in zip(count(), sim, svol):
                        print('APPLYING Bilateral Solver')
                        if simparams[k]['modalityWeight'][i] > 0:
                            quant = 0.99 * ssim.max() # ssim.quantile(q=0.99)
                            print('quant:', quant, 'thresh:', ramp_min * quant)
                            crops, mima = crop_pad([ssim, mvol], thresh=ramp_min * quant, pad=5)
                            csim, cvol = crops
                            mask = filter_dilation(make_5d(csim), thresh=ramp_min * quant, size=7).squeeze()
                            csim = apply_bilateral_solver3d(make_4d(csim), cvol, grid_params=bls_params) * simparams[k]['modalityWeight'][i]
                            csim[~mask] = 0.0
                            quant = 0.99 * csim.max()  # csim.quantile(q=0.9999)
                            ssim = write_crop_into(torch.zeros_like(ssim), csim, mima)
                            print('Wrote crop into original similarity map', csim.shape, '->', ssim.shape)
                            blsim += ssim

                    sim_split[k] = (255.0 / quant * blsim).clamp(0, 254.0).cpu().to(torch.uint8).squeeze()
                    # print(f'sim_split[{k}]', sim_split[k].shape, sim_split[k].min(), sim_split[k].max())
                else:
                    msim = 0.0
                    # Merge similarities of different modalities
                    for i, ssim in enumerate(sim):
                        if simparams[k]['modalityWeight'][i] > 0:
                            msim += ssim * simparams[k]['modalityWeight'][i]
                    sim_split[k] = (255.0 / msim.float().quantile(q=0.9999) * msim).clamp(0, 254.0).cpu().to(torch.uint8).squeeze()

                # log(k, sim_split[k])
                # Set similarity to 1 where the volume is annotated
                print('Setting similarity to 1 where the volume is annotated')
                print(lr_abs_coords[k].shape)
                an_indices = tuple(a.squeeze(-1) for a in lr_abs_coords[k].squeeze().split(1, dim=-1))
                print(an_indices)
                print(k, sim_split[k].shape)
                # sim_split[k][an_indices] = 255
                print('Returning shit')
            return sim_split

    def updateSimilarities(self):
        print('updateSimilarities()')
        annotations, requires_update = get_annotations(self.dinoProcessorIdentifier.value, return_requpdate=True)
        print(' -> requires_update', requires_update.keys())
        print(' -> annotations.keys()', annotations.keys())
        print(' -> self.similarities.keys()', self.similarities.keys())
        total_todo = requires_update.keys()
        if len(requires_update) > 0 or len(annotations.keys()) > len(self.similarities):
            to_update = set(requires_update.keys()).union(set(k for k in annotations.keys() if k not in self.similarities.keys()))
            print(' -> to_update: ', to_update)
            self.similarities.update(self.computeSimilarities({ k: v for k,v in annotations.items() if k in to_update}))
            for ntf in requires_update.values():
                ntf.requiresUpdate = False
            print(' -> self.similarities: ', { k: v.shape for k,v in self.similarities.items() })
            return to_update
        else:
            print('Nothing to update')
            return set()

    def clearSimilarity(self):
        self.similarities.clear()
        self.invalidateOutput(force=True)

    def initializeResources(self):
        print('initializeResources()')
        # Takes care of computing, caching and loading self.feat_vol
        if self.cache_path is None: return # No cache path means no feat_vol to load
        if self.cache_path.exists(): # Load feat_vol if it exists
            if self.feat_vol is None or self.cache_path != self.loaded_cache_path:
                self.loadCache(self.cache_path) # Loads self.feat_vol
                self.addAndConnectOutports() # Add volume outports for similarities if necessary and connect to DINOVolumeRenderer
                self.registerCallbacks() # Registers callbacks on DINOVolumeRenderer's NTF properties
        elif self.inport.hasData(): # Compute and cache feat_vol
            if is_path_creatable(str(self.cache_path)): # only if we can save to cache_path
                print(f'Computing features and saving cache to {self.cache_path}')
                # Save incoming volume temporarily as .npy
                vol_np = self.inport.getData().data
                tmpvol_path = str(self.cache_path.parent/'tmpvol.npy')
                np.save(tmpvol_path, np.ascontiguousarray(vol_np))
                # Run infer.py script to produce feat_vol cache
                args = f'--data-path "{tmpvol_path}" --cache-path "{self.cache_path}" --slice-along {self.sliceAlong.selectedValue} --feature-output-size 96'
                if vol_np.ndim == 3:
                    cmd = f'{sys.executable} {NTF_REPO}/infer.py {args}'
                elif vol_np.ndim == 4:
                    cmd = f'{sys.executable} {NTF_REPO}/infer_multi.py {args}'
                else:
                    raise Exception(f'Invalid volume dimension: {vol_np.shape} is neither 3D nor 4D')
                print(f'Running command: {cmd}')
                comp = subprocess.run(cmd, encoding='UTF-8', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                if comp.returncode == 0: # infer.py was successful, load cache
                    self.loadCache(self.cache_path) # Load self.feat_vol
                    self.addAndConnectOutports()    # Update volume outports and connect automatically
                else: # infer.py failed, log error
                    print(f'Something went wrong with computing the features (Return Code {comp.returncode}):')
                    print(comp.stdout)
                if self.cleanupTemporaryVolume.value: # Remove temporary .npy file
                    os.remove(tmpvol_path)
            else:
                print(f'Use valid Cache Path. ("{self.cache_path}" is invalid or cannot be written to)')

    def process(self):
        print('process()')
        self.updateSims.value = False
        if self.feat_vol is None:
            self.initializeResources() # If there's no feat_vol, try to load it
        else:
            # annotations_to_compute = get_annotations(self.dinoProcessorIdentifier.value)
            # if len(annotations_to_compute) > 0:
            #     self.similarities.update(self.computeSimilarities(annotations_to_compute))
            ports_to_update = self.updateSimilarities()
            print(' -> ports_to_update', ports_to_update)
            if len(self.similarities) == 0:
                print('Could not compute self.similarities. Did you annotate anything?')
            else: # Output similarity volumes through volume outports
                in_vol = self.inport.getData() # ivw.Volume
                print(' -> self.outs:')
                print(self.outs)
                for k in ports_to_update: #self.outs.keys():
                    sim_np = self.similarities[k].numpy()
                    print('sim_np.max()', sim_np.max())
                    for _ in range(self.openCloseIterations.value):
                        sim_np = grey_closing(grey_opening(sim_np, (3,3,3)), (3,3,3))
                    volume = ivw.data.Volume(np.asfortranarray(sim_np))
                    volume.modelMatrix = in_vol.modelMatrix
                    volume.worldMatrix = in_vol.worldMatrix
                    volume.dataMap.dataRange = ivw.glm.dvec2(0.0, 255.0)
                    volume.dataMap.valueRange= ivw.glm.dvec2(0.0, 255.0)
                    # volume.interpolation = ivw.data.InterpolationType.Nearest
                    print(f'Setting data for {k} to {sim_np.shape}')
                    self.outs[k].setData(volume)
