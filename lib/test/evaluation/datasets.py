from collections import namedtuple
import importlib
from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

# Useful abbreviations to reduce the clutter
pt = "lib.test.evaluation.%sdataset"

dataset_dict = dict(
    otb=DatasetInfo(module=pt % "otb", class_name="OTBDataset", kwargs=dict()),
    nfs=DatasetInfo(module=pt % "nfs", class_name="NFSDataset", kwargs=dict()),
    uav=DatasetInfo(module=pt % "uav", class_name="UAVDataset", kwargs=dict()),
    tc128=DatasetInfo(module=pt %
                      "tc128", class_name="TC128Dataset", kwargs=dict()),
    tc128ce=DatasetInfo(module=pt % "tc128ce",
                        class_name="TC128CEDataset", kwargs=dict()),
    trackingnet=DatasetInfo(module=pt % "trackingnet",
                            class_name="TrackingNetDataset", kwargs=dict()),
    got10k_test=DatasetInfo(
        module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='test')),
    got10k_val=DatasetInfo(
        module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='val')),
    got10k_ltrval=DatasetInfo(
        module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='ltrval')),
    lasot=DatasetInfo(module=pt %
                      "lasot", class_name="LaSOTDataset", kwargs=dict()),
    lasot_lmdb=DatasetInfo(module=pt % "lasot_lmdb",
                           class_name="LaSOTlmdbDataset", kwargs=dict()),

    vot18=DatasetInfo(module=pt %
                      "vot", class_name="VOTDataset", kwargs=dict()),
    vot22=DatasetInfo(module=pt %
                      "vot", class_name="VOTDataset", kwargs=dict(year=22)),
    itb=DatasetInfo(module=pt % "itb", class_name="ITBDataset", kwargs=dict()),
    tnl2k=DatasetInfo(module=pt %
                      "tnl2k", class_name="TNL2kDataset", kwargs=dict()),
    lasot_extension_subset=DatasetInfo(module=pt % "lasotextensionsubset", class_name="LaSOTExtensionSubsetDataset",
                                       kwargs=dict()),
    antiuav410_test=DatasetInfo(
        module=pt % "antiuav410", class_name="AntiUAV410Dataset", kwargs=dict(split='test')),
    antiuav410_val=DatasetInfo(
        module=pt % "antiuav410", class_name="AntiUAV410Dataset", kwargs=dict(split='val')),
    antiuav410_ltrval=DatasetInfo(
        module=pt % "antiuav410", class_name="AntiUAV410Dataset", kwargs=dict(split='ltrval')),
    dutantiuav_track=DatasetInfo(module=pt % "dutantiuavtrack", class_name="DUTAntiUAVTrack", kwargs=dict(split='test')),
    antiuav318_rgb_test = DatasetInfo(module=pt % "antiuav318_rgb", class_name="AntiUAV318_RGBDataset", kwargs=dict(split='test')),
    antiuav318_rgb_val = DatasetInfo(module=pt % "antiuav318_rgb", class_name="AntiUAV318_RGBDataset", kwargs=dict(split='val')),
    ard100_test = DatasetInfo(module=pt % "ard100", class_name="ARD100Dataset", kwargs=dict(split='test')),
    u2u_dt_test = DatasetInfo(module=pt % "u2u_dt", class_name="U2U_DTDataset", kwargs=dict(split='test')),
    drone_vs_bird_test = DatasetInfo(module=pt % "drone_vs_bird", class_name="DroneVsBirdDataset", kwargs=dict(split='test')),
    uav_anti_uav_test = DatasetInfo(module=pt % "uav_anti_uav", class_name="UAVAntiUAVDataset", kwargs=dict(split='test')),
)


def load_dataset(name: str, tracker_name: str = None, tracker_param: str = None):
    """ Import and load a single dataset."""
    name=name.lower()
    dset_info=dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m=importlib.import_module(dset_info.module)
    
    kwargs=dset_info.kwargs.copy()
    if name == 'dutantiuav_track':
        kwargs.update({
            'tracker_name': tracker_name, 
            'tracker_param': tracker_param
        })
    
    dataset = getattr(m, dset_info.class_name)(**kwargs)
    return dataset.get_sequence_list()


def get_dataset(name, tracker_name=None, tracker_param=None):
    """ Get a single or set of datasets."""
    dset=SequenceList()
    names = [name] if isinstance(name, str) else name
    for n in names:
        dset.extend(load_dataset(n, tracker_name, tracker_param))
    return dset
