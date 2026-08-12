def classFactory(iface):
    from .plugin import VolumeFinderPlugin
    return VolumeFinderPlugin(iface)
