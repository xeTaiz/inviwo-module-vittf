# Name: SegmentationVolumeSplitter

import inviwopy as ivw
from inviwopy.data       import VolumeInport, VolumeOutport, Volume
from inviwopy.glm        import dvec2

import numpy as np

class SegmentationVolumeSplitter(ivw.Processor):
    def __init__(self, id, name):
        ivw.Processor.__init__(self, id, name)
        self.inport = VolumeInport("inport")
        self.inport.onChange(self.updateOutports)
        self.addInport(self.inport, owner=False)
        self.outs = {}
        self.num_classes = 0

    def updateOutports(self):
        ''' Generates outports based on the number of components in the input volume. '''
        if self.inport.hasData():
            volume = self.inport.getData()
            self.num_classes = round(volume.dataMap.valueRange[1]) + 1

            if self.num_classes != len(self.outs):
                for out in self.outs.values():
                    self.removeOutport(out)
                self.outs.clear()
                self.outs = { f'ntf{i}': VolumeOutport(f'ntf{i}')
                                 for i in range(self.num_classes-1) }
                for out in self.outs.values():
                    self.addOutport(out, owner=False)

    @staticmethod
    def processorInfo():
        return ivw.ProcessorInfo(
            classIdentifier = "org.inviwo.segmentationvolumesplitter",
            displayName = "Segmentation Volume Splitter",
            category = "Python",
            codeState = ivw.CodeState.Stable,
            tags = ivw.Tags.PY
        )

    def getProcessorInfo(self):
        return SegmentationVolumeSplitter.processorInfo()

    def initializeResources(self):
        pass

    def process(self):
        if self.inport.hasData():
            print('has data')
            in_vol = self.inport.getData()
            print(f'looping over {self.num_classes} classes')
            val_range = self.inport.getData().dataMap.valueRange
            data_range = self.inport.getData().dataMap.dataRange
            if in_vol.data.dtype != np.uint8:
                rounded_vol = ((in_vol.data - data_range[0]) / (data_range[1] - data_range[0]))
                rounded_vol = np.round(rounded_vol * (val_range[1] - val_range[0]) + val_range[0]).astype(np.uint8)
            else:
                rounded_vol = in_vol.data
            # np.save('/run/media/dome/SSD/Data/Volumes/CT-ORG/labels-10.npy', rounded_vol)
            for i in range(1,self.num_classes):
                vol = Volume(255 * (rounded_vol == i).astype(np.uint8))
                # vol.interpolation = ivw.data.InterpolationType.Nearest
                vol.dataMap.dataRange = dvec2(0, 1)
                vol.dataMap.valueRange = dvec2(0, 1)
                vol.modelMatrix = in_vol.modelMatrix
                vol.worldMatrix = in_vol.worldMatrix
                print('Setting Outport')
                self.outports[f'ntf{i-1}'].setData(vol)
