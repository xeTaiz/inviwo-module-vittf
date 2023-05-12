/*********************************************************************************
 *
 * Inviwo - Interactive Visualization Workshop
 *
 * Copyright (c) 2023 Inviwo Foundation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *********************************************************************************/

#pragma once

#include <inviwo/neuraltf/neuraltfmoduledefine.h>
#include <inviwo/core/io/serialization/deserializer.h>        // for Deserializer
#include <inviwo/core/util/callback.h>
#include <inviwo/core/processors/processor.h>
#include <inviwo/core/processors/processorinfo.h>             // for ProcessorInfo
#include <inviwo/core/ports/volumeport.h>                     // for VolumeInport
#include <inviwo/core/ports/imageport.h>                      // for ImageInport, ImageOutport
#include <modules/opengl/shader/shader.h>                     // for Shader

#include <inviwo/core/properties/listproperty.h>
#include <inviwo/neuraltf/properties/ntfproperty.h>
#include <inviwo/core/properties/transferfunctionproperty.h>  // for TransferFunctionProperty
#include <inviwo/core/properties/simplelightingproperty.h>    // for SimpleLightingProperty
#include <inviwo/core/properties/simpleraycastingproperty.h>  // for SimpleRaycastingProperty
#include <inviwo/core/properties/cameraproperty.h>            // for CameraProperty
#include <inviwo/core/properties/volumeindicatorproperty.h>   // for VolumeIndicatorProperty
#include <inviwo/core/properties/eventproperty.h>                       // for EventProperty

#include <vector>
#include <functional>

namespace inviwo {

class IVW_MODULE_NEURALTF_API DINOVolumeRenderer : public Processor {
public:
    virtual const ProcessorInfo getProcessorInfo() const override;
    static const ProcessorInfo processorInfo_;
    DINOVolumeRenderer();
    virtual ~DINOVolumeRenderer() = default;

    virtual void initializeResources() override;
    virtual void process() override;

    void updateCurrentSimilarityTF();
    void updateButtons();
    void addAnnotation();
    void removeAnnotation();
private:
    Shader shader_;

    VolumeInport volumePort_;
    DataInport<Volume, 0, true> similarityPort_;
    ImageInport entryPort_;
    ImageInport exitPort_;
    ImageInport backgroundPort_;
    ImageOutport outport_;
    VolumeOutport simOutport_;

    NTFPropertyList ntfs_;
    ListProperty annotationButtons_;
    OptionPropertyInt selectedModality_;
    OptionPropertySize_t selectedClass_;
    FloatProperty brushSize_;
    BoolProperty brushMode_;
    BoolProperty eraseMode_;
    BoolProperty updateSims_;

    TransferFunctionProperty rawTransferFunction_;
    SimpleRaycastingProperty raycasting_;
    CameraProperty camera_;
    SimpleLightingProperty lighting_;
    VolumeIndicatorProperty positionIndicator_;
    IntProperty currentVoxelSelectionX_;
    IntProperty currentVoxelSelectionY_;
    IntProperty currentVoxelSelectionZ_;
    IntVec3Property currentVoxelSelection_;
    TransferFunctionProperty currentSimilarityTF_;
    EventProperty cycleModalitySelection_;
    EventProperty cycleClassSelection_;
    EventProperty addAnnotation_;
};

}  // namespace inviwo
