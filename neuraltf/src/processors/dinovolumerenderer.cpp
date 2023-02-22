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

#include <inviwo/neuraltf/processors/dinovolumerenderer.h>

#include <inviwo/core/algorithm/boundingbox.h>                // for boundingBox

#include <inviwo/core/properties/valuewrapper.h>              // for PropertySerializa...
#include <inviwo/core/properties/transferfunctionproperty.h>  // for TransferFunctionProperty
#include <inviwo/core/properties/listproperty.h>
#include <inviwo/neuraltf/properties/ntfproperty.h>

#include <modules/opengl/shader/shader.h>                     // for Shader, Shader::Build
#include <modules/opengl/shader/shaderobject.h>               // for ShaderObject
#include <modules/opengl/shader/shaderutils.h>                // for addShaderDefines, addShader...
#include <modules/opengl/texture/textureunit.h>               // for TextureUnitContainer
#include <modules/opengl/texture/textureutils.h>              // for bindAndSetUniforms, activat...
#include <modules/opengl/volume/volumeutils.h>                // for bindAndSetUniforms

#include <fmt/core.h>
#include <inviwo/core/util/zip.h>                             // for enumerate, zipIterator, zipper

namespace inviwo {

// The Class Identifier has to be globally unique. Use a reverse DNS naming scheme
const ProcessorInfo DINOVolumeRenderer::processorInfo_{
    "org.inviwo.DINOVolumeRenderer",  // Class identifier
    "DINOVolume Renderer",        // Display name
    "NeuralNet",                   // Category
    CodeState::Experimental,       // Code state
    Tags::None,                    // Tags
    R"(<Explanation of how to use the processor.>)"_unindentHelp};

const ProcessorInfo DINOVolumeRenderer::getProcessorInfo() const { return processorInfo_; }

DINOVolumeRenderer::DINOVolumeRenderer()
    : Processor{}
    , shader_("neuraltfraycaster.frag", Shader::Build::No)
    , volumePort_{"volume", "Input Volume"_help}
    , similarityPort_{"similarity", "Similarity Volumes"_help}
    , entryPort_{"entry", "Entry-point image"_help}
    , exitPort_{"exit", "Exit-point image"_help}
    , backgroundPort_{"bg", "Input Image to write into"_help}
    , outport_{"outport", "Rendered Image"_help}
    , ntfs_{"tfs", "Classes", 
        std::make_unique<NTFProperty>("ntf0", "Class 1"),} 
    , annotationButtons_{"annotationButtons", "Add Annotations",
        std::make_unique<ButtonProperty>("addCoord", "Add Annotation"), 
        0, ListPropertyUIFlag::Static, InvalidationLevel::Valid}
    , raycasting_{"raycaster", "Raycasting"}
    , camera_{"camera", "Camera", util::boundingBox(volumePort_)}
    , lighting_{"lighting", "Lighting", &camera_}
    , positionIndicator_("positionindicator", "Position Indicator")
    , currentVoxelSelectionX_("currentVoxelX", "Current Voxel Selection X", 0, 0, 2048)
    , currentVoxelSelectionY_("currentVoxelY", "Current Voxel Selection Y", 0, 0, 2048)
    , currentVoxelSelectionZ_("currentVoxelZ", "Current Voxel Selection Z", 0, 0, 2048){

    volumePort_.onChange([this]() { initializeResources(); });
    similarityPort_.onChange([this]() { initializeResources(); });
    ntfs_.setSerializationMode(PropertySerializationMode::All);
    ntfs_.setInvalidationLevel(InvalidationLevel::Valid);
    ntfs_.onChange([this](){ 
        if (ntfs_.getProperties().size() != annotationButtons_.getProperties().size()) {
            updateButtons();
        }
        invalidate(InvalidationLevel::InvalidResources); 
    });
    shader_.onReload([this]() { initializeResources(); });
    backgroundPort_.onConnect([&]() { this->invalidate(InvalidationLevel::InvalidResources); });
    backgroundPort_.onDisconnect([&]() { this->invalidate(InvalidationLevel::InvalidResources); });

    addPorts(volumePort_, entryPort_, exitPort_, similarityPort_, backgroundPort_, outport_);
    similarityPort_.setOptional(true);
    backgroundPort_.setOptional(true);

    addProperties(raycasting_, ntfs_, annotationButtons_, camera_, lighting_, positionIndicator_, 
        currentVoxelSelectionX_, currentVoxelSelectionY_, currentVoxelSelectionZ_);
    currentVoxelSelectionX_.setVisible(false);
    currentVoxelSelectionY_.setVisible(false);
    currentVoxelSelectionZ_.setVisible(false);
}

void DINOVolumeRenderer::initializeResources() {
    utilgl::addShaderDefines(shader_, raycasting_);
    utilgl::addShaderDefines(shader_, camera_);
    utilgl::addShaderDefines(shader_, lighting_);
    utilgl::addShaderDefines(shader_, positionIndicator_);
    utilgl::addShaderDefinesBGPort(shader_, backgroundPort_);

    size_t numClasses = similarityPort_.getVectorData().size();
    shader_.getFragmentShaderObject()->addShaderDefine("NUM_CLASSES", std::to_string(numClasses));

    if (similarityPort_.hasData()) {
        // old numClasses ntfs_.getProperties().size()
        if (numClasses > 0){
            StrBuffer str3dsampler, str2dsampler, strApply;
            for (size_t i = 0; i < numClasses; ++i) {
                str3dsampler.append("uniform sampler3D ntf{0};", i);
                str2dsampler.append("uniform sampler2D transferFunction{0};", i);
                strApply.append("color[{0}] = applyTF(transferFunction{0}, texture(ntf{0}, samplePos).x);", i);
            }
            shader_.getFragmentShaderObject()->addShaderDefine("DEFINE_NTF_SAMPLERS", str3dsampler.view());
            shader_.getFragmentShaderObject()->addShaderDefine("DEFINE_TF_SAMPLERS", str2dsampler.view());
            shader_.getFragmentShaderObject()->addShaderDefine("APPLY_NTFS", strApply.view());
        }
    }
    if(volumePort_.hasData()) {
        shader_.build();
    }
}

void DINOVolumeRenderer::process() {
    utilgl::activateAndClearTarget(outport_);
    shader_.activate();

    TextureUnitContainer units;
    utilgl::bindAndSetUniforms(shader_, units, volumePort_);
    for (auto [p,v] : similarityPort_.getSourceVectorData()){
        utilgl::bindAndSetUniforms(shader_, units, *v, p->getIdentifier());
    }
    utilgl::bindAndSetUniforms(shader_, units, entryPort_, ImageType::ColorDepthPicking);
    utilgl::bindAndSetUniforms(shader_, units, exitPort_, ImageType::ColorDepth);
    if (backgroundPort_.hasData()) {
        utilgl::bindAndSetUniforms(shader_, units, backgroundPort_, ImageType::ColorDepthPicking);
    }

    std::vector<Property*> ntfProps = ntfs_.getProperties();
    for (const Property* prop : ntfProps) {
        const TransferFunctionProperty& tfProp = static_cast<const NTFProperty*>(prop)->tf_;
        utilgl::bindAndSetUniforms(shader_, units, tfProp);
    }
    utilgl::setUniforms(shader_, outport_, camera_, lighting_, raycasting_, positionIndicator_);

    utilgl::singleDrawImagePlaneRect();

    shader_.deactivate();
    utilgl::deactivateCurrentTarget();
}


void DINOVolumeRenderer::deserialize(Deserializer& d) {
    Processor::deserialize(d);
    for (auto [i, ntf] : util::enumerate(ntfs_.getProperties())) {
        static_cast<NTFProperty*>(ntf)->tf_.setIdentifier(std::string("transferFunction") + std::to_string(i+1));
    }
}

void DINOVolumeRenderer::updateButtons() {
    const std::vector<Property*> ntfProps = ntfs_.getProperties();
    annotationButtons_.clear();
    for (Property* prop : ntfProps) {
        NTFProperty* ntfProp = static_cast<NTFProperty*>(prop);
        std::string propId = ntfProp->getIdentifier();
        ButtonProperty* btn = new ButtonProperty(
            ntfProp->getIdentifier() + "-addCoord", 
            "Add to " + ntfProp->getDisplayName(),
            InvalidationLevel::Valid);
        // ButtonProperty* btn = static_cast<ButtonProperty*>(annotationButtons_.constructProperty(0));
        btn->onChange([&, propId](){
            NTFProperty* ntfProp = static_cast<NTFProperty*>(ntfs_.getPropertyByIdentifier(propId));
            size3_t coord (currentVoxelSelectionX_.get(), 
                           currentVoxelSelectionY_.get(), 
                           currentVoxelSelectionZ_.get());
            ntfProp->addAnnotation(coord);
        });
        annotationButtons_.addProperty(btn, true);
    }
}

}  // namespace inviwo
