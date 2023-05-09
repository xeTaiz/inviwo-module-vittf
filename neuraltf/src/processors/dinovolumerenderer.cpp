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
#include <inviwo/core/network/networklock.h>

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
#include <modules/opengl/image/layergl.h>                               // for LayerGL
#include <modules/opengl/texture/texture2d.h>                           // for Texture2D
#include <inviwo/core/interaction/events/keyboardkeys.h>                // for IvwKey, KeyState
#include <inviwo/core/network/processornetwork.h>    // IWYU pragma: keep

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
    , simOutport_{"simOutport", "Selected Similarity Volume"_help}
    , ntfs_{"tfs", "Classes",
        std::make_unique<NTFProperty>("ntf0", "Class 1", &volumePort_)}
    , annotationButtons_{"annotationButtons", "Add Annotations",
        std::make_unique<ButtonProperty>("addCoord", "Add Annotation"),
        0, ListPropertyUIFlag::Static, InvalidationLevel::Valid}
    , selectedModality_("selectedModaltiy", "Selected Modality", {
        {"channel0", "Channel 1", 0}, {"channel1", "Channel 2", 1},
        {"channel2", "Channel 3", 2}, {"channel3", "Channel 4", 3} })
    , selectedClass_("selectedClass", "Selected Class")
    , brushSize_("brushSize", "Brush Size", 1.0f, 1.0f, 16.0f, 1.0f, InvalidationLevel::Valid)
    , brushMode_("brushMode", "Brush Mode", false, InvalidationLevel::Valid)
    , eraseMode_("eraseMode", "Erase Mode", false, InvalidationLevel::Valid)
    , updateSims_("updateSims", "Update Similarities", false, InvalidationLevel::Valid)
    , rawTransferFunction_{"rawTransferFunction", "Raw Data Transfer Function", &volumePort_}
    , raycasting_{"raycaster", "Raycasting"}
    , camera_{"camera", "Camera", util::boundingBox(volumePort_)}
    , lighting_{"lighting", "Lighting", &camera_}
    , positionIndicator_("positionindicator", "Position Indicator")
    , currentVoxelSelectionX_("currentVoxelX", "Current Voxel Selection X", 0, 0, 2048, 1, InvalidationLevel::InvalidOutput)
    , currentVoxelSelectionY_("currentVoxelY", "Current Voxel Selection Y", 0, 0, 2048, 1, InvalidationLevel::InvalidOutput)
    , currentVoxelSelectionZ_("currentVoxelZ", "Current Voxel Selection Z", 0, 0, 2048, 1, InvalidationLevel::InvalidOutput)
    , currentVoxelSelection_("currentVoxel", "Current Voxel Selection", ivec3(128), ivec3(1), ivec3(256), ivec3(1),
                InvalidationLevel::Valid)
    , currentSimilarityTF_("currentSimilarityTF", "Current Similarity TF")
    , cycleModalitySelection_("cycleModality", "Cycle Modality", [this](Event* e){
        if (selectedModality_.size() > 0) {
            auto numChannels = volumePort_.getData()->getDataFormat()->getComponents();
            selectedModality_.setSelectedValue((selectedModality_.getSelectedValue() + 1) % std::min(selectedModality_.size(), numChannels));
        }
        e->markAsUsed();
    }, IvwKey::C, KeyState::Press)
    , cycleClassSelection_("cycleClass", "Cycle Class", [this](Event* e){
        if (selectedClass_.size() > 0) {
            selectedClass_.setSelectedValue((selectedClass_.getSelectedValue() + 1) % selectedClass_.size());
        }
        e->markAsUsed();
    }, IvwKey::X, KeyState::Press)
    , addAnnotation_("addAnnotation", "Add Annotation", [this](Event* e){
        addAnnotation();
        updateSims_.set(true);
        e->markAsUsed();
    }, IvwKey::Space, KeyState::Press)
    {
    // Invalidation Levels
    camera_.setInvalidationLevel(InvalidationLevel::InvalidOutput);
    rawTransferFunction_.setInvalidationLevel(InvalidationLevel::InvalidOutput);
    lighting_.setInvalidationLevel(InvalidationLevel::InvalidOutput);

    volumePort_.onChange([this]() { initializeResources(); });
    similarityPort_.onChange([this]() { initializeResources(); });
    ntfs_.setSerializationMode(PropertySerializationMode::All);
    ntfs_.setInvalidationLevel(InvalidationLevel::Valid);
    ntfs_.onChange([this](){ initializeResources(); });

    shader_.onReload([this]() { initializeResources(); });
    backgroundPort_.onConnect([&]() { this->invalidate(InvalidationLevel::InvalidResources); });
    backgroundPort_.onDisconnect([&]() { this->invalidate(InvalidationLevel::InvalidResources); });

    addPorts(volumePort_, entryPort_, exitPort_, similarityPort_, backgroundPort_, outport_, simOutport_);
    similarityPort_.setOptional(true);
    backgroundPort_.setOptional(true);

    addProperties(rawTransferFunction_, ntfs_, annotationButtons_, selectedModality_, selectedClass_, brushSize_, brushMode_, eraseMode_, updateSims_, raycasting_, camera_, lighting_, positionIndicator_);
    addProperties(currentVoxelSelectionX_, currentVoxelSelectionY_, currentVoxelSelectionZ_, currentVoxelSelection_, currentSimilarityTF_, addAnnotation_, cycleModalitySelection_, cycleClassSelection_);
    currentVoxelSelectionX_.setVisible(false).setReadOnly(true);
    currentVoxelSelectionY_.setVisible(false).setReadOnly(true);
    currentVoxelSelectionZ_.setVisible(false).setReadOnly(true);
    currentVoxelSelection_.setVisible(false).setReadOnly(true);
    currentSimilarityTF_.setVisible(false).setReadOnly(true);
    currentVoxelSelection_.onChange([this](){
        if (selectedClass_.size() > 0 && brushMode_.get()) {
            NTFProperty* ntfProp = static_cast<NTFProperty*>(ntfs_.getPropertyByIdentifier(selectedClass_.getSelectedIdentifier()));
            if (eraseMode_.get()){
                ntfProp->removeAnnotation(size3_t(currentVoxelSelection_.get()), brushSize_.get() / 2.0);
            } else {
                size3_t volDim = volumePort_.getData()->getDimensions();
                ntfProp->addAnnotation(size3_t(currentVoxelSelection_.get()), volDim, brushSize_.get() / 2.0);
            }
        }
    });
    selectedClass_.onChange([this](){
        if (selectedClass_.size() > 0) {
            NTFProperty* ntfProp = static_cast<NTFProperty*>(ntfs_.getPropertyByIdentifier(selectedClass_.getSelectedIdentifier()));
            vec2 simRange = ntfProp->getSimilarityRamp();
            if (simRange.y < 0.99) {
                currentSimilarityTF_.set(TransferFunction(
                    {{simRange.x, vec4(0.f, 1.f, 0.f, 0.f)}, {simRange.y, vec4(0.f, 1.f, 0.f, 1.f)},
                     {0.99,       vec4(0.f, 1.f, 0.f, 1.f)}, {1.0,        vec4(0.f, 0.f, 1.f, 1.f)}}));
            } else {
                currentSimilarityTF_.set(TransferFunction(
                    {{simRange.x, vec4(0.f, 1.f, 0.f, 0.f)}, {simRange.y, vec4(0.f, 1.f, 0.f, 1.f)}}));
            }
            // currentSimilarityTF_.set(ntfProp->simTf_.get());
        }
    });
}

void DINOVolumeRenderer::initializeResources() {
    updateButtons();
    utilgl::addShaderDefines(shader_, raycasting_);
    utilgl::addShaderDefines(shader_, camera_);
    utilgl::addShaderDefines(shader_, lighting_);
    utilgl::addShaderDefines(shader_, positionIndicator_);
    utilgl::addShaderDefinesBGPort(shader_, backgroundPort_);

    size_t numComponents = volumePort_.getData()->getDataFormat()->getComponents();
    size_t numClasses = similarityPort_.getVectorData().size();
    shader_.getFragmentShaderObject()->addShaderDefine("NUM_CLASSES", std::to_string(numClasses));

    auto ntfProps = ntfs_.getProperties();
    if (similarityPort_.hasData() && numClasses > 0) {
        StrBuffer str3dsampler, str2dsampler, strApply, strTfChannel;
        for (size_t i = 0; i < numClasses; ++i) {
            // Define Uniforms
            auto ntfProp (static_cast<const NTFProperty*>(ntfProps[i]));
            int modalityChannel = ntfProp->modality_.getSelectedValue();
            vec2 simRange = ntfProp->getSimilarityRamp();
            str3dsampler.append("uniform sampler3D ntf{0};", i);
            str2dsampler.append("uniform sampler2D transferFunction{0};", i);
            // str2dsampler.append("uniform sampler2D similarityFunction{0};", i);
            // Generate code to use the transfer functions
            strApply.append("sim[{0}] = texture(ntf{0}, samplePos).x;", i);
            // strApply.append("alpha[{0}] = applyTF(similarityFunction{0}, sim[{0}]).a;", i);
            strApply.append("alpha[{0}] = mix(0.0, 1.0, clamp((sim[{0}] - {1}) / {2}, 0.0, 1.0));", i, simRange.x, std::max(simRange.y - simRange.x, 1e-5f));
            strApply.append("grad[{0}] = gradientCentralDiff(voxel, volume, volumeParameters, samplePos, {1});", i, modalityChannel);
            if (numComponents < 4) {
                strApply.append("color[{0}] = vec4(1,1,1,alpha[{0}]) * applyTF(transferFunction{0}, voxel, {2});", i, numClasses, modalityChannel);
            } else if (numComponents == 4) {
                strApply.append("color[{0}] = vec4(voxel.rgb, alpha[{0}]);", i);
            }
            // sim, alpha, color have length numClasses + 1, the numClasses (=last) value is used for TF on raw volume data
        }
        shader_.getFragmentShaderObject()->addShaderDefine("DEFINE_NTF_SAMPLERS", str3dsampler.view());
        shader_.getFragmentShaderObject()->addShaderDefine("DEFINE_TF_SAMPLERS", str2dsampler.view());
        shader_.getFragmentShaderObject()->addShaderDefine("APPLY_NTFS", strApply.view());
    } else {
        shader_.getFragmentShaderObject()->addShaderDefine("DEFINE_NTF_SAMPLERS", "");
        shader_.getFragmentShaderObject()->addShaderDefine("DEFINE_TF_SAMPLERS", "");
        shader_.getFragmentShaderObject()->addShaderDefine("APPLY_NTFS", "");
    }
    if(volumePort_.hasData()) {
        shader_.build();
    }
}

void DINOVolumeRenderer::process() {
    utilgl::activateAndClearTarget(outport_);
    shader_.activate();

    TextureUnitContainer units;
    // Bind Volumes
    utilgl::bindAndSetUniforms(shader_, units, volumePort_);
    auto similarityMap = similarityPort_.getSourceVectorData();
    for (auto [p,v] : similarityMap){
        utilgl::bindAndSetUniforms(shader_, units, *v, p->getIdentifier());
    }
    // Bind Images
    utilgl::bindAndSetUniforms(shader_, units, entryPort_, ImageType::ColorDepthPicking);
    utilgl::bindAndSetUniforms(shader_, units, exitPort_, ImageType::ColorDepth);
    if (auto normals = entryPort_.getData()->getColorLayer(1)) {
        utilgl::bindAndSetUniforms(shader_, units,
                                   *normals->getRepresentation<LayerGL>()->getTexture(),
                                   std::string_view{"entryNormal"});
        shader_.setUniform("useNormals", true);
    } else {
        shader_.setUniform("useNormals", false);
    }
    if (backgroundPort_.hasData()) {
        utilgl::bindAndSetUniforms(shader_, units, backgroundPort_, ImageType::ColorDepthPicking);
    }
    // Bind Transfer Functions
    utilgl::bindAndSetUniforms(shader_, units, rawTransferFunction_);
    std::vector<Property*> ntfProps = ntfs_.getProperties();
    for (const Property* prop : ntfProps) {
        const TransferFunctionProperty& tfProp = static_cast<const NTFProperty*>(prop)->tf_;
        // const TransferFunctionProperty& simTfProp = static_cast<const NTFProperty*>(prop)->simTf_;
        utilgl::bindAndSetUniforms(shader_, units, tfProp);
        // utilgl::bindAndSetUniforms(shader_, units, simTfProp);
    }
    // Bind remaining stuff
    utilgl::setUniforms(shader_, outport_, camera_, lighting_, raycasting_, positionIndicator_);
    // Draw
    utilgl::singleDrawImagePlaneRect();

    shader_.deactivate();
    utilgl::deactivateCurrentTarget();

    // Pass through selected similarity Volume
    size_t selection = selectedClass_.getSelectedValue();
    if (similarityMap.size() > 0 && selection < similarityMap.size()) {
        // auto ntfID = similarityMap[selection].first->getIdentifier();
        // auto simVol = similarityMap[selection].second;
        // NTFProperty* ntfProp = static_cast<NTFProperty*>(ntfs_.getPropertyByIdentifier(ntfID));
        // Volume v = Volume(*similarityMap[selection].second);
        // auto vram = std::shared_ptr<VolumeRAM>(simVol->getRepresentation<VolumeRAM>()->clone());
        // double val = simVol->getDataFormat()->getMax();
        // vec3 factor = vec3(simVol->getDimensions()) / vec3(volumePort_.getData()->getDimensions());
        // for (const size3_t pos : ntfProp->getAnnotatedVoxels()){
        //     size3_t lowresPos = size3_t(glm::round(factor * vec3(pos)));
        //     vram->setFromDouble(lowresPos, val);
        //     LogInfo("Setting output Volume's voxel at " << lowresPos << " to " << val);
        // }
        simOutport_.setData(similarityMap[selection].second);
    } else {
        simOutport_.setData(Volume(size3_t(8,8,8)));
    }
}

void DINOVolumeRenderer::updateButtons() {
    NetworkLock lock(this);
    std::map<std::string, Property*> ntfPropMap; // Maps ID -> Property*
    std::map<std::string, Property*> btnMap;     // Maps NTF-ID -> Property*  (modifies button ID to contain NTF ID)
    for (Property* prop : ntfs_.getProperties()) {
        ntfPropMap.insert(std::make_pair(prop->getIdentifier(), prop));
    }
    for (Property* prop : annotationButtons_.getProperties()) {
        std::string_view btnId = prop->getIdentifier(); // Crop button ID to the ntf id (remove "-addCoord")
        btnMap.insert(std::make_pair(btnId.substr(0, btnId.size()-9), prop));
    }
    // Remove old buttons
    for (auto& [id, p] : btnMap) {
        if (ntfPropMap.count(id) == 0) { // Button not in NTF properties, remove
            annotationButtons_.removeProperty(id+"-addCoord");
        }
    }
    // Update existing and add new buttons
    for (auto entry : ntfPropMap) {
        std::string id = entry.first;
        Property* p = entry.second;
        if (btnMap.count(id) == 1) { // There is a button for this NTF Property, update Name
            btnMap[id]->setDisplayName("Add to " + p->getDisplayName());
        } else if (btnMap.count(id) == 0) { // Add button for new NTF Property
            ButtonProperty* btn = new ButtonProperty(
                p->getIdentifier() + "-addCoord",
                "Add to " + p->getDisplayName(),
                InvalidationLevel::Valid
            );
            NTFProperty* ntfProp = static_cast<NTFProperty*>(p);
            btn->onChange([&, id, ntfProp](){
                if (getNetwork()->isDeserializing()) return;
                size3_t volDim = volumePort_.getData()->getDimensions();
                size3_t coord (currentVoxelSelection_.get());
                ntfProp->addAnnotation(coord, volDim);
                selectedClass_.setSelectedIdentifier(id);
            });
            annotationButtons_.addProperty(btn, true);
            // add callback to ntfProp's modality change that updates selectedModality_
            ntfProp->modality_.onChange([&, ntfProp](){
                if (getNetwork()->isDeserializing()) return;
                selectedModality_.setSelectedValue(ntfProp->modality_.getSelectedValue());
            });
            ntfProp->similarityRamp_.onChange([&, ntfProp](){
                if (getNetwork()->isDeserializing()) return;
                updateSims_.set(true);
            });
        }
    }
    // Update selectedClass dropdown
    if (getNetwork()->isDeserializing()) return;
    // Update selectedClass dropdown if necessary, then set selected class to last added
    if (ntfPropMap.size() != selectedClass_.size()) {
        selectedClass_.clearOptions();
        for (auto [i, entry] : util::enumerate(ntfPropMap)){
            selectedClass_.addOption(entry.first, entry.first, i);
        }
        if (ntfPropMap.size() > 0) {
            selectedClass_.setSelectedIdentifier(ntfPropMap.rbegin()->first);
        }
    }
}

void DINOVolumeRenderer::addAnnotation() {
    if (selectedClass_.size() > 0) {
        size3_t volDim = volumePort_.getData()->getDimensions();
        size3_t coord (currentVoxelSelection_.get());
        NTFProperty* ntfProp = static_cast<NTFProperty*>(ntfs_.getPropertyByIdentifier(selectedClass_.getSelectedIdentifier()));
        ntfProp->addAnnotation(coord, volDim, brushSize_.get() / 2.0);
    }
}

void DINOVolumeRenderer::removeAnnotation() {
    if (selectedClass_.size() > 0) {
        size3_t coord (currentVoxelSelection_.get());
        NTFProperty* ntfProp = static_cast<NTFProperty*>(ntfs_.getPropertyByIdentifier(selectedClass_.getSelectedIdentifier()));
        ntfProp->removeAnnotation(coord, brushSize_.get() / 2.0);
    }
}
}  // namespace inviwo
