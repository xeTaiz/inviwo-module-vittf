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

#include "inviwo/core/properties/compositeproperty.h"
#include <inviwo/neuraltf/properties/ntfproperty.h>
#include <inviwo/core/properties/valuewrapper.h>              // for PropertySerializa...
#include <inviwo/core/properties/propertysemantics.h>         // for PropertySemantics
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_set>


namespace inviwo {

void NTFPropertyList::insertProperty(size_t index, Property* property, bool owner) {
    property->addObserver(this);
    CompositeProperty::insertProperty(index, property, owner);
}

void NTFPropertyList::onSetDisplayName(Property* property, const std::string& displayName) {
    auto btnListProp = this->getOwner()->getPropertyByIdentifier("annotationButtons");
    auto btn = dynamic_cast<PropertyOwner*>(btnListProp)->getPropertyByIdentifier(property->getIdentifier()+"-addCoord");
    btn->setDisplayName("Add to " + displayName);
}

void NTFProperty::init() {
    modality_.onChange([&](){
        HistogramSelection selection{};
        selection[modality_.getSelectedValue()] = true;
        tf_.setHistogramSelection(selection);
        vec4 modalityWeight (0.0f);
        modalityWeight[modality_.getSelectedValue()] = 1.0f;
        modalityWeight_.set(modalityWeight);
    });
    tf_.setSerializationMode(PropertySerializationMode::All);
    color_.setSemantics(PropertySemantics::Color);
    enableBLS_.addProperties(blsSigmaSpatial_, blsSigmaChroma_, blsSigmaLuma_);
    enableBLS_.onChange([&](){
        requiresUpdate_ = true;
    });
    color_.onChange([&](){ requiresUpdate_ = true; });
    blsSigmaSpatial_.onChange([&](){requiresUpdate_ = true;});
    blsSigmaChroma_.onChange([&](){requiresUpdate_ = true;});
    blsSigmaLuma_.onChange([&](){requiresUpdate_ = true;});
    isoValue_.onChange([&](){requiresUpdate_ = true;});
    proximity_.onChange([&](){requiresUpdate_ = true;});
    contrastFactor_.onChange([&](){requiresUpdate_ = true;});
    enableBLS_.setCollapsed(!enableBLS_.isChecked());
    addProperties(color_, similarityReduction_, modality_, modalityWeight_, isoValue_, proximity_, contrastFactor_, connectedComponent_, clearAnnotationButton_, clearLastAnnotationButton_, enableBLS_);

    modality_.setVisible(false);
    modalityWeight_.setVisible(false);
    similarityReduction_.setVisible(false);
    // Hide currently deprecated properties TODO: remove or use
    connectedComponent_.setVisible(false);
    contrastFactor_.setVisible(false);
    setAnnotationCount(annotatedVoxels_.size());
}

NTFProperty::NTFProperty(std::string_view identifier,
    std::string_view displayName,
    VolumeInport* inport,
    InvalidationLevel invalidationLevel,
    PropertySemantics semantics)
    : CompositeProperty(identifier, displayName, invalidationLevel, semantics)
    , tf_("transferFunction0", "Transfer Function", TransferFunction(
        {{0.01, vec4(1.0f, 1.0f, 1.0f, 0.0f)}, {0.02, vec4(1.0f, 1.0f, 1.0f, 1.0f)},
         {0.99, vec4(1.0f, 1.0f, 1.0f, 1.0f)}, {1.0, vec4(1.0f, 1.0f, 1.0f, 0.0f)}}), inport)
    , color_("color", "Color", vec4(1.0f), vec4(0.0f), vec4(1.0f), vec4(0.01f), InvalidationLevel::InvalidResources)
    , isoValue_("isoValue", "Iso Value", 0.4, 0.0, 1.0, 0.01, InvalidationLevel::InvalidResources)
    , proximity_("proximity", "Proximity", 0.1, 0.0, 1.0, 0.1, InvalidationLevel::InvalidResources)
    , similarityReduction_("simreduction", "Reduction", { {"mean", "Mean", "mean"}, {"max", "Max", "max"} })
    , modality_("modality", "Modality", {
        {"channel0", "Channel 1", 0}, {"channel1", "Channel 2", 1},
        {"channel2", "Channel 3", 2}, {"channel3", "Channel 4", 3} }, 0, InvalidationLevel::InvalidResources)
    , modalityWeight_("modalityWeight", "Modality Weighting", vec4(1.0, 0,0,0), vec4(0), vec4(1))
    , clearAnnotationButton_("clearAnnotations", "Clear Annotations", [&](){NTFProperty::clearAnnotations();})
    , clearLastAnnotationButton_("clearLastAnnotation", "Undo Last Annotation", [&](){NTFProperty::removeLastAnnotation();})
    , enableBLS_("enableBLS", "Bilateral Solver", false, InvalidationLevel::InvalidResources)
    , blsSigmaSpatial_("blsSigmaSpatial", "Sigma Spatial", 5, 1, 32)
    , blsSigmaChroma_("blsSigmaChroma", "Sigma Chroma", 5, 1, 16)
    , blsSigmaLuma_("blsSigmaLuma", "Sigma Luma", 5, 1, 16)
    , contrastFactor_("contrastFactor", "Contrast Factor", 2.0f, 1.0f, 8.0f)
    , connectedComponent_("connectedComponent", "Connected Component", false, InvalidationLevel::InvalidResources)
    , volumeInport_(inport)
    , requiresUpdate_(false)
    { init(); }

NTFProperty::NTFProperty(const NTFProperty& other)
    : CompositeProperty(other)
    , tf_(other.tf_)
    , color_(other.color_)
    , isoValue_(other.isoValue_)
    , proximity_(other.proximity_)
    , similarityReduction_(other.similarityReduction_)
    , modality_(other.modality_)
    , modalityWeight_(other.modalityWeight_)
    , clearAnnotationButton_("clearAnnotations", "Clear Annotations", [&](){NTFProperty::clearAnnotations();})
    , clearLastAnnotationButton_("clearLastAnnotation", "Undo Last Annotation", [&](){NTFProperty::removeLastAnnotation();})
    , enableBLS_(other.enableBLS_)
    , blsSigmaSpatial_(other.blsSigmaSpatial_)
    , blsSigmaChroma_(other.blsSigmaChroma_)
    , blsSigmaLuma_(other.blsSigmaLuma_)
    , contrastFactor_(other.contrastFactor_)
    , connectedComponent_(other.connectedComponent_)
    , volumeInport_(other.volumeInport_)
   , requiresUpdate_(other.requiresUpdate_)
    { init(); }

Property& NTFProperty::setIdentifier(const std::string_view identifier){
    std::string num = std::string(identifier.size() > 0 ? identifier.substr(3) : "");
    tf_.setIdentifier("transferFunction" + num);
    return Property::setIdentifier(identifier);
}

CompositeProperty& NTFProperty::setCollapsed(bool val) {
    // TODO: do or remove
    // LogInfo("Collapsing " << this->getIdentifier());
    return CompositeProperty::setCollapsed(val);
}

CompositeProperty& NTFProperty::resetToDefaultState() {
    clearLastAnnotationButton_.pressButton();
    return *this;
}

void NTFProperty::deserialize(Deserializer& d) {
    Property::deserialize(d);
    std::string num = std::string(getIdentifier().size() > 0 ? getIdentifier().substr(3) : "");
    tf_.setIdentifier("transferFunction" + num);
    PropertyOwner::deserialize(d);
}

void NTFProperty::addAnnotation(const size3_t coord, const size3_t volDims){
    size_t oldSz = annotatedVoxels_.size();
    annotatedVoxels_.push_back(coord);

    if (oldSz < annotatedVoxels_.size()) {
        requiresUpdate_ = true;
        setAnnotationCount(annotatedVoxels_.size());
    }
}

void NTFProperty::removeAnnotation(const size3_t coord, const float distanceThreshold){
    size_t oldSz = annotatedVoxels_.size();
    for (auto it = annotatedVoxels_.begin(); it != annotatedVoxels_.end();) {
        if (glm::distance(vec3(*it), vec3(coord)) <= distanceThreshold) {
            it = annotatedVoxels_.erase(it);
        } else {
            ++it;
        }
    }
    if (oldSz > annotatedVoxels_.size()) {
        requiresUpdate_ = true;
        setAnnotationCount(annotatedVoxels_.size());
    }
}

void NTFProperty::removeLastAnnotation(){
    if (annotatedVoxels_.size() > 0) {
        annotatedVoxels_.pop_back();
    }
    setAnnotationCount(annotatedVoxels_.size());
    requiresUpdate_ = true;
}

void NTFProperty::clearAnnotations(){
    annotatedVoxels_.clear();
    setAnnotationCount(size_t(0));
    requiresUpdate_ = true;
}

void NTFProperty::setAnnotationCount(const size_t annotationCount) {
    if (annotationCount > 0) {
        clearAnnotationButton_.setDisplayName("Clear Annotations (" + std::to_string(annotationCount) +")");
        clearAnnotationButton_.setReadOnly(false);
        clearLastAnnotationButton_.setReadOnly(false);
    } else {
        clearAnnotationButton_.setDisplayName("Clear Annotations");
        clearAnnotationButton_.setReadOnly(true);
        clearLastAnnotationButton_.setReadOnly(true);
    }
}

void NTFProperty::setAnnotations(const std::vector<size3_t>& annotations){
    annotatedVoxels_.clear();
    for (const size3_t a : annotations){
        annotatedVoxels_.push_back(a);
    }
    setAnnotationCount(annotatedVoxels_.size());
}

const std::vector<size3_t> NTFProperty::getAnnotatedVoxels() const {
    return annotatedVoxels_;
}

const std::shared_ptr<std::vector<size3_t>> NTFProperty::getAnnotatedVoxelsPtr() const {
    return std::make_shared<std::vector<size3_t>>(annotatedVoxels_.begin(), annotatedVoxels_.end());
}

void NTFProperty::showModalityProperties(bool show) {
    modality_.setVisible(show);
    modalityWeight_.setVisible(show);
}


const std::string NTFProperty::classIdentifier = "org.inviwo.NTFProperty";
std::string NTFProperty::getClassIdentifier() const { return classIdentifier; }

NTFProperty* NTFProperty::clone() const {
    return new NTFProperty(*this);
}

}  // namespace inviwo
