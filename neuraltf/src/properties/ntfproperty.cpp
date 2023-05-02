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

#include <inviwo/neuraltf/properties/ntfproperty.h>
#include <inviwo/core/properties/valuewrapper.h>              // for PropertySerializa...
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
    simTf_.setSerializationMode(PropertySerializationMode::All);
    addProperties(tf_, simTf_, similarityExponent_, similarityThreshold_, similarityReduction_, modality_, modalityWeight_, annotations_);
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
    , simTf_("similarityFunction0", "Similarity Function", TransferFunction(
        {{0.6, vec4(0.0f, 1.0f, 0.0f, 0.0f)}, {0.7, vec4(0.0f, 1.0f, 0.0f, 1.0f)}}))
    , similarityExponent_("simexponent", "Exponent", 2.5, 1.0, 10.0)
    , similarityThreshold_("simthresh", "Threshold", 0.25, 0.0, 1.0)
    , similarityReduction_("simreduction", "Reduction", { {"mean", "Mean", "mean"}, {"max", "Max", "max"} })
    , modality_("modality", "Modality", {
        {"channel0", "Channel 1", 0}, {"channel1", "Channel 2", 1},
        {"channel2", "Channel 3", 2}, {"channel3", "Channel 4", 3} }, 0, InvalidationLevel::InvalidResources)
    , modalityWeight_("modalityWeight", "Modality Weighting", vec4(1.0, 0,0,0), vec4(0), vec4(1))
    , annotations_("annotations", "Annotations",
        std::make_unique<IntSize3Property>("coord", "Coordinate", size3_t(0), size3_t(0), size3_t(2048)),
        0, ListPropertyUIFlag::Remove, InvalidationLevel::Valid)
    , volumeInport_(inport)
    { init(); }

NTFProperty::NTFProperty(const NTFProperty& other)
    : CompositeProperty(other)
    , tf_(other.tf_)
    , simTf_(other.simTf_)
    , similarityExponent_(other.similarityExponent_)
    , similarityThreshold_(other.similarityThreshold_)
    , similarityReduction_(other.similarityReduction_)
    , modality_(other.modality_)
    , modalityWeight_(other.modalityWeight_)
    , annotations_(other.annotations_)
    { init(); }

Property& NTFProperty::setIdentifier(const std::string_view identifier){
    std::string num = std::string(identifier.size() > 0 ? identifier.substr(3) : "");
    tf_.setIdentifier("transferFunction" + num);
    simTf_.setIdentifier("similarityFunction" + num);
    return Property::setIdentifier(identifier);
}

void NTFProperty::deserialize(Deserializer& d) {
    Property::deserialize(d);
    std::string num = std::string(getIdentifier().size() > 0 ? getIdentifier().substr(3) : "");
    tf_.setIdentifier("transferFunction" + num);
    simTf_.setIdentifier("similarityFunction" + num);
    PropertyOwner::deserialize(d);
}

void NTFProperty::addAnnotation(const size3_t coord, const size3_t volDims, const float distanceThreshold){
    // static_cast<IntSize3Property*>(annotations_.constructProperty(0))->set(coord);
    if (distanceThreshold > 1.0f) {
        size_t distFloor = std::floor(distanceThreshold);
        size3_t minCoord = glm::clamp(coord - size3_t(distFloor), size3_t(0), volDims - size3_t(1));
        size3_t maxCoord = glm::clamp(coord + size3_t(distFloor), size3_t(0), volDims - size3_t(1));
        LogInfo("distFloor: " << distFloor);
        for (size_t x = minCoord.x; x <= maxCoord.x; ++x) {
            for (size_t y = minCoord.y; y <= maxCoord.y; ++y) {
                for (size_t z = minCoord.z; z <= maxCoord.z ; ++z) {
                    if (glm::distance(vec3(coord), vec3(x, y, z)) <= distanceThreshold) {
                        annotatedVoxels_.insert(size3_t(x, y, z));
                        LogInfo(size3_t(x, y, z) << " inserted.");
                    }
                }
            }
        }
    } else {
        annotatedVoxels_.insert(coord);
    }
    LogInfo("Annotated Voxels: " << annotatedVoxels_.size());
}

void NTFProperty::removeAnnotation(const size3_t coord, const float distanceThreshold){
    for (auto it = annotatedVoxels_.begin(); it != annotatedVoxels_.end();) {
        if (glm::distance(vec3(*it), vec3(coord)) <= distanceThreshold) {
            LogInfo(*it << " removed.");
            it = annotatedVoxels_.erase(it);
        } else {
            ++it;
        }
    }
    LogInfo("Annotated Voxels: " << annotatedVoxels_.size());
}

const std::vector<size3_t> NTFProperty::getAnnotatedVoxels() const {
    return std::vector<size3_t>(annotatedVoxels_.begin(), annotatedVoxels_.end());
}

const std::string NTFProperty::classIdentifier = "org.inviwo.NTFProperty";
std::string NTFProperty::getClassIdentifier() const { return classIdentifier; }

NTFProperty* NTFProperty::clone() const {
    return new NTFProperty(*this);
}

}  // namespace inviwo
