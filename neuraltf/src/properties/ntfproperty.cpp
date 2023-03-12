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

NTFProperty::NTFProperty(std::string_view identifier,
    std::string_view displayName,
    VolumeInport* inport,
    InvalidationLevel invalidationLevel,
    PropertySemantics semantics) 
    : CompositeProperty(identifier, displayName, invalidationLevel, semantics)
    , tf_("transferFunction0", "Transfer Function", inport)
    , simTf_("similarityFunction0", "Similarity Function", "Maps similarity to opacity"_help)
    , similarityExponent_("simexponent", "Similarity Exponent", 2.0, 1.0, 10.0)
    , similarityThreshold_("simthresh", "Similarity Threshold", 0.25, 0.0, 1.0)
    , normalizeBeforeBilateral_("normBeforeBilat", "Normalize before Bilateral Solver", false)
    , annotations_("annotations", "Annotations", 
        std::make_unique<IntSize3Property>("coord", "Coordinate", size3_t(0), size3_t(0), size3_t(2048)),
        0, ListPropertyUIFlag::Remove, InvalidationLevel::Valid)
    , volumeInport_(inport)
    {
        tf_.setSerializationMode(PropertySerializationMode::All);
        simTf_.setSerializationMode(PropertySerializationMode::All);
        addProperties(tf_, simTf_, similarityExponent_, similarityThreshold_, normalizeBeforeBilateral_, annotations_);
}

NTFProperty::NTFProperty(const NTFProperty& other)
    : CompositeProperty(other)
    , tf_(other.tf_)
    , simTf_(other.simTf_)
    , similarityExponent_(other.similarityExponent_)
    , similarityThreshold_(other.similarityThreshold_)
    , normalizeBeforeBilateral_(other.normalizeBeforeBilateral_)
    , annotations_(other.annotations_) {
        addProperties(tf_, simTf_, similarityExponent_, similarityThreshold_, normalizeBeforeBilateral_, annotations_);
}

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

void NTFProperty::addAnnotation(const size3_t coord){
    static_cast<IntSize3Property*>(annotations_.constructProperty(0))->set(coord);
}

const std::string NTFProperty::classIdentifier = "org.inviwo.NTFProperty";
std::string NTFProperty::getClassIdentifier() const { return classIdentifier; }

NTFProperty* NTFProperty::clone() const {
    return new NTFProperty(*this);
}

}  // namespace inviwo
