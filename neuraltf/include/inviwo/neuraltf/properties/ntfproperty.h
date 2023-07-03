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
#include <inviwo/core/properties/compositeproperty.h>  // for CompositeProperty
#include <inviwo/core/properties/listproperty.h>
#include <inviwo/core/properties/invalidationlevel.h>  // for InvalidationLevel, InvalidationLev...
#include <inviwo/core/properties/propertysemantics.h>  // for PropertySemantics, PropertySemanti...
#include <inviwo/core/properties/stringproperty.h>
#include <inviwo/core/properties/ordinalproperty.h>
#include <inviwo/core/properties/minmaxproperty.h>
#include <inviwo/core/properties/boolproperty.h>
#include <inviwo/core/properties/buttonproperty.h>
#include <inviwo/core/properties/optionproperty.h>
#include <inviwo/core/properties/transferfunctionproperty.h>  // for TransferFunctionProperty

#include <string>
#include <string_view>
#include <unordered_set>

namespace inviwo {

class IVW_MODULE_NEURALTF_API NTFPropertyList : public ListProperty, public PropertyObserver {
public:
    NTFPropertyList(std::string_view identifier, std::string_view displayName, Document help = {},
                 std::vector<std::unique_ptr<Property>> prefabs = {},
                 size_t maxNumberOfElements = 0,
                 ListPropertyUIFlags uiFlags = ListPropertyUIFlag::Add | ListPropertyUIFlag::Remove,
                 InvalidationLevel invalidationLevel = InvalidationLevel::InvalidResources,
                 PropertySemantics semantics = PropertySemantics::Default)
        : ListProperty(identifier, displayName, help, std::move(prefabs), maxNumberOfElements, uiFlags, invalidationLevel, semantics) {};

    NTFPropertyList(std::string_view identifier, std::string_view displayName,
                 std::vector<std::unique_ptr<Property>> prefabs, size_t maxNumberOfElements = 0,
                 ListPropertyUIFlags uiFlags = ListPropertyUIFlag::Add | ListPropertyUIFlag::Remove,
                 InvalidationLevel invalidationLevel = InvalidationLevel::InvalidResources,
                 PropertySemantics semantics = PropertySemantics::Default)
        : ListProperty(identifier, displayName, std::move(prefabs), maxNumberOfElements, uiFlags, invalidationLevel, semantics) {};

    NTFPropertyList(std::string_view identifier, std::string_view displayName,
                 std::unique_ptr<Property> prefab, size_t maxNumberOfElements = 0,
                 ListPropertyUIFlags uiFlags = ListPropertyUIFlag::Add | ListPropertyUIFlag::Remove,
                 InvalidationLevel invalidationLevel = InvalidationLevel::InvalidResources,
                 PropertySemantics semantics = PropertySemantics::Default)
        : ListProperty(identifier, displayName, std::move(prefab), maxNumberOfElements, uiFlags, invalidationLevel, semantics) {};

    NTFPropertyList(std::string_view identifier, std::string_view displayName,
                 size_t maxNumberOfElements,
                 ListPropertyUIFlags uiFlags = ListPropertyUIFlag::Add | ListPropertyUIFlag::Remove,
                 InvalidationLevel invalidationLevel = InvalidationLevel::InvalidResources,
                 PropertySemantics semantics = PropertySemantics::Default)
        : ListProperty(identifier, displayName, maxNumberOfElements, uiFlags, invalidationLevel, semantics) {};

    NTFPropertyList(const NTFPropertyList& rhs) : ListProperty(rhs) {};
    // virtual ListProperty* clone() const override { return ListProperty::clone(this); }
    // virtual ~NTFPropertyList() = default;
    virtual void insertProperty(size_t index, Property* property, bool owner = true) override;
    virtual void onSetDisplayName(Property* property, const std::string& displayName) override;
};

/**
 * \brief VERY_BRIEFLY_DESCRIBE_THE_CLASS
 * DESCRIBE_THE_CLASS_FROM_A_DEVELOPER_PERSPECTIVE
 */
class IVW_MODULE_NEURALTF_API NTFProperty : public CompositeProperty {
public:
    NTFProperty(std::string_view identifier, std::string_view displayName,
                VolumeInport* inport = nullptr,
                InvalidationLevel invalidationLevel = InvalidationLevel::InvalidOutput,
                PropertySemantics semantics = PropertySemantics::Default);
    NTFProperty(const NTFProperty& other);
    ~NTFProperty() = default;

    virtual std::string getClassIdentifier() const override;
    static const std::string classIdentifier;

    virtual NTFProperty* clone() const override;
    virtual Property& setIdentifier(const std::string_view identifier) override;
    virtual void deserialize(Deserializer&) override;

    void addAnnotation(const size3_t coord, const size3_t volDims, const float distanceThreshold = 1e-4f);
    void removeAnnotation(const size3_t coord, const float distanceThreshold = 1e-4f);
    void clearAnnotations();
    const std::vector<size3_t> getAnnotatedVoxels() const;
    void init();

    void showModalityProperties(bool show);

    // Getter & Setter
    float getIsoValue() const { return isoValue_.get(); }
    void setIsoValue(float value) { isoValue_.set(value); }
    std::string getSimilarityReduction() const { return similarityReduction_.getSelectedValue(); }
    void setSimilarityReduction(std::string value) { similarityReduction_.setSelectedValue(value); }
    int getModality() const { return modality_.getSelectedValue(); }
    void setModality(int value) { modality_.setSelectedValue(value); }
    vec4 getModalityWeight() const { return modalityWeight_.get(); }
    void setModalityWeight(vec4 value) { modalityWeight_.set(value); }
    bool getRequiresUpdate() const { return requiresUpdate_; }
    void setRequiresUpdate(bool value) { requiresUpdate_ = value; }
    vec2 getSimilarityRamp() const { return similarityRamp_.get(); }
    void setSimilarityRamp(vec2 value) { similarityRamp_.set(value); }
    vec4 getColor() const { return color_.get(); }
    void setColor(vec4 value) { color_.set(value); }

    TransferFunctionProperty tf_;
    FloatVec4Property color_;
    FloatProperty isoValue_;
    FloatMinMaxProperty similarityRamp_;
    OptionPropertyString similarityReduction_;
    OptionPropertyInt modality_;
    FloatVec4Property modalityWeight_;
    IntSizeTProperty annotationCount_;
    ButtonProperty clearAnnotationButton_;

    VolumeInport* volumeInport_;

private:
    std::unordered_set<size3_t> annotatedVoxels_;
    bool requiresUpdate_;
};

}  // namespace inviwo
