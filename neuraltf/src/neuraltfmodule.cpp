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

#include <inviwo/neuraltf/neuraltfmodule.h>
#include <inviwo/neuraltf/processors/dinovolumerenderer.h>
#include <inviwo/neuraltf/properties/ntfproperty.h>
#include <inviwo/neuraltf/bindings/pyneuraltf.h>

#include <inviwo/neuraltf/processors/brushvolumeslicegl.h>
#include <modules/basegl/shader_resources.h>
#include <modules/opengl/shader/shadermanager.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace inviwo {

NeuralTFModule::NeuralTFModule(InviwoApplication* app)
    : InviwoModule(app, "NeuralTF")
    , scripts_(getPath() + "/python")
    , pythonProcessorFolderObserver_(app, getPath() + "/python/processors", *this)
    {
    // Add a directory to the search path of the Shadermanager
    ShaderManager::getPtr()->addShaderSearchPath(getPath(ModulePath::GLSL));

    // Register objects that can be shared with the rest of inviwo here:

    // Processors
    registerProcessor<BrushVolumeSliceGL>();
    registerProcessor<DINOVolumeRenderer>();
    // registerProcessor<NeuralTFProcessor>();

    // Properties
    registerProperty<NTFProperty>();

    // Readers and writes
    // registerDataReader(std::make_unique<NeuralTFReader>());
    // registerDataWriter(std::make_unique<NeuralTFWriter>());

    // Data converters
    // registerRepresentationConverter(std::make_unique<NeuralTFDisk2RAMConverter>());

    // Ports
    // registerPort<NeuralTFOutport>();
    // registerPort<NeuralTFInport>();

    // PropertyWidgets
    // registerPropertyWidget<NeuralTFPropertyWidget, NeuralTFProperty>("Default");

    // Dialogs
    // registerDialog<NeuralTFDialog>(NeuralTFOutport);

    // Other things
    // registerCapabilities(std::make_unique<NeuralTFCapabilities>());
    // registerSettings(std::make_unique<NeuralTFSettings>());
    // registerMetaData(std::make_unique<NeuralTFMetaData>());
    // registerPortInspector("NeuralTFOutport", "path/workspace.inv");
    // registerProcessorWidget(std::string processorClassName, std::unique_ptr<ProcessorWidget> processorWidget);
    // registerDrawer(util::make_unique_ptr<NeuralTFDrawer>());
    try {
        py::module ivwpy = pybind11::module::import("inviwopy");
        py::module ivwProps = ivwpy.attr("properties");
        exposeNTFBindings(ivwProps);
    } catch (const std::exception& e) {
        throw ModuleInitException(e.what(), IVW_CONTEXT);
    }
}

}  // namespace inviwo
