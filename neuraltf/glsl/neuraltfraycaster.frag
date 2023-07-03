/*********************************************************************************
 *
 * Inviwo - Interactive Visualization Workshop
 *
 * Copyright (c) 2014-2022 Inviwo Foundation
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

#include "utils/structs.glsl"
#include "utils/sampler2d.glsl"
#include "utils/sampler3d.glsl"

#include "utils/classification.glsl"
#include "utils/compositing.glsl"
#include "utils/depth.glsl"
#include "utils/gradients.glsl"
#include "utils/shading.glsl"
#include "utils/raycastgeometry.glsl"

uniform VolumeParameters volumeParameters;
uniform sampler3D volume;
DEFINE_NTF_SAMPLERS

uniform ImageParameters entryParameters;
uniform sampler2D entryColor;
uniform sampler2D entryDepth;
uniform sampler2D entryNormal;
uniform bool useNormals = false;

uniform ImageParameters exitParameters;
uniform sampler2D exitColor;
uniform sampler2D exitDepth;

uniform ImageParameters bgParameters;
uniform sampler2D bgColor;
uniform sampler2D bgPicking;
uniform sampler2D bgDepth;

uniform ImageParameters outportParameters;

uniform LightParameters lighting;
uniform CameraParameters camera;
uniform VolumeIndicatorParameters positionindicator;
uniform RaycastingParameters raycaster;

uniform sampler2D rawTransferFunction;
DEFINE_TF_SAMPLERS

#define ERT_THRESHOLD 0.99  // threshold for early ray termination

vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec4 rayTraversal(vec3 entryPoint, vec3 exitPoint, vec2 texCoords, float backgroundDepth, vec3 entryNormal) {
    vec4 result = vec4(0.0);
    vec3 rayDirection = exitPoint - entryPoint;
    float tEnd = length(rayDirection);
    float tIncr = min(
        tEnd, tEnd / (raycaster.samplingRate * length(rayDirection * volumeParameters.dimensions)));
    float samples = ceil(tEnd / tIncr);
    tIncr = tEnd / samples;
    float raySegmentLen = tIncr;
    float t = 0.5f * tIncr;
    rayDirection = normalize(rayDirection);
    float tDepth = -1.0;
    vec4 color[NUM_CLASSES + 1];
    vec3 gradient;
    vec4 voxel;
    vec3 hsv;
    vec3 grad[NUM_CLASSES + 1];
    float sim[NUM_CLASSES + 1];
    float alpha[NUM_CLASSES + 1];
    float isoValue[NUM_CLASSES +1]; // TODO: uniform per class??
    isoValue[NUM_CLASSES] = 0.9;
    bool surfaceFound[NUM_CLASSES + 1];
    for(int i = 0; i < NUM_CLASSES+1; ++i) { surfaceFound[i] = false; }
    vec3 samplePos;
    vec3 toCameraDir =
        normalize(camera.position - (volumeParameters.textureToWorld * vec4(entryPoint, 1.0)).xyz);

    vec4 backgroundColor = vec4(0);
    float bgTDepth = -1;
#ifdef BACKGROUND_AVAILABLE
    backgroundColor = texture(bgColor, texCoords);
    // convert to raycasting depth
    bgTDepth = tEnd * calculateTValueFromDepthValue(
        camera, backgroundDepth, texture(entryDepth, texCoords).x, texture(exitDepth, texCoords).x);

    if (bgTDepth < 0) {
        result = backgroundColor;
    }
#endif // BACKGROUND_AVAILABLE
    bool first = true;
    bool setFHD = true;
    int stop = 2000;
    while (t < tEnd && stop-- > 0) {
        samplePos = entryPoint + t * rayDirection;
        voxel = getNormalizedVoxel(volume, volumeParameters, samplePos);
        hsv = rgb2hsv(voxel.rgb);
        float hue = hsv.x; //(1.0 - hsv.y) * hsv.z;
        SET_ISO_VALUES
        sim[NUM_CLASSES] = voxel.x;
        color[NUM_CLASSES] = applyTF(rawTransferFunction, voxel.x);
        grad[NUM_CLASSES] = gradientCentralDiff(voxel, volume, volumeParameters, samplePos, 0);

        // macro defined in MultichanlnelRaycaster::initializeResources()
        APPLY_NTFS
        result = DRAW_BACKGROUND(result, t, tIncr, backgroundColor, bgTDepth, tDepth);
        result = DRAW_PLANES(result, samplePos, rayDirection, tIncr, positionindicator, t, tDepth);

        // World space position
        vec3 worldSpacePosition = (volumeParameters.textureToWorld * vec4(samplePos, 1.0)).xyz;
        // DVR standard 1d TF
        if(color[NUM_CLASSES].a > 0.01) {
            color[NUM_CLASSES].rgb = APPLY_LIGHTING(lighting, color[NUM_CLASSES].rgb, color[NUM_CLASSES].rgb, vec3(1.0),
                worldSpacePosition, normalize(-grad[NUM_CLASSES]), toCameraDir);
            result = APPLY_COMPOSITING(result, color[NUM_CLASSES], samplePos, vec4(sim[NUM_CLASSES]), gradient, camera,
                                            raycaster.isoValue, t, tDepth, tIncr);
            //result.rgb = result.rgb + (1.0 - result.a) * color[NUM_CLASSES].a * color[NUM_CLASSES].rgb;
            //result.a   = result.a   + (1.0 - result.a) * color[NUM_CLASSES].a;
        }
        // Render NTF Iso Surfaces
        for (int i = 0; i < NUM_CLASSES; ++i) {
            if (alpha[i] < 0.01 || surfaceFound[i]) {continue;}
            if (first) {grad[i] = -entryNormal; }
            float diff = alpha[i] - isoValue[i];
            bool sampInside = alpha[i] > isoValue[i];
            if (abs(diff) < 0.01 || (sampInside && first)) {
                color[i].rgb = APPLY_LIGHTING(lighting, color[i].rgb, color[i].rgb, vec3(1.0),
                    worldSpacePosition, normalize(-grad[i]), toCameraDir);
                result.rgb = result.rgb + (1.0 - result.a) * color[i].a * color[i].rgb;
                result.a   = result.a   + (1.0 - result.a) * color[i].a;
                // result.rgb = 0.5*normalize(grad[i]) + 0.5;
                tDepth = t;
                surfaceFound[i] = true;
                tIncr = tEnd / samples;
                break;
            } else if (sampInside) {
                // result = vec4(1.0f, 0.0f ,0.0f, 0.5f);
                t -= tIncr;
                tIncr /= 2.0;
                break;
            }
        }
        first = false;
        // early ray termination
        if (result.a > ERT_THRESHOLD) {
            t = tEnd;
        } else {
            t += tIncr;
        }
    }

    if (bgTDepth > tEnd) {
        result =
            DRAW_BACKGROUND(result, bgTDepth, tIncr, backgroundColor, bgTDepth, tDepth);
    }


    if (tDepth == -1.0) {
        tDepth = 1.0;
    }

    gl_FragDepth = min(backgroundDepth, tDepth);
    return result;
}

void main() {
    vec2 texCoords = gl_FragCoord.xy * outportParameters.reciprocalDimensions;
    vec3 entryPoint = texture(entryColor, texCoords).rgb;
    vec3 exitPoint = texture(exitColor, texCoords).rgb;

    vec4 color = vec4(0);

    float backgroundDepth = 1;
#ifdef BACKGROUND_AVAILABLE
    color = texture(bgColor, texCoords);
    gl_FragDepth = backgroundDepth = texture(bgDepth, texCoords).x;
    PickingData = texture(bgPicking, texCoords);
#else // BACKGROUND_AVAILABLE
    PickingData = vec4(0);
    if (entryPoint == exitPoint) {
        discard;
    }
#endif // BACKGROUND_AVAILABLE
    if (entryPoint != exitPoint) {
        vec3 normal = useNormals ? texture(entryNormal, texCoords).xyz : vec3(0,0,0);
        color = rayTraversal(entryPoint, exitPoint, texCoords, backgroundDepth, normal);
    }
    FragData0 = color;
}
