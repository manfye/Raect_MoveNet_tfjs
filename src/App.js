import * as tfJsCore from '@tensorflow/tfjs-core';
import * as tfJsConverter from '@tensorflow/tfjs-converter';
import * as tfJsBackendWebgl from '@tensorflow/tfjs-backend-webgl';
import { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import { isMobile } from 'react-device-detect';
import * as poseDetection from '@tensorflow-models/pose-detection';
import { MOVENET_CONFIGS, LINE_WIDTH, DEFAULT_RADIUS } from "./config"
import * as ml5 from "ml5";
import modeljson from "./models/model.json";
import metadata from "./models/model_meta.json";
import asanas from "./models/asanas.json";

let detector;
let start_inference_time;
let num_of_inferences = 0;
let total_inference_time = 0;
let last_panel_update = 0;
let rafId;
let canvas;
let contex2d;
let model;
let modelType;
let brain;
let poseLabel = "pose";
let text = "oki";
let poseScore = "0";
let lastPoseLabel;

const options = {
    task: 'classification' // or 'regression'
}
const modelDetails = {
    model: modeljson,
    metadata: metadata,
    weights: "https://cdn.glitch.global/e6659bd5-94b1-4dbc-94f9-5468bd8f317d/model.weights.bin?v=1642380976991"
}

// this loads the ml5 Neural Network model with the options specified and the files uploaded
brain = ml5.neuralNetwork(options);
console.log(modelDetails)

brain.load(modelDetails, modelLoaded)

function modelLoaded() {
    // continue on your neural network journey
    // use nn.classify() for classifications or nn.predict() for regressions
}

console.log("after loading the model", brain)

const VIDEO_CONFIGS = {
    facingMode: "user",
    deviceId: "",
    frameRate: { max: 60, ideal: 30 },
    width: isMobile ? 360 : 640,
    height: isMobile ? 270 : 480
};


function App() {
    const [cameraReady, setCameraReady] = useState(false);
    const [displayFps, setDisplayFps] = useState(0);
    const webcamRef = useRef({});

    useEffect(() => {
        _loadPoseNet().then();
    }, []);

    const _loadPoseNet = async () => {
        if (rafId) {
            window.cancelAnimationFrame(rafId);
            detector.dispose();
        }

        detector = await createDetector();
        await renderPrediction();
    }

    const createDetector = async () => {
        model = poseDetection.SupportedModels.MoveNet;
        modelType = poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING; //or SINGLEPOSE_THUNDER
        return await poseDetection.createDetector(model, { modelType: modelType });
    }

    const renderPrediction = async () => {
        await renderResult();
        rafId = requestAnimationFrame(renderPrediction);
    }

    const renderResult = async () => {
        const video = webcamRef.current && webcamRef.current['video'];

        if (!cameraReady && !video) {
            return;
        }

        if (video.readyState < 2) {
            return;
        }

        startEstimatePoses();
        const poses = await detector.estimatePoses(video, {
            maxPoses: MOVENET_CONFIGS.maxPoses, //When maxPoses = 1, a single pose is detected
            flipHorizontal: false
        });
        stopEstimatePoses();
        getContexFullScreen(video);

        if (poses.length > 0) {
            getResultsFullScreen(poses);
        }
    }

    const startEstimatePoses = () => {
        start_inference_time = (performance || Date).now();
    }

    const stopEstimatePoses = () => {
        const endInferenceTime = (performance || Date).now();
        total_inference_time += endInferenceTime - start_inference_time;
        ++num_of_inferences;
        const panelUpdateMilliseconds = 1000;

        if (endInferenceTime - last_panel_update >= panelUpdateMilliseconds) {
            const averageInferenceTime = total_inference_time / num_of_inferences;
            total_inference_time = 0;
            num_of_inferences = 0;
            setDisplayFps(1000.0 / averageInferenceTime, 120);
            last_panel_update = endInferenceTime;
        }
    }

    const getContexFullScreen = (video) => {
        canvas = document.getElementById('canvas');
        contex2d = canvas.getContext('2d');

        const videoWidth = video.videoWidth;
        const videoHeight = video.videoHeight;

        video.width = videoWidth;
        video.height = videoHeight;

        canvas.width = videoWidth;
        canvas.height = videoHeight;
        contex2d.fillRect(0, 0, videoWidth, videoHeight);

        contex2d.translate(video.videoWidth, 0);
        contex2d.scale(-1, 1);
        contex2d.drawImage(video, 0, 0, videoWidth, videoHeight);
    }

    const getResultsFullScreen = (poses) => {
        for (const pose of poses) {
            getResults(pose);
        }
    }

    const getResults = (pose) => {
        if (pose.keypoints != undefined && pose.keypoints != null) {
            let inputs = [];
            for (let i = 0; i < pose.keypoints.length; i++) {
                let x = pose.keypoints[i].x;
                let y = pose.keypoints[i].y;
                inputs.push(x);
                inputs.push(y);
            }
            getKeyPoints(pose.keypoints);
            drawSkeleton(pose.keypoints);
            brain.classify(inputs, handleclassify);
            // console.log("classify", brain)
        }

    }

    function handleclassify(error, results) {
        console.log("resulst",results)
        if (results && results[0].confidence > 0.70) {
            poseLabel = results[0].label;
            poseScore = results[0].confidence.toFixed(2);
        
            // this tells it to run the writePose & writeInfo functions
            if (lastPoseLabel !== poseLabel) {
              console.log("we got to this point")
            }
            lastPoseLabel = results[0].label;
          }
          
          // here it calls for classifyPose again with a timeout
    
          setTimeout(getResults, 1000);
    }

    function handleResults(error, result) {
        if (error) {
            console.error(error);
            return;
        }
        console.log(result); // {label: 'red', confidence: 0.8};
    }



    const getKeyPoints = (keypoints) => {
        const keypointInd = poseDetection.util.getKeypointIndexBySide(model);
        contex2d.fillStyle = '#fff';
        contex2d.strokeStyle = '#fff';
        contex2d.lineWidth = LINE_WIDTH;

        for (const i of keypointInd.middle) {
            getKeyPoint(keypoints[i]);
        }

        contex2d.fillStyle = '#00ff00';

        for (const i of keypointInd.left) {
            getKeyPoint(keypoints[i]);
        }

        contex2d.fillStyle = '#ffff00';

        for (const i of keypointInd.right) {
            getKeyPoint(keypoints[i]);
        }
    }

    const getKeyPoint = (keypoint) => {
        // If score is null, just show the keypoint.
        const score = keypoint.score != null ? keypoint.score : 1;
        const scoreThreshold = MOVENET_CONFIGS.scoreThreshold || 0;

        if (score >= scoreThreshold) {
            const circle = new Path2D();
            circle.arc(keypoint.x, keypoint.y, DEFAULT_RADIUS, 0, 2 * Math.PI);
            contex2d.fill(circle);
            contex2d.stroke(circle);
        }
    }

    const drawSkeleton = (keypoints) => {
        contex2d.fillStyle = 'White';
        contex2d.strokeStyle = 'White';
        contex2d.lineWidth = LINE_WIDTH;
        poseDetection.util.getAdjacentPairs(model).forEach(([i, j]) => {
            const keypoint1 = keypoints[i];
            const keypoint2 = keypoints[j]; // If score is null, just show the keypoint.

            const score1 = keypoint1.score != null ? keypoint1.score : 1;
            const score2 = keypoint2.score != null ? keypoint2.score : 1;
            const scoreThreshold = MOVENET_CONFIGS.scoreThreshold || 0;

            if (score1 >= scoreThreshold && score2 >= scoreThreshold) {
                contex2d.beginPath();
                contex2d.moveTo(keypoint1.x, keypoint1.y);
                contex2d.lineTo(keypoint2.x, keypoint2.y);
                contex2d.stroke();
            }
        });
    }

    const onUserMediaError = () => {
        console.log('ERROR Occured');
    };

    const onUserMedia = () => {
        console.log('Camera loaded!');
        setCameraReady(true);
    };

    return (
        <section >
            <div>
                <Webcam
                    style={{ visibility: "hidden" }}
                    ref={webcamRef}
                    audio={false}
                    height={isMobile ? 270 : 480}
                    width={isMobile ? 360 : 640}
                    videoConstraints={VIDEO_CONFIGS}
                    onUserMediaError={onUserMediaError}
                    onUserMedia={onUserMedia} />
                <canvas id="canvas" />

            </div>
        </section>
    );
}

export default App;
