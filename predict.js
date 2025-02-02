const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');

let predictions = [];
const classifier = knnClassifier.create();
let sentence = [];
let sentences = []; // Store multiple sentences
let modelLoaded = false; // Prevent double loading

const LABELS_MAP = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
    'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'Stop': 26
};

async function loadModel() {
    if (modelLoaded) {
        console.warn("⚠️ Model already loaded. Skipping reloading...");
        return;
    }

    console.log("📥 Loading trained model...");
    try {
        const modelResponse = await fetch('asl_modelv4.json');
        const modelData = await modelResponse.json();

        classifier.clearAllClasses(); // Clear before loading

        Object.keys(modelData).forEach(key => {
            const data = modelData[key];
            const tensor = tf.tensor(data, [data.length / 63, 63]); // Reshape properly
            classifier.setClassifierDataset({ ...classifier.getClassifierDataset(), [key]: tensor });

            console.log(`✅ Loaded class ${key} with ${data.length / 63} examples.`);
        });

        modelLoaded = true; // Mark as loaded
        console.log("✅ Model loaded successfully.");
    } catch (error) {
        console.error("🚨 Model loading failed:", error);
    }
}

async function setup() {
    console.log("🔧 Setup initializing...");
    await loadModel(); // Ensure model is loaded only once

    const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.1/${file}`
    });

    hands.setOptions({
        maxNumHands: 2,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    hands.onResults((results) => onResults(results));

    const camera = new Camera(videoElement, {
        onFrame: async () => await hands.send({ image: videoElement }),
        width: 720,
        height: 630
    });

    camera.start().then(() => console.log("🎥 Camera started.")).catch(console.error);

    videoElement.style.display = "none";
    canvasElement.style.display = "block";

    document.getElementById('startTranslating').addEventListener('click', classify);
}

function onResults(results) {
    predictions = results || {};
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    if (results.image) {
        canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    }

    if (results.multiHandLandmarks) {
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 5 });
            drawLandmarks(canvasCtx, landmarks, { color: '#FF0000', lineWidth: 2 });
        }
    }
}

async function classify() {
    if (classifier.getNumClasses() <= 0) {
        console.error("🚨 No trained signs detected!");
        alert("Please load the trained model first!");
        return;
    }

    if (!predictions.multiHandLandmarks || predictions.multiHandLandmarks.length === 0) {
        console.warn("⚠️ No hand detected, retrying...");
        setTimeout(classify, 2000);
        return;
    }

    const features = predictions.multiHandLandmarks[0].flatMap(lm => [lm.x, lm.y, lm.z]);
    const tensorFeatures = tf.tensor(features, [1, 63]); // Reshape

    try {
        const results = await classifier.predictClass(tensorFeatures);
        console.log("✅ Prediction complete!", results);

        const predictedLabel = results.label;
        const confidence = results.confidences[predictedLabel] * 100;

        if (!predictedLabel) return;

        console.log(`🎯 Recognized Sign: ${predictedLabel}`);
        console.log(`📈 Confidence: ${confidence.toFixed(2)}%`);

        document.getElementById('result').innerText = predictedLabel;
        document.getElementById('confidence').innerText = confidence.toFixed(2);

        // ✅ Only add letter to sentence if confidence is 100%
        if (confidence === 100) {
            if (predictedLabel !== 'Stop') {
                if (sentence.length === 0 || sentence[sentence.length - 1] !== predictedLabel) {
                    sentence.push(predictedLabel);
                    console.log(`📝 Sentence so far: ${sentence.join(' ')}`);
                }
            } else {
                // Stop detected → Save sentence & reset
                sentences.push(sentence.join(' ') + '.');
                sentence = [];
                console.log(`🛑 Sentences: ${sentences.join('\n')}`);
            }
        } else {
            console.warn(`⚠️ Prediction confidence is ${confidence.toFixed(2)}%, waiting for 100%`);
        }

        document.getElementById('sentenceOutput').innerText = sentences.join('\n');
    } catch (error) {
        console.error("🚨 Prediction error:", error);
    }

    setTimeout(classify, 1000);
}


setup();
