const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');

let predictions = [];
const classifier = knnClassifier.create();
let state = 'waiting';
let currentLabel = '';
let sentence = [];
let lastPredictedLabel = ''; // Stores the last predicted label for correction

const LABELS_MAP = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
    'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15,
    'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
    'Y': 24, 'Z': 25, 'Stop': 26
};

function setup() {
    console.log("🔧 Setup initialized...");

    const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.1/${file}`
    });

    hands.setOptions({
        maxNumHands: 2,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    hands.onResults(onResults);

    const camera = new Camera(videoElement, {
        onFrame: async () => {
            await hands.send({ image: videoElement });
        },
        width: 720,
        height: 630
    });

    camera.start();
    videoElement.style.display = "none";
    createButtons();
}

function onResults(results) {
    predictions = results || {};

    if (state === 'collecting') {
        addExample(currentLabel);
    }

    canvasCtx.save();
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
    canvasCtx.restore();
}

function addExample(label) {
    if (predictions.multiHandLandmarks && predictions.multiHandLandmarks.length > 0) {
        console.log(`🟡 Training started for: ${label}`);

        const features = predictions.multiHandLandmarks[0].map(lm => [lm.x, lm.y, lm.z]);
        const tensorFeatures = tf.tensor(features).reshape([1, features.length * 3]);

        classifier.addExample(tensorFeatures, label);
        console.log(`🟢 Training complete for: ${label}`);
    } else {
        console.warn("⚠️ No hand detected. Try again.");
    }
}

async function classify() {
    if (classifier.getNumClasses() <= 0) {
        alert("Please train the model first!");
        return;
    }

    if (predictions.multiHandLandmarks && predictions.multiHandLandmarks.length > 0) {
        const features = predictions.multiHandLandmarks[0].map(lm => [lm.x, lm.y, lm.z]);
        const tensorFeatures = tf.tensor(features).reshape([1, features.length * 3]);

        const results = await classifier.predictClass(tensorFeatures);

        if (results.label) {
            lastPredictedLabel = results.label;
            document.getElementById('result').innerText = results.label;
            document.getElementById('confidence').innerText = (results.confidences[results.label] * 100).toFixed(2);

            if (results.label !== 'Stop') {
                if (sentence.length === 0 || sentence[sentence.length - 1] !== results.label) {
                    sentence.push(results.label);
                }
            } else {
                document.getElementById('sentenceOutput').innerText = sentence.join(' ');
                sentence = [];
            }
        }

        setTimeout(classify, 1000);
    } else {
        setTimeout(classify, 1000);
    }
}

function retrain(correctLabel) {
    if (!predictions.multiHandLandmarks || predictions.multiHandLandmarks.length === 0) {
        alert("No hand detected! Try again.");
        return;
    }

    const features = predictions.multiHandLandmarks[0].map(lm => [lm.x, lm.y, lm.z]);
    const tensorFeatures = tf.tensor(features).reshape([1, features.length * 3]);

    classifier.addExample(tensorFeatures, correctLabel);
    console.log(`✅ Retrained on letter ${correctLabel}`);

    alert(`Model retrained on letter ${correctLabel}. Keep training until accuracy improves.`);
}

function saveModel() {
    if (classifier.getNumClasses() <= 0) {
        alert("Please train the model first!");
        return;
    }

    console.log("💾 Saving model...");
    const dataset = classifier.getClassifierDataset();
    const datasetObj = {};

    Object.keys(dataset).forEach(key => {
        datasetObj[key] = Array.from(dataset[key].dataSync());
    });

    const json = JSON.stringify(datasetObj);
    const blob = new Blob([json], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'asl_model.json';
    link.click();
    console.log("✅ Model saved successfully.");
}

function createButtons() {
    const buttonContainer = document.getElementById('trainingButtons');
    buttonContainer.innerHTML = '';

    Object.keys(LABELS_MAP).forEach(label => {
        const button = document.createElement('button');
        button.innerText = `Train: ${label}`;
        button.onclick = () => {
            console.log(`🎬 Training for ${label} started.`);
            currentLabel = label;
            state = 'collecting';
            setTimeout(() => {
                state = 'waiting';
                console.log(`✅ Training for ${label} completed.`);
            }, 5000);
        };
        buttonContainer.appendChild(button);
    });

    const predictButton = document.createElement('button');
    predictButton.innerText = 'Start Predicting';
    predictButton.onclick = classify;
    buttonContainer.appendChild(predictButton);

    const saveButton = document.createElement('button');
    saveButton.innerText = 'Save Model';
    saveButton.onclick = saveModel;
    buttonContainer.appendChild(saveButton);

    const correctButton = document.createElement('button');
    correctButton.innerText = 'Correct';
    correctButton.onclick = () => alert("✅ Prediction was correct!");
    buttonContainer.appendChild(correctButton);

    const incorrectButton = document.createElement('button');
    incorrectButton.innerText = 'Incorrect';
    incorrectButton.onclick = () => {
        const correctLetter = prompt("Enter the correct letter:");
        if (correctLetter && LABELS_MAP.hasOwnProperty(correctLetter.toUpperCase())) {
            retrain(correctLetter.toUpperCase());
        } else {
            alert("Invalid letter. Please enter a valid ASL letter.");
        }
    };
    buttonContainer.appendChild(incorrectButton);
}

setup();
