importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest');

const MODEL_PATH = `yolov5n_web_model/model.json`;
const LABELS_PATH = `yolov5n_web_model/labels.json`;
const INPUT_MODEL_DIMENSIONS = 640;
const CLASS_THRESHOLD = 0.4;

let _labels = [];
let _model = null;

async function loadModelAndLabels() {
    await tf.ready();

    _labels = await (await fetch(LABELS_PATH)).json();
    _model = await tf.loadGraphModel(MODEL_PATH);

    // warmup
    const dummyInput = tf.ones(_model.inputs[0].shape);
    await _model.executeAsync(dummyInput);
    tf.dispose(dummyInput);

    postMessage({ type: 'model-loaded' })
    
}
function preprocessImage(input) {
    return tf.tidy(() => {
        const image = tf.browser.fromPixels(input);

        return tf.image
            .resizeBilinear(image, [INPUT_MODEL_DIMENSIONS, INPUT_MODEL_DIMENSIONS])
            .div(255)
            .expandDims(0);
    });
}

self.onmessage = async ({ data }) => {
    if (data.type !== 'predict') return

    postMessage({
        type: 'prediction',
        x: 400,
        y: 400,
        score: 0
    });


};

console.log('🧠 YOLOv5n Web Worker initialized');
