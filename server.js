const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const ort = require('onnxruntime-node');
const sharp = require('sharp');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 5000;

const HEWAN_INFO = {
    'sapi': {
        nama: 'Sapi',
        emoji: '🐄',
        hukum: 'Sunnah Muakkad',
        syarat: 'Minimal 2 tahun, sehat, tidak cacat',
        dalil: 'QS. Al-Hajj: 36',
        niat: 'Niatkan qurban karena Allah SWT',
        jumlah: '1 ekor untuk 7 orang'
    },

    'kerbau': {
        nama: 'Kerbau',
        emoji: '🐃',
        hukum: 'Sunnah Muakkad',
        syarat: 'Minimal 2 tahun, sehat, tidak cacat',
        dalil: 'QS. Al-Hajj: 36',
        niat: 'Niatkan qurban karena Allah SWT',
        jumlah: '1 ekor untuk 7 orang'
    },

    'kambing': {
        nama: 'Kambing',
        emoji: '🐐',
        hukum: 'Sunnah Muakkad',
        syarat: 'Minimal 1 tahun (masuk tahun ke-2), sehat',
        dalil: 'QS. Al-Kautsar: 2',
        niat: 'Niatkan qurban karena Allah SWT',
        jumlah: '1 ekor untuk 1 orang'
    },

    'domba': {
        nama: 'Domba',
        emoji: '🐑',
        hukum: 'Sunnah Muakkad',
        syarat: 'Minimal 1 tahun (masuk tahun ke-2), sehat',
        dalil: 'QS. Al-Kautsar: 2',
        niat: 'Niatkan qurban karena Allah SWT',
        jumlah: '1 ekor untuk 1 orang'
    },

    'unta': {
        nama: 'Unta',
        emoji: '🐪',
        hukum: 'Sunnah Muakkad',
        syarat: 'Minimal 5 tahun, sehat, tidak cacat',
        dalil: 'QS. Al-Hajj: 36',
        niat: 'Niatkan qurban karena Allah SWT',
        jumlah: '1 ekor untuk 7 orang'
    }
};

const CLASSES = [
    'kerbau',
    'kambing',
    'sapi',
    'domba',
    'unta'
];

app.use(cors());

app.use(
    express.static(
        path.join(__dirname, 'public')
    )
);

const upload = multer({
    storage: multer.memoryStorage(),
    limits: {
        fileSize: 10 * 1024 * 1024
    }
});

let session;
let inputName;
let outputName;

async function startServer() {

    try {

        console.log(
            '[STARTUP] Memuat model ONNX...'
        );

        const modelPath = path.join(
            __dirname,
            'model_int8.onnx'
        );

        console.log(
            '[CHECK] File model ada?',
            fs.existsSync(modelPath)
        );

        session =
            await ort.InferenceSession.create(
                modelPath
            );

        inputName = session.inputNames[0];
        outputName = session.outputNames[0];

        console.log(
            '[STARTUP] Model siap!'
        );

        console.log(
            `[SERVER] Jalankan di browser: http://localhost:${PORT}`
        );

        app.listen(PORT);

    } catch (err) {

        console.error(
            '[ERROR] Gagal memuat model:',
            err
        );
    }
}

async function preprocessImage(imageBuffer) {

    const rawData = await sharp(imageBuffer)
        .resize(224, 224)
        .removeAlpha()
        .raw()
        .toBuffer();

    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    const input = new Float32Array(
        1 * 3 * 224 * 224
    );

    for (let i = 0; i < 224 * 224; i++) {

        const r = rawData[i * 3] / 255.0;
        const g = rawData[i * 3 + 1] / 255.0;
        const b = rawData[i * 3 + 2] / 255.0;

        // RED
        input[i] =
            (r - mean[0]) / std[0];

        // GREEN
        input[224 * 224 + i] =
            (g - mean[1]) / std[1];

        // BLUE
        input[2 * 224 * 224 + i] =
            (b - mean[2]) / std[2];
    }

    return new ort.Tensor(
        'float32',
        input,
        [1, 3, 224, 224]
    );
}

function softmax(arr) {

    const maxVal = Math.max(...arr);

    const expArr = arr.map(x =>
        Math.exp(x - maxVal)
    );

    const sumExp = expArr.reduce(
        (a, b) => a + b,
        0
    );

    return expArr.map(x => x / sumExp);
}

app.post(
    '/predict',
    upload.single('file'),

    async (req, res) => {

        if (!req.file) {

            return res.status(400).json({
                error: 'Tidak ada file'
            });
        }

        try {

            const inputTensor =
                await preprocessImage(
                    req.file.buffer
                );

            const feeds = {};

            feeds[inputName] =
                inputTensor;

            const results =
                await session.run(feeds);

            const probabilities =
                softmax(
                    Array.from(
                        results[outputName].data
                    )
                );

            const semua_prob =
                CLASSES.map((cls, idx) => ({
                    kelas: cls,
                    nama:
                        HEWAN_INFO[cls].nama,
                    emoji:
                        HEWAN_INFO[cls].emoji,
                    probabilitas:
                        probabilities[idx] * 100
                }))
                .sort(
                    (a, b) =>
                        b.probabilitas -
                        a.probabilitas
                );

            // HASIL TERBAIK
            const hasil =
                semua_prob[0];

            res.json({

                prediksi:
                    hasil.kelas,

                confidence:
                    hasil.probabilitas,

                semua_prob,

                info:
                    HEWAN_INFO[
                        hasil.kelas
                    ]
            });

        } catch (error) {

            console.error(
                '[ERROR] Detail:',
                error
            );

            res.status(500).json({

                error:
                    'Gagal memproses gambar: ' +
                    error.message
            });
        }
    }
);

const downloadModel = require("./download-model");

(async () => {
    await downloadModel();

    // baru load ONNX setelah model ada
    const ort = require("onnxruntime-node");

    const session = await ort.InferenceSession.create(
        "./models/model_int8.onnx"
    );

    console.log("Model siap digunakan 🚀");
})();


startServer();