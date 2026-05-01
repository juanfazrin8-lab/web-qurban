const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const ort = require('onnxruntime-node');
const sharp = require('sharp');

const app = express();
const PORT = 5000;

const HEWAN_INFO = {
    'sapi':   { nama: 'Sapi',   emoji: '🐄', hukum: 'Sunnah Muakkad', syarat: 'Minimal 2 tahun, sehat, tidak cacat', dalil: 'QS. Al-Hajj: 36', niat: 'Niatkan qurban karena Allah SWT', jumlah: '1 ekor untuk 7 orang' },
    'kerbau': { nama: 'Kerbau', emoji: '🐃', hukum: 'Sunnah Muakkad', syarat: 'Minimal 2 tahun, sehat, tidak cacat', dalil: 'QS. Al-Hajj: 36', niat: 'Niatkan qurban karena Allah SWT', jumlah: '1 ekor untuk 7 orang' },
    'kambing': { nama: 'Kambing', emoji: '🐐', hukum: 'Sunnah Muakkad', syarat: 'Minimal 1 tahun (masuk tahun ke-2), sehat', dalil: 'QS. Al-Kautsar: 2', niat: 'Niatkan qurban karena Allah SWT', jumlah: '1 ekor untuk 1 orang' },
    'domba':  { nama: 'Domba',  emoji: '🐑', hukum: 'Sunnah Muakkad', syarat: 'Minimal 1 tahun (masuk tahun ke-2), sehat', dalil: 'QS. Al-Kautsar: 2', niat: 'Niatkan qurban karena Allah SWT', jumlah: '1 ekor untuk 1 orang' },
    'unta':   { nama: 'Unta',  emoji: '🐪', hukum: 'Sunnah Muakkad', syarat: 'Minimal 5 tahun, sehat, tidak cacat', dalil: 'QS. Al-Hajj: 36', niat: 'Niatkan qurban karena Allah SWT', jumlah: '1 ekor untuk 7 orang' }
};

const CLASSES = ['kerbau', 'kambing', 'sapi', 'domba', 'unta'];

app.use(cors());
app.use(express.static(path.join(__dirname, 'public')));

const upload = multer({ 
    storage: multer.memoryStorage(),
    limits: { fileSize: 10 * 1024 * 1024 } 
});

let session;
let inputName;
let outputName;

async function startServer() {
    try {
        console.log('[STARTUP] Memuat model ONNX...');
        session = await ort.InferenceSession.create(path.join(__dirname, 'model_gabungan_super.onnx'));
        
        inputName = session.inputNames[0];
        outputName = session.outputNames[0];
        
        const dummy = new ort.Tensor('float32', new Float32Array(3 * 224 * 224).fill(0), [1, 3, 224, 224]);
        await session.run({ [inputName]: dummy });
        
        console.log('[STARTUP] Model siap!');
        app.listen(PORT, () => console.log(`[SERVER] Jalankan di browser: http://localhost:${PORT}`));
    } catch (err) {
        console.error('[ERROR] Gagal memuat model:', err);
    }
}

async function preprocessImage(imageBuffer) {
    const rawData = await sharp(imageBuffer)
        .resize(224, 224, { fit: 'fill' }) 
        .removeAlpha()
        .raw()
        .toBuffer();

    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    const float32Data = new Float32Array(3 * 224 * 224);
    
    for (let i = 0; i < 224 * 224; i++) {
        float32Data[i] = (rawData[i * 3] / 255.0 - mean[0]) / std[0];
        float32Data[224 * 224 + i] = (rawData[i * 3 + 1] / 255.0 - mean[1]) / std[1];
        float32Data[2 * 224 * 224 + i] = (rawData[i * 3 + 2] / 255.0 - mean[2]) / std[2];
    }
    
    return new ort.Tensor('float32', float32Data, [1, 3, 224, 224]);
}

function softmax(arr) {
    const maxVal = Math.max(...arr);
    const expArr = arr.map(x => Math.exp(x - maxVal));
    const sumExp = expArr.reduce((a, b) => a + b, 0);
    return expArr.map(x => x / sumExp);
}

app.post('/predict', upload.single('file'), async (req, res) => {
    if (!req.file) return res.status(400).json({ error: 'Tidak ada file' });
    
    try {
        const inputTensor = await preprocessImage(req.file.buffer);
        const feeds = {};
        feeds[inputName] = inputTensor;
        const results = await session.run(feeds);
        
        const probabilities = softmax(Array.from(results[outputName].data));
        
        let semua_prob = CLASSES.map((cls, idx) => ({
            kelas: cls, nama: HEWAN_INFO[cls].nama, emoji: HEWAN_INFO[cls].emoji,
            probabilitas: probabilities[idx] * 100
        })).sort((a, b) => b.probabilitas - a.probabilitas);
               // JIKA KEYAKINAN AI DIBAWAH 60%, BERARTI BUKAN HEWAN QURBAN!
        if (semua_prob[0].probabilitas < 80.0) {
            return res.json({
                prediksi: 'unknown',
                confidence: semua_prob[0].probabilitas,
                error: 'Maaf, gambar ini bukan termasuk 5 hewan qurban yang dikenali sistem.'
            });
        }

        res.json({
            prediksi: semua_prob[0].kelas,
            confidence: semua_prob[0].probabilitas,
            semua_prob,
            info: HEWAN_INFO[semua_prob[0].kelas]
        });
    } catch (error) {
        console.error('[ERROR] Detail:', error.message);
        res.status(500).json({ error: 'Gagal memproses gambar: ' + error.message });
    } 
});

startServer();