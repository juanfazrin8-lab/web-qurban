const axios = require("axios");
const fs = require("fs");
const path = require("path");

const URL = "https://drive.google.com/uc?export=download&id=1fW7ZbIzCXO7YAiMSEvWA_CM6sSRJih_3";

const output = path.join(__dirname, "models/model_int8.onnx");

async function downloadModel() {
    console.log("[DOWNLOAD] Ambil model dari Google Drive...");

    const res = await axios({
        url: URL,
        method: "GET",
        responseType: "stream",
    });

    const writer = fs.createWriteStream(output);
    res.data.pipe(writer);

    return new Promise((resolve, reject) => {
        writer.on("finish", () => {
            console.log("[OK] Model berhasil di-download ✔");
            resolve();
        });
        writer.on("error", reject);
    });
}

module.exports = downloadModel;