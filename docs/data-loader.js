/**
 * GeoTIFF Data Loader for LandScan Population Density
 */

class DataLoader {
    constructor() {
        this.cache = new Map();
    }

    async loadGeoTIFF(file) {
        try {
            const arrayBuffer = await file.arrayBuffer();
            const tiff = await GeoTIFF.fromArrayBuffer(arrayBuffer);
            const image = await tiff.getImage();
            const rasters = await image.readRasters();

            // Get metadata
            const width = image.getWidth();
            const height = image.getHeight();
            const bbox = image.getBoundingBox();

            const data = {
                raster: rasters[0], // First band
                width,
                height,
                bbox,
                filename: file.name
            };

            return data;
        } catch (error) {
            console.error('Failed to load GeoTIFF:', error);
            throw error;
        }
    }

    resampleToGrid(data, targetSize) {
        const { raster, width, height } = data;
        const resampled = new Float32Array(targetSize * targetSize);

        const scaleX = width / targetSize;
        const scaleY = height / targetSize;

        for (let i = 0; i < targetSize; i++) {
            for (let j = 0; j < targetSize; j++) {
                const srcX = Math.floor(j * scaleX);
                const srcY = Math.floor(i * scaleY);
                const srcIdx = srcY * width + srcX;

                let value = raster[srcIdx];

                // Handle NaN and negative values
                if (isNaN(value) || value < 0) value = 0;

                resampled[i * targetSize + j] = value;
            }
        }

        // Normalize to preserve total population
        const sum = resampled.reduce((a, b) => a + b, 0);
        if (sum > 0) {
            for (let i = 0; i < resampled.length; i++) {
                resampled[i] /= sum;
            }
        }

        return resampled;
    }

    extractRegion(data, bbox) {
        // Extract specific geographic region
        // bbox: { west, south, east, north }
        const { raster, width, height, bbox: fullBbox } = data;

        const xStart = Math.floor(((bbox.west - fullBbox[0]) / (fullBbox[2] - fullBbox[0])) * width);
        const xEnd = Math.floor(((bbox.east - fullBbox[0]) / (fullBbox[2] - fullBbox[0])) * width);
        const yStart = Math.floor(((bbox.north - fullBbox[3]) / (fullBbox[1] - fullBbox[3])) * height);
        const yEnd = Math.floor(((bbox.south - fullBbox[3]) / (fullBbox[1] - fullBbox[3])) * height);

        const regionWidth = xEnd - xStart;
        const regionHeight = yEnd - yStart;
        const region = new Float32Array(regionWidth * regionHeight);

        for (let i = 0; i < regionHeight; i++) {
            for (let j = 0; j < regionWidth; j++) {
                const srcIdx = (yStart + i) * width + (xStart + j);
                region[i * regionWidth + j] = raster[srcIdx];
            }
        }

        return {
            raster: region,
            width: regionWidth,
            height: regionHeight
        };
    }

    computeStatistics(data) {
        const { raster } = data;
        const valid = Array.from(raster).filter(v => !isNaN(v) && v >= 0);

        if (valid.length === 0) {
            return { min: 0, max: 0, mean: 0, std: 0 };
        }

        const min = Math.min(...valid);
        const max = Math.max(...valid);
        const mean = valid.reduce((a, b) => a + b, 0) / valid.length;
        const variance = valid.reduce((a, b) => a + (b - mean) ** 2, 0) / valid.length;
        const std = Math.sqrt(variance);

        return { min, max, mean, std };
    }

    async loadMultipleYears(files) {
        // Load multiple GeoTIFFs for temporal analysis
        const promises = files.map(file => this.loadGeoTIFF(file));
        const datasets = await Promise.all(promises);

        // Sort by filename (assuming year in filename)
        datasets.sort((a, b) => a.filename.localeCompare(b.filename));

        return datasets;
    }

    computeTemporalChange(data1, data2, targetSize) {
        // Compute year-over-year change
        const field1 = this.resampleToGrid(data1, targetSize);
        const field2 = this.resampleToGrid(data2, targetSize);

        const change = new Float32Array(targetSize * targetSize);
        for (let i = 0; i < change.length; i++) {
            change[i] = field2[i] - field1[i];
        }

        return change;
    }

    // Compute empirical power spectrum
    computeSpectrum(field, size) {
        // Simple radial power spectrum
        const spectrum = new Float32Array(Math.floor(size / 2));
        const counts = new Uint32Array(Math.floor(size / 2));

        for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
                const ki = i < size/2 ? i : i - size;
                const kj = j < size/2 ? j : j - size;
                const k = Math.sqrt(ki*ki + kj*kj);
                const bin = Math.min(Math.floor(k), spectrum.length - 1);

                const val = field[i * size + j];
                spectrum[bin] += val * val;
                counts[bin]++;
            }
        }

        // Average
        for (let i = 0; i < spectrum.length; i++) {
            if (counts[i] > 0) {
                spectrum[bin] /= counts[i];
            }
        }

        return spectrum;
    }

    // Find dominant wavenumber from spectrum
    findDominantK(spectrum) {
        let maxIdx = 0;
        let maxVal = 0;

        // Ignore k=0 (DC component)
        for (let i = 1; i < spectrum.length; i++) {
            if (spectrum[i] > maxVal) {
                maxVal = spectrum[i];
                maxIdx = i;
            }
        }

        return maxIdx;
    }
}

export { DataLoader };
